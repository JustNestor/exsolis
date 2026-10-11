//! Builds a [`Skeleton`] and joint-space geometry from a glTF document.
//!
//! This is where the node hierarchy becomes something the renderer can
//! animate:
//!
//! 1. Walk the scene graph, composing every node's rest world matrix.
//! 2. Collect the **joint set**: nodes driven by some clip, plus every
//!    ancestor needed to compose through. For `models/girl.glb` that is 62
//!    nodes out of 1908.
//! 3. Rewrite each mesh primitive's vertices into its joint's local space,
//!    baking the static part of the hierarchy into them.
//! 4. Merge primitives that share a material into one primitive.
//!
//! Because this rig is *rigid* — every primitive is a child of exactly one
//! bone — each vertex gets a single influence (`JOINTS_0 = j, WEIGHTS_0 = 1`).
//! Joint indices still go into the vertex so the GPU can skin, but there is
//! no weight blend to compute and the batch stays a single draw call.

use bevy_log::{debug, warn};
use glam::{Mat4, Vec3};

use crate::anim::{
    AnimChannel, AnimPath, AnimationClip, ClipRole, ClipStats, Joint, MAX_JOINTS, Skeleton,
    Transform,
};
use crate::gltf::{GltfMesh, GltfPrimitive, GltfVertex};

/// Joint index reserved for geometry that no clip can move.
///
/// Its palette entry is always the identity matrix, so static geometry needs
/// no special case in the shader — it just binds to joint 0.
pub const STATIC_JOINT: usize = 0;

/// One primitive after the rig has placed it in joint space.
#[derive(Clone)]
struct BakedPrimitive {
    vertices: Vec<GltfVertex>,
    indices: Vec<u32>,
    material_index: Option<usize>,
    joint: u16,
}

/// Bone-name prefixes that make up the first-person viewmodel.
///
/// Upper-arm joints the first-person arms hang from.
///
/// The arm cut starts here and walks down the joint tree (see
/// [`arm_joint_set`]), so this only has to name the two tops of the arms —
/// never the forearm, hand or finger bones, which is what made the old
/// name-prefix-only cut tear the hands off.
const ARM_ROOT_PREFIXES: &[&str] = &["LeftArm", "RightArm"];

/// Garment bones that follow the body arms but are parented to the chest
/// rather than hanging off them.
///
/// Each entry is a prefix of a bone name (`LeftArm_964`), and the trailing
/// `_<id>` that the exporter appends is what makes prefix matching stable
/// across re-exports. Bones *below* an upper-arm joint do not need an entry
/// here — [`arm_joint_set`] takes those by hierarchy.
const ARM_JOINT_PREFIXES: &[&str] = &[
    // Hair and garment arms that follow the body arms; without these the
    // sleeves and long hair would stay behind in the world.
    "H_LeftArm",
    "H_RightArm",
    "H_above_Left",
    "H_above_Right",
    "H_hm_",
    "Hoodie_Left",
    "Hoodie_Right",
];

/// True when a joint belongs to the arms.
fn is_arm_joint(name: &str) -> bool {
    ARM_JOINT_PREFIXES
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

/// A decoded, ready-to-upload model: joint-space geometry plus its skeleton.
pub struct Rig {
    pub skeleton: Skeleton,
    primitives: Vec<BakedPrimitive>,
    /// The same geometry with everything outside the arms removed, for the
    /// first-person viewmodel. `None` when no joint matched, which is
    /// reported as a warning rather than silently rendering nothing.
    pub arms: Option<Vec<BakedPrimitive>>,
    /// Where the arms pivot, in model space: the shoulder line.
    ///
    /// Applied once, as `T(-arms_pivot)` at the front of the viewmodel's
    /// instance matrix. The geometry itself is left where the body has it, and
    /// the bone palette is left exactly as the body's — see [`arm_shoulder`]
    /// for why doing it any other way explodes the mesh.
    pub arms_pivot: [f32; 3],
    /// Names of the joints the arm cut kept, after the far-away garment bones
    /// were pruned. Diagnostics and the regression tests read this.
    pub arm_joints: Vec<String>,
}

impl Rig {
    /// Decode a document into a rig.
    ///
    /// Models with no animation still come back with a one-entry skeleton
    /// (just [`STATIC_JOINT`]) and geometry baked to world space, so callers
    /// never need a separate unskinned path at the geometry level.
    pub fn build(document: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Result<Rig, String> {
        let scene = document
            .default_scene()
            .ok_or("GLTF has no default scene")?;

        let nodes: Vec<gltf::Node> = document.nodes().collect();
        let parent = parent_map(&nodes);

        // 1. Rest-pose world matrices, top-down from every scene root.
        let mut world = vec![Mat4::IDENTITY; nodes.len()];
        for root in scene.nodes().map(|n| n.index()) {
            compute_world(&nodes, root, Mat4::IDENTITY, &mut world);
        }

        // 2. Joints: animated nodes plus their ancestors, in an order where a
        //    parent always precedes its children.
        let animated = animated_nodes(document);
        let (joints, joint_of_node) = assign_joints(&nodes, &parent, &animated, &world)?;

        // 3. Geometry, per mesh-bearing node, rewritten into joint space.
        let mut primitives = Vec::new();
        for (index, node) in nodes.iter().enumerate() {
            let Some(mesh) = node.mesh() else { continue };
            let joint = joint_of_node[index];
            // Vertices are baked into *model* space (the space the rest pose
            // lives in), which is what `palette[j] = posed_world *
            // inverse(bind_world)` consumes. Baking into joint-local space
            // instead would need `inverse(world[joint])` folded into the
            // bind matrix, and silently squashes the model instead.
            let to_joint = world[index];
            // Take the upper-left 3x3 of the inverse-transpose by hand: glam's
            // `Mat4` has no `mat3()` in this version.
            let it = to_joint.inverse().transpose().to_cols_array_2d();
            let normal_matrix = glam::Mat3::from_cols_array(&[
                it[0][0], it[0][1], it[0][2], it[1][0], it[1][1], it[1][2], it[2][0], it[2][1],
                it[2][2],
            ]);

            // A node with a negative-determinant world matrix (a mirrored
            // part) flips triangle winding once its vertices are baked. The
            // indices must be reversed to compensate, or back-face culling
            // throws away the outside of the part and shows its inside.
            let mirrored = to_joint.determinant() < 0.0;

            // Skinned exports (one mesh, a `skin`, JOINTS_0/WEIGHTS_0) store
            // vertices already in bind space, where the rest pose IS model
            // space, and the mesh node's own transform is ignored by the
            // spec. The engine skins rigidly (one joint per vertex), so each
            // vertex goes to its dominant influence; for box-modelled
            // avatars that is weight 1.0 anyway.
            let skin_nodes: Option<Vec<usize>> = node
                .skin()
                .map(|skin| skin.joints().map(|j| j.index()).collect());

            for primitive in mesh.primitives() {
                if let Some(skin_nodes) = &skin_nodes {
                    let loaded = crate::gltf::load_gltf_primitive(&primitive, buffers)?;
                    let reader = primitive.reader(|b| Some(&buffers[b.index()]));
                    let joints_attr: Option<Vec<[u16; 4]>> =
                        reader.read_joints(0).map(|j| j.into_u16().collect());
                    let weights_attr: Option<Vec<[f32; 4]>> =
                        reader.read_weights(0).map(|w| w.into_f32().collect());
                    if let (Some(js), Some(ws)) = (joints_attr, weights_attr)
                        && js.len() == loaded.vertices.len()
                        && ws.len() == loaded.vertices.len()
                    {
                        let mut vertices = loaded.vertices;
                        for (v, (j, w)) in vertices.iter_mut().zip(js.iter().zip(ws.iter())) {
                            let mut best = 0;
                            for k in 1..4 {
                                if w[k] > w[best] {
                                    best = k;
                                }
                            }
                            v.joint = skin_nodes
                                .get(j[best] as usize)
                                .and_then(|&n| joint_of_node[n])
                                .unwrap_or(STATIC_JOINT)
                                as u16;
                        }
                        // Blockbench pads every mesh element with zero-area
                        // triangles; drop them now that each vertex has its
                        // joint, so they never reach the GPU.
                        let mut indices = loaded.indices;
                        drop_degenerate_triangles(&vertices, &mut indices);
                        primitives.extend(split_by_joint(vertices, indices, loaded.material_index));
                        continue;
                    }
                }
                let mut loaded = crate::gltf::load_gltf_primitive(&primitive, buffers)?;
                if mirrored {
                    for tri in loaded.indices.chunks_exact_mut(3) {
                        tri.swap(1, 2);
                    }
                }
                primitives.push(BakedPrimitive {
                    vertices: loaded
                        .vertices
                        .into_iter()
                        .map(|v| GltfVertex {
                            position: to_joint
                                .transform_point3(Vec3::from_array(v.position))
                                .to_array(),
                            joint: joint.unwrap_or(STATIC_JOINT) as u16,
                            // Normals need the inverse-transpose, not the
                            // transform: baked-in non-uniform scale is
                            // legal glTF and reusing the transform here
                            // reads as broken shading.
                            normal: (normal_matrix * Vec3::from_array(v.normal))
                                .normalize()
                                .to_array(),
                            ..v
                        })
                        .collect(),
                    indices: loaded.indices,
                    material_index: loaded.material_index,
                    joint: joint.unwrap_or(STATIC_JOINT) as u16,
                });
            }
        }

        // 4. Merge by material: the girl is single-material, so 795
        //    primitives collapse to one draw call.
        let (arms, arms_pivot, arm_joints) = match build_arms(&joints, &world, &primitives) {
            Some((arms, pivot, arm_joints)) => (Some(arms), pivot, arm_joints),
            None => (None, [0.0; 3], Vec::new()),
        };
        let primitives = merge_by_material(primitives);

        let skeleton = Skeleton {
            joints: joints.iter().map(|j| j.joint.clone()).collect(),
            clips: decode_clips(document, buffers, &joint_of_node, &joints)?,
            upper_body: upper_body_flags(&joints),
        };

        debug!(
            "rig: {} joints, {} merged meshes (from {} primitives), {} clips",
            skeleton.joints.len(),
            primitives.len(),
            primitives
                .iter()
                .map(|p| p.joint as usize)
                .collect::<Vec<_>>()
                .len(),
            skeleton.clips.len()
        );

        Ok(Rig {
            skeleton,
            primitives,
            arms,
            arms_pivot,
            arm_joints,
        })
    }

    /// Arms-only geometry for the first-person viewmodel.
    pub fn into_viewmodels(self) -> (Vec<GltfMesh>, Option<Vec<GltfMesh>>) {
        let arms = self.arms.map(merge_by_material).map(to_meshes);
        (to_meshes(self.primitives), arms)
    }

    /// Take the skeleton out, leaving the geometry. [`Rig::meshes`] consumes
    /// what is left.
    pub fn into_parts(self) -> (Skeleton, Vec<GltfMesh>, Option<Vec<GltfMesh>>) {
        let skeleton = self.skeleton.clone();
        let (body, arms) = self.into_viewmodels();
        (skeleton, body, arms)
    }

    /// Flatten the rig into engine meshes, one per material.
    pub fn meshes(self) -> Vec<GltfMesh> {
        to_meshes(self.primitives)
    }
}

fn to_meshes(primitives: Vec<BakedPrimitive>) -> Vec<GltfMesh> {
    primitives
        .into_iter()
        .enumerate()
        .map(|(i, p)| GltfMesh {
            name: Some(format!("rig_material_{i}")),
            primitives: vec![GltfPrimitive {
                vertices: p.vertices,
                indices: p.indices,
                material_index: p.material_index,
            }],
        })
        .collect()
}

/// Extract the primitives bound to arm joints, for the first-person
/// viewmodel.
///
/// The same joint indices are kept, so the viewmodel animates with the very
/// same bone palette as the body — the arms swing because the walk clip says
/// so, not because a second animation was authored for them.
///
/// The geometry is **not** moved and the palette is **not** rewritten: the
/// caller translates the whole instance by the returned pivot instead. See
/// [`arms_pivot`] for why doing it per-bone is the thing that exploded.
fn build_arms(
    joints: &[JointBuild],
    world: &[glam::Mat4],
    primitives: &[BakedPrimitive],
) -> Option<(Vec<BakedPrimitive>, [f32; 3], Vec<String>)> {
    let arm_set = arm_joint_set(&joints.iter().map(|j| &j.joint).collect::<Vec<_>>()[..]);
    let arm_set = prune_far_garments(joints, world, arm_set);
    let pivot = arm_shoulder(joints, world);

    let kept: Vec<BakedPrimitive> = primitives
        .iter()
        .filter(|p| arm_set.contains(&p.joint))
        .cloned()
        .collect();

    if kept.is_empty() {
        warn!(
            "viewmodel: no arm joints matched ({} prefixes, {} joints scanned); \
             first-person hands will be empty",
            ARM_JOINT_PREFIXES.len(),
            joints.len()
        );
        return None;
    }
    let kept_joints: std::collections::HashSet<u16> = kept.iter().map(|p| p.joint).collect();
    let total_verts: usize = primitives.iter().map(|p| p.vertices.len()).sum();
    let kept_verts: usize = kept.iter().map(|p| p.vertices.len()).sum();
    let share = kept_verts as f32 / total_verts.max(1) as f32;
    // An arms-only viewmodel should be a small fraction of the body. If it
    // is most of the model the bone names must have changed, and silently
    // drawing a headless-but-mostly-intact avatar in first person is worse
    // than drawing nothing.
    if share > 0.25 {
        let mut names: Vec<&str> = kept_joints
            .iter()
            .filter_map(|j| joints.get(*j as usize))
            .map(|j| j.joint.name.as_str())
            .collect();
        names.sort_unstable();
        warn!(
            "viewmodel: arm joints matched {:.0}% of the geometry ({} joints) - bone \
             names probably changed; drawing NO first-person arms rather than the \
             whole avatar. Matched: {:?}",
            share * 100.0,
            kept_joints.len(),
            names
        );
        return None;
    }
    debug!(
        "viewmodel: {} arm joints, {} of {} verts, shoulder at ({:.2}, {:.2}, {:.2})",
        kept_joints.len(),
        kept_verts,
        total_verts,
        pivot[0],
        pivot[1],
        pivot[2]
    );
    let mut names: Vec<String> = arm_set
        .iter()
        .filter_map(|&index| joints.get(index as usize))
        .map(|joint| joint.joint.name.clone())
        .collect();
    names.sort();
    Some((kept, pivot, names))
}

/// The joints that hang below the two upper-arm joints — the arms, by
/// hierarchy, whatever the exporter named anything.
///
/// `girl.glb` binds its left hand/finger cubes to `spLeft_963` and its right
/// ones to `RightHand_1006`; both sit between the upper arm and the forearm
/// and neither looks like an "arm" bone. A name list alone can never cover
/// that, which is exactly how the hands ended up torn off.
fn arm_subtree(joints: &[&crate::anim::Joint]) -> std::collections::HashSet<u16> {
    let roots: Vec<u16> = joints
        .iter()
        .enumerate()
        .filter(|(_, j)| {
            ARM_ROOT_PREFIXES
                .iter()
                .any(|prefix| j.name.starts_with(prefix))
        })
        .map(|(index, _)| index as u16)
        .collect();
    if roots.is_empty() {
        warn!(
            "viewmodel: no upper-arm joint matched {:?} ({} joints scanned); \
             falling back to name prefixes only",
            ARM_ROOT_PREFIXES,
            joints.len()
        );
    }
    let mut subtree: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for root in roots {
        // `Joint::parent` is the nearest *joint* ancestor, so the joint tree is
        // closed under descendants and this walk reaches the whole arm subtree
        // without needing the node graph.
        mark_subtree(root, joints, &mut subtree);
    }
    subtree
}

/// Every joint that belongs to the first-person arms, by hierarchy and name.
///
/// [`arm_subtree`] plus the garment bones that follow the arms but are
/// parented to the chest (`Hoodie_Left`, `H_above_Left`, `H_hm_*`).
///
fn arm_joint_set(joints: &[&crate::anim::Joint]) -> std::collections::HashSet<u16> {
    let mut arm_set = arm_subtree(joints);
    for (index, joint) in joints.iter().enumerate() {
        if is_arm_joint(&joint.name) {
            arm_set.insert(index as u16);
        }
    }
    arm_set
}

/// Keep only the garment joints that actually sit next to the arm.
///
/// `H_hm_*` and friends are parented to the hoodie/chest, so a few of them
/// live at the hip or the ankles (`H_hm_Left_895` rests at y ≈ 0.86,
/// `H_above_Left_c2_869` at y ≈ −0.09, on a rig whose shoulders are at
/// y ≈ 1.99). Keeping them dragged that geometry into the viewmodel, which is
/// what made the first-person hands look like the whole lower body arriving.
///
/// Only the **name-matched** joints are tested, and only against the rest
/// world origins of the *hierarchy* arm joints. Testing against the full
/// candidate set (as this first did) is tautological — every joint is
/// trivially near itself.
fn prune_far_garments(
    joints: &[JointBuild],
    world: &[glam::Mat4],
    mut arm_set: std::collections::HashSet<u16>,
) -> std::collections::HashSet<u16> {
    let subtree = arm_subtree(&joints.iter().map(|j| &j.joint).collect::<Vec<_>>()[..]);
    let arm_origins: Vec<glam::Vec3> = subtree
        .iter()
        .filter_map(|&index| joints.get(index as usize))
        .filter_map(|joint| world.get(joint.node_index))
        .map(|matrix| matrix.transform_point3(glam::Vec3::ZERO))
        .collect();
    if arm_origins.is_empty() {
        return arm_set;
    }
    // Reach scales off the arm itself, so this is not a magic number: longer
    // arms keep longer sleeves.
    let reach = {
        let top = arm_origins
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let bottom = arm_origins
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        ((top - bottom).abs() * 0.75).max(0.35)
    };

    let far: Vec<u16> = arm_set
        .iter()
        .copied()
        .filter(|&index| !subtree.contains(&index))
        .filter(|&index| is_arm_joint(&joints[index as usize].joint.name))
        .filter(|&index| {
            let Some(at) = world
                .get(joints[index as usize].node_index)
                .map(|matrix| matrix.transform_point3(glam::Vec3::ZERO))
            else {
                return false;
            };
            !arm_origins
                .iter()
                .any(|origin| origin.distance(at) <= reach)
        })
        .collect();
    if !far.is_empty() {
        let names: Vec<&str> = far
            .iter()
            .filter_map(|&index| joints.get(index as usize))
            .map(|joint| joint.joint.name.as_str())
            .collect();
        debug!(
            "viewmodel: dropped {} garment joint(s) that sit away from the arms: {:?}",
            names.len(),
            names
        );
        for index in &far {
            arm_set.remove(index);
        }
    }
    arm_set
}

/// Collect `root` and every joint that hangs below it.
fn mark_subtree(
    root: u16,
    joints: &[&crate::anim::Joint],
    out: &mut std::collections::HashSet<u16>,
) {
    out.insert(root);
    for (index, joint) in joints.iter().enumerate() {
        if joint.parent == Some(root as usize) {
            mark_subtree(index as u16, joints, out);
        }
    }
}

/// Where the arms pivot, in model space: the shoulder line.
///
/// The rest **world** translation of the two upper-arm joints is the exact
/// point the arm rotates about, so it is the only origin the viewmodel can be
/// placed from. The earlier version estimated it as the centroid of the top
/// 10% slice of the arm *geometry*, which drifts with every sleeve cube the
/// exporter adds above the shoulder.
///
/// This is applied ONCE, as a `T(-pivot)` at the front of the viewmodel's
/// instance matrix — never to the geometry, and never to the bone palette.
/// Doing it to both (shift the vertices, then conjugate every bone by
/// `T(-p) * A * T(p)`) nets to the same place only when the two pivots agree
/// exactly; let them drift and the error is `(A - I) * (p_geom - p_palette)`,
/// which is a different translation for every bone — the mesh comes apart
/// into long needles. That is what produced the exploded viewmodel.
fn arm_shoulder(joints: &[JointBuild], world: &[glam::Mat4]) -> [f32; 3] {
    let mut sum = [0f64; 3];
    let mut count = 0u32;
    for joint in joints {
        if !ARM_ROOT_PREFIXES
            .iter()
            .any(|prefix| joint.joint.name.starts_with(prefix))
        {
            continue;
        }
        let Some(matrix) = world.get(joint.node_index) else {
            continue;
        };
        let at = matrix.transform_point3(glam::Vec3::ZERO);
        for axis in 0..3 {
            sum[axis] += at[axis] as f64;
        }
        count += 1;
    }
    if count == 0 {
        return [0.0; 3];
    }
    [
        sum[0] as f32 / count as f32,
        sum[1] as f32 / count as f32,
        sum[2] as f32 / count as f32,
    ]
}

/// A joint plus the bookkeeping the builder needs but the runtime does not.
struct JointBuild {
    joint: Joint,
    node_index: usize,
}

/// Drop triangles that can never cover a pixel.
///
/// Blockbench exports every mesh element as 32 triangles over 8 corner
/// positions, 20 of which are zero-area (a cube needs 12). They draw nothing
/// but make up ~70% of the index stream.
///
/// A triangle is only dropped when all three vertices are bound to the same
/// joint: they then share one palette matrix, so zero area at rest stays zero
/// area in every pose. A triangle spanning two bones can grow area under
/// animation even if it is flat at rest, so those are always kept.
fn drop_degenerate_triangles(vertices: &[GltfVertex], indices: &mut Vec<u32>) {
    let mut kept = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let (Some(a), Some(b), Some(c)) = (
            vertices.get(tri[0] as usize),
            vertices.get(tri[1] as usize),
            vertices.get(tri[2] as usize),
        ) else {
            // Out of range: leave it for the caller to trip over, not hide it.
            kept.extend_from_slice(tri);
            continue;
        };
        let flat = {
            let pa = Vec3::from_array(a.position);
            let pb = Vec3::from_array(b.position);
            let pc = Vec3::from_array(c.position);
            (pb - pa).cross(pc - pa).length_squared() < 1e-12
        };
        if flat && a.joint == b.joint && b.joint == c.joint {
            continue;
        }
        kept.extend_from_slice(tri);
    }
    *indices = kept;
}

/// Split one skinned primitive into one primitive per joint, so the
/// per-primitive `joint` that `build_arms` filters on means something.
/// A triangle goes to the group of its first vertex; vertices keep their own
/// joint, so a triangle straddling two bones still skins correctly.
fn split_by_joint(
    vertices: Vec<GltfVertex>,
    indices: Vec<u32>,
    material_index: Option<usize>,
) -> Vec<BakedPrimitive> {
    struct Group {
        remap: std::collections::HashMap<u32, u32>,
        vertices: Vec<GltfVertex>,
        indices: Vec<u32>,
    }
    let mut groups: std::collections::HashMap<u16, Group> = std::collections::HashMap::new();
    let mut order: Vec<u16> = Vec::new();
    for tri in indices.chunks_exact(3) {
        let key = vertices[tri[0] as usize].joint;
        if !groups.contains_key(&key) {
            order.push(key);
            groups.insert(
                key,
                Group {
                    remap: Default::default(),
                    vertices: Vec::new(),
                    indices: Vec::new(),
                },
            );
        }
        let group = groups.get_mut(&key).unwrap();
        for &i in tri {
            let new = match group.remap.get(&i) {
                Some(n) => *n,
                None => {
                    group.vertices.push(vertices[i as usize]);
                    let n = (group.vertices.len() - 1) as u32;
                    group.remap.insert(i, n);
                    n
                }
            };
            group.indices.push(new);
        }
    }
    order
        .into_iter()
        .map(|key| {
            let g = groups.remove(&key).unwrap();
            BakedPrimitive {
                vertices: g.vertices,
                indices: g.indices,
                material_index,
                joint: key,
            }
        })
        .collect()
}

/// Concatenate primitives sharing a material into one primitive each,
/// preserving first-seen material order so upload order is deterministic.
fn merge_by_material(primitives: Vec<BakedPrimitive>) -> Vec<BakedPrimitive> {
    let mut order: Vec<Option<usize>> = Vec::new();
    let mut buckets: Vec<Vec<BakedPrimitive>> = Vec::new();
    for p in primitives {
        let slot = match order.iter().position(|m| *m == p.material_index) {
            Some(s) => s,
            None => {
                order.push(p.material_index);
                buckets.push(Vec::new());
                buckets.len() - 1
            }
        };
        buckets[slot].push(p);
    }
    buckets
        .into_iter()
        .zip(order)
        .map(|(members, material_index)| {
            // Primitives bound to different joints merge without trouble:
            // the joint index lives on each *vertex*, so one draw call can
            // cover a whole skinned body regardless of how many bones it
            // spans. That is what collapses the girl's 795 primitives (one
            // per bone) into a single draw.
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            for m in &members {
                let base = vertices.len() as u32;
                vertices.extend_from_slice(&m.vertices);
                indices.extend(m.indices.iter().map(|i| i + base));
            }
            BakedPrimitive {
                vertices,
                indices,
                material_index,
                // Reference only; the shader reads `GltfVertex::joint`.
                joint: members.first().map(|m| m.joint).unwrap_or(0),
            }
        })
        .collect()
}

fn parent_map(nodes: &[gltf::Node]) -> Vec<Option<usize>> {
    let mut parent = vec![None; nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        for child in node.children() {
            parent[child.index()] = Some(i);
        }
    }
    parent
}

/// Compose rest world matrices for `index` and its subtree.
fn compute_world(nodes: &[gltf::Node], index: usize, parent_world: Mat4, world: &mut [Mat4]) {
    let node = &nodes[index];
    // `transform()` already resolves a node's `matrix` or its TRS triple, so
    // the girl (which uses `matrix`) needs no special handling here.
    let w = parent_world * Mat4::from_cols_array_2d(&node.transform().matrix());
    world[index] = w;
    for child in node.children() {
        compute_world(nodes, child.index(), w, world);
    }
}

fn animated_nodes(document: &gltf::Document) -> std::collections::HashSet<usize> {
    let mut set = std::collections::HashSet::new();
    for animation in document.animations() {
        for channel in animation.channels() {
            let node = channel.target().node();
            set.insert(node.index());
        }
    }
    set
}

/// Build the joint list and the node -> joint mapping.
///
/// Returns `joint_of_node[i]` = the joint a node's geometry is bound to (its
/// nearest joint ancestor, which is itself when it is one), or `None` when
/// the node is entirely static.
fn assign_joints(
    nodes: &[gltf::Node],
    parent: &[Option<usize>],
    animated: &std::collections::HashSet<usize>,
    world: &[Mat4],
) -> Result<(Vec<JointBuild>, Vec<Option<usize>>), String> {
    // A node needs to be a joint if it is animated, or if it has an animated
    // descendant and is therefore on the path to one.
    let mut needed = animated.clone();
    for &node in animated {
        let mut cursor = parent[node];
        while let Some(p) = cursor {
            if !needed.insert(p) {
                break; // already claimed this chain
            }
            cursor = parent[p];
        }
    }
    if needed.len() > MAX_JOINTS {
        return Err(format!(
            "model needs {} joints, limit is {MAX_JOINTS}",
            needed.len()
        ));
    }

    // Order parents before children. Sorting by depth guarantees it: a parent
    // is always strictly shallower than its child. (A hand-rolled DFS would
    // also work, but it needs the roots picked correctly and a mistake there
    // silently produces an unordered skeleton.)
    // Depth by walking up with memoization: a parent's index is not
    // guaranteed to be lower than its child's (the girl's scene root is
    // node 1907 of 1908), so a single ascending pass would read
    // not-yet-computed depths.
    let mut depth = vec![usize::MAX; nodes.len()];
    for start in 0..nodes.len() {
        if depth[start] != usize::MAX {
            continue;
        }
        let mut chain = Vec::new();
        let mut cursor = Some(start);
        while let Some(c) = cursor {
            if depth[c] != usize::MAX {
                break;
            }
            chain.push(c);
            cursor = parent[c];
        }
        // The shallowest node in `chain` is a *child* of the node we stopped at,
        // so the base is one deeper than that node's own depth.
        let base = cursor.map_or(0, |c| depth[c] + 1);
        // `chain` runs deepest-first, so the shallower a node sits in it the
        // smaller its depth: `chain[i]` is `len - 1 - i` levels below the
        // node we resumed from.
        let len = chain.len();
        for (i, c) in chain.iter().enumerate() {
            depth[*c] = base + (len - 1 - i);
        }
    }
    let mut order: Vec<usize> = needed.iter().copied().collect();
    order.sort_by_key(|n| (depth[*n], *n));

    // Slot 0 is the reserved static joint.
    let mut joints = vec![JointBuild {
        joint: Joint {
            name: "__static".to_string(),
            parent: None,
            rest_local: Transform::default(),
            static_prefix: Mat4::IDENTITY,
            inv_bind: Mat4::IDENTITY,
        },
        node_index: usize::MAX,
    }];
    let mut index_of = std::collections::HashMap::with_capacity(order.len());
    for (slot, &n) in order.iter().enumerate() {
        index_of.insert(n, slot + 1);
    }

    // Rest world matrices, composed exactly the way `compose_palette` will
    // at runtime. Computing `inv_bind` any other way silently desynchronises
    // the two and the model comes apart in the rest pose.
    let mut rest_world = vec![Mat4::IDENTITY; order.len() + 1];
    for &n in &order {
        let rest_local = transform_of(&nodes[n]);
        let parent_joint = parent[n].and_then(|p| index_of.get(&p).copied());
        // If the immediate parent is itself a joint, `compose_palette`
        // already multiplies its world matrix in and the prefix is
        // identity. Otherwise the whole static chain above this joint has to
        // be folded into a constant.
        let static_prefix = match parent_joint {
            Some(_) => Mat4::IDENTITY,
            None => parent[n].map_or(Mat4::IDENTITY, |p| world[p]),
        };
        let parent_world = parent_joint.map_or(Mat4::IDENTITY, |j| rest_world[j]);
        let w = parent_world * static_prefix * rest_local.to_mat4();
        let slot = index_of[&n];
        rest_world[slot] = w;
        joints.push(JointBuild {
            joint: Joint {
                name: nodes[n].name().unwrap_or("joint").to_string(),
                parent: parent_joint,
                rest_local,
                static_prefix,
                inv_bind: w.inverse(),
            },
            node_index: n,
        });
    }

    // Every node maps to its nearest joint ancestor.
    let mut joint_of_node = vec![None; nodes.len()];
    for n in 0..nodes.len() {
        let mut cursor = Some(n);
        while let Some(c) = cursor {
            if let Some(j) = index_of.get(&c) {
                joint_of_node[n] = Some(*j);
                break;
            }
            cursor = parent[c];
        }
    }

    Ok((joints, joint_of_node))
}

/// A node's local transform, decomposed into TRS.
///
/// `girl.glb` stores node transforms as raw `matrix` values rather than TRS
/// triples, so this decomposition runs on every joint at load. It is the
/// crate's own routine, so the round-trip is the one glTF defines.
fn transform_of(node: &gltf::Node) -> Transform {
    let (translation, rotation, scale) = node.transform().decomposed();
    Transform {
        translation: Vec3::from_array(translation),
        rotation: glam::Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
        scale: Vec3::from_array(scale),
    }
}

/// Decode every clip, keeping only channels that target a joint.
fn decode_clips(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    joint_of_node: &[Option<usize>],
    joints: &[JointBuild],
) -> Result<Vec<AnimationClip>, String> {
    let mut clips = Vec::new();
    for animation in document.animations() {
        let mut channels: Vec<AnimChannel> = Vec::new();
        let mut duration: f32 = 0.0;
        let mut max_keys = 0usize;

        for channel in animation.channels() {
            let node_index = channel.target().node().index();
            // A channel on a node with no joint cannot be expressed in the
            // palette, so it is dropped rather than silently ignored later.
            let Some(joint) = joint_of_node[node_index] else {
                continue;
            };
            let sampler = channel.sampler();
            let interpolation = crate::anim::Interpolation::from_gltf(sampler.interpolation());
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));
            let Some(times) = reader.read_inputs() else {
                continue;
            };
            let times: Vec<f32> = times.collect();
            if times.is_empty() {
                continue;
            }
            let path = match channel.target().property() {
                gltf::animation::Property::Translation => AnimPath::Translation,
                gltf::animation::Property::Rotation => AnimPath::Rotation,
                gltf::animation::Property::Scale => AnimPath::Scale,
                // Morph weights need a per-primitive weight buffer we do not
                // have; ignoring them keeps the rest of the clip playable.
                _ => continue,
            };
            let Some(outputs) = reader.read_outputs() else {
                continue;
            };
            // CUBICSPLINE packs each key as [in_tangent, value, out_tangent],
            // so the buffer is 3x wider than the key count. The other modes
            // are a single value per key. Either way the flat layout is
            // written straight through and `AnimChannel::sample_into` reads
            // it with the right stride.
            let values: Vec<f32> = match outputs {
                // Quaternion keys come back as [x, y, z, w].
                gltf::animation::util::ReadOutputs::Rotations(r) => match interpolation {
                    crate::anim::Interpolation::CubicSpline => r
                        .into_f32()
                        .flatten()
                        .collect(),
                    _ => r
                        .into_f32()
                        .flat_map(|q| [q[0], q[1], q[2], q[3]])
                        .collect(),
                },
                gltf::animation::util::ReadOutputs::Translations(t) => match interpolation {
                    crate::anim::Interpolation::CubicSpline => t.flatten().collect(),
                    _ => t.flatten().collect(),
                },
                gltf::animation::util::ReadOutputs::Scales(s) => s.flatten().collect(),
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
            };
            let components = match path {
                AnimPath::Rotation => 4,
                _ => 3,
            };
            // Expected length: key_count * components * (3 for CUBICSPLINE).
            let stride = match interpolation {
                crate::anim::Interpolation::CubicSpline => 3,
                _ => 1,
            };
            if values.len() < times.len() * components * stride {
                warn!(
                    "clip {:?}: channel output shorter than its key count; skipping",
                    animation.name()
                );
                continue;
            }
            duration = duration.max(times[times.len() - 1]);
            max_keys = max_keys.max(times.len());
            channels.push(AnimChannel {
                joint,
                path,
                interpolation,
                times,
                values,
            });
        }

        let stats = clip_stats(&channels, &joints, duration, max_keys);
        let natural_speed = natural_speed(&channels, &joints, duration);
        clips.push(AnimationClip {
            name: animation.name().unwrap_or("clip").to_string(),
            role: crate::anim::classify_clip(&stats),
            duration,
            channels,
            stats,
            natural_speed,
        });
    }
    Ok(clips)
}

/// Structural facts about a clip, used to infer its role.
fn clip_stats(
    channels: &[AnimChannel],
    joints: &[JointBuild],
    duration: f32,
    max_keys: usize,
) -> ClipStats {
    let mut stats = ClipStats {
        duration,
        max_keys,
        ..Default::default()
    };
    for channel in channels {
        let Some(joint) = joints.get(channel.joint) else {
            continue;
        };
        let name = joint.joint.name.clone();
        if !stats.driven.contains(&name) {
            stats.driven.push(name);
        }
        // Root/hip translation is what carries locomotion in this rig; its
        // peak-to-peak vertical span separates walk from run.
        if channel.path == AnimPath::Translation
            && matches!(joint.joint.name.as_str(), n if n.starts_with("Root") || n.starts_with("DownBody") || n.starts_with("AllBody"))
        {
            let ys: Vec<f32> = channel.values.iter().skip(1).step_by(3).copied().collect();
            let lo = ys.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if ys.iter().all(|v| v.is_finite()) {
                stats.root_motion = stats.root_motion.max(hi - lo);
            }
        }
    }
    stats
}

/// Blocks per second this clip represents at 1x playback.
///
/// Horizontal root displacement over one loop divided by its duration. Zero
/// when the clip is in-place, which is the honest answer: playback then
/// scales with the authoritative speed directly.
fn natural_speed(channels: &[AnimChannel], joints: &[JointBuild], duration: f32) -> f32 {
    if duration <= f32::EPSILON {
        return 0.0;
    }
    let mut stride: f32 = 0.0;
    for channel in channels {
        if channel.path != AnimPath::Translation {
            continue;
        }
        let Some(joint) = joints.get(channel.joint) else {
            continue;
        };
        if !(joint.joint.name.starts_with("Root") || joint.joint.name.starts_with("DownBody")) {
            continue;
        }
        let xs: Vec<f32> = channel.values.iter().step_by(3).copied().collect();
        let zs: Vec<f32> = channel.values.iter().skip(2).step_by(3).copied().collect();
        if xs.len() < 2 || zs.len() < 2 {
            continue;
        }
        let dx = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - xs.iter().cloned().fold(f32::INFINITY, f32::min);
        let dz = zs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - zs.iter().cloned().fold(f32::INFINITY, f32::min);
        stride = stride.max((dx * dx + dz * dz).sqrt());
    }
    stride / duration
}

/// Bone-name convention of this rig: everything that is not a leg or a foot
/// is upper body. A [`crate::anim::Mask::LowerBody`] transition uses this to
/// blend legs while the head and arms hold their pose.
fn upper_body_flags(joints: &[JointBuild]) -> Vec<bool> {
    joints
        .iter()
        .map(|j| {
            let n = &j.joint.name;
            !(n.starts_with("LeftLeg")
                || n.starts_with("RightLeg")
                || n.starts_with("LeftFoot")
                || n.starts_with("RightFoot"))
        })
        .collect()
}

/// Convenience for tests and callers that want the default role for a model.
pub fn default_roles(path: &str) -> &'static [(&'static str, ClipRole)] {
    crate::anim::builtin_roles(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The avatar's world-space extent, computed from the file by composing
    /// every node matrix over its primitives' accessor bounds. Guards the
    /// static bake.
    const BOUNDS: [f32; 3] = [1.3697, 3.1928, 3.1391];
    /// Lowest point of the figure in the rest pose.
    const BOUNDS_FLOOR: f32 = -0.0053;

    fn load(
        name: &str,
    ) -> (
        gltf::Document,
        Vec<gltf::buffer::Data>,
        Vec<gltf::image::Data>,
    ) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../assets/models")
            .join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        // Mirror what the server does at catalog build time: the glTF
        // importer only understands PNG/JPEG, and girl.glb's texture is
        // WebP.
        let bytes = crate::gltf::normalize_model_bytes(name, &bytes);
        gltf::import_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
    }

    /// The avatar is the whole reason the rig exists: a hierarchical model
    /// whose geometry is spread over 795 nodes.
    #[test]
    fn girl_rig_bakes_the_hierarchy_into_a_handful_of_draws() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let (skeleton, meshes, _) = rig.into_parts();

        // 795 primitives collapse to one draw call because the model has a
        // single material.
        assert_eq!(meshes.len(), 1, "expected a single merged mesh");
        let prim = &meshes[0].primitives[0];
        assert_eq!(
            prim.vertices.len(),
            19_080,
            "vertex count must survive merge"
        );
        // Every one of girl.glb's 795 primitives is a `TRIANGLE_STRIP` of 24
        // indices that triangulates to 12 triangles (34 indices minus the
        // degenerate restart windows). Reading them as a triangle list used
        // to give 11 wrong triangles per box, many of them spanning opposite
        // faces — the exploded-mesh symptom.
        assert_eq!(prim.indices.len(), 795 * 12 * 3);

        // Only the animated nodes and their ancestors become joints; the
        // other 1800-odd nodes are baked into vertex positions.
        assert!(
            skeleton.joints.len() <= MAX_JOINTS,
            "joint count {} exceeds the palette",
            skeleton.joints.len()
        );
        assert!(
            skeleton.joints.len() < 200,
            "expected the static hierarchy to be baked away, got {} joints",
            skeleton.joints.len()
        );
    }

    #[test]
    fn girl_clips_decode_with_the_measured_shapes() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let clips = &rig.skeleton.clips;
        assert_eq!(clips.len(), 11, "the asset ships 11 clips");

        // Durations measured out of the file; see the plan's asset audit.
        let walk = clips.iter().find(|c| c.name == "walk").expect("walk clip");
        assert!(
            (walk.duration - 1.0).abs() < 1e-3,
            "walk duration was {}",
            walk.duration
        );
        assert!(
            walk.channels.iter().any(|c| c.path == AnimPath::Rotation),
            "walk must drive rotations"
        );

        // Every clip except two had its name destroyed by the exporter, so the
        // structural classifier supplies roles and the manifest is what
        // actually ships. Both agree on `walk`.
        assert_eq!(walk.role, ClipRole::Walk);
        let manifest: Vec<(String, ClipRole)> = crate::anim::builtin_roles("models/girl.glb")
            .iter()
            .map(|(k, r)| {
                (
                    k.rsplit('#')
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap()
                        .to_string(),
                    *r,
                )
            })
            .collect();
        assert_eq!(manifest.len(), clips.len(), "manifest covers every clip");
        let mut classified = clips.to_vec();
        crate::anim::apply_manifest(&mut classified, &manifest);
        assert_eq!(classified[1].role, ClipRole::Walk);
        assert_eq!(classified[7].role, ClipRole::Run);
        assert!(
            classified[3].role.is_additive(),
            "clip 3 drives only the tail"
        );
    }

    #[test]
    fn every_joint_reaches_the_static_joint_through_its_parents() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let skeleton = rig.skeleton;
        for (i, joint) in skeleton.joints.iter().enumerate() {
            if let Some(parent) = joint.parent {
                assert!(parent < i, "joint {i} must be ordered after its parent");
            }
            // Walking up must terminate rather than loop.
            let mut cursor = Some(i);
            let mut steps = 0;
            while let Some(c) = cursor {
                cursor = skeleton.joints[c].parent;
                steps += 1;
                assert!(steps <= skeleton.joints.len(), "joint parent cycle");
            }
        }
    }

    /// A rest-pose palette must be the identity: that is what makes the
    /// inverse bind matrices correct, and a mistake here shows up as a model
    /// exploding or collapsing to a point.
    #[test]
    fn the_rest_pose_palette_is_the_identity() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let skeleton = rig.skeleton;
        let n = skeleton.joints.len();
        let locals = vec![crate::anim::Transform::default(); n];
        // Rest locals, not identity defaults.
        let mut rest = vec![crate::anim::Transform::default(); n];
        for (slot, joint) in rest.iter_mut().zip(skeleton.joints.iter()) {
            *slot = joint.rest_local;
        }
        let mut worlds = vec![glam::Mat4::IDENTITY; n];
        let mut palette = vec![glam::Mat4::IDENTITY; n];
        skeleton.compose_palette(&rest, &mut worlds, &mut palette);
        for (i, m) in palette.iter().enumerate() {
            let m = m.to_cols_array();
            let max_err = (0..16)
                .map(|k| (m[k] - if k % 5 == 0 { 1.0 } else { 0.0 }).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-2,
                "joint {i} ({}) rest palette is not identity, max err {max_err}",
                skeleton.joints[i].name
            );
        }
        let _ = locals;
    }

    /// The rest pose must reproduce the original model bounds: if the
    /// baking dropped a node transform, the avatar would collapse toward the
    /// origin and this would catch it.
    #[test]
    fn the_baked_geometry_stays_inside_the_original_bounds() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let skeleton = rig.skeleton.clone();
        let (_, meshes, _) = rig.into_parts();
        let palette = rest_palette(&skeleton);

        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in &meshes[0].primitives[0].vertices {
            let m = palette[v.joint as usize];
            let p = m.transform_point3(Vec3::from_array(v.position));
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        // The avatar spans about 1.37 x 3.19 x 3.14 units and sits between
        // y = -0.005 and y = 3.19. A dropped node transform collapses or
        // stretches one of these.
        let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        for (axis, got) in size.iter().enumerate() {
            assert!(
                (got - BOUNDS[axis]).abs() < 0.02,
                "axis {axis} is {got:.3}, expected about {:.3} — a dropped node \
                 transform collapses or stretches the figure",
                BOUNDS[axis]
            );
        }
        assert!(
            (lo[1] - -0.0053).abs() < 0.02 && (hi[1] - 3.1875).abs() < 0.02,
            "vertical placement moved: {:?}..{:?}",
            lo[1],
            hi[1]
        );
    }

    /// The first-person viewmodel must come out of the same rig so the
    /// hands animate from the same bone palette as the body.
    #[test]
    fn the_avatar_yields_an_arms_only_viewmodel() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let pivot = rig.arms_pivot;
        let (_, body, arms) = rig.into_parts();

        let arms = arms.expect("the avatar must have arms to extract");
        assert_eq!(arms.len(), 1, "arms should merge into one draw call");

        let body_verts: usize = body
            .iter()
            .flat_map(|m| m.primitives.iter())
            .map(|p| p.vertices.len())
            .sum();
        let arm_verts: usize = arms
            .iter()
            .flat_map(|m| m.primitives.iter())
            .map(|p| p.vertices.len())
            .sum();
        assert!(arm_verts > 0, "arms viewmodel has no vertices");
        assert!(
            arm_verts < body_verts,
            "arms viewmodel ({arm_verts}) is not smaller than the body ({body_verts}) \
             — the arm joint prefixes probably stopped matching"
        );
        // An arms-only slice should be a modest share of the body, not most
        // of it.
        let share = arm_verts as f32 / body_verts as f32;
        assert!(
            share < 0.35,
            "arms are {share:.0}% of the body, which means the bone names changed"
        );
        // Every arm vertex must still point at a real joint, or the palette
        // lookup reads out of bounds.
        let joint_count = 82usize; // measured: 81 animated nodes + the static slot
        for v in arms[0].primitives[0].vertices.iter() {
            assert!((v.joint as usize) < joint_count);
        }

        // The arms are NOT moved to the viewmodel origin on the CPU any more:
        // `viewmodel_matrix` applies `T(-pivot)` in one go, so the geometry
        // stays in true model space and only the instance matrix knows about
        // the pivot. Assert against that: relative to the pivot, the arms must
        // hang below the shoulder line and straddle it in x, so that when the
        // instance matrix subtracts the pivot they end up below the eye.
        let (lo, hi) = arms
            .iter()
            .flat_map(|m| m.primitives.iter())
            .flat_map(|p| p.vertices.iter())
            .fold(
                ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
                |(mut lo, mut hi), v| {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v.position[k]);
                        hi[k] = hi[k].max(v.position[k]);
                    }
                    (lo, hi)
                },
            );
        let pivot_y = pivot[1];
        let top = hi[1] - pivot_y;
        let bottom = lo[1] - pivot_y;
        assert!(
            top < 0.6 && bottom > -2.0,
            "relative to the shoulder the arms span y {:.2}..{:.2} — they are not an arm-sized \
             piece hanging off the shoulder",
            bottom,
            top
        );
        // ...and they sit *below* the pivot, mostly, so the hands hang down.
        let below = arms
            .iter()
            .flat_map(|m| m.primitives.iter())
            .flat_map(|p| p.vertices.iter())
            .filter(|v| v.position[1] < pivot_y)
            .count();
        let total = arms
            .iter()
            .flat_map(|m| m.primitives.iter())
            .flat_map(|p| p.vertices.iter())
            .count();
        assert!(
            below * 2 > total,
            "only {below} of {total} arm verts are below the shoulder — the arms are not \
             hanging down"
        );
    }

    /// The viewmodel arms must move *as one body* under a posed clip.
    ///
    /// This is the regression test for the exploded viewmodel. The arms are
    /// drawn with the body's bone palette and one instance matrix, so a vertex
    /// on a forearm cube moves by exactly as much as its neighbour on the same
    /// bone. What it must NOT do is scatter, which is what a per-bone offset
    /// error produces: `T(-p) * A * T(p)` applied to geometry that was never
    /// shifted by `-p` leaves `(A - I) * p` of extra translation, and `p` is
    /// shoulder height — metres, different for every bone. Every triangle then
    /// stretches into a needle.
    ///
    /// Checked through the same math the shader runs: `instance * bone *
    /// vertex`, at the rest pose and half-way through the walk clip. With the
    /// pose driven, a vertex may move some distance (the arm swings), but a
    /// *pair* of vertices sharing a bone must move by nearly the same amount.
    #[test]
    fn posed_arms_move_as_one_body() {
        use crate::anim::Transform as JointTransform;
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let skeleton = &rig.skeleton;
        let arms = rig.arms.as_ref().expect("arms");
        let n = skeleton.joints.len();

        let mut world = vec![glam::Mat4::IDENTITY; n];
        let posed =
            |skeleton: &crate::anim::Skeleton, locals: &[JointTransform]| -> Vec<glam::Mat4> {
                let mut world = vec![glam::Mat4::IDENTITY; n];
                let mut bones = vec![glam::Mat4::IDENTITY; n];
                skeleton.compose_palette(locals, &mut world, &mut bones);
                bones
            };

        // Rest pose.
        let mut locals = vec![JointTransform::default(); n];
        for (slot, joint) in locals.iter_mut().zip(skeleton.joints.iter()) {
            *slot = joint.rest_local;
        }
        let rest_bones = posed(skeleton, &locals);

        // Half-way into the walk clip (clip 1, measured at 1.0 s).
        let walk = skeleton
            .clips
            .iter()
            .position(|c| c.role == ClipRole::Walk)
            .expect("the manifest marks a walk clip");
        skeleton.sample(walk as u32, 0.5, &mut locals);
        let posed_bones = posed(skeleton, &locals);

        // Group the arm vertices by their bone and look at how far each group
        // moved. A per-bone offset error shows up as a group displacement of
        // metres; real animation keeps an arm-sized group within an arm.
        let mut by_bone: std::collections::HashMap<u16, Vec<(glam::Vec3, glam::Vec3)>> =
            std::collections::HashMap::new();
        for m in arms {
            for v in &m.vertices {
                let joint = v.joint;
                if joint as usize >= n {
                    panic!("arm vertex references joint {joint}, only {n} exist");
                }
                let from =
                    rest_bones[joint as usize].transform_point3(glam::Vec3::from_array(v.position));
                let to = posed_bones[joint as usize]
                    .transform_point3(glam::Vec3::from_array(v.position));
                by_bone.entry(joint).or_default().push((from, to));
            }
        }

        for (joint, verts) in &by_bone {
            let mut max_drift = 0.0f32;
            for (from, to) in verts {
                max_drift = max_drift.max((to - from).length());
            }
            assert!(
                max_drift < 2.0,
                "joint {joint}: {max_drift:.2} units of per-vertex drift under the walk \
                 clip — the pose is scattering the vertices (a per-bone offset error)"
            );
        }
    }

    #[test]
    fn arm_prefixes_match_this_rigs_bone_names() {
        // Name prefixes are only for the garment bones that follow the arms
        // but hang off the chest; the arms themselves are found by hierarchy.
        for name in [
            "H_LeftArm_918",
            "H_RightArm_1017",
            "H_above_Left_b_874",
            "H_hm_above2_899",
            "Hoodie_Left_886",
        ] {
            assert!(is_arm_joint(name), "{name} should be an arm joint");
        }
        for name in [
            "Head_822",
            "LeftLeg_52",
            "RightFoot_95",
            "Breast_1064",
            "Tail_182",
            // The arm bones themselves are *not* in the name list any more,
            // so a rig that names them anything else still works: the walk
            // from the upper-arm root picks them up.
            "LeftArm_964",
            "LeftForeArm_960",
            "Rig_Elbow_7",
        ] {
            assert!(!is_arm_joint(name), "{name} should not be an arm joint");
        }
    }

    #[test]
    fn the_arm_cut_includes_the_hand_geometry() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");

        // The arm cut is the subtree of the two upper-arm joints plus the
        // garment bones. `girl.glb` binds its finger cubes to `spLeft_963` and
        // `spRight_1020`, which sit between the upper arm and the forearm, so
        // only a hierarchy walk finds them: the old name-prefix cut came out
        // with forearms and no hands.
        let set = arm_joint_set(&rig.skeleton.joints.iter().collect::<Vec<_>>());
        let names: Vec<&str> = rig
            .skeleton
            .joints
            .iter()
            .enumerate()
            .filter(|(index, _)| set.contains(&(*index as u16)))
            .map(|(_, joint)| joint.name.as_str())
            .collect();
        for required in [
            "LeftArm_964",
            "LeftForeArm_960",
            "RightArm_1021",
            "RightForeArm_1009",
            "spLeft_963",
            "spRight_1020",
        ] {
            assert!(
                names.contains(&required),
                "arm cut is missing {required}; it has {}",
                names.join(", ")
            );
        }
        // `spLeft2_959` / `spRight2_1008` are *not* animated, so they never
        // become joints: their geometry binds to the forearm, which is above.
        assert!(
            !names.contains(&"spLeft2_959"),
            "a static bone should not be a joint; it has {}",
            names.join(", ")
        );
        // ...and it must not swallow the rest of the body.
        for excluded in [
            "Head_822",
            "LeftLeg_52",
            "RightFoot_95",
            "Breast_1064",
            "Tail_182",
        ] {
            assert!(
                !names.contains(&excluded),
                "arm cut swallowed {excluded}; it has {}",
                names.join(", ")
            );
        }
    }

    /// What the runtime actually does every frame: sample a locomotion clip
    /// and compose its palette, then check the figure still holds together.
    ///
    /// This is the check that catches a bone assigning a transform to only
    /// part of the body. Because the model is 795 separate boxes, a wrong
    /// per-vertex bone scatters boxes rather than distorting a mesh, so a
    /// posed figure must stay within roughly the rest pose's extent.
    #[test]
    fn a_walk_pose_keeps_the_figure_together() {
        let (document, buffers, _images) = load("girl.glb");
        let rig = Rig::build(&document, &buffers).expect("rig");
        let skeleton = rig.skeleton.clone();
        let (_, meshes, _) = rig.into_parts();
        let vertices = &meshes[0].primitives[0].vertices;

        let bounds = |palette: &[Mat4]| {
            let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
            for v in vertices {
                let p = palette[v.joint as usize].transform_point3(Vec3::from_array(v.position));
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        };

        let walk = skeleton
            .clip_with_role(ClipRole::Walk)
            .expect("the avatar has a walk clip");

        for &(clip, time) in &[(walk, 0.0), (walk, 0.25), (walk, 0.5), (walk, 0.75)] {
            let n = skeleton.joints.len();
            let mut locals = vec![crate::anim::Transform::default(); n];
            let mut worlds = vec![Mat4::IDENTITY; n];
            let mut palette = vec![Mat4::IDENTITY; n];
            skeleton.sample(clip, time, &mut locals);
            skeleton.compose_palette(&locals, &mut worlds, &mut palette);

            for (i, m) in palette.iter().enumerate() {
                assert!(
                    m.to_cols_array().iter().all(|c| c.is_finite()),
                    "clip {clip} t={time}: bone {i} ({}) has a non-finite matrix",
                    skeleton.joints[i].name
                );
            }

            let (lo, hi) = bounds(&palette);
            let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
            for axis in 0..3 {
                assert!(
                    size[axis] < BOUNDS[axis] * 1.5 && size[axis] > BOUNDS[axis] * 0.5,
                    "clip {clip} t={time}: axis {axis} is {:.3}, rest is {:.3} — \
                     the pose is exploding or collapsing",
                    size[axis],
                    BOUNDS[axis]
                );
            }
            // And the feet must not sink through the floor or levitate.
            assert!(
                lo[1] > BOUNDS_FLOOR - 0.5 && lo[1] < BOUNDS_FLOOR + 0.5,
                "clip {clip} t={time}: feet at y={:.3}, expected near {BOUNDS_FLOOR}",
                lo[1]
            );
        }
    }

    // Helpers for the bound check above.
    fn rest_palette(skeleton: &Skeleton) -> Vec<Mat4> {
        let n = skeleton.joints.len();
        let rest: Vec<_> = skeleton.joints.iter().map(|j| j.rest_local).collect();
        let mut worlds = vec![Mat4::IDENTITY; n];
        let mut palette = vec![Mat4::IDENTITY; n];
        skeleton.compose_palette(&rest, &mut worlds, &mut palette);
        palette
    }
}
