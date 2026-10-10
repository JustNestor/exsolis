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
struct BakedPrimitive {
    vertices: Vec<GltfVertex>,
    indices: Vec<u32>,
    material_index: Option<usize>,
    joint: u16,
}

/// A decoded, ready-to-upload model: joint-space geometry plus its skeleton.
pub struct Rig {
    pub skeleton: Skeleton,
    primitives: Vec<BakedPrimitive>,
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

            for primitive in mesh.primitives() {
                let loaded = crate::gltf::load_gltf_primitive(&primitive, buffers)?;
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
        })
    }

    /// Take the skeleton out, leaving the geometry. [`Rig::meshes`] consumes
    /// what is left.
    pub fn into_parts(self) -> (Skeleton, Vec<GltfMesh>) {
        let skeleton = self.skeleton.clone();
        (skeleton, self.meshes())
    }

    /// Flatten the rig into engine meshes, one per material.
    pub fn meshes(self) -> Vec<GltfMesh> {
        self.primitives
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
}

/// A joint plus the bookkeeping the builder needs but the runtime does not.
struct JointBuild {
    joint: Joint,
    node_index: usize,
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
            if sampler.interpolation() != gltf::animation::Interpolation::Linear {
                warn!(
                    "clip {:?}: non-LINEAR interpolation is not supported; skipping channel",
                    animation.name()
                );
                continue;
            }
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
            let values: Vec<f32> = match outputs {
                // Quaternion keys come back as [x, y, z, w].
                gltf::animation::util::ReadOutputs::Rotations(r) => r
                    .into_f32()
                    .flat_map(|q| [q[0], q[1], q[2], q[3]])
                    .collect(),
                gltf::animation::util::ReadOutputs::Translations(t) => t.flatten().collect(),
                gltf::animation::util::ReadOutputs::Scales(s) => s.flatten().collect(),
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => continue,
            };
            let components = match path {
                AnimPath::Rotation => 4,
                _ => 3,
            };
            if values.len() < times.len() * components {
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
        let (skeleton, meshes) = rig.into_parts();

        // 795 primitives collapse to one draw call because the model has a
        // single material.
        assert_eq!(meshes.len(), 1, "expected a single merged mesh");
        let prim = &meshes[0].primitives[0];
        assert_eq!(
            prim.vertices.len(),
            19_080,
            "vertex count must survive merge"
        );
        assert_eq!(prim.indices.len(), 27_030);

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
        let (_, meshes) = rig.into_parts();
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
