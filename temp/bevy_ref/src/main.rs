//! Reference renderer for the exsolis GLTF assets, using upstream Bevy PBR.
//!
//! Purpose: decide whether the "blurry / smooth" look on dragon & girl is an
//! exsolis pipeline bug or a property of the assets themselves. Bevy is the
//! reference implementation, so if these same .glb files look crisp here and
//! soft in exsolis, the bug is in exsolis; if they look equally soft here, it
//! is the assets.
//!
//! Usage: REF_FOCUS=<0..4> cargo run
//!   0 (or unset) = side-by-side row of all four models
//!   1 = duck, 2 = avocado, 3 = dragon, 4 = girl -- framed full screen so a
//!       screenshot can be inspected at texel level.

use std::collections::HashMap;

use bevy::camera::primitives::Aabb;
use bevy::prelude::*;
use bevy::world_serialization::prelude::WorldAssetRoot;

/// Every model is normalized to this largest-edge size in world units.
const TARGET_SIZE: f32 = 2.0;

/// Which single model is framed full-screen, if any.
#[derive(Resource, Default)]
struct Focus(Option<usize>);

// Reflect + register: bevy_world_serialization's spawner reflects every
// component in the world and panics on unregistered types.
#[derive(Component, Reflect)]
struct ModelRoot {
    slot: usize,
}

fn main() {
    // Read focus from the environment rather than key bindings: this session
    // is Wayland, where synthetic key events are unreliable.
    let focus_slot: Option<usize> = std::env::var("REF_FOCUS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| (1..=4).contains(n))
        .map(|n| n - 1);

    // Run from anywhere: the asset root is pinned rather than inferred from
    // CARGO_MANIFEST_DIR / the executable location.
    let asset_root = std::env::var("REF_ASSETS")
        .unwrap_or_else(|_| format!("{}/assets", env!("CARGO_MANIFEST_DIR")));

    App::new()
        .add_plugins(DefaultPlugins.set(AssetPlugin {
            file_path: asset_root,
            ..default()
        }).set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (1920, 1080).into(),
                title: "bevy_ref - exsolis gltf reference".into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Focus(focus_slot))
        .register_type::<ModelRoot>()
        .add_systems(Startup, setup)
        .add_systems(Update, (normalize_models, camera_control, report).chain())
        .run();
}

fn setup(mut commands: Commands, server: Res<AssetServer>) {
    let models = [
        "models/duck.glb",
        "models/avocado.glb",
        "models/dragon/result.glb",
        "models/girl/result.glb",
    ];

    for (slot, path) in models.iter().enumerate() {
        let x = (slot as f32 - 1.5) * 3.0;
        commands.spawn((
            ModelRoot { slot },
            WorldAssetRoot(server.load(GltfAssetLabel::Scene(0).from_asset(*path))),
            Transform::from_xyz(x, 0.0, 0.0),
        ));
    }

    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn(AmbientLight {
        color: Color::srgb(0.6, 0.6, 0.65),
        brightness: 1.2,
        ..default()
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 3.2, 12.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y),
    ));
}

/// Scale every model so its largest bounding-box edge is `TARGET_SIZE`.
///
/// Bounds are accumulated in each root's *local* space (world corners mapped
/// back through the root's inverse transform) so the measured extent does not
/// depend on the scale we are about to set -- otherwise the correction would
/// oscillate every frame.
fn normalize_models(
    mut roots: Query<(&mut Transform, &GlobalTransform, &ModelRoot), With<ModelRoot>>,
    aabbs: Query<(Entity, &GlobalTransform, &Aabb), Without<ModelRoot>>,
    parents: Query<&ChildOf>,
    root_set: Query<(), With<ModelRoot>>,
    mut seen: Local<HashMap<usize, usize>>,
) {
    // slot -> (root-local min, root-local max, mesh count)
    let mut acc: HashMap<usize, ([Vec3; 2], usize)> = HashMap::new();

    for (entity, gt, aabb) in &aabbs {
        // Walk up the hierarchy until we find a ModelRoot.
        let mut entity = entity;
        let root = loop {
            match parents.get(entity) {
                Ok(child_of) => {
                    entity = child_of.parent();
                    if root_set.contains(entity) {
                        break entity;
                    }
                }
                Err(_) => break entity,
            }
        };
        let Ok((_, root_gt, model_root)) = roots.get(root) else {
            continue;
        };
        let inv = root_gt.affine().inverse();
        let entry = acc.entry(model_root.slot).or_insert((
            [Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)],
            0,
        ));
        entry.1 += 1;
        for i in 0..8 {
            let corner = Vec3::new(
                if i & 1 == 0 {
                    -aabb.half_extents.x
                } else {
                    aabb.half_extents.x
                },
                if i & 2 == 0 {
                    -aabb.half_extents.y
                } else {
                    aabb.half_extents.y
                },
                if i & 4 == 0 {
                    -aabb.half_extents.z
                } else {
                    aabb.half_extents.z
                },
            ) + Vec3::from(aabb.center);
            // Into root-local space, so the extent is scale-independent.
            let local = inv.transform_point3(gt.transform_point(corner));
            entry.0[0] = entry.0[0].min(local);
            entry.0[1] = entry.0[1].max(local);
        }
    }

    for (slot, ([mn, mx], count)) in acc {
        // Re-apply only if the mesh count changed (scene still streaming in).
        if seen.get(&slot).is_some_and(|c| *c == count) {
            continue;
        }
        let extent = (mx - mn).max_element();
        if extent <= 0.0 || !extent.is_finite() {
            continue;
        }
        let scale = TARGET_SIZE / extent;
        for (mut t, _, model_root) in &mut roots {
            if model_root.slot == slot {
                t.scale = Vec3::splat(scale);
            }
        }
        seen.insert(slot, count);
    }
}

fn camera_control(
    focus: Res<Focus>,
    // Disjoint on Transform: Bevy 0.19 rejects read+write access to the same
    // component in one system even when the filters differ per-query.
    roots: Query<(&Transform, &ModelRoot), (With<ModelRoot>, Without<Camera3d>)>,
    mut cams: Query<&mut Transform, (With<Camera3d>, Without<ModelRoot>)>,
) {
    let Ok(mut cam) = cams.single_mut() else { return };
    match focus.0 {
        None => {
            *cam = Transform::from_xyz(0.0, 3.2, 12.0).looking_at(Vec3::new(0.0, 1.2, 0.0), Vec3::Y);
        }
        Some(slot) => {
            let Some((t, _)) = roots.iter().find(|(_, r)| r.slot == slot) else {
                return;
            };
            let centre = Vec3::new(t.translation.x, TARGET_SIZE * 0.5, 0.0);
            // Fill the vertical FOV with the normalized model.
            let dist = TARGET_SIZE / (2.0 * (27.0f32.to_radians() * 0.5).tan()) * 1.25;
            *cam = Transform::from_xyz(centre.x, centre.y, dist).looking_at(centre, Vec3::Y);
        }
    }
}

/// One-shot report of the normalization actually applied to each model.
fn report(roots: Query<(&Transform, &ModelRoot), With<ModelRoot>>, mut done: Local<bool>) {
    if *done || roots.is_empty() {
        return;
    }
    *done = true;
    let mut lines: Vec<String> = roots
        .iter()
        .map(|(t, r)| {
            format!(
                "slot {} x={:.2} uniform_scale={:.6} -> max_edge={:.3}",
                r.slot, t.translation.x, t.scale.x, TARGET_SIZE
            )
        })
        .collect();
    lines.sort();
    for l in lines {
        println!("[bevy_ref] {l}");
    }
}