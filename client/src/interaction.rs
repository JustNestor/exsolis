//! Client-side block targeting: raycasts from the camera through loaded
//! chunks to find what the player is aiming at, and sends
//! `PlaceBlock`/`BreakBlock` requests on click or continuous press.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::MouseButton;
use bevy_log::{info, warn};
use bevy_renet::RenetClient;
use bevy_renet::renet::DefaultChannel;
use bevy_time::{Time, Timer, TimerMode};
use bevy_transform::prelude::Transform;
use exsolis_common::{BlockPos, BlockType, ClientMessage, HOTBAR_SIZE};
use glam::{IVec3, Vec3};

use crate::chunk::{CHUNK_SIZE_I32, Chunk};
use crate::chunk_registry::ChunkRegistry;
use crate::player::{FlyCamera, LocalPlayer};

/// Blocks. Purely advisory on the client (affects what gets highlighted /
/// what a click tries to do); the server's own `MAX_REACH` is what
/// actually matters.
const REACH: f32 = 8.0;
const RAYCAST_STEP_LIMIT: i32 = 64;

/// Cooldown resource to prevent per-frame network packet spam when holding action buttons.
#[derive(Resource)]
pub struct InteractionCooldown {
    pub break_timer: Timer,
    pub place_timer: Timer,
}

impl Default for InteractionCooldown {
    fn default() -> Self {
        Self {
            break_timer: Timer::from_seconds(0.2, TimerMode::Once),
            place_timer: Timer::from_seconds(0.2, TimerMode::Once),
        }
    }
}

/// What the player is currently aiming at, if anything: the targeted
/// block, and the empty cell just before it along the ray.
#[derive(Component, Default)]
pub struct BlockTargeting {
    pub target: Option<TargetedBlock>,
}

#[derive(Clone, Copy, Debug)]
pub struct TargetedBlock {
    pub pos: IVec3,
    pub place_pos: IVec3,
}

#[derive(Resource, Default, Debug)]
pub struct ClientInventory {
    pub slots: [Option<(BlockType, u8)>; HOTBAR_SIZE],
    pub selected: u8,
}

impl ClientInventory {
    pub fn selected_item(&self) -> Option<BlockType> {
        self.slots[self.selected as usize].map(|(item, _)| item)
    }
}

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ClientInventory>()
            .init_resource::<InteractionCooldown>()
            .add_systems(
                Update,
                (
                    ensure_targeting_component,
                    select_hotbar_by_number_input,
                    update_targeting,
                    handle_break_input,
                    handle_place_input,
                ),
            );
    }
}

/// Ensures `BlockTargeting` is automatically inserted whenever a `LocalPlayer`
/// is spawned (fixing issues if `LocalPlayer` spawns after `Startup`).
fn ensure_targeting_component(
    mut commands: Commands,
    query: Query<Entity, (With<LocalPlayer>, Without<BlockTargeting>)>,
) {
    for entity in &query {
        info!(
            "[Targeting] Attaching BlockTargeting component to LocalPlayer entity {:?}",
            entity
        );
        commands.entity(entity).insert(BlockTargeting::default());
    }
}

const DIGIT_KEYS: [KeyCode; HOTBAR_SIZE] = [
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
];

fn select_hotbar_by_number_input(keys: Res<ButtonInput<KeyCode>>, mut client: ResMut<RenetClient>) {
    let Some(index) = DIGIT_KEYS.iter().position(|k| keys.just_pressed(*k)) else {
        return;
    };
    if !client.is_connected() {
        warn!("[SelectHotbar] Key digit press detected, but RenetClient is not connected");
        return;
    }
    info!("[SelectHotbar] Sending SelectHotbar({}) packet", index);
    let msg = bincode::serialize(&ClientMessage::SelectHotbar(index as u8)).unwrap();
    client.send_message(DefaultChannel::ReliableOrdered, msg);
}

fn read_block(registry: &ChunkRegistry, chunks: &Query<&Chunk>, world: IVec3) -> Option<BlockType> {
    let coord = exsolis_common::ChunkCoord::new(
        world.x.div_euclid(CHUNK_SIZE_I32),
        world.y.div_euclid(CHUNK_SIZE_I32),
        world.z.div_euclid(CHUNK_SIZE_I32),
    );
    let Some(entity) = registry.entity_at(coord) else {
        // Chunk is not registered in ChunkRegistry
        return None;
    };
    let Ok(chunk) = chunks.get(entity) else {
        warn!(
            "[read_block] Chunk entity {:?} registered at {:?}, but missing Chunk component",
            entity, coord
        );
        return None;
    };
    let lx = world.x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let ly = world.y.rem_euclid(CHUNK_SIZE_I32) as usize;
    let lz = world.z.rem_euclid(CHUNK_SIZE_I32) as usize;
    Some(chunk.get(lx, ly, lz))
}

fn raycast_voxels(
    registry: &ChunkRegistry,
    chunks: &Query<&Chunk>,
    origin: Vec3,
    dir: Vec3,
) -> Option<TargetedBlock> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }

    let mut cell = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );
    let mut prev_cell = cell;

    let step = IVec3::new(
        dir.x.signum() as i32,
        dir.y.signum() as i32,
        dir.z.signum() as i32,
    );

    let t_delta = Vec3::new(
        if dir.x != 0.0 {
            (1.0 / dir.x).abs()
        } else {
            f32::INFINITY
        },
        if dir.y != 0.0 {
            (1.0 / dir.y).abs()
        } else {
            f32::INFINITY
        },
        if dir.z != 0.0 {
            (1.0 / dir.z).abs()
        } else {
            f32::INFINITY
        },
    );

    let next_boundary = |o: f32, c: i32, s: i32| -> f32 {
        if s > 0 {
            (c + 1) as f32 - o
        } else {
            o - c as f32
        }
    };
    let mut t_max = Vec3::new(
        if dir.x != 0.0 {
            next_boundary(origin.x, cell.x, step.x) / dir.x.abs()
        } else {
            f32::INFINITY
        },
        if dir.y != 0.0 {
            next_boundary(origin.y, cell.y, step.y) / dir.y.abs()
        } else {
            f32::INFINITY
        },
        if dir.z != 0.0 {
            next_boundary(origin.z, cell.z, step.z) / dir.z.abs()
        } else {
            f32::INFINITY
        },
    );

    for _ in 0..RAYCAST_STEP_LIMIT {
        let block = read_block(registry, chunks, cell)?;
        if block != BlockType::Air {
            return Some(TargetedBlock {
                pos: cell,
                place_pos: prev_cell,
            });
        }
        prev_cell = cell;

        if t_max.x < t_max.y && t_max.x < t_max.z {
            if t_max.x > REACH {
                return None;
            }
            cell.x += step.x;
            t_max.x += t_delta.x;
        } else if t_max.y < t_max.z {
            if t_max.y > REACH {
                return None;
            }
            cell.y += step.y;
            t_max.y += t_delta.y;
        } else {
            if t_max.z > REACH {
                return None;
            }
            cell.z += step.z;
            t_max.z += t_delta.z;
        }
    }
    None
}

fn update_targeting(
    registry: Res<ChunkRegistry>,
    chunks: Query<&Chunk>,
    mut query: Query<(&Transform, &FlyCamera, &mut BlockTargeting), With<LocalPlayer>>,
) {
    for (transform, cam, mut targeting) in &mut query {
        let origin = cam.eye_position(transform);
        let dir = transform.rotation * Vec3::NEG_Z;
        targeting.target = raycast_voxels(&registry, &chunks, origin, dir);
    }
}

fn handle_break_input(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cooldown: ResMut<InteractionCooldown>,
    mut client: ResMut<RenetClient>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.pressed(MouseButton::Left) {
        cooldown.break_timer.reset();
        return;
    }

    cooldown.break_timer.tick(time.delta());

    if !mouse.just_pressed(MouseButton::Left) && !cooldown.break_timer.is_finished() {
        return;
    }

    cooldown.break_timer.reset();

    if !client.is_connected() {
        warn!("[BreakBlock] Left click pressed, but RenetClient is NOT connected!");
        return;
    }

    if query.is_empty() {
        warn!(
            "[BreakBlock] Left click pressed, but query for LocalPlayer + BlockTargeting returned NO entities!"
        );
        return;
    }

    for targeting in &query {
        let Some(target) = targeting.target else {
            warn!(
                "[BreakBlock] Left click pressed, but BlockTargeting.target is None (raycast hit air, unloaded chunk, or reached max distance)"
            );
            continue;
        };

        info!(
            "[BreakBlock] Sending BreakBlock packet for pos {:?}",
            target.pos
        );
        let msg = bincode::serialize(&ClientMessage::BreakBlock {
            pos: BlockPos::new(target.pos.x, target.pos.y, target.pos.z),
        })
        .unwrap();
        client.send_message(DefaultChannel::ReliableOrdered, msg);
    }
}

fn handle_place_input(
    time: Res<Time>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cooldown: ResMut<InteractionCooldown>,
    mut client: ResMut<RenetClient>,
    inventory: Res<ClientInventory>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.pressed(MouseButton::Right) {
        cooldown.place_timer.reset();
        return;
    }

    cooldown.place_timer.tick(time.delta());

    if !mouse.just_pressed(MouseButton::Right) && !cooldown.place_timer.is_finished() {
        return;
    }

    cooldown.place_timer.reset();

    if !client.is_connected() {
        warn!("[PlaceBlock] Right click pressed, but RenetClient is NOT connected!");
        return;
    }

    let Some(item) = inventory.selected_item() else {
        warn!(
            "[PlaceBlock] Right click pressed, but ClientInventory selected slot {} is empty/None! (slots: {:?})",
            inventory.selected, inventory.slots
        );
        return;
    };

    if query.is_empty() {
        warn!(
            "[PlaceBlock] Right click pressed, but query for LocalPlayer + BlockTargeting returned NO entities!"
        );
        return;
    }

    for targeting in &query {
        let Some(target) = targeting.target else {
            warn!(
                "[PlaceBlock] Right click pressed, but BlockTargeting.target is None (raycast hit air, unloaded chunk, or reached max distance)"
            );
            continue;
        };

        info!(
            "[PlaceBlock] Sending PlaceBlock packet for item {:?} at place_pos {:?}",
            item, target.place_pos
        );
        let msg = bincode::serialize(&ClientMessage::PlaceBlock {
            pos: BlockPos::new(target.place_pos.x, target.place_pos.y, target.place_pos.z),
            kind: item,
        })
        .unwrap();
        client.send_message(DefaultChannel::ReliableOrdered, msg);
    }
}
