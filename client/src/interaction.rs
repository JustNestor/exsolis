//! Client-side block targeting: raycasts from the camera through loaded
//! chunks to find what the player is aiming at, and sends
//! `PlaceBlock`/`BreakBlock` requests on click. The server re-validates
//! everything (loaded chunk, reach, occupancy, inventory) in
//! `interaction.rs` on the server — this file only decides what to ASK
//! for, it has no authority.
//!
//! NOTE ON PROVENANCE: this wires into `player.rs`'s `LocalPlayer` /
//! `FlyCamera` (yaw/pitch/eye_height, `eye_position()`) and the
//! Move-throttling pattern in `send_movement_input`, per the version of
//! that file from earlier in this conversation. I do not have that file
//! on disk to check against right now, so this is written from the
//! verbatim contents shown then, not re-verified. If real `player.rs` has
//! since drifted (renamed fields, different plugin set), the breakage
//! will be localized to the `use crate::player::{...}` line and the
//! `FlyCamera`/`LocalPlayer` field accesses below — everything else
//! (the DDA, the inventory tracking, the wire messages) is independent
//! of it.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::KeyCode;
use bevy_input::mouse::MouseButton;
use bevy_log::info;
use bevy_renet::RenetClient;
use bevy_renet::renet::DefaultChannel;
use bevy_transform::prelude::Transform;
use exsolis_common::{BlockPos, BlockType, ClientMessage, HOTBAR_SIZE};
use glam::{IVec3, Vec3};

use crate::chunk::{CHUNK_SIZE_I32, Chunk};
use crate::chunk_registry::ChunkRegistry;
use crate::player::{FlyCamera, LocalPlayer};

/// Blocks. Purely advisory on the client (affects what gets highlighted /
/// what a click tries to do); the server's own `MAX_REACH` is what
/// actually matters. Kept equal to avoid "I'm aiming at it but nothing
/// happens" confusion from a client/server mismatch.
const REACH: f32 = 8.0;
const RAYCAST_STEP_LIMIT: i32 = 64; // world-unit steps; generous vs REACH

/// What the player is currently aiming at, if anything: the targeted
/// block, and the empty cell just before it along the ray (where a
/// placement would go). Recomputed every frame in `update_targeting`.
#[derive(Component, Default)]
pub struct BlockTargeting {
    pub target: Option<TargetedBlock>,
}

#[derive(Clone, Copy)]
pub struct TargetedBlock {
    pub pos: IVec3,
    pub place_pos: IVec3,
}

/// Client-side mirror of the server's authoritative hotbar, populated
/// from `ServerMessage::YourInventory`. Never written to directly by
/// input code — see `streaming.rs`-style "server tells us, we display"
/// (there's no display yet, per this pass's scope, but `PlaceBlock`
/// needs to know which `BlockType` is selected to put on the wire).
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
            .add_systems(Startup, spawn_targeting)
            .add_systems(
                Update,
                (
                    select_hotbar_by_number_input,
                    update_targeting,
                    handle_break_input,
                    handle_place_input,
                ),
            );
    }
}

fn spawn_targeting(mut commands: Commands, query: Query<Entity, With<LocalPlayer>>) {
    for entity in &query {
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

/// Sends `SelectHotbar` immediately on keypress; does NOT update
/// `ClientInventory.selected` itself — that only changes once the server
/// confirms via `YourInventory` (see `streaming.rs`'s receive loop, where
/// this message needs to be handled). Optimistically updating here would
/// let a rejected/out-of-order select desync the client's idea of what's
/// selected from the server's.
fn select_hotbar_by_number_input(keys: Res<ButtonInput<KeyCode>>, mut client: ResMut<RenetClient>) {
    let Some(index) = DIGIT_KEYS.iter().position(|k| keys.just_pressed(*k)) else {
        return;
    };
    if !client.is_connected() {
        return;
    }
    let msg = bincode::serialize(&ClientMessage::SelectHotbar(index as u8)).unwrap();
    client.send_message(DefaultChannel::ReliableOrdered, msg);
}

/// Reads a block through the client's currently loaded chunks. Unlike the
/// server's `world_query::get_block_at` (which I don't have the new
/// client equivalent of), this returns `None` — not `Air` — for anything
/// in an unloaded chunk, so the raycast can tell "definitely air" apart
/// from "don't know, chunk isn't loaded" and stop rather than see through
/// the edge of the loaded world.
fn read_block(registry: &ChunkRegistry, chunks: &Query<&Chunk>, world: IVec3) -> Option<BlockType> {
    let coord = exsolis_common::ChunkCoord::new(
        world.x.div_euclid(CHUNK_SIZE_I32),
        world.y.div_euclid(CHUNK_SIZE_I32),
        world.z.div_euclid(CHUNK_SIZE_I32),
    );
    let entity = registry.entity_at(coord)?;
    let chunk = chunks.get(entity).ok()?;
    let lx = world.x.rem_euclid(CHUNK_SIZE_I32) as usize;
    let ly = world.y.rem_euclid(CHUNK_SIZE_I32) as usize;
    let lz = world.z.rem_euclid(CHUNK_SIZE_I32) as usize;
    Some(chunk.get(lx, ly, lz))
}

/// Amanatides & Woo voxel traversal: steps exactly one voxel boundary at a
/// time along the ray, so it can't tunnel through a thin (1-block) wall
/// the way a fixed-step marcher can at grazing angles. Returns the first
/// solid block hit and the empty cell immediately before it (= where a
/// placed block would go), or `None` if nothing solid is hit within
/// `REACH` / before leaving loaded chunks.
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

    // Distance along the ray to cross one full voxel on each axis.
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

    // Distance along the ray to the first voxel boundary on each axis.
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
        let block = read_block(registry, chunks, cell)?; // None = unloaded: stop, don't guess
        if block != BlockType::Air {
            return Some(TargetedBlock {
                pos: cell,
                place_pos: prev_cell,
            });
        }
        prev_cell = cell;

        // Advance along whichever axis reaches its next boundary soonest.
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
    mouse: Res<ButtonInput<MouseButton>>,
    mut client: ResMut<RenetClient>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.just_pressed(MouseButton::Left) || !client.is_connected() {
        return;
    }

    for targeting in &query {
        let Some(target) = targeting.target else {
            continue;
        };
        let msg = bincode::serialize(&ClientMessage::BreakBlock {
            pos: BlockPos::new(target.pos.x, target.pos.y, target.pos.z),
        })
        .unwrap();
        client.send_message(DefaultChannel::ReliableOrdered, msg);
    }
}

fn handle_place_input(
    mouse: Res<ButtonInput<MouseButton>>,
    mut client: ResMut<RenetClient>,
    inventory: Res<ClientInventory>,
    query: Query<&BlockTargeting, With<LocalPlayer>>,
) {
    if !mouse.just_pressed(MouseButton::Right) || !client.is_connected() {
        return;
    }
    // Nothing selected (or empty slot): nothing to place, and nothing to
    // send — the server would reject it anyway, this just avoids the
    // wasted packet.

    let Some(item) = inventory.selected_item() else {
        return;
    };
    for targeting in &query {
        let Some(target) = targeting.target else {
            continue;
        };
        let msg = bincode::serialize(&ClientMessage::PlaceBlock {
            pos: BlockPos::new(target.place_pos.x, target.place_pos.y, target.place_pos.z),
            kind: item,
        })
        .unwrap();
        client.send_message(DefaultChannel::ReliableOrdered, msg);
    }
}
