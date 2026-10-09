//! Server-authoritative block placement/breaking + hotbar selection.
//!
//! `PlaceBlock`/`BreakBlock`/`SelectHotbar` arrive on the client->server
//! ReliableOrdered channel (same as `Move`), get bridged out of
//! `events.rs::dispatch_incoming` as `Message`s (same pattern `chunks.rs`
//! uses for `RenderDistanceRequested`), and are handled here so this file
//! doesn't need to know about packet framing. Outbound traffic goes
//! through `net::OutgoingPacket`, flushed centrally by `net.rs` — nothing
//! here touches `RenetServer` directly.
//!
//! Everything the client claims is treated as a request, not a fact:
//! - the target chunk must actually be loaded server-side (`ChunkWorld`);
//! - the target must be within `MAX_REACH` of the player's OWN server
//!   position (`PlayerPosition`), not wherever the client says it's aiming;
//! - break requires the target to be non-air; place requires it to be air
//!   AND the player's selected hotbar slot to actually hold that item.

use bevy_app::prelude::*;
use bevy_ecs::message::Message;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use exsolis_common::{BlockPos, BlockType, ChunkCoord, PlayerId, ServerMessage};
use glam::Vec3;

use crate::Inventory;
use crate::PlayerMap;
use crate::chunks::PlayerPosition;
use crate::chunks::{ChunkRenderState, ChunkWorld};
use crate::net::{self, OutgoingPacket};

/// Blocks. A block is 1 world unit, so this is also world units. Generous
/// on purpose — this exists to catch cheating/desync, not to be a strict
/// gameplay reach limit; tighten once real hand/camera reach is settled.
const MAX_REACH: f32 = 8.0;

pub struct InteractionPlugin;

impl Plugin for InteractionPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PlaceBlockRequested>()
            .add_message::<BreakBlockRequested>()
            .add_message::<SelectHotbarRequested>()
            .add_systems(
                Update,
                (
                    handle_select_hotbar,
                    (handle_place_block, handle_break_block),
                )
                    .chain()
                    .after(crate::events::dispatch_incoming)
                    .in_set(net::GameplaySet),
            );
    }
}

/// Bridged from `events.rs::dispatch_incoming`, same shape as
/// `chunks::RenderDistanceRequested`.
#[derive(Message)]
pub struct PlaceBlockRequested {
    pub player_id: PlayerId,
    pub pos: BlockPos,
    pub kind: BlockType,
}

#[derive(Message)]
pub struct BreakBlockRequested {
    pub player_id: PlayerId,
    pub pos: BlockPos,
}

#[derive(Message)]
pub struct SelectHotbarRequested {
    pub player_id: PlayerId,
    pub index: u8,
}

fn handle_select_hotbar(
    mut events: MessageReader<SelectHotbarRequested>,
    player_map: Res<PlayerMap>,
    mut inventories: Query<&mut Inventory>,
) {
    for ev in events.read() {
        let Some(&entity) = player_map.playerid_to_entity.get(&ev.player_id) else {
            warn!(
                "[SelectHotbar] Player ID {:?} not in PlayerMap",
                ev.player_id
            );
            continue;
        };
        let Ok(mut inventory) = inventories.get_mut(entity) else {
            warn!(
                "[SelectHotbar] Entity {:?} missing Inventory component",
                entity
            );
            continue;
        };
        inventory.select(ev.index);
    }
}

fn world_to_chunk(pos: BlockPos) -> (ChunkCoord, usize) {
    const S: i32 = exsolis_common::CHUNK_SIZE as i32;
    let cx = pos.x.div_euclid(S);
    let cy = pos.y.div_euclid(S);
    let cz = pos.z.div_euclid(S);
    let lx = pos.x.rem_euclid(S) as usize;
    let ly = pos.y.rem_euclid(S) as usize;
    let lz = pos.z.rem_euclid(S) as usize;
    let idx = lx
        + ly * exsolis_common::CHUNK_SIZE
        + lz * exsolis_common::CHUNK_SIZE * exsolis_common::CHUNK_SIZE;
    (ChunkCoord::new(cx, cy, cz), idx)
}

fn within_reach(player_pos: Vec3, target: BlockPos) -> (bool, f32) {
    // Center of the target block, not its corner.
    let center = Vec3::new(
        target.x as f32 + 0.5,
        target.y as f32 + 0.5,
        target.z as f32 + 0.5,
    );
    let dist = player_pos.distance(center);
    (dist <= MAX_REACH, dist)
}

/// Applies the mutation and queues a `BlockUpdate` `OutgoingPacket` to
/// exactly the players who actually have this chunk loaded
/// (`ChunkRenderState.loaded`) — sending it to everyone would leak block
/// data outside a player's own render distance and waste bandwidth on
/// players who can't see it anyway.
fn apply_and_broadcast(
    coord: ChunkCoord,
    idx: usize,
    new_block: BlockType,
    pos: BlockPos,
    world: &mut ChunkWorld,
    outgoing: &mut MessageWriter<OutgoingPacket>,
    loaded_by: &Query<(&crate::Player, &ChunkRenderState)>,
) -> bool {
    let Some(existing) = world.ready.get(&coord) else {
        warn!(
            "[apply_and_broadcast] Target chunk {:?} not loaded in ChunkWorld.ready",
            coord
        );
        return false;
    };
    let Some(updated) = existing.set_block(idx, new_block) else {
        warn!(
            "[apply_and_broadcast] Out of range index {} or corrupt chunk data for chunk {:?}",
            idx, coord
        );
        return false;
    };
    world.ready.insert(coord, updated);

    for (player, state) in loaded_by {
        if state.loaded.contains(&coord) {
            outgoing.write(OutgoingPacket::to(
                player.0,
                ServerMessage::BlockUpdate {
                    pos,
                    kind: new_block,
                },
            ));
        }
    }
    true
}

fn handle_place_block(
    mut events: MessageReader<PlaceBlockRequested>,
    player_map: Res<PlayerMap>,
    positions: Query<&PlayerPosition>,
    mut inventories: Query<&mut Inventory>,
    mut world: ResMut<ChunkWorld>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    loaded_by: Query<(&crate::Player, &ChunkRenderState)>,
) {
    for ev in events.read() {
        let Some(&entity) = player_map.playerid_to_entity.get(&ev.player_id) else {
            warn!("[PlaceBlock] Player ID {:?} not in PlayerMap", ev.player_id);
            continue;
        };
        let Ok(player_pos) = positions.get(entity) else {
            warn!(
                "[PlaceBlock] Entity {:?} (PlayerId: {:?}) missing PlayerPosition component",
                entity, ev.player_id
            );
            continue;
        };
        let (in_reach, dist) = within_reach(player_pos.0, ev.pos);
        if !in_reach {
            warn!(
                "[PlaceBlock] Reach check failed for PlayerId {:?}: distance {:.2} > MAX_REACH {:.1} (player_pos: {:?}, target_pos: {:?})",
                ev.player_id, dist, MAX_REACH, player_pos.0, ev.pos
            );
            continue;
        }
        let Ok(mut inventory) = inventories.get_mut(entity) else {
            warn!(
                "[PlaceBlock] Entity {:?} missing Inventory component",
                entity
            );
            continue;
        };
        let Some(selected) = inventory.selected_slot() else {
            warn!(
                "[PlaceBlock] Selected hotbar slot {} is empty for PlayerId {:?}",
                inventory.selected_index(),
                ev.player_id
            );
            continue;
        };
        if selected.item != ev.kind {
            warn!(
                "[PlaceBlock] Item mismatch for PlayerId {:?}: hotbar holds {:?}, request claims {:?}",
                ev.player_id, selected.item, ev.kind
            );
            continue;
        }

        let (coord, idx) = world_to_chunk(ev.pos);
        let Some(target) = world.ready.get(&coord) else {
            warn!(
                "[PlaceBlock] Target chunk {:?} not loaded/ready in ChunkWorld for target pos {:?}",
                coord, ev.pos
            );
            continue;
        };
        let Some(unpacked) = target.unpack() else {
            warn!("[PlaceBlock] Failed to unpack target chunk {:?}", coord);
            continue;
        };
        if unpacked[idx] != BlockType::Air {
            warn!(
                "[PlaceBlock] Target block at {:?} is not Air (currently {:?}, attempted place {:?}",
                ev.pos, unpacked[idx], ev.kind
            );
            continue;
        }

        if !apply_and_broadcast(
            coord,
            idx,
            ev.kind,
            ev.pos,
            &mut world,
            &mut outgoing,
            &loaded_by,
        ) {
            warn!(
                "[PlaceBlock] apply_and_broadcast failed for chunk {:?}, pos {:?}",
                coord, ev.pos
            );
        } else {
            let selected_index = inventory.selected_index();
            inventory.remove_one(selected_index, ev.kind);
        }
    }
}

fn handle_break_block(
    mut events: MessageReader<BreakBlockRequested>,
    player_map: Res<PlayerMap>,
    positions: Query<&PlayerPosition>,
    mut inventories: Query<&mut Inventory>,
    mut world: ResMut<ChunkWorld>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    loaded_by: Query<(&crate::Player, &ChunkRenderState)>,
) {
    for ev in events.read() {
        let Some(&entity) = player_map.playerid_to_entity.get(&ev.player_id) else {
            warn!("[BreakBlock] Player ID {:?} not in PlayerMap", ev.player_id);
            continue;
        };
        let Ok(player_pos) = positions.get(entity) else {
            warn!(
                "[BreakBlock] Entity {:?} (PlayerId: {:?}) missing PlayerPosition component",
                entity, ev.player_id
            );
            continue;
        };
        let (in_reach, dist) = within_reach(player_pos.0, ev.pos);
        if !in_reach {
            warn!(
                "[BreakBlock] Reach check failed for PlayerId {:?}: distance {:.2} > MAX_REACH {:.1} (player_pos: {:?}, target_pos: {:?})",
                ev.player_id, dist, MAX_REACH, player_pos.0, ev.pos
            );
            continue;
        }

        let (coord, idx) = world_to_chunk(ev.pos);
        let Some(target) = world.ready.get(&coord) else {
            warn!(
                "[BreakBlock] Target chunk {:?} not loaded/ready in ChunkWorld for target pos {:?}",
                coord, ev.pos
            );
            continue;
        };
        let Some(unpacked) = target.unpack() else {
            warn!("[BreakBlock] Failed to unpack target chunk {:?}", coord);
            continue;
        };
        let broken = unpacked[idx];
        if broken == BlockType::Air {
            warn!(
                "[BreakBlock] Target block at {:?} is Air (nothing to break)",
                ev.pos
            );
            continue;
        }

        if !apply_and_broadcast(
            coord,
            idx,
            BlockType::Air,
            ev.pos,
            &mut world,
            &mut outgoing,
            &loaded_by,
        ) {
            warn!(
                "[BreakBlock] apply_and_broadcast failed for chunk {:?}, pos {:?}",
                coord, ev.pos
            );
        } else if let Ok(mut inventory) = inventories.get_mut(entity) {
            inventory.add_item(broken, 1);
        }
    }
}
