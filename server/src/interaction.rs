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
use bevy_log::{debug, warn};
use exsolis_common::{AIR_ID, BlockId, BlockPos, ChunkCoord, PlayerId, ServerMessage};
use exsolis_common::{EntityKind, EntityState};
use glam::Vec3;

use crate::Inventory;
use crate::PlayerMap;
use crate::blocks::ServerBlocks;
use crate::chunks::PlayerPosition;
use crate::chunks::{ChunkRenderState, ChunkWorld};
use crate::creative::send_inventory_sync;
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
    pub kind: String,
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
    blocks: Res<ServerBlocks>,
    mut outgoing: MessageWriter<OutgoingPacket>,
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
        let inv = inventory.clone();
        send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing);
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
    new_block: BlockId,
    pos: BlockPos,
    world: &mut ChunkWorld,
    outgoing: &mut MessageWriter<OutgoingPacket>,
    loaded_by: &Query<(&crate::Player, &ChunkRenderState)>,
    blocks: &ServerBlocks,
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

    let kind = blocks
        .name(new_block)
        .unwrap_or("exsolis:air")
        .to_string();
    for (player, state) in loaded_by {
        if state.loaded.contains(&coord) {
            outgoing.write(OutgoingPacket::to(
                player.0,
                ServerMessage::BlockUpdate { pos, kind: kind.clone() },
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
    mut spawns: MessageWriter<crate::entities::SpawnEntityRequest>,
    loaded_by: Query<(&crate::Player, &ChunkRenderState)>,
    blocks: Res<ServerBlocks>,
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
        // The server's selected slot is authoritative: the client's
        // `kind` guess is never trusted. A mismatch is logged but the
        // server item is what gets placed and broadcast.
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
        let resolved_id = selected.item;
        if blocks.is_air(resolved_id) {
            warn!(
                "[PlaceBlock] Cannot place air for PlayerId {:?}",
                ev.player_id
            );
            continue;
        };
        let server_kind = blocks.name(resolved_id).unwrap_or("exsolis:air");
        if server_kind != ev.kind {
            debug!(
                "[PlaceBlock] Kind mismatch for PlayerId {:?}: server selected {:?}, request claims {:?} — proceeding with server item",
                ev.player_id, server_kind, ev.kind
            );
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
        if unpacked[idx] != AIR_ID {
            warn!(
                "[PlaceBlock] Target block at {:?} is not Air (currently {:?}, attempted place {:?}",
                ev.pos, unpacked[idx], ev.kind
            );
            continue;
        }

        if !apply_and_broadcast(
            coord,
            idx,
            resolved_id,
            ev.pos,
            &mut world,
            &mut outgoing,
            &loaded_by,
            &blocks,
        ) {
            warn!(
                "[PlaceBlock] apply_and_broadcast failed for chunk {:?}, pos {:?}",
                coord, ev.pos
            );
        } else {
            let selected_index = inventory.selected_index();
            inventory.remove_one(selected_index, resolved_id);
            let inv = inventory.clone();
            send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing);
            // Placing a container block brings its block entity into being.
            // Keyed by position, so replacing the block at a position that
            // already has an entity is handled by the break path first.
            if crate::entities::is_block_entity(resolved_id, &blocks.0) {
                spawns.write(crate::entities::SpawnEntityRequest {
                    kind: EntityKind::Block,
                    state: block_entity_state(ev.pos),
                    // A block entity renders through its block, not a model.
                    model: None,
                    scale: 1.0,
                    block_pos: Some(ev.pos),
                    extra: crate::entities::SpawnExtra::BlockEntity(resolved_id),
                    player_token: None,
                });
            }
        }
    }
}

/// A block entity's transform is the block's own position. Yaw is derived
/// from the position hash so a chest consistently faces one of four
/// directions instead of every chest pointing the same way.
fn block_entity_state(pos: exsolis_common::BlockPos) -> EntityState {
    EntityState {
        x: pos.x as f32 + 0.5,
        y: pos.y as f32,
        z: pos.z as f32 + 0.5,
        yaw: ((pos.x * 7 + pos.z * 13) & 3) as f32 * std::f32::consts::FRAC_PI_2,
        ..Default::default()
    }
}

fn handle_break_block(
    mut events: MessageReader<BreakBlockRequested>,
    player_map: Res<PlayerMap>,
    positions: Query<&PlayerPosition>,
    mut inventories: Query<&mut Inventory>,
    mut containers: Query<&mut crate::Container>,
    registry: Res<crate::entities::EntityRegistry>,
    mut despawns: MessageWriter<crate::entities::DespawnEntityRequest>,
    mut world: ResMut<ChunkWorld>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    loaded_by: Query<(&crate::Player, &ChunkRenderState)>,
    blocks: Res<ServerBlocks>,
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
        if broken == AIR_ID {
            warn!(
                "[BreakBlock] Target block at {:?} is Air (nothing to break)",
                ev.pos
            );
            continue;
        }

        if !apply_and_broadcast(
            coord,
            idx,
            AIR_ID,
            ev.pos,
            &mut world,
            &mut outgoing,
            &loaded_by,
            &blocks,
        ) {
            warn!(
                "[BreakBlock] apply_and_broadcast failed for chunk {:?}, pos {:?}",
                coord, ev.pos
            );
        } else {
            // Breaking a container block tears down its block entity. The
            // contents are handed to the breaker rather than dropped, so a
            // chest can never be destroyed to destroy its loot.
            if crate::entities::is_block_entity(broken, &blocks.0)
                && let Some(block_entity_id) = registry.at_block(ev.pos)
            {
                if let Some(be) = registry.entity_of(block_entity_id) {
                    if let Ok(mut container) = containers.get_mut(be)
                        && let Ok(mut inventory) = inventories.get_mut(entity)
                    {
                        let leftover = inventory.0.absorb(&mut container.0);
                        if leftover > 0 {
                            warn!(
                                "[BreakBlock] {} items did not fit in the breaker inventory",
                                leftover
                            );
                        }
                    }
                }
                despawns.write(crate::entities::DespawnEntityRequest(block_entity_id));
            }
            if let Ok(mut inventory) = inventories.get_mut(entity) {
                inventory.add_item(broken, 1);
                let inv = inventory.clone();
                send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing);
            }
        }
    }
}
