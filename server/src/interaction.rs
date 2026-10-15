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
            .add_message::<BreakBlockApplied>()
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

/// Fired when a break succeeds, carrying what was broken. Scripting
/// forwards it to Lua (`break_block` with name + numeric id) so the Lua
/// display inventory can mirror the Rust authority.
#[derive(Message)]
pub struct BreakBlockApplied {
    pub player_id: PlayerId,
    pub pos: BlockPos,
    pub block: BlockId,
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

pub(crate) fn handle_place_block(
    mut events: MessageReader<PlaceBlockRequested>,
    player_map: Res<PlayerMap>,
    positions: Query<&PlayerPosition>,
    looks: Query<&crate::movement::PlayerLook>,
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
        // Items and multiblock parts are not placeable directly.
        if !blocks.0.is_placeable(resolved_id) {
            warn!(
                "[PlaceBlock] {:?} is not placeable for PlayerId {:?}",
                blocks.name(resolved_id),
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

        // Generic multiblock: the master cell plus every part cell (rotated
        // by placement facing) must be loaded air, or nothing is placed.
        let facing = looks
            .get(entity)
            .map(|look| exsolis_common::Facing::from_yaw(look.yaw))
            .unwrap_or(exsolis_common::Facing::North);
        let Some(cells) = multiblock_cells(ev.pos, resolved_id, facing, &world, &blocks) else {
            warn!(
                "[PlaceBlock] Multiblock footprint blocked for {:?} at {:?}",
                server_kind, ev.pos
            );
            continue;
        };

        let mut ok = true;
        for (pos, id) in &cells {
            let (coord, idx) = world_to_chunk(*pos);
            if !apply_and_broadcast(coord, idx, *id, *pos, &mut world, &mut outgoing, &loaded_by, &blocks) {
                warn!("[PlaceBlock] apply failed for multiblock cell {pos:?}");
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        {
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
            bevy_log::info!(
                "[PlaceBlock] {:?} placed {} cell(s) at {:?}",
                server_kind,
                cells.len(),
                ev.pos
            );
        }
    }
}

/// Read one cell, or `None` when its chunk is not loaded (treated as
/// "not air" by every multiblock check: never half-place into the void).
fn read_cell(world: &ChunkWorld, pos: BlockPos) -> Option<BlockId> {
    let (coord, idx) = world_to_chunk(pos);
    world
        .ready
        .get(&coord)
        .and_then(|c| c.unpack())
        .and_then(|u| u.get(idx).copied())
}

/// Master + part cells for placing `master` at `pos`, or `None` when any
/// footprint cell is not loaded air. Single blocks yield one cell.
fn multiblock_cells(
    pos: BlockPos,
    master: BlockId,
    facing: exsolis_common::Facing,
    world: &ChunkWorld,
    blocks: &ServerBlocks,
) -> Option<Vec<(BlockPos, BlockId)>> {
    let mut cells = vec![(pos, master)];
    let layout = blocks
        .0
        .def(master)
        .and_then(|d| d.multiblock.clone());
    let Some(layout) = layout else {
        return match read_cell(world, pos) {
            Some(id) if id == AIR_ID => Some(cells),
            _ => None,
        };
    };
    for part in &layout.parts {
        let Some(part_id) = blocks.id(&part.part) else {
            return None;
        };
        let o = facing.rotate_offset(part.offset);
        let cell = BlockPos::new(pos.x + o[0], pos.y + o[1], pos.z + o[2]);
        cells.push((cell, part_id));
    }
    // Master first, then parts: every cell must be loaded air.
    for (cell, _) in &cells {
        if read_cell(world, *cell) != Some(AIR_ID) {
            return None;
        }
    }
    Some(cells)
}

/// Cells to clear when `hit` (master or part) breaks. Facing is not stored,
/// so every facing is tried: the one whose part cells actually hold the
/// expected part ids wins. Degenerate hits (edited world, missing parts)
/// degrade to the single cell.
fn break_cells(
    hit: BlockPos,
    hit_id: BlockId,
    world: &ChunkWorld,
    blocks: &ServerBlocks,
) -> (BlockId, Vec<(BlockPos, BlockId)>) {
    // Case 1: a master was hit. Try each facing for full coverage.
    if let Some(layout) = blocks.0.def(hit_id).and_then(|d| d.multiblock.clone()) {
        for facing in [
            exsolis_common::Facing::North,
            exsolis_common::Facing::East,
            exsolis_common::Facing::South,
            exsolis_common::Facing::West,
        ] {
            let mut cells = vec![(hit, hit_id)];
            let mut ok = true;
            for part in &layout.parts {
                let Some(part_id) = blocks.id(&part.part) else {
                    ok = false;
                    break;
                };
                let o = facing.rotate_offset(part.offset);
                let cell = BlockPos::new(hit.x + o[0], hit.y + o[1], hit.z + o[2]);
                if read_cell(world, cell) != Some(part_id) {
                    ok = false;
                    break;
                }
                cells.push((cell, part_id));
            }
            if ok {
                return (hit_id, cells);
            }
        }
        return (hit_id, vec![(hit, hit_id)]);
    }
    // Case 2: a part was hit. Find the master cell covering it: for each
    // facing, the master sits at hit - rotated_offset holding the master id.
    if let Some(master_name) = blocks
        .0
        .def(hit_id)
        .and_then(|_| {
            blocks
                .0
                .iter()
                .find(|(_, d)| {
                    d.multiblock.as_ref().is_some_and(|layout| {
                        layout.parts.iter().any(|p| {
                            blocks.id(&p.part).is_some_and(|id| id == hit_id)
                        })
                    })
                })
                .map(|(_, d)| d.name.clone())
        })
    {
        let Some(master_id) = blocks.id(&master_name) else {
            return (hit_id, vec![(hit, hit_id)]);
        };
        let layout = blocks
            .0
            .def(master_id)
            .and_then(|d| d.multiblock.clone());
        if let Some(layout) = layout {
            for facing in [
                exsolis_common::Facing::North,
                exsolis_common::Facing::East,
                exsolis_common::Facing::South,
                exsolis_common::Facing::West,
            ] {
                for part in &layout.parts {
                    if blocks.id(&part.part) != Some(hit_id) {
                        continue;
                    }
                    let o = facing.rotate_offset(part.offset);
                    let master_pos =
                        BlockPos::new(hit.x - o[0], hit.y - o[1], hit.z - o[2]);
                    if read_cell(world, master_pos) != Some(master_id) {
                        continue;
                    }
                    // Found the master; clear whatever of the structure
                    // is actually present (tolerates partial edits).
                    let mut cells = vec![(master_pos, master_id), (hit, hit_id)];
                    for p2 in &layout.parts {
                        let Some(pid) = blocks.id(&p2.part) else {
                            continue;
                        };
                        if pid == hit_id {
                            continue;
                        }
                        let o2 = facing.rotate_offset(p2.offset);
                        let cell = BlockPos::new(
                            master_pos.x + o2[0],
                            master_pos.y + o2[1],
                            master_pos.z + o2[2],
                        );
                        if read_cell(world, cell) == Some(pid) {
                            cells.push((cell, pid));
                        }
                    }
                    return (master_id, cells);
                }
            }
        }
    }
    (hit_id, vec![(hit, hit_id)])
}

/// A block entity's transform is the block's own position. Yaw is derived
/// from the position hash so a chest consistently faces one of four
/// directions instead of every chest pointing the same way.
fn block_entity_state(pos: exsolis_common::BlockPos) -> EntityState {    EntityState {
        x: pos.x as f32 + 0.5,
        y: pos.y as f32,
        z: pos.z as f32 + 0.5,
        yaw: ((pos.x * 7 + pos.z * 13) & 3) as f32 * std::f32::consts::FRAC_PI_2,
        ..Default::default()
    }
}

pub(crate) fn handle_break_block(
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
    mut applied: MessageWriter<BreakBlockApplied>,
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

        // Generic multiblock: resolve the whole structure (master or part
        // hit — facing is rediscovered by coverage, never stored).
        let (drop_id, cells) = break_cells(ev.pos, broken, &world, &blocks);
        let mut ok = true;
        for (pos, _) in &cells {
            let (coord, idx) = world_to_chunk(*pos);
            if !apply_and_broadcast(
                coord, idx, AIR_ID, *pos, &mut world, &mut outgoing, &loaded_by, &blocks,
            ) {
                warn!("[BreakBlock] apply failed for multiblock cell {pos:?}");
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        {
            // Breaking a container block tears down its block entity. The
            // contents are handed to the breaker rather than dropped, so a
            // chest can never be destroyed to destroy its loot.
            if crate::entities::is_block_entity(drop_id, &blocks.0)
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
                inventory.add_item(drop_id, 1);
                let inv = inventory.clone();
                send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing);
            }
            applied.write(BreakBlockApplied {
                player_id: ev.player_id,
                pos: ev.pos,
                block: drop_id,
            });
            bevy_log::info!(
                "[BreakBlock] cleared {} cell(s), dropped {:?}",
                cells.len(),
                blocks.name(drop_id)
            );
        }
    }
}
