//! Creative-mode give + authoritative inventory sync.
//!
//! `ClientMessage::CreativeGive { name }` is bridged out of
//! `events.rs::dispatch_incoming` as a `CreativeGiveRequested` message
//! (same pattern `chunks.rs` uses for `RenderDistanceRequested`) and
//! handled here so this file owns all creative inventory mutation.
//! Outbound traffic goes through `net::OutgoingPacket`, flushed centrally
//! by `net.rs` — nothing here touches `RenetServer` directly.

use bevy_app::prelude::*;
use bevy_ecs::message::Message;
use bevy_ecs::prelude::*;
use bevy_log::{debug, info, warn};
use exsolis_common::{PlayerId, ServerMessage, inventory::MAX_STACK};

use crate::Inventory;
use crate::PlayerMap;
use crate::blocks::ServerBlocks;
use crate::net::{self, OutgoingPacket};

pub struct CreativePlugin;

impl Plugin for CreativePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CreativeGiveRequested>().add_systems(
            Update,
            (handle_creative_give, sync_new_inventories)
                .chain()
                .after(crate::events::dispatch_incoming)
                .in_set(net::GameplaySet),
        );
    }
}

/// Bridged from `events.rs::dispatch_incoming`, same shape as
/// `interaction::PlaceBlockRequested`.
#[derive(Message)]
pub struct CreativeGiveRequested {
    pub player_id: PlayerId,
    pub name: String,
}

/// Authoritative hotbar snapshot for one player: exactly HOTBAR_SIZE
/// `(name, count)` slots plus the selected index. The client renders
/// exactly this.
pub(crate) fn send_inventory_sync(
    player_id: PlayerId,
    inventory: &Inventory,
    blocks: &ServerBlocks,
    outgoing: &mut MessageWriter<OutgoingPacket>,
) {
    let slots = inventory
        .slots()
        .iter()
        .map(|slot| {
            slot.and_then(|s| {
                blocks
                    .name(s.item)
                    .map(|name| (name.to_string(), s.amount))
            })
        })
        .collect::<Vec<_>>();
    let selected = inventory.selected_index();
    debug!(
        "[InventorySync] to player {player_id}: {} slots, selected {selected}",
        slots.len()
    );
    outgoing.write(OutgoingPacket::to(
        player_id,
        ServerMessage::InventorySync { slots, selected },
    ));
}

/// Sorted block names for `ServerMessage::BlockList`. Snapshot the
/// registry so the join path owns a plain `Vec<String>`.
pub(crate) fn sorted_block_names(blocks: &ServerBlocks) -> Vec<String> {
    let mut names: Vec<String> = blocks.0.iter().map(|(_, def)| def.name.clone()).collect();
    names.sort();
    names
}

/// Add one stack (`MAX_STACK`) of `block` to `inventory`, returning
/// overflow that did not fit (dropped by the caller with a warning).
/// `add_item` fills same-type stacks first (including the selected slot
/// when it holds this item) and otherwise takes the first empty slot in
/// order, which is exactly the "selected-if-same else first-empty"
/// policy; anything beyond that is overflow.
pub(crate) fn give_stack(
    inventory: &mut Inventory,
    block: exsolis_common::BlockId,
) -> u8 {
    inventory.add_item(block, MAX_STACK)
}

fn handle_creative_give(
    mut events: MessageReader<CreativeGiveRequested>,
    player_map: Res<PlayerMap>,
    mut inventories: Query<&mut Inventory>,
    blocks: Res<ServerBlocks>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for ev in events.read() {
        let Some(block) = blocks.id(&ev.name) else {
            warn!(
                "[CreativeGive] Unknown block {:?} from player {:?} — dropped (client is untrusted)",
                ev.name, ev.player_id
            );
            continue;
        };
        if blocks.is_air(block) {
            warn!(
                "[CreativeGive] Refused air for player {:?} (name {:?})",
                ev.player_id, ev.name
            );
            continue;
        }
        let Some(&entity) = player_map.playerid_to_entity.get(&ev.player_id) else {
            warn!(
                "[CreativeGive] Player ID {:?} not in PlayerMap",
                ev.player_id
            );
            continue;
        };
        let Ok(mut inventory) = inventories.get_mut(entity) else {
            warn!(
                "[CreativeGive] Entity {:?} missing Inventory component",
                entity
            );
            continue;
        };
        let overflow = give_stack(&mut inventory, block);
        if overflow > 0 {
            warn!(
                "[CreativeGive] {} of {:?} did not fit in player {:?} inventory — dropped",
                overflow, ev.name, ev.player_id
            );
        }
        info!(
            "[CreativeGive] player {:?} gave {:?} (overflow {overflow})",
            ev.player_id, ev.name
        );
        let inv = inventory.clone();
        send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing);
    }
}

/// New player inventories (inserted by `attach_player_components`
/// during `entities::spawn_requested`) get their initial `InventorySync`
/// here. `PlayerMap` is already filled at that point, so the lookup
/// succeeds one frame after spawn.
fn sync_new_inventories(
    query: Query<(Entity, &Inventory), Added<Inventory>>,
    player_map: Res<PlayerMap>,
    blocks: Res<ServerBlocks>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for (entity, inventory) in query.iter() {
        let Some(&player_id) = player_map.entity_to_playerid.get(&entity) else {
            continue;
        };
        debug!("[InventorySync] initial sync for player {player_id}");
        send_inventory_sync(player_id, inventory, &blocks, &mut outgoing);
    }
}
