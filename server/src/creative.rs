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
use crate::ui::UiActionRequested;

pub struct CreativePlugin;

/// Last hotbar snapshot each player was last sent (full on join, then the
/// running diff baseline). Lets the inventory sync path send deltas instead
/// of re-sending the whole hotbar — and skip the HUD redraw entirely — when
/// nothing actually changed (no-op selects, failed clicks).
#[derive(Resource, Default)]
pub struct HotbarSyncLog {
    last: std::collections::HashMap<PlayerId, HotbarSync>,
}

/// One player's hotbar as mirrored on the wire: per-slot
/// `(name, amount)` (`None` = empty) plus the selected index.
pub(crate) type HotbarSync = (Vec<Option<(String, u8)>>, u8);

impl HotbarSyncLog {
    pub fn snapshot(&self, player_id: PlayerId) -> Option<HotbarSync> {
        self.last.get(&player_id).cloned()
    }

    pub fn record(&mut self, player_id: PlayerId, sync: &HotbarSync) {
        self.last.insert(player_id, sync.clone());
    }

    /// Drop a disconnected player's baseline, so the map does not grow with
    /// every join for the life of the server.
    pub fn forget(&mut self, player_id: PlayerId) {
        self.last.remove(&player_id);
    }
}

impl Plugin for CreativePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CreativeGiveRequested>().add_systems(
            Update,
            (handle_creative_give, sync_new_inventories)
                .chain()
                .after(crate::events::dispatch_incoming)
                .in_set(net::GameplaySet),
        );
        // Inventory click + give bookkeeping lives here so the hotbar sync
        // path can diff against the last sent snapshot.
        app.init_resource::<HotbarSyncLog>();
        app.add_systems(
            Update,
            apply_inventory_clicks
                .after(crate::ui::dispatch_ui_packets)
                .before(crate::scripting::forward_ui_action)
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

/// Authoritative hotbar snapshot for one player: the first HOTBAR_SIZE
/// slots as `(name, count)` plus the selected index. The client renders
/// exactly this (HUD + hotbar row); the full 36 live in Lua snapshots.
pub(crate) fn send_inventory_sync(
    player_id: PlayerId,
    inventory: &Inventory,
    blocks: &ServerBlocks,
    outgoing: &mut MessageWriter<OutgoingPacket>,
    hotbar_log: &mut HotbarSyncLog,
) {
    let slots = inventory
        .hotbar()
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
    let sync: HotbarSync = (slots.clone(), selected);

    if let Some((prev_slots, prev_selected)) = hotbar_log.snapshot(player_id) {
        let (changes, selected_change) = hotbar_diff(&prev_slots, &slots, prev_selected, selected);
        if changes.is_empty() && selected_change.is_none() {
            // Nothing moved (a no-op select, or a click that mutated
            // nothing): skip the resend *and* the HUD redraw entirely.
            return;
        }
        outgoing.write(OutgoingPacket::to(
            player_id,
            ServerMessage::InventorySyncDiff {
                changes,
                selected: selected_change,
            },
        ));
        hotbar_log.record(player_id, &sync);
    } else {
        outgoing.write(OutgoingPacket::to(
            player_id,
            ServerMessage::InventorySync { slots, selected },
        ));
        hotbar_log.record(player_id, &sync);
    }
    // Server-driven hotbar HUD: tiny non-modal screen, redrawn only when
    // the hotbar actually changed (or the selection moved). Render failures
    // only log (the wire sync above already went out; the HUD is redundant
    // display).
    send_hud(player_id, inventory, blocks, outgoing);
}

/// Diff two hotbar snapshots. Pure (no ECS, no I/O) so it can be unit
/// tested. Returns the changed slots as `(hotbar_index, new_contents)` —
/// `contents` is the slot's new value, `None` when it was emptied — plus
/// the new selection when it differs from the previous one (`None` =
/// unchanged).
fn hotbar_diff(
    prev: &[Option<(String, u8)>],
    next: &[Option<(String, u8)>],
    prev_selected: u8,
    next_selected: u8,
) -> (Vec<(u8, Option<(String, u8)>)>, Option<u8>) {
    let mut changes = Vec::new();
    for (i, (a, b)) in prev.iter().zip(next.iter()).enumerate() {
        if a != b {
            changes.push((i as u8, (*b).clone()));
        }
    }
    let selected = (prev_selected != next_selected).then_some(next_selected);
    (changes, selected)
}

/// Render `assets/xml/hud.xml` for one player and mount it as the
/// non-modal `hud` screen.
fn send_hud(
    player_id: PlayerId,
    inventory: &Inventory,
    blocks: &ServerBlocks,
    outgoing: &mut MessageWriter<OutgoingPacket>,
) {
    use crate::ui_template::DataValue;
    use std::collections::HashMap;
    let selected = inventory.selected_index() as usize;
    let mut cells = Vec::with_capacity(exsolis_common::HOTBAR_SIZE);
    for (i, slot) in inventory.hotbar().iter().enumerate() {
        let mut cell = HashMap::new();
        let is_selected = i == selected;
        cell.insert(
            "bg".to_string(),
            DataValue::Str(if is_selected { "#c8c8c8".into() } else { "#222222cc".into() }),
        );
        cell.insert(
            "border".to_string(),
            DataValue::Str(if is_selected { "#ffffff".into() } else { "#555555".into() }),
        );
        if let Some(s) = slot {
            if let Some(def) = blocks.0.def(s.item) {
                let prefix = if def.placeable { "block#" } else { "item#" };
                cell.insert(
                    "img".to_string(),
                    DataValue::Str(format!("{prefix}{}", def.name)),
                );
                cell.insert("count".to_string(), DataValue::Int(s.amount as i64));
            }
        }
        cells.push(DataValue::Map(cell));
    }
    let mut root = HashMap::new();
    root.insert("slots".to_string(), DataValue::Array(cells));
    match crate::ui_template::render_file("hud", &DataValue::Map(root)) {
        Ok(xml) => {
            outgoing.write(OutgoingPacket::to(
                player_id,
                ServerMessage::UiShow {
                    id: "hud".to_string(),
                    xml,
                },
            ));
        }
        Err(e) => warn!("[hud] render failed for player {player_id}: {e}"),
    }
}

/// Sorted block names for `ServerMessage::BlockList`. Snapshot the
/// registry so the join path owns a plain `Vec<String>`. Hidden
/// multiblock parts stay out (placed through their master).
pub(crate) fn sorted_block_names(blocks: &ServerBlocks) -> Vec<String> {
    blocks.0.creative_names()
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
    mut hotbar_log: ResMut<HotbarSyncLog>,
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
        send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing, &mut hotbar_log);
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
    mut hotbar_log: ResMut<HotbarSyncLog>,
) {
    for (entity, inventory) in query.iter() {
        let Some(&player_id) = player_map.entity_to_playerid.get(&entity) else {
            continue;
        };
        debug!("[InventorySync] initial sync for player {player_id}");
        send_inventory_sync(player_id, inventory, &blocks, &mut outgoing, &mut hotbar_log);
    }
}

/// Parse click data (`"3"`, `"left 12"`, `"right 7"`) into slot + button.
fn parse_click(data: Option<&str>) -> Option<(usize, bool)> {
    let data = data?;
    let mut index = None;
    let mut right = false;
    for token in data.split_whitespace() {
        if token.eq_ignore_ascii_case("right") {
            right = true;
        } else if let Ok(i) = token.parse::<usize>() {
            index = Some(i);
        }
    }
    index.map(|i| (i, right))
}

/// MC click mechanics, applied server-side: slot clicks mutate the Rust
/// inventory + cursor held stack, creative cells give. Runs before the
/// scripting forwarder, so the Lua snapshot already includes the result.
/// Unknown names/indices are warned and dropped (client is untrusted).
fn apply_inventory_clicks(
    mut events: MessageReader<crate::ui::UiActionRequested>,
    player_map: Res<PlayerMap>,
    mut inventories: Query<&mut Inventory>,
    blocks: Res<ServerBlocks>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut hotbar_log: ResMut<HotbarSyncLog>,
) {
    for ev in events.read() {
        if ev.ui_id != "inv" {
            continue;
        }
        let Some(&entity) = player_map.playerid_to_entity.get(&ev.player_id) else {
            continue;
        };
        let Ok(mut inventory) = inventories.get_mut(entity) else {
            warn!("[InvClick] Entity {:?} missing Inventory", entity);
            continue;
        };
        match ev.action.as_str() {
            "inventory:click" => {
                let Some((index, right)) = parse_click(ev.data.as_deref()) else {
                    warn!("[InvClick] Bad click data {:?}", ev.data);
                    continue;
                };
                if index >= exsolis_common::inventory::INVENTORY_SIZE {
                    warn!("[InvClick] Slot {index} out of range");
                    continue;
                }
                if inventory.click(index, right) {
                    let inv = inventory.clone();
                    send_inventory_sync(
                        ev.player_id,
                        &inv,
                        &blocks,
                        &mut outgoing,
                        &mut hotbar_log,
                    );
                }
            }
            "inventory:creative" => {
                let Some(name) = ev.data.as_deref() else {
                    continue;
                };
                let Some(block) = blocks.id(name) else {
                    warn!("[InvClick] Unknown creative block {name:?}");
                    continue;
                };
                // Air and hidden parts are not giveable; items are.
                if blocks.is_air(block)
                    || blocks.0.def(block).is_some_and(|d| d.hidden)
                {
                    warn!("[InvClick] Refused creative {name:?}");
                    continue;
                }
                // Creative grab: full stack to the cursor (MC-like), not
                // into storage. Whatever was held goes back first.
                let dropped = inventory.grab(block);
                if dropped > 0 {
                    warn!("[InvClick] {dropped} held items did not fit — dropped");
                }
                info!("[InvClick] player {:?} grabbed {name:?}", ev.player_id);
                let inv = inventory.clone();
                send_inventory_sync(ev.player_id, &inv, &blocks, &mut outgoing, &mut hotbar_log);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Slot = Option<(String, u8)>;

    fn snap(slots: &[Option<(&str, u8)>], selected: u8) -> HotbarSync {
        (
            slots
                .iter()
                .map(|s| s.map(|(n, a)| (n.to_string(), a)))
                .collect(),
            selected,
        )
    }

    #[test]
    fn hotbar_diff_selection_only_change() {
        let prev = snap(&[Some(("exsolis:stone", 64))], 0);
        let next = snap(&[Some(("exsolis:stone", 64))], 2);
        let (changes, selected) = hotbar_diff(&prev.0, &next.0, prev.1, next.1);
        assert!(changes.is_empty());
        assert_eq!(selected, Some(2));
    }

    #[test]
    fn hotbar_diff_reports_content_change() {
        let prev = snap(&[Some(("exsolis:stone", 64))], 0);
        let next = snap(&[Some(("exsolis:dirt", 32))], 0);
        let (changes, selected) = hotbar_diff(&prev.0, &next.0, prev.1, next.1);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].0, 0);
        assert_eq!(changes[0].1, Some(("exsolis:dirt".to_string(), 32)));
        assert_eq!(selected, None);
    }

    #[test]
    fn hotbar_diff_marks_emptied_slot_with_none_contents() {
        let prev = snap(&[Some(("exsolis:stone", 64)), None], 0);
        let next = snap(&[None, Some(("exsolis:dirt", 8))], 3);
        let (changes, selected) = hotbar_diff(&prev.0, &next.0, prev.1, next.1);
        assert_eq!(changes.len(), 2);
        assert_eq!(changes[0].0, 0);
        assert_eq!(changes[0].1, None);
        assert_eq!(changes[1].0, 1);
        assert_eq!(changes[1].1, Some(("exsolis:dirt".to_string(), 8)));
        assert_eq!(selected, Some(3));
    }

    #[test]
    fn hotbar_diff_nothing_changed_sends_nothing() {
        let prev = snap(&[None, Some(("exsolis:stone", 64))], 1);
        let next = snap(&[None, Some(("exsolis:stone", 64))], 1);
        let (changes, selected) = hotbar_diff(&prev.0, &next.0, prev.1, next.1);
        assert!(changes.is_empty());
        assert_eq!(selected, None);
    }
}
