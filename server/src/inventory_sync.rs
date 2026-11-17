//! Authoritative hotbar/inventory wire sync — the protocol contract every
//! game uses, independent of any creative/inventory UI.
//!
//! `HotbarSyncLog` remembers the last snapshot sent to each player so the
//! hotbar can be broadcast as a small diff (`InventorySyncDiff`) instead of a
//! full resend — and so a no-op select or a click that mutated nothing skips
//! the round trip entirely.
//!
//! The HUD itself (`send_hud`) lives in [`crate::ui`]; this module only owns
//! the *wire* state. Presentation is a game-layer concern.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::debug;
use exsolis_common::{PlayerId, ServerMessage};

use crate::Inventory;
use crate::PlayerMap;
use crate::blocks::ServerBlocks;
use crate::net::{self, OutgoingPacket};

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

pub struct InventorySyncPlugin;

impl Plugin for InventorySyncPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HotbarSyncLog>();
        app.add_systems(
            Update,
            sync_new_inventories
                .after(crate::events::dispatch_incoming)
                .in_set(net::GameplaySet),
        );
    }
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
    crate::ui::send_hud(player_id, inventory, blocks, outgoing);
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
