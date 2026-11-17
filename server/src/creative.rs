//! Game-layer inventory interactions: creative give + slot clicks.
//!
//! This is the remaining *game* logic that will move into authored TS
//! bundles when the engine core stops owning a creative mode. The wire
//! contract (`HotbarSyncLog`, `send_inventory_sync`, `hotbar_diff`) now lives
//! in [`crate::inventory_sync`]; HUD presentation lives in [`crate::ui`].
//! Nothing here touches `RenetServer` directly — outbound traffic goes
//! through `net::OutgoingPacket`, flushed centrally by `net.rs`.

use bevy_app::prelude::*;
use bevy_ecs::message::Message;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use exsolis_common::{PlayerId, inventory::MAX_STACK};

use crate::Inventory;
use crate::PlayerMap;
use crate::blocks::ServerBlocks;
use crate::inventory_sync::{HotbarSyncLog, send_inventory_sync};
use crate::net::{self, OutgoingPacket};

pub struct CreativePlugin;

impl Plugin for CreativePlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<CreativeGiveRequested>().add_systems(
            Update,
            handle_creative_give
                .after(crate::events::dispatch_incoming)
                .in_set(net::GameplaySet),
        );
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
