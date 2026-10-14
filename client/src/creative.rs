//! Creative-tab client state: server block catalog + creative screen.
//!
//! The server sends `ServerMessage::BlockList` once per join and drives the
//! hotbar via `ServerMessage::InventorySync`. This module stores the catalog
//! in [`ServerBlockList`] and applies syncs to [`ClientInventory`]. It also
//! owns the client-local creative screen (`clientui:creative`, toggled with
//! `E`): a paginated grid of `block#` previews mounted directly through
//! [`UiManager`] (no server XML involved). Item clicks send
//! `ClientMessage::CreativeGive`; paging/close stay local.

use bevy_app::prelude::*;
use bevy_ecs::message::{MessageReader, MessageWriter};
use bevy_ecs::prelude::*;
use bevy_input::{ButtonInput, keyboard::KeyCode};
use bevy_log::{debug, info, warn};
use exsolis_common::{ClientMessage, HOTBAR_SIZE, ServerMessage};

use crate::blocks::ClientBlocks;
use crate::interaction::ClientInventory;
use crate::net::{GameplaySet, IncomingPacket, NetConnected, OutgoingPacket};
use crate::ui::UiManager;
use crate::ui::events::LocalUiAction;

/// Sorted block-name catalog from the server (`ServerMessage::BlockList`).
/// Empty until the first list arrives.
#[derive(Resource, Default)]
pub struct ServerBlockList(pub Vec<String>);

/// Current creative-grid page (reset to 0 whenever the screen opens).
#[derive(Resource, Default)]
pub struct CreativePage(pub usize);

/// Set once the first `InventorySync` lands (drives one `info!`, then debug).
#[derive(Resource, Default)]
struct InventorySynced(bool);

pub const CREATIVE_UI_ID: &str = "clientui:creative";
const CREATIVE_TOGGLE_KEY: KeyCode = KeyCode::KeyE;
const COLS: usize = 9;
const ROWS: usize = 5;
const PER_PAGE: usize = COLS * ROWS;

pub struct CreativePlugin;

impl Plugin for CreativePlugin {
    fn build(&self, app: &mut App) {
        info!("creative: toggle key is E (KeyE is unused by the client)");
        app.init_resource::<ServerBlockList>()
            .init_resource::<CreativePage>()
            .init_resource::<InventorySynced>()
            .add_systems(
                Update,
                (
                    receive_creative_messages,
                    handle_creative_actions,
                    toggle_creative,
                )
                    .in_set(GameplaySet),
            );
    }
}

fn receive_creative_messages(
    mut incoming: MessageReader<IncomingPacket>,
    mut list: ResMut<ServerBlockList>,
    blocks: Option<Res<ClientBlocks>>,
    mut inventory: ResMut<ClientInventory>,
    mut synced: ResMut<InventorySynced>,
) {
    for packet in incoming.read() {
        match &packet.0 {
            ServerMessage::BlockList { blocks } => {
                let mut sorted = blocks.clone();
                sorted.sort();
                info!("creative: BlockList received ({} blocks)", sorted.len());
                list.0 = sorted;
            }
            ServerMessage::InventorySync { slots, selected } => {
                if slots.len() != HOTBAR_SIZE {
                    warn!(
                        "creative: InventorySync with {} slots (expected {HOTBAR_SIZE}), dropping",
                        slots.len()
                    );
                    continue;
                }
                if (*selected as usize) >= HOTBAR_SIZE {
                    warn!("creative: InventorySync selected={selected} out of range, dropping");
                    continue;
                }
                let mut new_slots: [Option<(u16, u8)>; HOTBAR_SIZE] = [None; HOTBAR_SIZE];
                for (i, slot) in slots.iter().enumerate() {
                    new_slots[i] = match slot {
                        None => None,
                        Some((name, count)) => {
                            match blocks.as_ref().and_then(|b| b.id(name)) {
                                Some(id) => Some((id, *count)),
                                None => {
                                    warn!(
                                        "creative: InventorySync unknown block `{name}`, leaving slot empty"
                                    );
                                    None
                                }
                            }
                        }
                    };
                }
                inventory.slots = new_slots;
                inventory.selected = *selected;
                if !synced.0 {
                    synced.0 = true;
                    info!("creative: first InventorySync applied (selected {selected})");
                } else {
                    debug!("creative: InventorySync applied (selected {selected})");
                }
            }
            _ => {}
        }
    }
}

/// `data` arrives merged with held modifiers (`"shift minecraft:stone"`),
/// so the block name / page delta is always the last whitespace token.
fn last_token(data: Option<&str>) -> Option<&str> {
    data?.split_whitespace().last()
}

fn handle_creative_actions(
    mut actions: MessageReader<LocalUiAction>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
    list: Res<ServerBlockList>,
    mut page: ResMut<CreativePage>,
) {
    for a in actions.read() {
        if a.ui_id != CREATIVE_UI_ID {
            continue;
        }
        match a.action.as_str() {
            "creative:give" => {
                let Some(name) = last_token(a.data.as_deref()) else {
                    warn!("creative: give clicked without a block name, ignoring");
                    continue;
                };
                if !list.0.iter().any(|b| b == name) {
                    warn!("creative: give for unknown block `{name}`, ignoring");
                    continue;
                }
                info!("creative: requesting `{name}`");
                outgoing.write(OutgoingPacket::reliable(ClientMessage::CreativeGive {
                    name: name.to_owned(),
                }));
            }
            "creative:page" => {
                let pages = page_count(list.0.len()).max(1);
                match last_token(a.data.as_deref()) {
                    Some("-1") => page.0 = page.0.saturating_sub(1),
                    Some("+1") => page.0 = (page.0 + 1).min(pages - 1),
                    other => {
                        warn!("creative: bad page delta `{other:?}`, ignoring");
                        continue;
                    }
                }
                ui.show(&mut commands, CREATIVE_UI_ID, &build_xml(&list.0, page.0));
            }
            "creative:close" => {
                ui.hide(&mut commands, CREATIVE_UI_ID);
                info!("creative: closed");
            }
            other => warn!("creative: unhandled action `{other}`"),
        }
    }
}

fn toggle_creative(
    keys: Res<ButtonInput<KeyCode>>,
    connected: Res<NetConnected>,
    list: Res<ServerBlockList>,
    mut page: ResMut<CreativePage>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if !keys.just_pressed(CREATIVE_TOGGLE_KEY) {
        return;
    }
    if ui.is_open(CREATIVE_UI_ID) {
        ui.hide(&mut commands, CREATIVE_UI_ID);
        info!("creative: closed");
        return;
    }
    if !connected.get() {
        debug!("creative: toggle pressed while disconnected, ignoring");
        return;
    }
    page.0 = 0;
    ui.show(&mut commands, CREATIVE_UI_ID, &build_xml(&list.0, page.0));
    info!("creative: opened ({} blocks)", list.0.len());
}

fn page_count(blocks: usize) -> usize {
    blocks.div_ceil(PER_PAGE)
}

fn build_xml(blocks: &[String], page: usize) -> String {
    let pages = page_count(blocks.len()).max(1);
    let page = page.min(pages - 1);
    let mut s = String::from(
        r##"<div anchor="center" modal="true" layout="column" align="center" justify="center" gap="8" bg="#101418ee" font-size="16" padding="16">"##,
    );
    s.push_str(r##"<label font-size="24">Creative</label>"##);
    if blocks.is_empty() {
        s.push_str(r##"<label>waiting for server list…</label>"##);
    } else {
        let start = page * PER_PAGE;
        for row in 0..ROWS {
            s.push_str(r##"<div layout="row" gap="4">"##);
            for col in 0..COLS {
                let Some(name) = blocks.get(start + row * COLS + col) else {
                    continue;
                };
                // Block names are `[A-Za-z0-9_:./-]` (validated server-side),
                // so they are safe verbatim in attributes.
                s.push_str(&format!(
                    r##"<button id="give-{name}" on-click="creative:give" data="{name}" padding="2" bg="#222a" hover-bg="#444" pressed-bg="#111"><img src="block#{name}" alt="{name}" width="48" height="48"/></button>"##
                ));
            }
            s.push_str("</div>");
        }
        s.push_str(&format!(
            r##"<div layout="row" gap="8" align="center"><button id="prev" on-click="creative:page" data="-1" padding="8" bg="#333" hover-bg="#555">Prev</button><label>Page {}/{}"##,
            page + 1,
            pages
        ));
        s.push_str("</label>");
        s.push_str(r##"<button id="next" on-click="creative:page" data="+1" padding="8" bg="#333" hover-bg="#555">Next</button></div>"##);
    }
    s.push_str(r##"<button id="close" on-click="creative:close" padding="8" bg="#a33" hover-bg="#c44">Close (E)</button></div>"##);
    s
}
