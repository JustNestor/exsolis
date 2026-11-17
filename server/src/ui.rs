//! Server-side handling of client-reported SDUI interactions.
//!
//! Mirrors `interaction.rs`'s shape (as seen from `events.rs`): a
//! `dispatch_incoming` match arm turns the raw `ClientMessage` into a
//! `*Requested` `Message`, and a system here owns everything downstream
//! of that. `UiActionRequested` / `UiDragRequested` are deliberately NOT
//! "the mod's problem to validate" — every field the client controls
//! (`ui_id`, `element_id`, `action`, `data`) is untrusted per the
//! doc-comments on `ClientMessage::UiAction`/`UiDrag` in `lib.rs`, and a
//! bug in a mod script must never be the only thing standing between a
//! malformed/spoofed packet and a scripting-layer crash. So the checks
//! that don't need any modding-specific knowledge — length limits, "is
//! this ui_id actually open for this player", "is this the reserved
//! `client:close` action" — happen here, in Rust, before a mod ever sees
//! the event. What a mod does in response (unlock a chest, move an item,
//! deny an interaction) is out of scope for this module entirely.
//!
//! What this module explicitly does NOT do, per the "inventory API is
//! being deprecated" note: no `Inventory` access, no slot semantics, no
//! opinion about what `source_id`/`target_id`/`action` *mean*. That is
//! 100% moved into modding space — see the note at the bottom of
//! `scripting.rs` for what's still needed there to make that work.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;

use exsolis_common::{PlayerId, ServerMessage, UI_DRAG_MAX_ID_LEN, UI_MAX_ACTION_LEN, UI_MAX_ID_LEN, UI_MAX_OPEN};

use crate::Inventory;
use crate::blocks::ServerBlocks;
use crate::net::{GameplaySet, IncomingPacket, OutgoingPacket};

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<UiActionRequested>()
            .add_message::<UiDragRequested>()
            .init_resource::<OpenScreens>()
            .add_systems(Update, dispatch_ui_packets.in_set(GameplaySet))
            // Needs to observe this frame's `UiShow`/`UiHide` traffic
            // (queued by scripting's `drain_script_requests`) before
            // `net::flush_outgoing` drains the same queue. See the fix in
            // `scripting.rs::ScriptingPlugin` — `drain_script_requests` is
            // now explicitly `.before(net::flush_outgoing)`, so anchoring
            // to it here (rather than trusting transitive ordering through
            // `GameplaySet`) is guaranteed correct.
            .add_systems(
                Update,
                track_open_screens
                    .after(crate::scripting::drain_script_requests)
                    .before(crate::net::flush_outgoing),
            );
    }
}

/// Render `assets/xml/hud.xml` for one player and mount it as the
/// non-modal `hud` screen. Called by the inventory wire-sync path only
/// when the hotbar actually changed; presentation belongs here in the UI
/// module, not in the protocol/inventory state module.
pub(crate) fn send_hud(
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
                ServerMessage::UiShow { id: "hud".to_string(), xml },
            ));
        }
        Err(e) => warn!("[hud] render failed for player {player_id}: {e}"),
    }
}

/// A validated `UiAction`, ready for a mod's `on("ui_action", ...)`
/// handler. Reaching a mod at all already means: the ids were within
/// length limits, the screen was actually open for this player per the
/// server's own bookkeeping, and it wasn't the reserved `client:close`
/// (which never leaves the client, see `events.rs::CLIENT_CLOSE` on the
/// client side — but see the note in `dispatch_ui_packets` about why the
/// server still can't fully trust that).
#[derive(Message, Debug, Clone)]
pub struct UiActionRequested {
    pub player_id: PlayerId,
    pub ui_id: String,
    pub element_id: String,
    pub action: String,
    pub data: Option<String>,
}

/// A validated `UiDrag`. Same trust boundary as above, plus: `source_id`
/// and `target_id` both passed length checks and neither is empty. This
/// does NOT mean the drag is semantically legal (that a slot with that id
/// exists, that the player owns it, that the item fits) — only that it's
/// well-formed enough to be worth a mod's attention. All of that
/// remaining validation belongs to whatever owns slot semantics in
/// modding space.
#[derive(Message, Debug, Clone)]
pub struct UiDragRequested {
    pub player_id: PlayerId,
    pub ui_id: String,
    pub source_id: String,
    pub target_id: String,
    pub data: Option<String>,
}

/// Which `ui_id`s the server believes are open for which players, purely
/// from watching outbound `UiShow`/`UiHide` traffic (see
/// `track_open_screens`). This is bookkeeping, not authority over what a
/// mod is allowed to show — it exists only so `dispatch_ui_packets` can
/// reject a `UiAction`/`UiDrag` for a screen the server never told this
/// client about (a spoofed or stale `ui_id`) without needing to ask
/// scripting's own state.
#[derive(Resource, Default)]
pub struct OpenScreens {
    /// player_id -> set of ui_ids currently shown to them.
    open: std::collections::HashMap<PlayerId, std::collections::HashSet<String>>,
}

impl OpenScreens {
    pub fn is_open(&self, player_id: PlayerId, ui_id: &str) -> bool {
        self.open
            .get(&player_id)
            .is_some_and(|set| set.contains(ui_id))
    }

    fn mark_open(&mut self, player_id: PlayerId, ui_id: &str) {
        let set = self.open.entry(player_id).or_default();
        if set.len() >= UI_MAX_OPEN && !set.contains(ui_id) {
            // Same cap the client enforces on itself (`UiManager::show`);
            // mirroring it here means a mod that leaks `UiShow`s without
            // matching `UiHide`s can't grow this unboundedly for a
            // long-connected player. The client would already be
            // rejecting the (UI_MAX_OPEN + 1)th screen on its own side,
            // so this doesn't change real behavior — it just keeps our
            // bookkeeping from drifting from what the client will
            // actually have mounted.
            warn!("ui: {player_id} has {UI_MAX_OPEN}+ screens tracked open, not tracking more");
            return;
        }
        set.insert(ui_id.to_owned());
    }

    fn mark_closed(&mut self, player_id: PlayerId, ui_id: &str) {
        if let Some(set) = self.open.get_mut(&player_id) {
            set.remove(ui_id);
        }
    }

    fn clear_player(&mut self, player_id: PlayerId) {
        self.open.remove(&player_id);
    }
}

/// Watches outgoing `UiShow`/`UiHide` (however they were queued — a mod's
/// `api.send`, a future non-mod system, anything going through
/// `OutgoingPacket`) so `OpenScreens` stays truthful without every call
/// site that shows UI having to remember to also update it. `OutgoingPacket`
/// supports multiple independent `MessageReader`s (same guarantee
/// `net.rs`'s own module doc relies on), so reading it here doesn't
/// interfere with `flush_outgoing`'s own drain of the same queue — see
/// `UiPlugin::build` for why this system is explicitly ordered between
/// `drain_script_requests` (the producer) and `flush_outgoing` (the other
/// consumer) rather than left to transitive ordering.
fn track_open_screens(
    mut outgoing: MessageReader<crate::net::OutgoingPacket>,
    mut screens: ResMut<OpenScreens>,
    mut disconnects: MessageReader<crate::events::PlayerLeaved>,
) {
    use crate::net::{Payload, SendTarget};
    use exsolis_common::ServerMessage;

    for packet in outgoing.read() {
        let Payload::Msg(msg) = &packet.payload else {
            continue;
        };
        let (id, opening) = match msg {
            ServerMessage::UiShow { id, .. } => (id, true),
            ServerMessage::UiHide { id } => (id, false),
            _ => continue,
        };
        match packet.target {
            SendTarget::One(player_id) => {
                if opening {
                    screens.mark_open(player_id, id);
                } else {
                    screens.mark_closed(player_id, id);
                }
            }
            // Broadcast UiShow/UiHide would need every connected player_id
            // to update per-player bookkeeping; not needed yet since every
            // UiShow call site so far targets one player (a screen is
            // inherently per-viewer). If a mod ever wants a broadcast
            // UiShow, extend this arm alongside it.
            SendTarget::All | SendTarget::AllExcept(_) => {
                warn!(
                    "ui: broadcast UiShow/UiHide isn't tracked by OpenScreens yet, \
                     UiAction/UiDrag validation may reject it incorrectly"
                );
            }
        }
    }

    for leave in disconnects.read() {
        screens.clear_player(leave.0);
    }
}

/// Turns `ClientMessage::UiAction`/`UiDrag` into their `*Requested`
/// counterparts, after the checks that don't need modding-specific
/// knowledge. This is a SEPARATE system from `events::dispatch_incoming`
/// (not another match arm there) so `interaction`/`movement`/etc never
/// need to know UI validation exists, and so this file is the one place
/// to look when tightening what's accepted from the wire.
// `pub(crate)` so `scripting.rs` can order its forwarding systems
// `.after(dispatch_ui_packets)` explicitly (same reasoning as
// `flush_outgoing`/`drain_script_requests` above).
pub(crate) fn dispatch_ui_packets(
    mut incoming: MessageReader<IncomingPacket>,
    screens: Res<OpenScreens>,
    mut ui_action: MessageWriter<UiActionRequested>,
    mut ui_drag: MessageWriter<UiDragRequested>,
) {
    use exsolis_common::ClientMessage;

    for packet in incoming.read() {
        let player_id = packet.player_id;

        match &packet.msg {
            ClientMessage::UiAction {
                ui_id,
                element_id,
                action,
                data,
            } => {
                // `client:close` is documented (client `events.rs`) as
                // "closes locally, nothing is sent" — a client that DOES
                // send it anyway is either an old build or not a real
                // client at all. Either way there's nothing for a mod to
                // do with it, so it's dropped here rather than forwarded
                // as a no-op event every mod would have to learn to
                // ignore.
                if action == "client:close" {
                    continue;
                }
                if !validate_ids(player_id, ui_id, &screens) {
                    continue;
                }
                if element_id.is_empty() || element_id.len() > UI_MAX_ID_LEN {
                    warn!("ui: {player_id} sent bad element_id length, dropping");
                    continue;
                }
                if action.len() > UI_MAX_ACTION_LEN
                    || data.as_ref().is_some_and(|d| d.len() > UI_MAX_ACTION_LEN)
                {
                    warn!("ui: {player_id} sent oversized action/data, dropping");
                    continue;
                }
                ui_action.write(UiActionRequested {
                    player_id,
                    ui_id: ui_id.clone(),
                    element_id: element_id.clone(),
                    action: action.clone(),
                    data: data.clone(),
                });
            }
            ClientMessage::UiDrag {
                ui_id,
                source_id,
                target_id,
                data,
            } => {
                if !validate_ids(player_id, ui_id, &screens) {
                    continue;
                }
                if source_id.is_empty()
                    || target_id.is_empty()
                    || source_id.len() > UI_DRAG_MAX_ID_LEN
                    || target_id.len() > UI_DRAG_MAX_ID_LEN
                {
                    warn!("ui: {player_id} sent bad drag id length, dropping");
                    continue;
                }
                if source_id == target_id {
                    // Client-side `on_drag_drop` already filters this, but
                    // the server can't trust the client did — a modified
                    // client could send it directly.
                    continue;
                }
                if data.as_ref().is_some_and(|d| d.len() > UI_MAX_ACTION_LEN) {
                    warn!("ui: {player_id} sent oversized drag data, dropping");
                    continue;
                }
                ui_drag.write(UiDragRequested {
                    player_id,
                    ui_id: ui_id.clone(),
                    source_id: source_id.clone(),
                    target_id: target_id.clone(),
                    data: data.clone(),
                });
            }
            _ => {}
        }
    }
}

/// Shared by both match arms: length check + "is this screen actually
/// open for this player" per `OpenScreens`. A `ui_id` a mod never sent
/// this player a `UiShow` for is either a stale client (they had it open,
/// server since hid it — a race, not an attack) or a forged packet; either
/// way there's no mod-side state for it to act on, so it's dropped before
/// ever reaching Lua.
fn validate_ids(player_id: PlayerId, ui_id: &str, screens: &OpenScreens) -> bool {
    if ui_id.is_empty() || ui_id.len() > UI_MAX_ID_LEN {
        warn!("ui: {player_id} sent bad ui_id length, dropping");
        return false;
    }
    if !screens.is_open(player_id, ui_id) {
        warn!("ui: {player_id} sent action for unopened screen `{ui_id}`, dropping");
        return false;
    }
    true
}
