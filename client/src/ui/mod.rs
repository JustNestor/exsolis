//! Server-driven UI.
//!
//! Server sends `UiShow { id, xml }` / `UiHide { id }`; the client parses
//! the XML (`parse`), maps attributes to `bevy_ui` (`style`), spawns the
//! entity tree (`spawn`) and reports clicks back as `UiAction` (`events`).
//!
//! Requires `crate::net::NetPlugin` to be added (uses `GameplaySet`).
//! Input code (mouse look, movement) should check
//! [`UiManager::captures_input`] and release the cursor while it is true.
//!
//! Reserved: action `client:close` closes the screen locally.

use std::collections::{HashMap, HashSet};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use exsolis_common::{ServerMessage, UI_MAX_ID_LEN, UI_MAX_OPEN};

use crate::net::{GameplaySet, IncomingPacket, NetConnected};

#[cfg(debug_assertions)]
pub mod dev_reload;
pub mod events;
pub mod parse;
pub mod render_bridge;
pub mod spawn;
pub mod style;

pub use spawn::UiRoot;

/// Tracks which named screens are mounted.
#[derive(Resource, Default)]
pub struct UiManager {
    roots: HashMap<String, Entity>,
    /// Screens whose root has `modal="true"`.
    modal: HashSet<String>,
}

impl UiManager {
    pub fn is_open(&self, ui_id: &str) -> bool {
        self.roots.contains_key(ui_id)
    }

    /// True while any `modal="true"` screen is mounted. Gameplay input
    /// (camera, block break/place) should back off while this holds.
    pub fn captures_input(&self) -> bool {
        !self.modal.is_empty()
    }

    /// Parse + mount (replacing any screen with the same id). On error the
    /// old screen is left untouched.
    pub fn show(&mut self, commands: &mut Commands, ui_id: &str, xml: &str) {
        if ui_id.is_empty() || ui_id.len() > UI_MAX_ID_LEN {
            warn!("ui: rejecting show with bad id length ({})", ui_id.len());
            return;
        }
        if !self.roots.contains_key(ui_id) && self.roots.len() >= UI_MAX_OPEN {
            warn!("ui: too many open screens, ignoring `{ui_id}`");
            return;
        }
        let parsed = match parse::parse_ui(xml) {
            Ok(p) => p,
            Err(e) => {
                warn!("ui `{ui_id}`: {e}; keeping previous version");
                return;
            }
        };

        self.hide(commands, ui_id);
        let root = spawn::spawn_ui(commands, ui_id, &parsed);
        self.roots.insert(ui_id.to_owned(), root);
        if parsed.root.attrs.get("modal").is_some_and(|v| v == "true") {
            self.modal.insert(ui_id.to_owned());
        }
        info!("ui `{ui_id}` mounted");
    }

    pub fn hide(&mut self, commands: &mut Commands, ui_id: &str) {
        self.modal.remove(ui_id);
        if let Some(root) = self.roots.remove(ui_id) {
            // Despawn is recursive over `Children`.
            commands.entity(root).despawn();
        }
    }

    pub fn hide_all(&mut self, commands: &mut Commands) {
        self.modal.clear();
        for (_, root) in self.roots.drain() {
            commands.entity(root).despawn();
        }
    }
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiManager>()
            .add_systems(
                Update,
                (
                    handle_ui_packets.in_set(GameplaySet),
                    hide_on_disconnect,
                    events::update_hover_styles,
                ),
            )
            .add_plugins(render_bridge::UiRenderBridgePlugin);

        // Debug-only local XML loader — see dev_reload.rs. Compiled out of
        // release builds entirely (not just inactive), so there's no risk
        // of it shipping or of assets/xml/ being expected at runtime.
        #[cfg(debug_assertions)]
        app.add_plugins(dev_reload::UiDevReloadPlugin);
    }
}

fn handle_ui_packets(
    mut incoming: MessageReader<IncomingPacket>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    for IncomingPacket(msg) in incoming.read() {
        match msg {
            ServerMessage::UiShow { id, xml } => ui.show(&mut commands, id, xml),
            ServerMessage::UiHide { id } => ui.hide(&mut commands, id),
            _ => {}
        }
    }
}

/// Screens are server state: when the connection drops they must not linger
/// (a stale modal would also keep the cursor captured forever).
fn hide_on_disconnect(
    connected: Res<NetConnected>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if connected.is_changed() && !connected.get() {
        ui.hide_all(&mut commands);
    }
}
