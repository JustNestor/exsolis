// Server-driven UI.
//
// Server sends `UiShow { id, xml }` / `UiHide { id }`; the client parses
// the XML (`parse`), maps attributes to `bevy_ui` (`style`), spawns the
// entity tree (`spawn`) and reports clicks back as `UiAction` (`events`).
//
// Requires `crate::net::NetPlugin` to be added (uses `GameplaySet`).
// Input code (mouse look, movement) should check
// [`UiManager::captures_input`] and release the cursor while it is true.
//
// Reserved: action `client:close` closes the screen locally.

use std::collections::{HashMap, HashSet};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_input::{ButtonInput, keyboard::KeyCode, mouse::MouseButton};
use bevy_log::{info, warn};
use bevy_ui::Interaction;
use bevy_window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use exsolis_common::{ServerMessage, UI_MAX_ID_LEN, UI_MAX_OPEN};

use crate::net::{GameplaySet, IncomingPacket, NetConnected};

pub const LOOPBACK_UI_PREFIX: &str = "clientui:";

pub mod client_ui;
#[cfg(debug_assertions)]
pub mod dev_reload;
pub mod drag;
pub mod events;
pub mod parse;
pub mod render_bridge;
pub mod spawn;
pub mod style;

pub mod img_assets;

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

    pub fn has_client_modal(&self) -> bool {
        self.modal
            .iter()
            .any(|id| id.starts_with(LOOPBACK_UI_PREFIX))
    }

    pub fn has_remote_modal(&self) -> bool {
        self.modal
            .iter()
            .any(|id| !id.starts_with(LOOPBACK_UI_PREFIX))
    }

    /// Closes every server-owned modal screen. Returns true if any was closed.
    pub fn close_remote_modals(&mut self, commands: &mut Commands) -> bool {
        let ids: Vec<String> = self
            .modal
            .iter()
            .filter(|id| !id.starts_with(LOOPBACK_UI_PREFIX))
            .cloned()
            .collect();
        let any = !ids.is_empty();
        for id in ids {
            self.hide(commands, &id);
        }
        any
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

    pub fn hide_remote(&mut self, commands: &mut Commands) {
        let ids: Vec<String> = self
            .roots
            .keys()
            .filter(|id| !id.starts_with(LOOPBACK_UI_PREFIX))
            .cloned()
            .collect();
        for id in ids {
            self.hide(commands, &id);
        }
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

pub fn gameplay_input_allowed(
    ui: Res<UiManager>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    mut prev_ok: Local<bool>,
) -> bool {
    let grabbed = windows
        .single()
        .map(|c| c.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    let ok_now = grabbed && !ui.captures_input();
    let allowed = ok_now && *prev_ok;
    *prev_ok = ok_now;
    allowed
}

fn grab(c: &mut CursorOptions) {
    c.grab_mode = CursorGrabMode::Locked;
    c.visible = false;
}

fn release(c: &mut CursorOptions) {
    c.grab_mode = CursorGrabMode::None;
    c.visible = true;
}

/// Owns Esc handling and cursor grab state (replaces `toggle_cursor_grab`).
fn escape_and_cursor(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    ui_interactions: Query<&Interaction>,
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
    mut out: MessageWriter<IncomingPacket>,
    mut had_modal: Local<bool>,
) {
    let Ok(mut cursor) = windows.single_mut() else {
        return;
    };
    let mut closed_by_esc = false;

    if keys.just_pressed(KeyCode::Escape) {
        if ui.has_client_modal() {
            client_ui::mount_main_menu(&mut out);
        } else if ui.has_remote_modal() {
            ui.close_remote_modals(&mut commands);
            release(&mut cursor);
            closed_by_esc = true;
        } else if cursor.grab_mode != CursorGrabMode::None {
            release(&mut cursor);
        } else {
            grab(&mut cursor);
        }
    }

    let modal = ui.captures_input();
    if modal {
        if cursor.grab_mode != CursorGrabMode::None || !cursor.visible {
            release(&mut cursor);
        }
    } else if *had_modal && !closed_by_esc {
        grab(&mut cursor);
    } else if mouse.just_pressed(MouseButton::Left)
        && cursor.grab_mode == CursorGrabMode::None
        && !ui_interactions.iter().any(|i| *i != Interaction::None)
    {
        grab(&mut cursor);
    }
    *had_modal = modal;
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<UiManager>()
            .add_systems(
                Update,
                (
                    handle_ui_packets.in_set(GameplaySet),
                    escape_and_cursor
                        .in_set(GameplaySet)
                        .before(handle_ui_packets),
                    hide_on_disconnect,
                    events::update_hover_styles,
                ),
            )
            .add_plugins(render_bridge::UiRenderBridgePlugin)
            .add_plugins(drag::UiDragPlugin)
            .add_message::<events::LocalUiAction>()
            .add_plugins(client_ui::ClientUiPlugin);

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

fn hide_on_disconnect(
    connected: Res<NetConnected>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if connected.is_changed() && !connected.get() {
        ui.hide_remote(&mut commands);
    }
}
