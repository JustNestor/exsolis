//! DEV-ONLY: load `assets/xml/ui.xml` straight off disk instead of waiting
//! on a server `UiShow` packet, so a UI screen can be iterated on without a
//! running server or a network round-trip.
//!
//! `#[cfg(debug_assertions)]`'d out of the module tree entirely in release
//! builds (see `ui/mod.rs`) — this is temporary scaffolding for testing
//! `parse.rs`/`style.rs`/`spawn.rs`, not part of the real SDUI protocol.
//! The mounted screen's `ui_id` is always `"dev"`, so it can't collide with
//! anything a real server sends.
//!
//! Bound to F5 (press to (re)load or reload after an edit) rather than
//! polling the file every frame, since this is meant to be a manual "check
//! my edit" action, not a hot-reload watcher.
//!
//! Path is resolved relative to the CURRENT WORKING DIRECTORY the way
//! `texture.rs`'s `assets/` scan already is (`Path::new("assets")`), i.e.
//! run `cargo run` from the workspace root, not from `client/`.

use std::path::Path;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_input::{ButtonInput, keyboard::KeyCode};
use bevy_log::{info, warn};

use super::UiManager;

/// Screen id used for everything loaded this way. Chosen so it can never
/// collide with a real server-sent `ui_id` — those are the mod's own
/// namespace, this is a fixed dev sentinel.
pub const DEV_UI_ID: &str = "dev";

/// Relative to the process's current working directory, same convention
/// `texture.rs` already uses for `assets/`.
const DEV_XML_PATH: &str = "assets/xml/ui.xml";

pub struct UiDevReloadPlugin;

impl Plugin for UiDevReloadPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, reload_on_key);
    }
}

fn reload_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    mut ui: ResMut<UiManager>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::F5) {
        return;
    }
    load_dev_ui(&mut ui, &mut commands);
}

/// Reads `assets/xml/ui.xml` and mounts it as `ui_id = "dev"`. Public so it
/// can also be called once at startup if you want the screen up
/// immediately rather than waiting for the first F5 — see the commented
/// system registration below.
pub fn load_dev_ui(ui: &mut UiManager, commands: &mut Commands) {
    let path = Path::new(DEV_XML_PATH);
    let xml = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            warn!("ui dev-reload: couldn't read {DEV_XML_PATH}: {e}");
            return;
        }
    };
    // `UiManager::show` already does parse + warn-on-error + replace-if-
    // already-mounted; nothing dev-reload-specific needed beyond the read.
    ui.show(commands, DEV_UI_ID, &xml);
    info!("ui dev-reload: loaded {DEV_XML_PATH}");
}

// To load on startup instead of waiting for the first F5, add this to
// `UiDevReloadPlugin::build` above:
//
//   app.add_systems(Startup, |mut ui: ResMut<UiManager>, mut commands: Commands| {
//       load_dev_ui(&mut ui, &mut commands);
//   });
//
// Left as a manual opt-in rather than the default: an always-loaded dev
// screen is easy to forget about and mistake for real server content
// while testing other things.
