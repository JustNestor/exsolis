//! Permanent bottom hotbar HUD (client-local, non-modal).
//!
//! Mirrors the server-authoritative [`ClientInventory`]: re-rendered whenever
//! it changes (join sync, give, place, break, wheel/number select). Display
//! only — no clicks, so it never interferes with world input.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::debug;
use exsolis_common::HOTBAR_SIZE;

use crate::{
    blocks::ClientBlocks,
    interaction::ClientInventory,
    net::GameplaySet,
    ui::UiManager,
};

const HUD_ID: &str = "clientui:hud";

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, sync_hotbar_hud.in_set(GameplaySet));
    }
}

fn sync_hotbar_hud(
    mut commands: Commands,
    mut ui: ResMut<UiManager>,
    inventory: Res<ClientInventory>,
    blocks: Option<Res<ClientBlocks>>,
    connected: Option<Res<crate::net::NetConnected>>,
    mut mounted: Local<bool>,
) {
    // No server, no hotbar: the display is purely server state (join sync
    // fills it). Showing the local default offline is what confused offline
    // players.
    let online = connected.as_ref().is_some_and(|c| c.get());
    if !online {
        if *mounted {
            ui.hide(&mut commands, HUD_ID);
            *mounted = false;
        }
        return;
    }
    if !*mounted || inventory.is_changed() {
        let mut cells = String::with_capacity(2048);
        for i in 0..HOTBAR_SIZE {
            let selected = inventory.selected as usize == i;
            let bg = if selected { "#c8c8c8" } else { "#222222cc" };
            let border = if selected { "#ffffff" } else { "#555555" };
            cells.push_str(&format!(
                "<div bg=\"{bg}\" border=\"2\" border-color=\"{border}\" \
                 width=\"44\" height=\"44\" align=\"center\" justify=\"center\">"
            ));
            if let Some((item, count)) = inventory.slots[i] {
                let prefix = match blocks.as_ref().and_then(|b| b.0.def(item)) {
                    Some(def) if !def.placeable => "item#",
                    _ => "block#",
                };
                let img = blocks
                    .as_ref()
                    .and_then(|b| b.name(item))
                    .map(|n| format!("<img src=\"{prefix}{n}\" alt=\"?\" width=\"36\" height=\"36\"/>"))
                    .unwrap_or_default();
                cells.push_str(&img);
                cells.push_str(&format!(
                    "<label position=\"absolute\" right=\"2\" bottom=\"0\" \
                     font-size=\"11\" color=\"#ffffff\">{count}</label>"
                ));
            } else if selected {
                cells.push_str("<label font-size=\"11\" color=\"#888888\">·</label>");
            }
            cells.push_str("</div>");
        }
        let xml = format!(
            "<div anchor=\"bottom\" layout=\"column\" align=\"center\" gap=\"2\" padding=\"0 0 8\">\
             <div layout=\"row\" gap=\"4\" padding=\"4\" bg=\"#00000088\" border=\"2\" border-color=\"#333333\">\
             {cells}</div></div>"
        );
        ui.show(&mut commands, HUD_ID, &xml);
        if !*mounted {
            debug!("hud: mounted permanent hotbar");
        }
        *mounted = true;
    }
}
