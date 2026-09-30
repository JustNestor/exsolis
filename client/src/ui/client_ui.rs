// src/ui/client_ui.rs

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use bevy_renet::netcode::{NetcodeClientTransport, NetcodeErrorEvent};
use exsolis_common::ServerMessage;

use super::events::LocalUiAction;
use crate::net::{GameplaySet, IncomingPacket, default_server_addr, start_connection};

pub const MAIN_MENU: &str = "clientui:main_menu";

const MAIN_MENU_XML: &str = r##"
<div anchor="center" modal="true" layout="column" width="100%" height="100%"
     align="center" justify="center" gap="12" bg="#101418ee" font-size="20">
  <label font-size="44">Exsolis</label>
  <button id="standard" on-click="clientui:standard_server" width="280" padding="12"
          bg="#2a7a4a" hover-bg="#35995c" pressed-bg="#1c5a36">Standard server</button>
  <button id="direct" on-click="clientui:direct_connect" width="280" padding="12"
          bg="#2a5a9a" hover-bg="#3574c2" pressed-bg="#1c3f6e">Direct connect</button>
</div>"##;

const DIRECT_MENU_XML: &str = r##"
<div anchor="center" modal="true" layout="column" width="100%" height="100%"
     align="center" justify="center" gap="12" bg="#101418ee" font-size="20">
  <label font-size="32">Direct connect</label>
  <button id="connect" on-click="clientui:connect" width="280" padding="12"
          bg="#2a7a4a" hover-bg="#35995c" pressed-bg="#1c5a36">Connect</button>
  <button id="back" on-click="clientui:back" width="280" padding="12"
          bg="#555" hover-bg="#777" pressed-bg="#333">Back</button>
</div>"##;

pub struct ClientUiPlugin;

impl Plugin for ClientUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, show_main_menu)
            .add_systems(
                Update,
                handle_actions
                    .in_set(GameplaySet)
                    .before(super::handle_ui_packets),
            )
            .add_observer(menu_on_net_error);
    }
}

pub fn mount(out: &mut MessageWriter<IncomingPacket>, xml: &str) {
    out.write(IncomingPacket(ServerMessage::UiShow {
        id: MAIN_MENU.to_owned(),
        xml: xml.to_owned(),
    }));
}

pub fn mount_main_menu(out: &mut MessageWriter<IncomingPacket>) {
    mount(out, MAIN_MENU_XML);
}

fn show_main_menu(mut out: MessageWriter<IncomingPacket>) {
    mount_main_menu(&mut out);
}

fn connect(commands: &mut Commands, out: &mut MessageWriter<IncomingPacket>) {
    match start_connection(commands, default_server_addr()) {
        Ok(()) => {
            out.write(IncomingPacket(ServerMessage::UiHide {
                id: MAIN_MENU.to_owned(),
            }));
        }
        Err(e) => {
            warn!("connect failed: {e}");
            mount_main_menu(out);
        }
    }
}

fn handle_actions(
    mut actions: MessageReader<LocalUiAction>,
    mut out: MessageWriter<IncomingPacket>,
    mut commands: Commands,
) {
    for a in actions.read() {
        match a.action.as_str() {
            "clientui:standard_server" | "clientui:connect" => connect(&mut commands, &mut out),
            "clientui:direct_connect" => mount(&mut out, DIRECT_MENU_XML),
            "clientui:back" => mount_main_menu(&mut out),
            other => warn!("unhandled client ui action `{other}`"),
        }
    }
}

fn menu_on_net_error(
    _: On<NetcodeErrorEvent>,
    mut commands: Commands,
    mut out: MessageWriter<IncomingPacket>,
) {
    commands.remove_resource::<NetcodeClientTransport>();
    mount_main_menu(&mut out);
}
