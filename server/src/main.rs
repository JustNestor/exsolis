use std::{
    collections::HashMap,
    net::{SocketAddr, UdpSocket},
    ops::{Deref, DerefMut},
    time::{Duration, SystemTime},
};

use bevy_app::{ScheduleRunnerPlugin, prelude::*};
use bevy_ecs::prelude::*;
use bevy_internal::MinimalPlugins;
use bevy_log::{LogPlugin, prelude::*};
use bevy_renet::{
    RenetServer, RenetServerEvent, RenetServerPlugin,
    netcode::{
        NetcodeErrorEvent, NetcodeServerPlugin, NetcodeServerTransport, ServerAuthentication,
        ServerConfig,
    },
    renet::{ChannelConfig, ConnectionConfig, DefaultChannel, SendType, ServerEvent},
};

use exsolis_common::{ClientMessage, PROTOCOL_ID, PlayerId, SERVER_ADDR, ServerMessage, TICK_RATE};

mod assets;
mod blocks;
mod chunks;
mod creative;
mod entities;
mod events;
mod interaction;
mod movement;
mod net;
mod network;
mod scripting;
#[cfg(feature = "advanced_worldgen")]
mod terrain;
mod ts_backend;
mod ts_ecs;
mod typedb;
mod ui;
mod ui_template;

use chunks::*;
use events::*;
use movement::*;
use network::*;

use crate::interaction::InteractionPlugin;
use crate::net::NetPlugin;
use crate::scripting::ScriptingPlugin;
use crate::ui::UiPlugin;
use exsolis_server::gltf_pack;
// mod level;

#[derive(Component)]
pub struct Player(pub PlayerId);

#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Inventory(pub exsolis_common::inventory::Inventory);

/// Item storage owned by something that is not a player: a chest's
/// block entity, a dropped item's stack, a future furnace output slot.
/// Same zero-based slot container the player's `Inventory` uses, so mod
/// code (`mods/container.lua`) treats them identically.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Container(pub exsolis_common::inventory::Inventory);

impl Deref for Container {
    type Target = exsolis_common::inventory::Inventory;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Container {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for Inventory {
    type Target = exsolis_common::inventory::Inventory;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Inventory {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Resource, Debug, Default)]
pub struct PlayerMap {
    pub playerid_to_entity: HashMap<PlayerId, Entity>,
    pub entity_to_playerid: HashMap<Entity, PlayerId>,
}
pub fn connection_config() -> ConnectionConfig {
    let mut config = ConnectionConfig::default();

    for channel in config
        .server_channels_config
        .iter_mut()
        .chain(config.client_channels_config.iter_mut())
    {
        // channel_id 2 == DefaultChannel::ReliableOrdered
        if channel.channel_id == DefaultChannel::ReliableOrdered as u8 {
            channel.max_memory_usage_bytes = 64 * 1024 * 1024;
        }
    }

    config
}

/// Inserts the resource when present (used for the optional type database).
trait InsertIfPresent {
    fn insert_resource_or_web<T: Resource>(&mut self, value: Option<T>) -> &mut Self;
}

impl InsertIfPresent for App {
    fn insert_resource_or_web<T: Resource>(&mut self, value: Option<T>) -> &mut Self {
        if let Some(value) = value {
            self.insert_resource(value);
        }
        self
    }
}

fn main() {
    // Headless server: no window / renderer, and the main loop is paced to
    // TICK_RATE (20 TPS) instead of spinning as fast as the CPU allows.
    // MinimalPlugins already includes TaskPoolPlugin (needed for
    // AsyncComputeTaskPool in chunks.rs) and TimePlugin.
    let tick = Duration::from_secs_f64(1.0 / TICK_RATE as f64);

    App::new()
        .add_plugins((
            MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(tick)),
            LogPlugin::default(),
        ))
        .add_plugins((RenetServerPlugin, NetcodeServerPlugin))
        .add_plugins(NetPlugin)
        .add_plugins(blocks::BlocksPlugin)
        .add_plugins(assets::AssetServerPlugin)
        .add_plugins(entities::EntityPlugin)
        .add_plugins(EventsPlugin)
        .add_plugins(MovementPlugin)
        .add_plugins(ChunksPlugin)
        .add_plugins(InteractionPlugin)
        .add_plugins(creative::CreativePlugin)
        .add_plugins(UiPlugin)
        .add_plugins(ScriptingPlugin)
        .add_plugins(ts_backend::TsScriptingPlugin)
        .insert_resource(RenetServer::new(connection_config()))
        .insert_resource(new_transport())
        .init_resource::<PlayerMap>()
        .insert_resource_or_web(typedb::ServerTypes::load())
        .add_observer(log_netcode_error)
        .run();
}
