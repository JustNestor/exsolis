use std::{
    collections::HashMap,
    net::{SocketAddr, UdpSocket},
    time::{Duration, SystemTime},
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_internal::DefaultPlugins;
use bevy_log::{LogPlugin, prelude::*};
use bevy_renet::{
    RenetServer, RenetServerEvent, RenetServerPlugin,
    netcode::{
        NetcodeErrorEvent, NetcodeServerPlugin, NetcodeServerTransport, ServerAuthentication,
        ServerConfig,
    },
    renet::{ChannelConfig, ConnectionConfig, DefaultChannel, SendType, ServerEvent},
};

use exsolis_common::{ClientMessage, PROTOCOL_ID, PlayerId, SERVER_ADDR, ServerMessage};

mod chunks;
mod events;
mod network;

use chunks::*;
use events::*;
use network::*;
// mod level;

#[derive(Component)]
pub struct Player(PlayerId);

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

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, LogPlugin::default()))
        .add_plugins((RenetServerPlugin, NetcodeServerPlugin))
        .add_plugins(EventsPlugin)
        .add_plugins(ChunksPlugin)
        .insert_resource(RenetServer::new(connection_config()))
        .insert_resource(new_transport())
        .init_resource::<PlayerMap>()
        .add_observer(log_netcode_error)
        .run();
}
