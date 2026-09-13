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
    renet::{ConnectionConfig, DefaultChannel, ServerEvent},
};

use exsolis_common::{ClientMessage, PROTOCOL_ID, PlayerId, SERVER_ADDR, ServerMessage};

mod events;
mod network;

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

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, LogPlugin::default()))
        .add_plugins((RenetServerPlugin, NetcodeServerPlugin))
        .add_plugins(EventsPlugin)
        .insert_resource(RenetServer::new(ConnectionConfig::default()))
        .insert_resource(new_transport())
        .init_resource::<PlayerMap>()
        .add_observer(log_netcode_error)
        .run();
}
