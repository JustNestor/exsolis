use std::{
    net::{SocketAddr, UdpSocket},
    time::{Duration, SystemTime},
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{LogPlugin, prelude::*};
use bevy_renet::{
    RenetClient, RenetClientPlugin, client_connected,
    netcode::{
        ClientAuthentication, NetcodeClientPlugin, NetcodeClientTransport, NetcodeErrorEvent,
    },
    renet::{ConnectionConfig, DefaultChannel},
};

use exsolis_common::{ClientMessage, PROTOCOL_ID, SERVER_ADDR, ServerMessage};

fn main() {
    App::new()
        .add_plugins((
            bevy_app::TaskPoolPlugin::default(),
            bevy_diagnostic::FrameCountPlugin,
            bevy_time::TimePlugin,
            bevy_app::ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(1.0 / 60.0)),
            LogPlugin::default(),
        ))
        .add_plugins((RenetClientPlugin, NetcodeClientPlugin))
        .insert_resource(RenetClient::new(ConnectionConfig::default()))
        .insert_resource(new_transport())
        .add_observer(log_netcode_error)
        .add_systems(
            Update,
            (send_fake_movement, receive_state)
                .chain()
                .run_if(client_connected),
        )
        .run();
}

fn new_transport() -> NetcodeClientTransport {
    let server_addr: SocketAddr = std::env::var("EXSOLIS_SERVER_ADDR")
        .unwrap_or_else(|_| SERVER_ADDR.to_string())
        .parse()
        .expect("bad EXSOLIS_SERVER_ADDR");
    let socket = UdpSocket::bind("0.0.0.0:0").expect("failed to bind client socket");
    let current_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap();

    let client_id = rand::random::<u64>();

    let authentication = ClientAuthentication::Unsecure {
        client_id,
        protocol_id: PROTOCOL_ID,
        server_addr,
        user_data: None,
    };

    NetcodeClientTransport::new(current_time, authentication, socket)
        .expect("failed to start transport")
}

fn log_netcode_error(error: On<NetcodeErrorEvent>) {
    warn!("netcode error: {}", *error);
}

fn send_fake_movement(mut client: ResMut<RenetClient>) {
    let msg = bincode::serialize(&ClientMessage::Move {
        x: 0.1,
        y: 0.0,
        z: 0.0,
    })
    .unwrap();
    client.send_message(DefaultChannel::Unreliable, msg);
}

fn receive_state(mut client: ResMut<RenetClient>) {
    while let Some(bytes) = client.receive_message(DefaultChannel::ReliableOrdered) {
        let Ok(msg) = bincode::deserialize::<ServerMessage>(&bytes) else {
            warn!("bad packet from server, dropping");
            continue;
        };
        match msg {
            ServerMessage::PlayerJoined(id) => info!("player {id} joined"),
            ServerMessage::PlayerLeft(id) => info!("player {id} left"),
            _ => {}
        }
    }

    // while let Some(bytes) = client.receive_message(DefaultChannel::Unreliable) {
    //     let Ok(ServerMessage::Positions(positions)) = bincode::deserialize::<ServerMessage>(&bytes)
    //     else {
    //         continue;
    //     };
    //     info!("positions: {positions:?}");
    // }
}
