use super::*;
use bevy_ecs::message::Message;
use bevy_renet::renet::SendType::Unreliable;
use glam::Vec3;

pub struct EventsPlugin;

impl Plugin for EventsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PlayerMoved>()
            .add_message::<PlayerJoined>()
            .add_message::<PlayerLeaved>()
            .add_message::<ChatMessage>()
            .add_systems(Update, (recive_packets, handle_player_join))
            .add_observer(on_server_event);
    }
}

#[derive(Message)]
pub struct PlayerMoved(pub Vec3);

#[derive(Message)]
pub struct PlayerJoined(pub PlayerId);

#[derive(Message)]
pub struct PlayerLeaved(pub PlayerId);

#[derive(Message)]
pub struct ChatMessage {
    author: Entity,
    message: String,
}

// #[derive(Message)]
// pub struct BlockPlaced {
//     author: Entity,
//     message: String,
// }

pub fn on_server_event(
    event: On<RenetServerEvent>,
    // server: Res<RenetServer>,
    mut commands: Commands,
    mut player_map: ResMut<PlayerMap>,
) {
    match **event {
        ServerEvent::ClientConnected { client_id } => {
            info!("client {client_id} connected");
            commands.write_message(PlayerJoined(client_id));

            // let msg = bincode::serialize(&ServerMessage::PlayerJoined(client_id)).unwrap();
            // server.broadcast_message(DefaultChannel::ReliableOrdered, msg);
        }
        ServerEvent::ClientDisconnected { client_id, reason } => {
            info!("client {client_id} disconnected: {reason}");

            commands.write_message(PlayerLeaved(client_id));

            // let msg = bincode::serialize(&ServerMessage::PlayerLeft(client_id)).unwrap();
            // server.broadcast_message(DefaultChannel::ReliableOrdered, msg);
        }
    }
}

pub fn recive_packets(mut server: ResMut<RenetServer>) {
    for client_id in server.clients_id() {
        while let Some(bytes) = server.receive_message(client_id, DefaultChannel::Unreliable) {
            let Ok(msg) = bincode::deserialize::<ClientMessage>(&bytes) else {
                warn!("bad packet from {client_id}, dropping");
                continue;
            };
            info!("recived packet from {client_id}");

            match msg {
                ClientMessage::Move { x, y, z } => {}
                _ => {}
            }
        }
    }
}

fn handle_player_join(mut messages: MessageReader<PlayerJoined>, mut server: ResMut<RenetServer>) {
    for message in messages.read() {
        let msg = bincode::serialize(&ServerMessage::PlayerJoined(message.0)).unwrap();
        server.broadcast_message_except(message.0, DefaultChannel::ReliableOrdered, msg);
    }
}
