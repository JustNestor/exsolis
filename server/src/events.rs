use super::*;
use bevy_ecs::message::Message;
use bevy_renet::renet::SendType::Unreliable;
use chunks::{PlayerPosition, RenderDistanceRequested};
use glam::{Vec3, Vec4};

pub struct EventsPlugin;

impl Plugin for EventsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PlayerMoved>()
            .add_message::<PlayerJoined>()
            .add_message::<PlayerLeaved>()
            .add_message::<ChatMessage>()
            .add_message::<RenderDistanceRequested>()
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

            // Without this, the player entity has no PlayerPosition, so
            // chunks.rs::compute_desired_chunks' query matches nothing for
            // them and they never get sent a single chunk — this was the
            // root cause of the blue-screen-with-no-blocks symptom.
            let entity = commands
                .spawn((Player(client_id), PlayerPosition(Vec3::new(0.0, 80.0, 0.0))))
                .id();
            player_map.playerid_to_entity.insert(client_id, entity);
            player_map.entity_to_playerid.insert(entity, client_id);

            commands.write_message(PlayerJoined(client_id));

            // let msg = bincode::serialize(&ServerMessage::PlayerJoined(client_id)).unwrap();
            // server.broadcast_message(DefaultChannel::ReliableOrdered, msg);
        }
        ServerEvent::ClientDisconnected { client_id, reason } => {
            info!("client {client_id} disconnected: {reason}");
            if let Some(entity) = player_map.playerid_to_entity.remove(&client_id) {
                player_map.entity_to_playerid.remove(&entity);
                commands.entity(entity).despawn();
            }
            commands.write_message(PlayerLeaved(client_id));
        }
    }
}

/// Units/second the server moves a player along the direction vector
/// their client sends. This is a placeholder authoritative speed —
/// swap for whatever your single-player FlyCamera's `speed`/`sprint_multiplier`
/// should map to server-side once sprint state is part of the packet too.
const SERVER_MOVE_SPEED: f32 = 8.0;

pub fn recive_packets(
    mut server: ResMut<RenetServer>,
    mut commands: Commands,
    player_map: Res<PlayerMap>,
    mut positions: Query<&mut PlayerPosition>,
    time: Res<bevy_time::Time>,
) {
    let dt = time.delta_secs();

    for client_id in server.clients_id() {
        while let Some(bytes) = server.receive_message(client_id, DefaultChannel::Unreliable) {
            let Ok(msg) = bincode::deserialize::<ClientMessage>(&bytes) else {
                warn!("bad packet from {client_id}, dropping");
                continue;
            };

            match msg {
                ClientMessage::Move { x, y, z } => {
                    let Some(&entity) = player_map.playerid_to_entity.get(&client_id) else {
                        continue;
                    };
                    let Ok(mut pos) = positions.get_mut(entity) else {
                        continue;
                    };
                    // x/y/z here is the normalized direction the client is
                    // pressing, not a destination — see player.rs on the
                    // client. No collision/physics yet, same as the rest
                    // of this server; this just gets something visibly
                    // moving so compute_desired_chunks has real input.
                    pos.0 += Vec3::new(x, y, z) * SERVER_MOVE_SPEED * dt;
                }
                ClientMessage::SetRenderDistance(radius) => {
                    commands.write_message(RenderDistanceRequested {
                        player_id: client_id,
                        radius,
                    });
                }
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
