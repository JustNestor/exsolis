use crate::interaction::{BreakBlockRequested, PlaceBlockRequested, SelectHotbarRequested};

use super::*;
use bevy_ecs::message::Message;
use bevy_renet::renet::SendType::Unreliable;
use chunks::{PlayerPosition, RenderDistanceRequested};
use exsolis_common::{BlockType, SPAWN_POS};
use glam::{Vec3, Vec4};
use movement::{PlayerInput, PositionSync};

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

            let mut inventory = crate::Inventory::default();
            for block in [
                BlockType::Dirt,
                BlockType::Stone,
                BlockType::Sand,
                BlockType::Snow,
                BlockType::Glass,
            ] {
                inventory.add_item(block, exsolis_common::inventory::MAX_STACK);
            }

            let entity = commands
                .spawn((
                    Player(client_id),
                    PlayerPosition(Vec3::from(SPAWN_POS)),
                    PlayerInput::default(),
                    PositionSync::default(),
                    inventory,
                ))
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

/// All client -> server messages arrive on ReliableOrdered. Clients only
/// send movement when it *changes*, so a lost or reordered packet would
/// leave the server with a stale direction (e.g. a dropped "stop" = the
/// player runs forever). Reliable + ordered rules that out, and the
/// client->server ReliableOrdered channel only carries these tiny messages
/// (chunk data uses the separate server->client channel), so there's no
/// head-of-line blocking behind chunk streaming.
pub fn recive_packets(
    mut server: ResMut<RenetServer>,
    mut commands: Commands,
    player_map: Res<PlayerMap>,
    mut inputs: Query<&mut PlayerInput>,
) {
    for client_id in server.clients_id() {
        while let Some(bytes) = server.receive_message(client_id, DefaultChannel::ReliableOrdered) {
            let Ok(msg) = bincode::deserialize::<ClientMessage>(&bytes) else {
                warn!("bad packet from {client_id}, dropping");
                continue;
            };

            match msg {
                ClientMessage::Move { x, y, z } => {
                    // x/y/z is the direction the client is moving in (zero =
                    // stopped), not a destination. It's stored and applied
                    // every tick by movement.rs::apply_player_input.
                    let dir = Vec3::new(x, y, z);
                    if !dir.is_finite() {
                        continue;
                    }
                    let Some(&entity) = player_map.playerid_to_entity.get(&client_id) else {
                        continue;
                    };
                    let Ok(mut input) = inputs.get_mut(entity) else {
                        continue;
                    };
                    // Clamp so a malicious client can't send (1000, 0, 0) to speed-hack.
                    input.0 = dir.clamp_length_max(1.0);
                }
                ClientMessage::SetRenderDistance(radius) => {
                    commands.write_message(RenderDistanceRequested {
                        player_id: client_id,
                        radius,
                    });
                }
                ClientMessage::PlaceBlock { pos, kind } => {
                    commands.write_message(PlaceBlockRequested {
                        player_id: client_id,
                        pos,
                        kind,
                    });
                }
                ClientMessage::BreakBlock { pos } => {
                    commands.write_message(BreakBlockRequested {
                        player_id: client_id,
                        pos,
                    });
                }
                ClientMessage::SelectHotbar(index) => {
                    commands.write_message(SelectHotbarRequested {
                        player_id: client_id,
                        index,
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
