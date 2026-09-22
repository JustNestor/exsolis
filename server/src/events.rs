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

pub fn on_server_event(
    event: On<RenetServerEvent>,
    mut server: ResMut<RenetServer>,
    mut commands: Commands,
    mut player_map: ResMut<PlayerMap>,
    positions: Query<&PlayerPosition>,
) {
    match **event {
        ServerEvent::ClientConnected { client_id } => {
            info!("client {client_id} connected");

            // 1. Send all existing connected players to the new client
            for (&existing_id, &existing_entity) in &player_map.playerid_to_entity {
                let join_msg =
                    bincode::serialize(&ServerMessage::PlayerJoined(existing_id)).unwrap();
                server.send_message(client_id, DefaultChannel::ReliableOrdered, join_msg);

                if let Ok(pos) = positions.get(existing_entity) {
                    let move_msg = bincode::serialize(&ServerMessage::PlayerMoved {
                        player_id: existing_id,
                        x: pos.0.x,
                        y: pos.0.y,
                        z: pos.0.z,
                    })
                    .unwrap();
                    server.send_message(client_id, DefaultChannel::ReliableOrdered, move_msg);
                }
            }

            // 2. Initialize inventory and spawn entity for new player
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

            // 3. Broadcast PlayerJoined for the new client to everyone else
            // let join_msg = bincode::serialize(&ServerMessage::PlayerJoined(client_id)).unwrap();
            // server.broadcast_message_except(client_id, DefaultChannel::ReliableOrdered, join_msg);
        }
        ServerEvent::ClientDisconnected { client_id, reason } => {
            info!("client {client_id} disconnected: {reason}");
            if let Some(entity) = player_map.playerid_to_entity.remove(&client_id) {
                player_map.entity_to_playerid.remove(&entity);
                commands.entity(entity).despawn();
            }
            commands.write_message(PlayerLeaved(client_id));

            // Broadcast disconnect packet to remaining peers
            let left_msg = bincode::serialize(&ServerMessage::PlayerLeft(client_id)).unwrap();
            server.broadcast_message(DefaultChannel::ReliableOrdered, left_msg);
        }
    }
}

pub fn recive_packets(
    mut server: ResMut<RenetServer>,
    mut commands: Commands,
    player_map: Res<PlayerMap>,
    mut inputs: Query<&mut PlayerInput>,
) {
    let channels: [u8; 2] = [
        DefaultChannel::ReliableOrdered.into(),
        DefaultChannel::Unreliable.into(),
    ];

    for client_id in server.clients_id() {
        for channel in channels {
            while let Some(bytes) = server.receive_message(client_id, channel) {
                let Ok(msg) = bincode::deserialize::<ClientMessage>(&bytes) else {
                    warn!("bad packet from {client_id}, dropping");
                    continue;
                };

                debug!("Recived packet {:?} from player {}", msg, client_id);

                match msg {
                    ClientMessage::Move { x, y, z } => {
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
}

fn handle_player_join(mut messages: MessageReader<PlayerJoined>, mut server: ResMut<RenetServer>) {
    for message in messages.read() {
        let msg = bincode::serialize(&ServerMessage::PlayerJoined(message.0)).unwrap();
        server.broadcast_message_except(message.0, DefaultChannel::ReliableOrdered, msg);
    }
}
