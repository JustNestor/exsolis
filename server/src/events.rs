use crate::interaction::{BreakBlockRequested, PlaceBlockRequested, SelectHotbarRequested};
use crate::net::{IncomingPacket, OutgoingPacket};

use super::*;
use bevy_ecs::message::Message;
use chunks::{PlayerPosition, RenderDistanceRequested};
use exsolis_common::{BlockType, SPAWN_POS};
use glam::Vec3;
use movement::{PlayerInput, PositionSync};

pub struct EventsPlugin;

impl Plugin for EventsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<PlayerMoved>()
            .add_message::<PlayerJoined>()
            .add_message::<PlayerLeaved>()
            .add_message::<ChatMessage>()
            .add_message::<RenderDistanceRequested>()
            .add_systems(
                Update,
                (dispatch_incoming, handle_player_join)
                    .chain()
                    .in_set(net::GameplaySet),
            )
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
    mut commands: Commands,
    mut player_map: ResMut<PlayerMap>,
    positions: Query<&PlayerPosition>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    match **event {
        ServerEvent::ClientConnected { client_id } => {
            info!("client {client_id} connected");

            // 1. Send all existing connected players to the new client
            for (&existing_id, &existing_entity) in &player_map.playerid_to_entity {
                outgoing.write(OutgoingPacket::to(
                    client_id,
                    ServerMessage::PlayerJoined(existing_id),
                ));

                if let Ok(pos) = positions.get(existing_entity) {
                    outgoing.write(OutgoingPacket::to(
                        client_id,
                        ServerMessage::PlayerMoved {
                            player_id: existing_id,
                            x: pos.0.x,
                            y: pos.0.y,
                            z: pos.0.z,
                        },
                    ));
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
            outgoing.write(OutgoingPacket::broadcast(ServerMessage::PlayerLeft(
                client_id,
            )));
        }
    }
}

/// Fans `IncomingPacket`s out into the per-feature request `Message`s
/// (`PlaceBlockRequested`, `RenderDistanceRequested`, ...). Replaces the
/// old `recive_packets`, which used to drain `RenetServer` itself; the
/// actual drain now lives in `net::drain_incoming`, so this only needs to
/// read `IncomingPacket` and route it — same shape as `net.rs` on the
/// client, where gameplay never touches the transport directly.
pub fn dispatch_incoming(
    mut incoming: MessageReader<IncomingPacket>,
    mut commands: Commands,
    player_map: Res<PlayerMap>,
    mut inputs: Query<&mut PlayerInput>,
) {
    for packet in incoming.read() {
        let client_id = packet.player_id;
        debug!("Recived packet {:?} from player {}", packet.msg, client_id);

        match &packet.msg {
            ClientMessage::Move { x, y, z } => {
                let dir = Vec3::new(*x, *y, *z);
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
                    radius: *radius,
                });
            }
            ClientMessage::PlaceBlock { pos, kind } => {
                commands.write_message(PlaceBlockRequested {
                    player_id: client_id,
                    pos: *pos,
                    kind: *kind,
                });
            }
            ClientMessage::BreakBlock { pos } => {
                commands.write_message(BreakBlockRequested {
                    player_id: client_id,
                    pos: *pos,
                });
            }
            ClientMessage::SelectHotbar(index) => {
                commands.write_message(SelectHotbarRequested {
                    player_id: client_id,
                    index: *index,
                });
            }
            _ => {}
        }
    }
}

fn handle_player_join(
    mut messages: MessageReader<PlayerJoined>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for message in messages.read() {
        outgoing.write(OutgoingPacket::broadcast_except(
            message.0,
            ServerMessage::PlayerJoined(message.0),
        ));
    }
}
