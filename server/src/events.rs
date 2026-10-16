use crate::entities::{DespawnEntityRequest, SpawnEntityRequest, SpawnExtra};
use crate::blocks::ServerBlocks;
use crate::creative::CreativeGiveRequested;
use crate::interaction::{BreakBlockRequested, PlaceBlockRequested, SelectHotbarRequested};
use crate::net::{IncomingPacket, OutgoingPacket};
use crate::scripting::KeybindActionRequested;

use super::*;
use bevy_ecs::message::Message;
use chunks::{PlayerPosition, RenderDistanceRequested};
use exsolis_common::{BlockRegistry, EntityKind, EntityState, SPAWN_POS};
use glam::Vec3;
use movement::{PlayerInput, PlayerLook, PositionSync};
use std::collections::HashSet;

pub struct EventsPlugin;

impl Plugin for EventsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JoiningPlayers>()
            .add_message::<PlayerMoved>()
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
    mut joining: ResMut<JoiningPlayers>,
    entity_registry: Res<crate::entities::EntityRegistry>,
    mut despawns: MessageWriter<DespawnEntityRequest>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    catalog: Res<crate::assets::AssetCatalog>,
    mut http: ResMut<crate::assets::AssetHttpService>,
    mut pending: ResMut<crate::assets::PendingAssetJoins>,
) {
    match **event {
        ServerEvent::ClientConnected { client_id } => {
            info!("client {client_id} connected");
            crate::assets::handle_connected(
                client_id,
                &catalog,
                &mut http,
                &mut pending,
                &mut outgoing,
            );
        }
        ServerEvent::ClientDisconnected { client_id, reason } => {
            info!("client {client_id} disconnected: {reason}");
            crate::assets::handle_disconnected(client_id, &mut pending);
            joining.requested.remove(&client_id);
            if let Some(entity) = player_map.playerid_to_entity.remove(&client_id) {
                player_map.entity_to_playerid.remove(&entity);
                // Despawn through the registry so peers get `DespawnEntity`
                // and the id is retired. The ECS entity itself is removed by
                // `entities::despawn_requested`.
                match entity_registry.id_of(entity) {
                    Some(id) => {
                        despawns.write(DespawnEntityRequest(id));
                    }
                    // No registry entry (a client that never finished the
                    // asset handshake, or one already despawned): take the
                    // entity down directly so it does not linger.
                    None => {
                        commands.entity(entity).despawn();
                    }
                }
            }
            commands.write_message(PlayerLeaved(client_id));

            // Broadcast disconnect packet to remaining peers
            outgoing.write(OutgoingPacket::broadcast(ServerMessage::PlayerLeft(
                client_id,
            )));
        }
    }
}

/// Called only after the client has downloaded, merged, and verified its
/// server-required asset set.
///
/// The player's entity is created through `SpawnEntityRequest` rather than
/// inline, so the id allocation, component set and `SpawnEntity` broadcast
/// all live in one place. That means this function cannot know the new
/// `EntityId` yet — it passes the `PlayerId` as a token and
/// `entities::spawn_requested` fills `PlayerMap` once the id exists.
pub(crate) fn spawn_joined_player(
    client_id: PlayerId,
    commands: &mut Commands,
    player_map: &mut PlayerMap,
    joining: &mut JoiningPlayers,
    positions: &Query<(&PlayerPosition, &PlayerLook)>,
    outgoing: &mut MessageWriter<OutgoingPacket>,
    spawns: &mut MessageWriter<SpawnEntityRequest>,
    blocks: &ServerBlocks,
) {
    if !joining.requested.insert(client_id) {
        return;
    }
    outgoing.write(OutgoingPacket::to(
        client_id,
        ServerMessage::YourPlayerId(client_id),
    ));
    let block_list = crate::creative::sorted_block_names(blocks);
    info!(
        "[BlockList] sending {} blocks to player {}",
        block_list.len(),
        client_id
    );
    outgoing.write(OutgoingPacket::to(
        client_id,
        ServerMessage::BlockList { blocks: block_list },
    ));
    // Everyone already here, so a joining client can render them before it
    // receives any of their movement updates.
    for (&existing_id, &existing_entity) in &player_map.playerid_to_entity {
        outgoing.write(OutgoingPacket::to(
            client_id,
            ServerMessage::PlayerJoined(existing_id),
        ));
        if let Ok((pos, look)) = positions.get(existing_entity) {
            outgoing.write(OutgoingPacket::to(
                client_id,
                ServerMessage::PlayerMoved {
                    player_id: existing_id,
                    x: pos.0.x,
                    y: pos.0.y,
                    z: pos.0.z,
                    yaw: look.yaw,
                    pitch: look.pitch,
                },
            ));
        }
    }

    spawns.write(SpawnEntityRequest {
        kind: EntityKind::Player,
        state: EntityState {
            x: SPAWN_POS[0],
            y: SPAWN_POS[1],
            z: SPAWN_POS[2],
            ..Default::default()
        },
        model: Some(entities::PLAYER_MODEL.to_string()),
        scale: entities::PLAYER_MODEL_SCALE,
        block_pos: None,
        extra: SpawnExtra::Player,
        player_token: Some(client_id),
    });
    commands.write_message(PlayerJoined(client_id));
}

/// Clients whose spawn has already been requested this session.
///
/// `PlayerMap` cannot be used for this guard: the entity is created a frame
/// later by `entities::spawn_requested`, so a second call in the same frame
/// would not see the first one's id yet.
#[derive(Resource, Default)]
pub struct JoiningPlayers {
    pub requested: HashSet<PlayerId>,
}

/// Components only a player entity carries: input, look, the authoritative
/// position, sync state and a starting inventory.
pub(crate) fn attach_player_components(
    commands: &mut Commands,
    entity: Entity,
    registry: &BlockRegistry,
) {
    let mut inventory = crate::Inventory::default();
    for name in [
        "exsolis:dirt",
        "exsolis:stone",
        "exsolis:sand",
        "exsolis:snow",
        "exsolis:glass",
    ] {
        if let Some(block) = registry.id(name) {
            inventory.add_item(block, exsolis_common::inventory::MAX_STACK);
        }
    }
    commands.entity(entity).insert((
        PlayerPosition(Vec3::from(SPAWN_POS)),
        PlayerInput::default(),
        PlayerLook::default(),
        PositionSync::default(),
        inventory,
    ));
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
    mut players: Query<(&mut PlayerInput, &mut PlayerLook)>,
) {
    for packet in incoming.read() {
        let client_id = packet.player_id;
        debug!("Recived packet {:?} from player {}", packet.msg, client_id);

        match &packet.msg {
            ClientMessage::MoveAndLook {
                x,
                y,
                z,
                yaw,
                pitch,
            } => {
                let dir = Vec3::new(*x, *y, *z);
                if !dir.is_finite() || !yaw.is_finite() || !pitch.is_finite() {
                    continue;
                }
                let Some(&entity) = player_map.playerid_to_entity.get(&client_id) else {
                    continue;
                };
                let Ok((mut input, mut look)) = players.get_mut(entity) else {
                    continue;
                };
                input.0 = dir.clamp_length_max(1.0);
                look.yaw = *yaw;
                look.pitch = pitch.clamp(-89_f32.to_radians(), 89_f32.to_radians());
            }
            ClientMessage::Look { yaw, pitch } => {
                if !yaw.is_finite() || !pitch.is_finite() {
                    continue;
                }
                let Some(&entity) = player_map.playerid_to_entity.get(&client_id) else {
                    continue;
                };
                let Ok((_, mut look)) = players.get_mut(entity) else {
                    continue;
                };
                look.yaw = *yaw;
                look.pitch = pitch.clamp(-89_f32.to_radians(), 89_f32.to_radians());
            }
            ClientMessage::SetRenderDistance(radius) => {
                commands.write_message(RenderDistanceRequested {
                    player_id: client_id,
                    radius: *radius,
                });
            }
            ClientMessage::PlaceBlock { pos, kind, face } => {
                commands.write_message(PlaceBlockRequested {
                    player_id: client_id,
                    pos: *pos,
                    kind: kind.clone(),
                    face: *face,
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
            ClientMessage::KeybindAction { id, state } => {
                commands.write_message(KeybindActionRequested {
                    id: id.clone(),
                    state: *state,
                    player_id: client_id,
                });
            }
            ClientMessage::CreativeGive { name } => {
                commands.write_message(CreativeGiveRequested {
                    player_id: client_id,
                    name: name.clone(),
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
