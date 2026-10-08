use crate::chunk::CHUNK_SIZE_I32;
use bevy_app::prelude::*;
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use exsolis_common::{ChunkCoord, PlayerId, SPAWN_POS, ServerMessage};
use glam::Vec3;

use crate::{
    chunk::Chunk,
    chunk_registry::ChunkRegistry,
    interaction::ClientInventory,
    net::{GameplaySet, IncomingPacket},
    player::ServerTarget,
};

#[derive(Component)]
pub struct RemotePlayer(pub PlayerId);

#[derive(Resource, Default)]
pub struct LocalPlayerId(pub Option<PlayerId>);

pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkRegistry>()
            .init_resource::<LocalPlayerId>()
            .add_systems(Update, receive_chunk_updates.in_set(GameplaySet));
    }
}

fn receive_chunk_updates(
    mut commands: Commands,
    mut incoming: MessageReader<IncomingPacket>,
    mut registry: ResMut<ChunkRegistry>,
    mut inventory: ResMut<ClientInventory>,
    mut local_player_id: ResMut<LocalPlayerId>,
    mut chunks: Query<&mut Chunk>,
    remote_players: Query<(Entity, &RemotePlayer)>,
) {
    for packet in incoming.read() {
        match &packet.0 {
            ServerMessage::YourPlayerId(id) => local_player_id.0 = Some(*id),
            ServerMessage::ChunkData { coord, blocks } => {
                let Some(chunk) = Chunk::from_packed(*coord, blocks) else {
                    warn!("malformed chunk {coord:?}, dropping");
                    continue;
                };
                if let Some(entity) = registry.entity_at(*coord) {
                    commands.entity(entity).insert(chunk);
                } else {
                    let entity = commands.spawn(chunk).id();
                    registry.insert(*coord, entity);
                }
            }
            ServerMessage::BlockUpdate { pos, kind } => {
                let coord = ChunkCoord::new(
                    pos.x.div_euclid(CHUNK_SIZE_I32),
                    pos.y.div_euclid(CHUNK_SIZE_I32),
                    pos.z.div_euclid(CHUNK_SIZE_I32),
                );
                let lx = pos.x.rem_euclid(CHUNK_SIZE_I32) as usize;
                let ly = pos.y.rem_euclid(CHUNK_SIZE_I32) as usize;
                let lz = pos.z.rem_euclid(CHUNK_SIZE_I32) as usize;

                if let Some(entity) = registry.entity_at(coord) {
                    if let Ok(mut chunk) = chunks.get_mut(entity) {
                        chunk.set(lx, ly, lz, *kind);
                    }
                }
            }
            ServerMessage::ChunkUnload { coords } => {
                if let Some(entity) = registry.entity_at(*coords) {
                    commands.entity(entity).despawn();
                    registry.remove(*coords);
                }
            }
            ServerMessage::YourInventory { slots, selected } => {
                inventory.slots = *slots;
                inventory.selected = *selected;
            }
            ServerMessage::PlayerJoined(id) => {
                let exists = remote_players.iter().any(|(_, p)| p.0 == *id);
                if !exists {
                    commands.spawn((
                        RemotePlayer(*id),
                        bevy_transform::prelude::Transform::from_translation(Vec3::from(SPAWN_POS)),
                        ServerTarget(Some(Vec3::from(SPAWN_POS))),
                    ));
                }
            }
            ServerMessage::PlayerLeft(id) => {
                for (entity, p) in &remote_players {
                    if p.0 == *id {
                        commands.entity(entity).despawn();
                    }
                }
            }
            _ => {}
        }
    }
}
