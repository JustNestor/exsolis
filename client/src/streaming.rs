use crate::chunk::CHUNK_SIZE_I32;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use bevy_renet::{RenetClient, renet::DefaultChannel};
use exsolis_common::{ChunkCoord, PlayerId, SPAWN_POS, ServerMessage};
use glam::Vec3;

use crate::{
    chunk::Chunk, chunk_registry::ChunkRegistry, interaction::ClientInventory, player::ServerTarget,
};

#[derive(Component)]
pub struct RemotePlayer(pub PlayerId);

pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkRegistry>()
            .add_systems(Update, receive_chunk_updates);
    }
}

fn receive_chunk_updates(
    mut commands: Commands,
    mut client: ResMut<RenetClient>,
    mut registry: ResMut<ChunkRegistry>,
    mut inventory: ResMut<ClientInventory>,
    mut chunks: Query<&mut Chunk>,
    remote_players: Query<(Entity, &RemotePlayer)>,
    mut targets: Query<&mut ServerTarget>,
) {
    while let Some(bytes) = client.receive_message(DefaultChannel::ReliableOrdered) {
        let Ok(msg) = bincode::deserialize::<ServerMessage>(&bytes) else {
            warn!("bad packet from server, dropping");
            continue;
        };

        match msg {
            ServerMessage::ChunkData { coord, blocks } => {
                let Some(chunk) = Chunk::from_packed(coord, &blocks) else {
                    warn!("malformed chunk {coord:?}, dropping");
                    continue;
                };
                if let Some(entity) = registry.entity_at(coord) {
                    commands.entity(entity).insert(chunk);
                } else {
                    let entity = commands.spawn(chunk).id();
                    registry.insert(coord, entity);
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
                        chunk.set(lx, ly, lz, kind);
                    }
                }
            }
            ServerMessage::ChunkUnload { coords } => {
                if let Some(entity) = registry.entity_at(coords) {
                    commands.entity(entity).despawn();
                    registry.remove(coords);
                }
            }
            ServerMessage::YourInventory { slots, selected } => {
                inventory.slots = slots;
                inventory.selected = selected;
            }
            ServerMessage::PlayerJoined(id) => {
                let exists = remote_players.iter().any(|(_, p)| p.0 == id);
                if !exists {
                    commands.spawn((
                        RemotePlayer(id),
                        bevy_transform::prelude::Transform::from_translation(Vec3::from(SPAWN_POS)),
                        ServerTarget(Some(Vec3::from(SPAWN_POS))),
                    ));
                }
            }
            ServerMessage::PlayerLeft(id) => {
                for (entity, p) in &remote_players {
                    if p.0 == id {
                        commands.entity(entity).despawn();
                    }
                }
            }
            _ => {}
        }
    }

    // Process unreliable messages (position updates)
    while let Some(bytes) = client.receive_message(DefaultChannel::Unreliable) {
        let Ok(msg) = bincode::deserialize::<ServerMessage>(&bytes) else {
            continue;
        };
        if let ServerMessage::PlayerMoved { player_id, x, y, z } = msg {
            let pos = Vec3::new(x, y, z);
            for (entity, p) in &remote_players {
                if p.0 == player_id {
                    if let Ok(mut target) = targets.get_mut(entity) {
                        target.0 = Some(pos);
                    }
                }
            }
        }
    }
}
