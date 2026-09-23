use crate::chunk::CHUNK_SIZE_I32;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use exsolis_common::{ChunkCoord, ServerMessage};

use crate::{
    chunk::Chunk,
    chunk_registry::ChunkRegistry,
    net::{GameplaySet, IncomingPacket},
};

pub struct StreamingPlugin;

impl Plugin for StreamingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkRegistry>()
            .add_systems(Update, receive_chunk_updates.in_set(GameplaySet));
    }
}

fn receive_chunk_updates(
    mut commands: Commands,
    mut packets: MessageReader<IncomingPacket>,
    mut registry: ResMut<ChunkRegistry>,
    mut chunks: Query<&mut Chunk>,
) {
    for packet in packets.read() {
        match &packet.0 {
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
            _ => {}
        }
    }
}
