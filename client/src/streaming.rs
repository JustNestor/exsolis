use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use bevy_renet::{RenetClient, renet::DefaultChannel};
use exsolis_common::{ChunkCoord, ServerMessage};

use crate::{
    chunk::{CHUNK_SIZE_I32, Chunk},
    chunk_registry::ChunkRegistry,
    interaction::ClientInventory,
};

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
    mut chunks: Query<&mut Chunk>, // existing: Query<(Entity, &Chunk)>
) {
    while let Some(bytes) = client.receive_message(DefaultChannel::ReliableOrdered) {
        // debug!("reciving {} bytes", bytes.len());
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
                        // Mutating `chunk` marks the component as Changed<Chunk>,
                        // triggering your client mesh building system query automatically.
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
            _ => {}
        }
    }

    // let _ = existing; // kept for future use (e.g. diffing); silences unused warning for now
}
