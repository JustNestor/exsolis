use bevy_ecs::prelude::*;
use exsolis_common::ChunkCoord;
use std::collections::HashMap;

/// Coord -> entity lookup. Needed so meshing can find the (up to 6, since
/// chunks are cubic now and neighbours exist above/below too, not just
/// horizontally) chunks adjacent to the one being meshed.
#[derive(Resource, Default)]
pub struct ChunkRegistry {
    entities: HashMap<ChunkCoord, Entity>,
}

impl ChunkRegistry {
    pub fn insert(&mut self, coord: ChunkCoord, entity: Entity) {
        self.entities.insert(coord, entity);
    }

    pub fn remove(&mut self, coord: ChunkCoord) {
        self.entities.remove(&coord);
    }

    pub fn entity_at(&self, coord: ChunkCoord) -> Option<Entity> {
        self.entities.get(&coord).copied()
    }
}
