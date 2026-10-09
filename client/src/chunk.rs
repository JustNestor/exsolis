//! Client-side chunk storage. 32^3 cubic chunks (cubic, unlike the older
//! 16x16x256-column project) — matches the shape streamed down by the
//! server in exsolis_common::ServerMessage::ChunkData.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use exsolis_common::{BlockType, ChunkBlocks, ChunkCoord};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;

#[derive(Component, Debug, Clone)]
pub struct Chunk {
    pub coord: ChunkCoord,
    blocks: Arc<[BlockType]>,
}

impl Chunk {
    pub fn from_packed(coord: ChunkCoord, packed: &ChunkBlocks) -> Option<Self> {
        Some(Self {
            coord,
            blocks: packed.unpack()?.into(),
        })
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, kind: BlockType) {
        if x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE {
            let idx = Self::index(x, y, z);
            Arc::make_mut(&mut self.blocks)[idx] = kind;
        }
    }

    #[inline]
    fn index(x: usize, y: usize, z: usize) -> usize {
        // x-major, matches the server's generate_chunk indexing
        // (x + y*SIZE + z*SIZE*SIZE) from the streaming server.
        x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        if x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return BlockType::Air;
        }
        self.blocks[Self::index(x, y, z)]
    }

    pub fn world_origin(&self) -> glam::Vec3 {
        glam::Vec3::new(
            (self.coord.x * CHUNK_SIZE_I32) as f32,
            (self.coord.y * CHUNK_SIZE_I32) as f32,
            (self.coord.z * CHUNK_SIZE_I32) as f32,
        )
    }
}

/// Maps a color to each solid block type. Swap for real textures later —
/// mesh.rs only depends on `block_color`, nothing else needs to change.
pub fn block_color(block: BlockType) -> [f32; 3] {
    match block {
        BlockType::Air => [0.0, 0.0, 0.0], // never actually emitted
        BlockType::Stone => [0.5, 0.5, 0.5],
        BlockType::Dirt => [0.45, 0.3, 0.18],
        BlockType::Grass => [0.25, 0.65, 0.25],
        BlockType::Sand => [0.87, 0.8, 0.55],
        BlockType::Water => [0.15, 0.4, 0.8],
        BlockType::Snow => [0.95, 0.95, 0.97],
        BlockType::Glass => [0.75, 0.9, 0.95],
        BlockType::Log => [0.4, 0.27, 0.13],
        BlockType::Leaves => [0.15, 0.45, 0.12],
        BlockType::Chest => [0.55, 0.38, 0.22],
    }
}
