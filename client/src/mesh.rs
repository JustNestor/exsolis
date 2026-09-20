//! Chunk mesh building with hardcoded per-block colors (no textures yet
//! — see `chunk::block_color`) and full 3D cross-chunk-boundary face
//! culling.
//!
//! This differs from the older 16x16x256-column project's
//! `ChunkNeighborhood` in one important way: those chunks were full
//! columns, so a chunk only ever had *horizontal* neighbours — Y was
//! never split across chunk boundaries, so `get_block` only needed to
//! wrap x/z and could treat any out-of-column y as a hard "clamp to air".
//! These chunks are cubic (32^3), so a face can be hidden by a block in
//! the chunk above/below too. `ChunkNeighborhood` here wraps x, y, *and*
//! z, and the neighbor snapshot carries up to 6 face-adjacent chunks
//! instead of up to 8 horizontally-adjacent ones (diagonals still aren't
//! needed for face visibility — only face-adjacency matters for culling
//! and this pass skips AO, which is the only thing that would want
//! corner/edge neighbours too).

use std::collections::HashMap;

use exsolis_common::BlockType;
use glam::IVec3;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::chunk::{CHUNK_SIZE, Chunk, block_color};

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
pub struct ChunkVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
        #version 450

        layout(push_constant) uniform PushConstants {
            mat4 view_proj;
        } push_constants;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;

        layout(location = 0) out vec3 v_color;

        void main() {
            gl_Position = push_constants.view_proj * vec4(position, 1.0);
            v_color = color;
        }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
        #version 450

        layout(location = 0) in vec3 v_color;
        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(v_color, 1.0);
        }
        ",
    }
}

#[derive(Clone, Copy)]
enum Face {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

const FACE_DIRS: [(Face, IVec3); 6] = [
    (Face::Top, IVec3::new(0, 1, 0)),
    (Face::Bottom, IVec3::new(0, -1, 0)),
    (Face::Left, IVec3::new(-1, 0, 0)),
    (Face::Right, IVec3::new(1, 0, 0)),
    (Face::Front, IVec3::new(0, 0, 1)),
    (Face::Back, IVec3::new(0, 0, -1)),
]; // Const array instead of a match-based iterator: this list gets walked
// once per solid block, so at 32^3 blocks/chunk keeping it a flat,
// branch-free table (rather than 6 separate maybe_push_face call sites
// like the textured version uses) cuts per-block dispatch overhead.

fn face_positions(face: Face, x: f32, y: f32, z: f32) -> [[f32; 3]; 4] {
    match face {
        Face::Top => [
            [x, y + 1.0, z],
            [x + 1.0, y + 1.0, z],
            [x + 1.0, y + 1.0, z + 1.0],
            [x, y + 1.0, z + 1.0],
        ],
        Face::Bottom => [
            [x, y, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x + 1.0, y, z],
            [x, y, z],
        ],
        Face::Back => [
            [x + 1.0, y + 1.0, z],
            [x, y + 1.0, z],
            [x, y, z],
            [x + 1.0, y, z],
        ],
        Face::Right => [
            [x + 1.0, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z],
            [x + 1.0, y, z],
            [x + 1.0, y, z + 1.0],
        ],
        Face::Front => [
            [x, y + 1.0, z + 1.0],
            [x + 1.0, y + 1.0, z + 1.0],
            [x + 1.0, y, z + 1.0],
            [x, y, z + 1.0],
        ],
        Face::Left => [
            [x, y + 1.0, z],
            [x, y + 1.0, z + 1.0],
            [x, y, z + 1.0],
            [x, y, z],
        ],
    }
}

fn push_face(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    x: f32,
    y: f32,
    z: f32,
    face: Face,
    color: [f32; 3],
) {
    let start = vertices.len() as u32;
    let positions = face_positions(face, x, y, z);

    for pos in positions {
        vertices.push(ChunkVertex {
            position: pos,
            color,
        });
    }

    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

/// Up to 6 face-adjacent chunks (plus the center one) around the chunk
/// being meshed, keyed by direction offset. A neighbour that isn't loaded
/// yet is simply absent from the map — treated as "no data", which
/// `is_face_visible` below resolves to "draw the face" (same don't-guess
/// behaviour as a genuine world edge).
pub(crate) struct ChunkNeighborhood {
    pub chunks: HashMap<IVec3, Chunk>,
}

impl ChunkNeighborhood {
    /// Splits a local coordinate that may have stepped outside
    /// `0..CHUNK_SIZE` into (which neighbour chunk to look in, the local
    /// coordinate within that chunk). Unlike the column-based project,
    /// this is applied on all three axes since neighbours exist in every
    /// direction now, not just horizontally.
    fn split(coord: i32) -> (i32, usize) {
        if coord < 0 {
            (-1, (coord + CHUNK_SIZE as i32) as usize)
        } else if coord >= CHUNK_SIZE as i32 {
            (1, (coord - CHUNK_SIZE as i32) as usize)
        } else {
            (0, coord as usize)
        }
    }

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        let (dx, lx) = Self::split(x);
        let (dy, ly) = Self::split(y);
        let (dz, lz) = Self::split(z);
        match self.chunks.get(&IVec3::new(dx, dy, dz)) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => BlockType::Air,
        }
    }
}

fn is_face_visible(
    neighborhood: &ChunkNeighborhood,
    x: i32,
    y: i32,
    z: i32,
    this: BlockType,
) -> bool {
    let neighbour = neighborhood.get_block(x, y, z);
    // Air/water-style "see through" check via is_solid(); same-type
    // transparent neighbours (e.g. two water blocks) don't get an
    // internal face between them.
    !neighbour.is_solid() && neighbour != this
}

#[derive(Default)]
pub(crate) struct ChunkMeshData {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

pub(crate) fn build_chunk_mesh(neighborhood: &ChunkNeighborhood) -> ChunkMeshData {
    let mut mesh = ChunkMeshData::default();
    let chunk = &neighborhood.chunks[&IVec3::ZERO];
    let origin = chunk.world_origin();

    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let block = chunk.get(x, y, z);
                if block.is_air() {
                    continue;
                }
                let color = block_color(block);
                let (fx, fy, fz) = (
                    origin.x + x as f32,
                    origin.y + y as f32,
                    origin.z + z as f32,
                );
                let (ix, iy, iz) = (x as i32, y as i32, z as i32);

                for &(face, dir) in &FACE_DIRS {
                    if is_face_visible(neighborhood, ix + dir.x, iy + dir.y, iz + dir.z, block) {
                        push_face(
                            &mut mesh.vertices,
                            &mut mesh.indices,
                            fx,
                            fy,
                            fz,
                            face,
                            color,
                        );
                    }
                }
            }
        }
    }

    mesh
}
