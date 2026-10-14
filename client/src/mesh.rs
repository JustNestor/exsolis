//! Chunk mesh building with bindless per-face textures and full 3D
//! cross-chunk-boundary face culling.
//!
//! Textures: each face resolves its texture name (`TextureManager::get`,
//! via `BlockTextures::texture_mapping`) to a bindless index at mesh time
//! and bakes it into every vertex as `sampled_text`. The fragment shader
//! samples through the `SamplerId` in the push constants. There is no AO
//! in this pass (see the note on corner neighbours below); instead each
//! face gets a fixed directional `shade` so cubes don't look flat.
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

use exsolis_common::{AIR_ID, BlockId, BlockRegistry};
use glam::IVec3;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::{
    block_textures,
    chunk::{CHUNK_SIZE, Chunk},
    texture::{TextureManager, TextureMapping, sampled_image_index},
};

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
pub struct ChunkVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32_UINT)]
    pub sampled_text: u32,
    /// Fixed per-face brightness multiplier (see `face_shade`).
    #[format(R32_SFLOAT)]
    pub shade: f32,
}

/// The push-constant block is declared identically in BOTH stages: Vulkan
/// merges per-stage ranges, and mismatched declarations of the same block
/// are a validation error. Only the fragment stage reads `sampler_id`.
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        root_path_env: "CARGO_MANIFEST_DIR",
        src: r"
        #version 450
        #include <vulkano.glsl>

        layout(push_constant) uniform PushConstants {
            mat4 view_proj;
            SamplerId sampler_id;
        } push_constants;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 uv;
        layout(location = 2) in uint sampled_text;
        layout(location = 3) in float shade;

        layout(location = 0) out vec2 v_uv;
        layout(location = 1) out flat uint v_text;
        layout(location = 2) out float v_shade;

        void main() {
            gl_Position = push_constants.view_proj * vec4(position, 1.0);
            v_uv = uv;
            v_text = sampled_text;
            v_shade = shade;
        }
        ",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        root_path_env: "CARGO_MANIFEST_DIR",
        src: r"
        #version 450
        #include <vulkano.glsl>

        layout(push_constant) uniform PushConstants {
            mat4 view_proj;
            SamplerId sampler_id;
        } push_constants;

        layout(location = 0) in vec2 v_uv;
        layout(location = 1) in flat uint v_text;
        layout(location = 2) in float v_shade;

        layout(location = 0) out vec4 f_color;

        void main() {
            SampledImageId g = SampledImageId(v_text, 256);
            vec4 color = texture(vko_sampler2D(g, push_constants.sampler_id), v_uv);
            // Cutout alpha (glass panes, leaves). No blending in this pass.
            if (color.a < 0.1) {
                discard;
            }
            f_color = vec4(color.rgb * v_shade, 1.0);
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
    sampled_text: u32,
) {
    let start = vertices.len() as u32;
    let positions = face_positions(face, x, y, z);
    // Corner order matches `face_positions` (top-left, top-right,
    // bottom-right, bottom-left as seen from outside), so v=0 is the top
    // edge and side textures come out upright.
    const UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let shade = face_shade(face);

    for (pos, uv) in positions.into_iter().zip(UVS) {
        vertices.push(ChunkVertex {
            position: pos,
            uv,
            sampled_text,
            shade,
        });
    }

    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

/// Fixed directional lighting: sky-lit top is full brightness, bottom is
/// darkest, and the two horizontal axes differ slightly so adjacent side
/// faces of a cube are distinguishable. Placeholder for real AO/lighting.
fn face_shade(face: Face) -> f32 {
    match face {
        Face::Top => 1.0,
        Face::Front | Face::Back => 0.8,
        Face::Left | Face::Right => 0.65,
        Face::Bottom => 0.5,
    }
}

fn texture_name_for_face(mapping: TextureMapping, face: Face) -> &'static str {
    match mapping {
        TextureMapping::Single(name) => name,
        TextureMapping::TopBottomSide { top, bottom, side } => match face {
            Face::Top => top,
            Face::Bottom => bottom,
            Face::Left | Face::Right | Face::Front | Face::Back => side,
        },
    }
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

    fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        let (dx, lx) = Self::split(x);
        let (dy, ly) = Self::split(y);
        let (dz, lz) = Self::split(z);
        match self.chunks.get(&IVec3::new(dx, dy, dz)) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => AIR_ID,
        }
    }
}

fn is_face_visible(
    neighborhood: &ChunkNeighborhood,
    x: i32,
    y: i32,
    z: i32,
    this: BlockId,
    registry: &BlockRegistry,
) -> bool {
    let neighbour = neighborhood.get_block(x, y, z);
    // Was `!neighbour.is_solid()`. That treats glass and leaves as solid,
    // which would cull the faces of whatever sits behind them and leave a
    // see-through hole to the void once they have alpha textures. Use
    // "transparent" instead; same-type neighbours (water/water,
    // glass/glass) still get no internal face between them.
    registry.is_transparent(neighbour) && neighbour != this
}

#[derive(Default)]
pub(crate) struct ChunkMeshData {
    pub vertices: Vec<ChunkVertex>,
    pub indices: Vec<u32>,
}

pub(crate) fn build_chunk_mesh(
    neighborhood: &ChunkNeighborhood,
    textures: &TextureManager,
    registry: &BlockRegistry,
) -> ChunkMeshData {
    let mut mesh = ChunkMeshData::default();
    let chunk = &neighborhood.chunks[&IVec3::ZERO];
    let origin = chunk.world_origin();

    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let block = chunk.get(x, y, z);
                if registry.is_air(block) {
                    continue;
                }
                let Some(mapping) = block_textures::texture_mapping(registry, block) else {
                    continue;
                };
                let (fx, fy, fz) = (
                    origin.x + x as f32,
                    origin.y + y as f32,
                    origin.z + z as f32,
                );
                let (ix, iy, iz) = (x as i32, y as i32, z as i32);

                for &(face, dir) in &FACE_DIRS {
                    if !is_face_visible(neighborhood, ix + dir.x, iy + dir.y, iz + dir.z, block, registry) {
                        continue;
                    }
                    // `get` only returns None when NO texture loaded at all,
                    // which render.rs already refuses to mesh through.
                    let Some(id) = textures.get(texture_name_for_face(mapping, face)) else {
                        continue;
                    };
                    push_face(
                        &mut mesh.vertices,
                        &mut mesh.indices,
                        fx,
                        fy,
                        fz,
                        face,
                        sampled_image_index(id),
                    );
                }
            }
        }
    }

    mesh
}
