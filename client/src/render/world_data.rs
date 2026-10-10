//! Read-only block data for effect shaders.
//!
//! Effects run entirely on the GPU, so anything they should react to (ground
//! to bounce on, water to splash in, a roof to stop rain) has to be there
//! already. This module keeps a window of chunks around the local camera
//! mirrored into one storage buffer that every effect pass can bind
//! (`@group(1) @binding(24)`, declared by the `engine/world` WGSL import).
//!
//! Layout (u32 words, indexed by a toroidal *cell*):
//! - `cells * 4` words of tags: `[chunk_x, chunk_y, chunk_z, state]`, where
//!   state 1 means the cell holds that chunk's blocks and 0 means empty.
//! - then `CHUNK_SIZE^3 / 2` words per cell: one u16 per block
//!   (bits 0..8 block id, bits 8..11 flags), two per word.
//!
//! A chunk at coordinate `c` always lives in cell `c mod WORLD_WINDOW_CELLS`,
//! so sliding the window never moves data; only chunks that newly enter it are
//! written, a bounded number per frame, nearest first. The shader checks the
//! tag, so a cell that still holds a chunk from the other side of the window
//! reads as "not loaded" instead of returning wrong blocks.
//!
//! Flexibility lives in two places: the window size and per-frame budget are
//! the constants below (shaders read the size from the frame buffer, so
//! changing them needs no shader edits), and the per-voxel encoding is
//! produced by [`pack_block`], the single place that decides what shaders can
//! know about a block.

use bevy_ecs::prelude::*;
use bevy_transform::prelude::Transform;
use exsolis_common::{BlockType, ChunkCoord};
use glam::IVec3;

use crate::{
    block_textures::BlockTextures,
    chunk::{CHUNK_SIZE, Chunk},
    chunk_registry::ChunkRegistry,
    player::LocalPlayer,
};

/// Size of the mirrored window in chunks (x, y, z). Even on every axis so the
/// window can centre on the camera chunk.
pub(super) const WORLD_WINDOW_CELLS: [u32; 3] = [8, 4, 8];
pub(super) const WORLD_CELL_COUNT: usize =
    (WORLD_WINDOW_CELLS[0] * WORLD_WINDOW_CELLS[1] * WORLD_WINDOW_CELLS[2]) as usize;
/// Chunks whose block data is converted and uploaded per frame. Conversion
/// walks every block of the chunk on the main thread, so keep this small;
/// the window fills nearest-first over a couple of seconds at the default.
const MAX_CHUNK_UPLOADS_PER_FRAME: usize = 2;
/// Bytes of tag data per cell: `[x, y, z, state]` as i32.
pub(super) const WORLD_TAG_BYTES: u64 = 16;

const _: () = assert!(CHUNK_SIZE % 2 == 0, "voxels are packed two per word");
const _: () = assert!(
    WORLD_WINDOW_CELLS[0] % 2 == 0 && WORLD_WINDOW_CELLS[1] % 2 == 0 && WORLD_WINDOW_CELLS[2] % 2 == 0
);

/// Voxel flag bits, shifted into bits 8..11 of the packed u16. Mirrored by
/// `WORLD_FLAG_*` in the `engine/world` WGSL import.
const FLAG_SOLID: u16 = 0x100;
const FLAG_TRANSPARENT: u16 = 0x200;
const FLAG_AIR: u16 = 0x400;

/// Bytes of block data per cell (CHUNK_SIZE^3 voxels of 2 bytes).
pub(super) fn world_cell_bytes() -> u64 {
    (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE * 2) as u64
}

/// Total size of the world buffer: all tags, then all cells' voxels.
pub(super) fn world_buffer_bytes() -> u64 {
    WORLD_CELL_COUNT as u64 * (WORLD_TAG_BYTES + world_cell_bytes())
}

/// What the frame buffer tells shaders about the window.
#[derive(Clone, Copy, Default)]
pub(super) struct WorldFrameInfo {
    /// False until the first uploads have been applied.
    pub enabled: bool,
    /// Chunk coordinate the window is centred on.
    pub anchor: [i32; 3],
}

/// One cell's worth of changes for the GPU.
pub(super) struct WorldUpload {
    pub cell: u32,
    /// `[chunk_x, chunk_y, chunk_z, state]` (state 1 loaded, 0 empty).
    pub tag: [i32; 4],
    /// Packed voxels, present only when the cell now holds a chunk.
    pub voxels: Option<Box<[u32]>>,
    desired: CellState,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CellState {
    /// Never written; the buffer contents are undefined.
    Unknown,
    Empty(IVec3),
    /// Chunk coordinate plus the change tick of the data that was packed.
    Loaded(IVec3, u32),
}

/// CPU-side record of what the world buffer holds, plus the uploads that
/// would bring it up to date. The mirror only advances in [`commit`], once a
/// frame carrying the uploads actually ran, so a dropped frame just repeats
/// them.
///
/// [`commit`]: WorldWindow::commit
#[derive(Resource, Default)]
pub struct WorldWindow {
    anchor: Option<IVec3>,
    committed_once: bool,
    mirror: Vec<CellState>,
    pending: Vec<WorldUpload>,
}

impl WorldWindow {
    pub(super) fn frame_info(&self) -> WorldFrameInfo {
        match self.anchor {
            Some(anchor) if self.committed_once => WorldFrameInfo {
                enabled: true,
                anchor: anchor.to_array(),
            },
            _ => WorldFrameInfo::default(),
        }
    }

    pub(super) fn take_pending(&mut self) -> Vec<WorldUpload> {
        std::mem::take(&mut self.pending)
    }

    pub(super) fn commit(&mut self, uploads: &[WorldUpload]) {
        for upload in uploads {
            if let Some(slot) = self.mirror.get_mut(upload.cell as usize) {
                *slot = upload.desired;
            }
        }
        self.committed_once |= !uploads.is_empty();
    }
}

/// Index of the cell chunk `coord` lives in. Must match `world_block` in the
/// `engine/world` WGSL import.
fn cell_index(coord: IVec3, dims: IVec3) -> usize {
    let m = coord.rem_euclid(dims);
    ((m.z * dims.y + m.y) * dims.x + m.x) as usize
}

/// Decides what shaders can know about a block: id (low byte) and flags.
/// Extend this (and the WGSL import) to expose more, e.g. a liquid flag.
fn pack_block(block: BlockType) -> u16 {
    let mut packed = (block as u32 & 0xFF) as u16;
    if block.is_solid() {
        packed |= FLAG_SOLID;
    }
    if block.is_transparent() {
        packed |= FLAG_TRANSPARENT;
    }
    if block.is_air() {
        packed |= FLAG_AIR;
    }
    packed
}

fn pack_chunk(chunk: &Chunk) -> Box<[u32]> {
    let mut words = vec![0u32; CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE / 2];
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let voxel = (y * CHUNK_SIZE + z) * CHUNK_SIZE + x;
                words[voxel / 2] |= u32::from(pack_block(chunk.get(x, y, z))) << ((voxel % 2) * 16);
            }
        }
    }
    words.into_boxed_slice()
}

/// Works out which cells differ from what the GPU holds and queues their
/// uploads. Runs every frame before the renderer; cheap when nothing changed
/// (a registry lookup and a tick compare per cell).
pub(super) fn update_world_window(
    mut window: ResMut<WorldWindow>,
    registry: Option<Res<ChunkRegistry>>,
    chunks: Query<Ref<Chunk>>,
    camera: Query<&Transform, With<LocalPlayer>>,
) {
    window.pending.clear();
    let (Some(registry), Ok(transform)) = (registry, camera.single()) else {
        return;
    };

    let dims = IVec3::new(
        WORLD_WINDOW_CELLS[0] as i32,
        WORLD_WINDOW_CELLS[1] as i32,
        WORLD_WINDOW_CELLS[2] as i32,
    );
    if window.mirror.len() != WORLD_CELL_COUNT {
        window.mirror = vec![CellState::Unknown; WORLD_CELL_COUNT];
    }
    let anchor = (transform.translation / CHUNK_SIZE as f32)
        .floor()
        .as_ivec3();
    window.anchor = Some(anchor);
    let low = anchor - dims / 2;

    let mut uploads = Vec::new();
    // (distance to the anchor, cell, chunk coordinate, entity, change tick)
    let mut jobs = Vec::new();
    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let coord = low + IVec3::new(x, y, z);
                let cell = cell_index(coord, dims);
                let loaded = registry
                    .entity_at(ChunkCoord::new(coord.x, coord.y, coord.z))
                    .and_then(|entity| {
                        chunks
                            .get(entity)
                            .ok()
                            // `Ref<Chunk>`’s Bevy change tick: fresh data since
                            // the mirror last uploaded this cell reads as new.
                            .map(|chunk| (entity, chunk.last_changed().get()))
                    });
                match loaded {
                    Some((entity, tick)) => {
                        if window.mirror[cell] != CellState::Loaded(coord, tick) {
                            jobs.push(((coord - anchor).length_squared(), cell, coord, entity, tick));
                        }
                    }
                    None => {
                        if window.mirror[cell] != CellState::Empty(coord) {
                            uploads.push(WorldUpload {
                                cell: cell as u32,
                                tag: [coord.x, coord.y, coord.z, 0],
                                voxels: None,
                                desired: CellState::Empty(coord),
                            });
                        }
                    }
                }
            }
        }
    }

    jobs.sort_by_key(|job| job.0);
    let mut jobs = jobs.into_iter();
    for (_, cell, coord, entity, tick) in jobs.by_ref().take(MAX_CHUNK_UPLOADS_PER_FRAME) {
        let Ok(chunk) = chunks.get(entity) else {
            continue;
        };
        uploads.push(WorldUpload {
            cell: cell as u32,
            tag: [coord.x, coord.y, coord.z, 1],
            voxels: Some(pack_chunk(&chunk)),
            desired: CellState::Loaded(coord, tick),
        });
    }
    // Chunks that missed this frame's budget still need their cell to hold
    // *something defined*: buffer memory starts uninitialised, and a garbage
    // tag could otherwise match a real chunk. Mark never-written cells empty;
    // the chunk itself follows on a later frame.
    for (_, cell, coord, _, _) in jobs {
        if window.mirror[cell] == CellState::Unknown {
            uploads.push(WorldUpload {
                cell: cell as u32,
                tag: [coord.x, coord.y, coord.z, 0],
                voxels: None,
                desired: CellState::Empty(coord),
            });
        }
    }
    window.pending = uploads;
}
