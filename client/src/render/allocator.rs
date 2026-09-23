//! Sub-allocation of the giant vertex/index buffers.
//!
//! The old bump pointer never freed, and its "wrap at 80%" check looked at
//! *this* mesh rather than total used, so re-meshes leaked GPU space until
//! they stomped live geometry. `offset-allocator` hands out vertex/index
//! *slots* (not bytes), coalesces on free, and is O(1).
//!
//! Add to the client `Cargo.toml`:
//! ```toml
//! offset-allocator = "0.2"
//! ```

use std::mem::size_of;

use offset_allocator::{Allocation, Allocator};

use crate::mesh::ChunkVertex;

use super::gpu::{MAX_INDICES, MAX_VERTICES};

/// One mesh's slice of the shared vertex + index buffers.
#[derive(Clone, Copy)]
pub struct GpuMeshAllocation {
    vertex: Allocation,
    pub vertex_count: u32,
    index: Allocation,
    pub index_count: u32,
}

impl GpuMeshAllocation {
    #[inline]
    pub fn vertex_offset_bytes(self) -> u64 {
        self.vertex.offset as u64 * size_of::<ChunkVertex>() as u64
    }

    #[inline]
    pub fn index_offset_bytes(self) -> u64 {
        self.index.offset as u64 * size_of::<u32>() as u64
    }

    /// `vkCmdDrawIndexed.firstIndex` — already in `u32` slots.
    #[inline]
    pub fn first_index(self) -> u32 {
        self.index.offset
    }

    /// `vkCmdDrawIndexed.vertexOffset` — added to every index.
    #[inline]
    pub fn vertex_offset(self) -> i32 {
        self.vertex.offset as i32
    }
}

pub struct MeshBufferAllocator {
    vertices: Allocator,
    indices: Allocator,
}

impl MeshBufferAllocator {
    pub fn new() -> Self {
        // Plenty of slots: a view-distance-8 world is ~4k chunks, plus a
        // handful of avatar meshes. 64k leaves headroom for re-mesh bursts
        // before frees land.
        const MAX_ALLOCS: u32 = 65_536;
        Self {
            vertices: Allocator::with_max_allocs(MAX_VERTICES as u32, MAX_ALLOCS),
            indices: Allocator::with_max_allocs(MAX_INDICES as u32, MAX_ALLOCS),
        }
    }

    /// `vertex_count` / `index_count` are slot counts, not bytes.
    pub fn allocate(&mut self, vertex_count: u32, index_count: u32) -> Option<GpuMeshAllocation> {
        if vertex_count == 0 || index_count == 0 {
            return None;
        }
        let vertex = self.vertices.allocate(vertex_count)?;
        let Some(index) = self.indices.allocate(index_count) else {
            self.vertices.free(vertex);
            return None;
        };
        Some(GpuMeshAllocation {
            vertex,
            vertex_count,
            index,
            index_count,
        })
    }

    pub fn free(&mut self, allocation: GpuMeshAllocation) {
        self.vertices.free(allocation.vertex);
        self.indices.free(allocation.index);
    }
}
