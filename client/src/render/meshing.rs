//! Async 3-stage chunk meshing: invalidate → queue → poll → budgeted upload.

use std::collections::HashMap;

use bevy_ecs::prelude::*;

use bevy_ecs::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use bevy_tasks::Task as AsyncTask;
use bevy_tasks::futures_lite::future;
use exsolis_common::ChunkCoord;
use glam::IVec3;

use crate::{
    chunk::Chunk,
    chunk_registry::ChunkRegistry,
    mesh::{ChunkMeshData, ChunkNeighborhood, ChunkVertex, build_chunk_mesh},
    texture::TextureManager,
    vulkan_prelude::VulkanContext,
};

use super::gpu::{PendingWrite, RenderState, upload_writes};

/// Chunks allowed to *start* meshing (spawn a background task) per frame.
/// Kept modest since 32^3 chunks are cubic — a player moving in any
/// direction, including up/down, can dirty a burst of neighbours at
/// once, not just on the horizontal ring the old column-chunk project
/// dealt with.
const MAX_MESH_TASKS_STARTED_PER_FRAME: usize = 8;

/// Finished meshes allowed to hit the GPU buffer write per frame — turns
/// a boundary-crossing burst into a smooth trickle instead of a frame
/// spike.
const MAX_UPLOADS_PER_FRAME: usize = 4;

#[derive(Component)]
pub(crate) struct ChunkMeshUploaded;

/// A chunk whose mesh is being built on the async compute pool.
#[derive(Component)]
pub(crate) struct MeshingChunk(pub(crate) Option<AsyncTask<ChunkMeshData>>);

/// A finished mesh waiting for its turn at the budgeted GPU upload.
#[derive(Component)]
pub struct ChunkMeshReady(ChunkMeshData);

/// The six face-adjacent chunk offsets. ORDER MATTERS: entries come in
/// opposite pairs (0,1), (2,3), (4,5), so `i ^ 1` is the opposite direction
/// (used by `invalidate_stale_meshes`).
const NEIGHBOR_DIRS: [IVec3; 6] = [
    IVec3::new(1, 0, 0),
    IVec3::new(-1, 0, 0),
    IVec3::new(0, 1, 0),
    IVec3::new(0, -1, 0),
    IVec3::new(0, 0, 1),
    IVec3::new(0, 0, -1),
];

/// Which of the six neighbours (bit `i` = `NEIGHBOR_DIRS[i]`) actually existed
/// in the snapshot this chunk's mesh was built from. A neighbour that wasn't
/// loaded reads as air, so faces toward it were DRAWN; when it later arrives
/// those faces are hidden internal surfaces and the chunk must be re-meshed.
/// Stays on the entity for the mesh's whole life (in flight, ready, uploaded)
/// and is overwritten each time the chunk is re-queued.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeshNeighbors(u8);

/// Stage 0: keep meshes consistent with the chunk data around them.
///
/// Meshes are snapshots: a not-yet-loaded neighbour reads as air, so faces
/// toward it get drawn. Nothing else revisits a chunk once it is uploaded,
/// so without this those boundary faces stay forever (visible as walls of
/// internal faces when flying underground).
///
/// Two cases, both driven by `Changed<Chunk>`:
/// - NEW chunk (`is_added`): re-mesh only neighbours whose `MeshNeighbors`
///   lacks it. Ones that already saw it are fine.
/// - RE-SENT chunk (same coord, new contents): `streaming.rs` does
///   `entity.insert(chunk)`, which Bevy reports as Changed but NOT Added,
///   so `Added<Chunk>` alone would miss it. Its own mesh is stale, and so
///   are its neighbours' boundary faces even if they "saw" it, so drop all
///   of them. This also covers future block edits (a `&mut Chunk` write).
///
/// Drops `ChunkMeshUploaded` / `MeshingChunk` / `ChunkMeshReady` together
/// rather than just the "uploaded" marker: a chunk can be mid-flight or
/// waiting for upload with a stale snapshot, and dropping `MeshingChunk`
/// cancels the task. Old GPU geometry keeps drawing until the replacement
/// uploads, so there is no flicker.
pub(super) fn invalidate_stale_meshes(
    mut commands: Commands,
    registry: Res<ChunkRegistry>,
    changed: Query<(Entity, Ref<Chunk>), Changed<Chunk>>,
    meshed: Query<&MeshNeighbors>,
) {
    for (entity, chunk) in &changed {
        let resent = !chunk.is_added();
        if resent {
            commands
                .entity(entity)
                .try_remove::<(ChunkMeshUploaded, MeshingChunk, ChunkMeshReady)>();
        }

        for (i, dir) in NEIGHBOR_DIRS.into_iter().enumerate() {
            let coord = ChunkCoord::new(
                chunk.coord.x + dir.x,
                chunk.coord.y + dir.y,
                chunk.coord.z + dir.z,
            );
            let Some(neighbor) = registry.entity_at(coord) else {
                continue;
            };
            // No `MeshNeighbors` = never queued yet; it will snapshot fresh.
            let Ok(seen) = meshed.get(neighbor) else {
                continue;
            };
            // From the neighbour's side, this chunk sits in the opposite
            // direction. A brand-new chunk it already saw needs no re-mesh;
            // a re-sent one always does (its contents may have changed).
            if !resent && seen.0 & (1 << (i ^ 1)) != 0 {
                continue;
            }
            commands
                .entity(neighbor)
                .try_remove::<(ChunkMeshUploaded, MeshingChunk, ChunkMeshReady)>();
        }
    }
}

/// Stage 1 of 3: snapshot the 6 face-adjacent neighbours (up, down, and
/// the 4 horizontal — cubic chunks need the vertical pair too, unlike
/// the old column-based project) and hand meshing to the async compute
/// pool.
pub(super) fn queue_chunk_meshing(
    mut commands: Commands,
    textures: Option<Res<TextureManager>>,
    registry: Res<ChunkRegistry>,
    all_chunks: Query<&Chunk>,
    needs_mesh: Query<
        (Entity, &Chunk),
        (
            Without<ChunkMeshUploaded>,
            Without<MeshingChunk>,
            Without<ChunkMeshReady>,
        ),
    >,
) {
    // Faces bake a bindless texture index into each vertex, so meshing
    // before `TextureManager` exists would bake in nothing. Chunks stay
    // un-meshed (no `ChunkMeshUploaded`) and are picked up next frame.
    let Some(textures) = textures else {
        return;
    };
    if !textures.has_textures() {
        // assets/ had no loadable PNGs. Without this, the world would just
        // silently never appear, which is miserable to debug.
        bevy_log::warn_once!(
            "TextureManager has no textures (no PNGs found under assets/?); \
             no chunks will be meshed"
        );
        return;
    }
    let pool = AsyncComputeTaskPool::get();

    for (entity, chunk) in needs_mesh.iter().take(MAX_MESH_TASKS_STARTED_PER_FRAME) {
        let mut neighbor_chunks = HashMap::with_capacity(7);
        neighbor_chunks.insert(IVec3::ZERO, chunk.clone());
        let mut seen_mask = 0u8;
        for (i, dir) in NEIGHBOR_DIRS.into_iter().enumerate() {
            let coord = ChunkCoord::new(
                chunk.coord.x + dir.x,
                chunk.coord.y + dir.y,
                chunk.coord.z + dir.z,
            );
            if let Some(neighbor_entity) = registry.entity_at(coord) {
                // Record presence from the query result, NOT from the registry:
                // the registry can name an entity whose components aren't
                // visible yet, and what matters is what the mesh really saw.
                if let Ok(neighbor_chunk) = all_chunks.get(neighbor_entity) {
                    neighbor_chunks.insert(dir, neighbor_chunk.clone());
                    seen_mask |= 1 << i;
                }
            }
        }

        // `Res<T>` derefs to T; `.clone()` on the Res itself would clone the
        // borrow wrapper (not 'static, can't move into the task). TextureManager
        // wraps its map in an Arc, so this is a refcount bump, not a HashMap copy.
        let textures: TextureManager = (*textures).clone();
        let task = pool.spawn(async move {
            let neighborhood = ChunkNeighborhood {
                chunks: neighbor_chunks,
            };
            build_chunk_mesh(&neighborhood, &textures)
        });
        commands
            .entity(entity)
            .insert((MeshingChunk(Some(task)), MeshNeighbors(seen_mask)));
    }
}

/// Stage 2 of 3: non-blocking poll of in-flight meshing tasks.
pub(super) fn collect_finished_meshes(
    mut commands: Commands,
    mut meshing: Query<(Entity, &mut MeshingChunk)>,
) {
    for (entity, mut task) in &mut meshing {
        let Some(inner_task) = task.0.as_mut() else {
            continue;
        };
        if let Some(mesh) = future::block_on(future::poll_once(inner_task)) {
            commands
                .entity(entity)
                .remove::<MeshingChunk>()
                .insert(ChunkMeshReady(mesh));
        }
    }
}

/// Stage 3 of 3: budgeted GPU upload, one wait + one batched execute for
/// however many chunks came due this frame.
pub(super) fn upload_ready_chunk_meshes(
    mut commands: Commands,
    vk: Res<VulkanContext>,
    mut render_state: ResMut<RenderState>,
    mut ready: Query<(Entity, &mut ChunkMeshReady)>,
) {
    struct Pending {
        entity: Entity,
        vertices: Vec<ChunkVertex>,
        indices: Vec<u32>,
    }

    let mut pending: Vec<Pending> = Vec::new();
    for (entity, mut mesh_ready) in ready.iter_mut().take(MAX_UPLOADS_PER_FRAME) {
        let mesh = std::mem::take(&mut mesh_ready.0);
        pending.push(Pending {
            entity,
            vertices: mesh.vertices,
            indices: mesh.indices,
        });
    }

    if pending.is_empty() {
        return;
    }

    // GPU may still be drawing last frame's slices. Wait before we free
    // anything so a reused range isn't written over in-flight draws.
    vk.resources.flight(vk.flight_id).wait(None).unwrap();

    let mut empties = Vec::new();
    let mut retries = Vec::new();
    let mut reserved = Vec::new();

    for item in pending {
        if item.vertices.is_empty() || item.indices.is_empty() {
            empties.push(item.entity);
            continue;
        }
        match render_state
            .allocator
            .allocate(item.vertices.len() as u32, item.indices.len() as u32)
        {
            Some(alloc) => reserved.push((item, alloc)),
            None => {
                bevy_log::warn!(
                    "mesh buffer exhausted; retrying chunk {:?} next frame ({} verts, {} indices)",
                    item.entity,
                    item.vertices.len(),
                    item.indices.len()
                );
                retries.push(item);
            }
        }
    }

    let mut pending_vertex_writes: Vec<PendingWrite<ChunkVertex>> = Vec::new();
    let mut pending_index_writes: Vec<PendingWrite<u32>> = Vec::new();
    let mut uploaded = Vec::new();
    let mut stale = Vec::new();

    {
        let mut shared = render_state.shared.write().unwrap();
        for entity in empties {
            if let Some(old) = shared.allocations.remove(&entity) {
                stale.push(old);
            }
            uploaded.push(entity);
        }
        for (item, alloc) in reserved {
            if let Some(old) = shared.allocations.insert(item.entity, alloc) {
                stale.push(old);
            }
            pending_vertex_writes.push(PendingWrite {
                offset_bytes: alloc.vertex_offset_bytes(),
                data: item.vertices,
            });
            pending_index_writes.push(PendingWrite {
                offset_bytes: alloc.index_offset_bytes(),
                data: item.indices,
            });
            uploaded.push(item.entity);
        }
    }

    for old in stale {
        render_state.allocator.free(old);
    }

    for item in retries {
        commands
            .entity(item.entity)
            .insert(ChunkMeshReady(ChunkMeshData {
                vertices: item.vertices,
                indices: item.indices,
            }));
    }

    let vertex_buffer_id = render_state.vertex_buffer_id;
    let index_buffer_id = render_state.index_buffer_id;
    upload_writes(
        &vk,
        vertex_buffer_id,
        index_buffer_id,
        &pending_vertex_writes,
        &pending_index_writes,
    );

    for entity in uploaded {
        commands
            .entity(entity)
            .remove::<ChunkMeshReady>()
            .insert(ChunkMeshUploaded);
    }
}

pub(super) fn cleanup_removed_chunk_meshes(
    mut removed: RemovedComponents<Chunk>,
    mut render_state: ResMut<RenderState>,
) {
    let mut stale = Vec::new();
    {
        let mut shared = render_state.shared.write().unwrap();
        for entity in removed.read() {
            if let Some(old) = shared.allocations.remove(&entity) {
                stale.push(old);
            }
        }
    }
    for old in stale {
        render_state.allocator.free(old);
    }
}
