//! Chunk mesh building + the Vulkano task-graph pipeline that draws
//! them. No textures yet — blocks are flat-colored per `chunk::block_color`
//! (swap that function out later for a real TextureManager without
//! touching this file's structure).
//!
//! Mirrors the established voxel-engine project's async 3-stage meshing
//! pipeline (queue -> poll -> budgeted GPU upload), adapted for:
//! - 32^3 cubic chunks instead of 16x16x256 columns, so cross-boundary
//!   culling (`mesh::ChunkNeighborhood`) now has to look in all 6
//!   face directions, not just the 4 horizontal ones — every chunk has
//!   a "neighbour above" and "neighbour below" to worry about now.
//! - A single vertex stream (position + color) instead of separate
//!   opaque/translucent streams with AO/UV/texture-index — there's
//!   nothing to blend or sample yet.

use std::{
    collections::HashMap,
    mem::{align_of, size_of},
    num::NonZero,
    sync::{Arc, Mutex, RwLock},
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_tasks::AsyncComputeTaskPool;
use bevy_tasks::Task as AsyncTask;
use bevy_tasks::futures_lite::future;
use bevy_transform::prelude::Transform;
use bevy_window::{PrimaryWindow, Window};
use exsolis_common::ChunkCoord;
use glam::IVec3;

use vulkano::{
    DeviceSize, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    format::{ClearValue, Format},
    image::{Image, ImageCreateInfo, ImageUsage},
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
    },
    swapchain::{PresentMode, Swapchain, SwapchainCreateInfo},
};
use vulkano_taskgraph::{
    ClearValues, Id, Task, TaskContext, TaskResult,
    command_buffer::RecordingCommandBuffer,
    descriptor_set::SamplerId,
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, HostAccessType, ImageLayoutType::Optimal},
    resource_map,
};

use crate::{
    chunk::Chunk,
    chunk_registry::ChunkRegistry,
    mesh::{ChunkMeshData, ChunkNeighborhood, ChunkVertex, build_chunk_mesh, fs, vs},
    texture::{TextureManager, TexturePlugin},
    vulkan_prelude::VulkanContext,
};

// 32^3 chunks pack more geometry per chunk than the old 16x16x256 columns
// did on average once boundary culling removes most internal faces, and
// with no textures/AO there's less per-vertex bandwidth pressure, so this
// keeps the same generous headroom as the established project.
const MAX_VERTICES: u64 = 40_000_000;
const MAX_INDICES: u64 = 60_000_000;

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
struct ChunkMeshReady(ChunkMeshData);

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

#[derive(Debug, Clone, Copy)]
struct GpuMeshAllocation {
    vertex_offset_bytes: u64,
    vertex_count: u32,
    index_offset_bytes: u64,
    index_count: u32,
}

struct RenderStateShared {
    pipeline: Option<Arc<GraphicsPipeline>>,
    allocations: HashMap<Entity, GpuMeshAllocation>,
}

pub struct FrameData {
    pub view_proj: [[f32; 4]; 4],
    pub viewport: vulkano::pipeline::graphics::viewport::Viewport,
    /// Bindless sampler used to read every block texture. Same sampler for
    /// all chunks; per-texture state lives in the image, not here.
    pub sampler_id: SamplerId,
}

struct ChunkRenderTask {
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_depth_id: Id<Image>,
    shared: Arc<RwLock<RenderStateShared>>,
}

impl Task for ChunkRenderTask {
    type World = FrameData;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        clear_values.set(
            self.virtual_swapchain_id.current_image_id(),
            [2.0 / 255.0, 6.0 / 255.0, 24.0 / 255.0, 1.0],
        );
        clear_values.set(self.virtual_depth_id, ClearValue::Depth(1.0));
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, std::slice::from_ref(&world.viewport));

        let shared = self.shared.read().unwrap();
        if let Some(pipeline) = shared.pipeline.as_ref() {
            cbf.push_constants(
                pipeline.layout(),
                0,
                &vs::PushConstants {
                    view_proj: world.view_proj,
                    sampler_id: world.sampler_id,
                },
            );
            cbf.bind_pipeline(pipeline);
            cbf.bind_index_buffer(
                self.index_buffer_id,
                0,
                None,
                vulkano::buffer::IndexType::U32,
            );
            cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

            for alloc in shared.allocations.values() {
                if alloc.index_count == 0 {
                    continue;
                }
                let first_index = (alloc.index_offset_bytes / size_of::<u32>() as u64) as u32;
                let vertex_offset =
                    (alloc.vertex_offset_bytes / size_of::<ChunkVertex>() as u64) as i32;
                unsafe {
                    cbf.draw_indexed(alloc.index_count, 1, first_index, vertex_offset, 0);
                }
            }
        }

        Ok(())
    }
}

#[derive(Resource)]
pub struct RenderState {
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    depth_image_id: Id<Image>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_depth_id: Id<Image>,
    task_graph: Mutex<ExecutableTaskGraph<FrameData>>,
    shared: Arc<RwLock<RenderStateShared>>,
    next_vertex_offset: u64,
    next_index_offset: u64,
}

pub struct RendererPlugin;

impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(TexturePlugin).add_systems(
            Update,
            (
                setup_renderer,
                invalidate_stale_meshes.run_if(resource_exists::<RenderState>),
                queue_chunk_meshing.run_if(resource_exists::<RenderState>),
                collect_finished_meshes.run_if(resource_exists::<RenderState>),
                upload_ready_chunk_meshes.run_if(resource_exists::<RenderState>),
                cleanup_removed_chunk_meshes.run_if(resource_exists::<RenderState>),
                render_frame.run_if(resource_exists::<RenderState>),
            )
                .chain(),
        );
    }
}

fn setup_renderer(
    mut commands: Commands,
    vk: Option<Res<VulkanContext>>,
    render_state: Option<Res<RenderState>>,
) {
    let Some(vk) = vk else {
        return;
    };
    if render_state.is_some() {
        return;
    }

    let vertex_buffer_size: DeviceSize = MAX_VERTICES * size_of::<ChunkVertex>() as u64;
    let index_buffer_size: DeviceSize = MAX_INDICES * size_of::<u32>() as u64;

    let vertex_layout = DeviceLayout::new(
        NonZero::new(vertex_buffer_size).expect("vertex buffer size must be non-zero"),
        DeviceAlignment::new(align_of::<ChunkVertex>() as u64).unwrap(),
    )
    .expect("vertex layout exceeds device limits");

    let index_layout = DeviceLayout::new(
        NonZero::new(index_buffer_size).expect("index buffer size must be non-zero"),
        DeviceAlignment::new(align_of::<u32>() as u64).unwrap(),
    )
    .expect("index layout exceeds device limits");

    let vertex_buffer_id = vk
        .resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_layout,
        )
        .unwrap();

    let index_buffer_id = vk
        .resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            index_layout,
        )
        .unwrap();

    let depth_image_id = vk
        .resources
        .create_image(
            &ImageCreateInfo {
                format: Format::D32_SFLOAT,
                extent: [
                    vk.viewport.extent[0] as u32,
                    vk.viewport.extent[1] as u32,
                    1,
                ],
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .unwrap();

    let shared = Arc::new(RwLock::new(RenderStateShared {
        pipeline: None,
        allocations: HashMap::new(),
    }));

    let mut task_graph: TaskGraph<FrameData> = TaskGraph::new(&vk.resources);

    let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
        image_format: vk.swapchain_format,
        present_mode: PresentMode::Mailbox,
        ..Default::default()
    });
    let virtual_framebuffer_id = task_graph.add_framebuffer();
    let virtual_depth_id = task_graph.add_image(&ImageCreateInfo {
        format: Format::D32_SFLOAT,
        ..Default::default()
    });

    let task = ChunkRenderTask {
        vertex_buffer_id,
        index_buffer_id,
        virtual_swapchain_id,
        virtual_depth_id,
        shared: Arc::clone(&shared),
    };

    let mut node =
        task_graph.create_task_node("Chunks", vulkano_taskgraph::QueueFamilyType::Graphics, task);
    node.framebuffer(virtual_framebuffer_id)
        .color_attachment(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            Optimal,
            &AttachmentInfo {
                clear: true,
                ..Default::default()
            },
        )
        .depth_stencil_attachment(
            virtual_depth_id,
            AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Optimal,
            &AttachmentInfo {
                clear: true,
                ..Default::default()
            },
        )
        .buffer_access(index_buffer_id, AccessTypes::INDEX_READ)
        .buffer_access(vertex_buffer_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
        .build();

    let mut task_graph = unsafe {
        task_graph
            .compile(&CompileInfo {
                queues: &[&vk.queue],
                present_queue: Some(&vk.queue),
                flight_id: vk.flight_id,
                ..Default::default()
            })
            .unwrap()
    };

    let mut node = task_graph.task_nodes().next().unwrap();
    // let node = task_graph.task_node_mut(node_id).unwrap();
    let subpass = node.subpass().unwrap().clone();

    let vs = unsafe { vs::load(&vk.device).unwrap().entry_point("main").unwrap() };
    let fs = unsafe { fs::load(&vk.device).unwrap().entry_point("main").unwrap() };

    let vertex_input_state = ChunkVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(&vs),
        PipelineShaderStageCreateInfo::new(&fs),
    ];

    let bcx = vk.resources.bindless_context().unwrap();
    let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

    let pipeline = GraphicsPipeline::new(
        &vk.device,
        None,
        &GraphicsPipelineCreateInfo {
            stages: &stages,
            vertex_input_state: Some(&vertex_input_state),
            input_assembly_state: Some(&InputAssemblyState::default()),
            viewport_state: Some(&ViewportState::default()),
            rasterization_state: Some(&RasterizationState {
                cull_mode: CullMode::Back,
                // mesh.rs emits faces clockwise when seen from outside, and
                // FlyCamera::view_projection flips Y for Vulkan's Y-down NDC,
                // so the rasterizer sees front faces as clockwise.
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(&MultisampleState::default()),
            color_blend_state: Some(&ColorBlendState {
                attachments: &[ColorBlendAttachmentState::default()],
                ..Default::default()
            }),
            depth_stencil_state: Some(&DepthStencilState {
                depth: Some(DepthState {
                    write_enable: true,
                    compare_op: CompareOp::Less,
                }),
                ..Default::default()
            }),
            dynamic_state: &[DynamicState::Viewport],
            subpass: Some((&subpass).into()),
            ..GraphicsPipelineCreateInfo::new(&layout)
        },
    )
    .unwrap();

    shared.write().unwrap().pipeline = Some(pipeline);

    commands.insert_resource(RenderState {
        vertex_buffer_id,
        index_buffer_id,
        depth_image_id,
        virtual_swapchain_id,
        virtual_depth_id,
        task_graph: Mutex::new(task_graph),
        shared,
        next_vertex_offset: 0,
        next_index_offset: 0,
    });
}

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
fn invalidate_stale_meshes(
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
fn queue_chunk_meshing(
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
fn collect_finished_meshes(
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

struct PendingWrite<T> {
    offset_bytes: u64,
    data: Vec<T>,
}

fn reserve_mesh(
    render_state: &mut RenderState,
    vertices: Vec<ChunkVertex>,
    indices: Vec<u32>,
    pending_vertex_writes: &mut Vec<PendingWrite<ChunkVertex>>,
    pending_index_writes: &mut Vec<PendingWrite<u32>>,
) -> Option<GpuMeshAllocation> {
    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    let vb_size = (vertices.len() * size_of::<ChunkVertex>()) as u64;
    let ib_size = (indices.len() * size_of::<u32>()) as u64;

    let start_voffset = render_state.next_vertex_offset;
    let start_ioffset = render_state.next_index_offset;
    let end_voffset = start_voffset + vb_size;
    let end_ioffset = start_ioffset + ib_size;

    let allocation = GpuMeshAllocation {
        vertex_offset_bytes: start_voffset,
        vertex_count: vertices.len() as u32,
        index_offset_bytes: start_ioffset,
        index_count: indices.len() as u32,
    };

    if vertices.len() as f32 >= 0.8 * MAX_VERTICES as f32
        || indices.len() as f32 >= 0.8 * MAX_INDICES as f32
    {
        render_state.next_vertex_offset = 0;
        render_state.next_index_offset = 0;
    } else {
        render_state.next_vertex_offset = end_voffset;
        render_state.next_index_offset = end_ioffset;
    }

    pending_vertex_writes.push(PendingWrite {
        offset_bytes: start_voffset,
        data: vertices,
    });
    pending_index_writes.push(PendingWrite {
        offset_bytes: start_ioffset,
        data: indices,
    });

    Some(allocation)
}

/// Stage 3 of 3: budgeted GPU upload, one wait + one batched execute for
/// however many chunks came due this frame.
fn upload_ready_chunk_meshes(
    mut commands: Commands,
    vk: Res<VulkanContext>,
    mut render_state: ResMut<RenderState>,
    mut ready: Query<(Entity, &mut ChunkMeshReady)>,
) {
    let mut pending_vertex_writes: Vec<PendingWrite<ChunkVertex>> = Vec::new();
    let mut pending_index_writes: Vec<PendingWrite<u32>> = Vec::new();
    let mut pending_allocations: Vec<(Entity, GpuMeshAllocation)> = Vec::new();

    for (entity, mut mesh_ready) in ready.iter_mut().take(MAX_UPLOADS_PER_FRAME) {
        let mesh = std::mem::take(&mut mesh_ready.0);
        if let Some(alloc) = reserve_mesh(
            &mut render_state,
            mesh.vertices,
            mesh.indices,
            &mut pending_vertex_writes,
            &mut pending_index_writes,
        ) {
            pending_allocations.push((entity, alloc));
        } else {
            // No geometry: either a fully-air chunk, or (after a re-mesh) a
            // chunk that is now completely hidden by its neighbours, e.g.
            // buried stone. Its PREVIOUS mesh must be released here, or the
            // stale boundary faces we just re-meshed to remove keep drawing.
            render_state
                .shared
                .write()
                .unwrap()
                .allocations
                .remove(&entity);
            // Still mark it done so it doesn't get requeued every frame.
            commands
                .entity(entity)
                .remove::<ChunkMeshReady>()
                .insert(ChunkMeshUploaded);
        }
    }

    if pending_allocations.is_empty() {
        return;
    }

    let vertex_buffer_id = render_state.vertex_buffer_id;
    let index_buffer_id = render_state.index_buffer_id;

    if !pending_vertex_writes.is_empty() || !pending_index_writes.is_empty() {
        vk.resources.flight(vk.flight_id).wait(None).unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &vk.queue,
                &vk.resources,
                vk.flight_id,
                |_cbf, tcx| {
                    for write in &pending_vertex_writes {
                        let end = write.offset_bytes
                            + (write.data.len() * size_of::<ChunkVertex>()) as u64;
                        tcx.try_write_buffer::<[ChunkVertex]>(
                            vertex_buffer_id,
                            write.offset_bytes..end,
                        )?
                        .copy_from_slice(&write.data);
                    }
                    for write in &pending_index_writes {
                        let end = write.offset_bytes + (write.data.len() * size_of::<u32>()) as u64;
                        tcx.try_write_buffer::<[u32]>(index_buffer_id, write.offset_bytes..end)?
                            .copy_from_slice(&write.data);
                    }
                    Ok(())
                },
                [
                    (vertex_buffer_id, HostAccessType::Write),
                    (index_buffer_id, HostAccessType::Write),
                ],
                [],
                [],
            )
            .unwrap();
        }
    }

    for (entity, allocation) in pending_allocations {
        render_state
            .shared
            .write()
            .unwrap()
            .allocations
            .insert(entity, allocation);
        commands
            .entity(entity)
            .remove::<ChunkMeshReady>()
            .insert(ChunkMeshUploaded);
    }
}

fn cleanup_removed_chunk_meshes(
    mut removed: RemovedComponents<Chunk>,
    render_state: Option<Res<RenderState>>,
) {
    let Some(render_state) = render_state else {
        return;
    };
    let mut shared = render_state.shared.write().unwrap();
    for entity in removed.read() {
        shared.allocations.remove(&entity);
    }
}

fn render_frame(
    mut vk: ResMut<VulkanContext>,
    mut render_state: ResMut<RenderState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Transform, &crate::player::FlyCamera), With<crate::player::LocalPlayer>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let width = window.resolution.width() as u32;
    let height = window.resolution.height() as u32;
    if width == 0 || height == 0 {
        return;
    }

    let needs_resize = vk.recreate_swapchain
        || vk.viewport.extent[0] as u32 != width
        || vk.viewport.extent[1] as u32 != height;

    if needs_resize {
        vk.swapchain_id = vk
            .resources
            .recreate_swapchain(vk.swapchain_id, |info| SwapchainCreateInfo {
                image_extent: [width, height],
                ..*info
            })
            .expect("failed to recreate swapchain");

        vk.viewport.extent = [width as f32, height as f32];
        vk.recreate_swapchain = false;

        render_state.depth_image_id = vk
            .resources
            .create_image(
                &ImageCreateInfo {
                    format: Format::D32_SFLOAT,
                    extent: [width, height, 1],
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            )
            .unwrap();
    }

    let aspect = vk.viewport.extent[0] / vk.viewport.extent[1];
    let view_proj = camera
        .iter()
        .next()
        .map(|(transform, cam)| cam.view_projection(transform, aspect))
        .unwrap_or(glam::Mat4::IDENTITY);

    let frame = FrameData {
        view_proj: view_proj.to_cols_array_2d(),
        viewport: vk.viewport.clone(),
        // Created once in vulkan_prelude::setup_vulkan (nearest-mag / trilinear).
        sampler_id: vk.block_sampler_id,
    };

    vk.resources.flight(vk.flight_id).wait(None).unwrap();

    let task_graph = render_state.task_graph.lock().unwrap();

    let resource_map = resource_map!(
        &*task_graph,
        render_state.virtual_swapchain_id => vk.swapchain_id,
        render_state.virtual_depth_id => render_state.depth_image_id,
    )
    .unwrap();

    match unsafe { task_graph.execute(resource_map, &frame, || {}) } {
        Ok(()) => {}
        Err(ExecuteError::Swapchain {
            error: VulkanError::OutOfDate,
            ..
        }) => {
            vk.recreate_swapchain = true;
        }
        Err(e) => panic!("failed to execute frame: {e:?}"),
    }
}
