//! Vulkan resources, the task-graph draw, and buffer uploads.

use std::{
    collections::HashMap,
    mem::{align_of, size_of},
    num::NonZero,
    sync::{Arc, Mutex, RwLock},
};

use bevy_ecs::prelude::*;
use bevy_time::Time;
use bevy_transform::prelude::Transform;
use bevy_window::{PrimaryWindow, Window};
use glam::{Mat4, Vec3};
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

use crate::streaming::{LocalPlayerId, RemotePlayer};
use crate::{
    mesh::{ChunkVertex, fs, vs},
    player::{FlyCamera, LocalPlayer},
    vulkan_prelude::VulkanContext,
};

use super::allocator::{GpuMeshAllocation, MeshBufferAllocator};
use super::ui_pass::{self, UiDrawList, UiPassGpu};

// 32^3 chunks pack more geometry per chunk than the old 16x16x256 columns
// did on average once boundary culling removes most internal faces, and
// with no textures/AO there's less per-vertex bandwidth pressure, so this
// keeps the same generous headroom as the established project.
pub(super) const MAX_VERTICES: u64 = 40_000_000;
pub(super) const MAX_INDICES: u64 = 60_000_000;

pub(super) struct RenderStateShared {
    pub pipeline: Option<Arc<GraphicsPipeline>>,
    /// UI quads (boxes + glyphs). Drawn after world/avatars in the same
    /// subpass with alpha blending and depth test/write disabled.
    pub ui_pipeline: Option<Arc<GraphicsPipeline>>,
    /// Chunk meshes only. Avatars share one [`avatar`] slice drawn with
    /// per-player model matrices.
    pub allocations: HashMap<Entity, GpuMeshAllocation>,
    pub avatar: Option<GpuMeshAllocation>,
    pub effects_gpu: Option<super::effects_gpu::EffectGpuState>,
}

pub(super) struct FrameData {
    pub frame_index: u64,
    pub time_seconds: f32,
    pub delta_seconds: f32,
    pub view_proj: Mat4,
    pub camera_position: Vec3,
    pub viewport: vulkano::pipeline::graphics::viewport::Viewport,
    /// Bindless sampler used to read every block texture. Same sampler for
    /// all chunks; per-texture state lives in the image, not here.
    pub sampler_id: SamplerId,
    /// One model matrix per remote player. Mesh is local-space in [`avatar`].
    pub avatar_models: Vec<Mat4>,
    /// Player ID -> model transform, for actions that explicitly request it.
    pub model_transforms: HashMap<u64, Mat4>,
    /// Validated actions received since the preceding rendered frame.
    pub mod_actions: Vec<super::effects::ModActionEvent>,
    /// UI instance buffer + live count (synced from `UiDrawList` each frame).
    pub ui_buffer_id: Id<Buffer>,
    pub ui_count: u32,
}

struct ChunkRenderTask {
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    ui_buffer_id: Id<Buffer>,
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
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        cbf.set_viewport(0, std::slice::from_ref(&world.viewport));

        let mut shared = self.shared.write().unwrap();
        if let Some(pipeline) = shared.pipeline.as_ref() {
            let push = |cbf: &mut RecordingCommandBuffer<'_>, view_proj: Mat4| {
                cbf.push_constants(
                    pipeline.layout(),
                    0,
                    &vs::PushConstants {
                        view_proj: view_proj.to_cols_array_2d(),
                        sampler_id: world.sampler_id,
                    },
                );
            };

            push(cbf, world.view_proj);
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
                unsafe {
                    cbf.draw_indexed(
                        alloc.index_count,
                        1,
                        alloc.first_index(),
                        alloc.vertex_offset(),
                        0,
                    );
                }
            }

            if let Some(avatar) = shared.avatar {
                if avatar.index_count > 0 {
                    for model in &world.avatar_models {
                        push(cbf, world.view_proj * *model);
                        unsafe {
                            cbf.draw_indexed(
                                avatar.index_count,
                                1,
                                avatar.first_index(),
                                avatar.vertex_offset(),
                                0,
                            );
                        }
                    }
                }
            }
        }

        if let Some(effects) = shared.effects_gpu.as_mut() {
            effects.record_graphics(
                cbf,
                tcx,
                world,
                super::effects::RenderPhase::WorldTransparent,
            )?;
            effects.record_graphics(cbf, tcx, world, super::effects::RenderPhase::WorldOverlay)?;
        }

        // UI pass: same subpass, after world geometry. Alpha-blended, no
        // depth test/write — see `ui_pass::create_pipeline`.
        if let Some(ui_pipeline) = shared.ui_pipeline.as_ref() {
            ui_pass::record_draw(
                cbf,
                ui_pipeline,
                world.ui_buffer_id,
                world.ui_count,
                [world.viewport.extent[0], world.viewport.extent[1]],
                world.sampler_id,
            );
        }
        if let Some(effects) = shared.effects_gpu.as_mut() {
            effects.record_graphics(cbf, tcx, world, super::effects::RenderPhase::UiOverlay)?;
        }

        Ok(())
    }
}

#[derive(Resource)]
pub struct RenderState {
    pub(super) vertex_buffer_id: Id<Buffer>,
    pub(super) index_buffer_id: Id<Buffer>,
    pub(super) depth_image_id: Id<Image>,
    pub(super) virtual_swapchain_id: Id<Swapchain>,
    pub(super) virtual_depth_id: Id<Image>,
    pub(super) task_graph: Mutex<ExecutableTaskGraph<FrameData>>,
    pub(super) shared: Arc<RwLock<RenderStateShared>>,
    pub(super) allocator: MeshBufferAllocator,
    pub(super) ui_gpu: UiPassGpu,
    pub(super) frame_index: u64,
}

pub(super) struct PendingWrite<T> {
    pub offset_bytes: u64,
    pub data: Vec<T>,
}

/// One wait + one batched host write for a list of sub-allocations.
pub(super) fn upload_writes(
    vk: &VulkanContext,
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    vertex_writes: &[PendingWrite<ChunkVertex>],
    index_writes: &[PendingWrite<u32>],
) {
    if vertex_writes.is_empty() && index_writes.is_empty() {
        return;
    }

    vk.resources.flight(vk.flight_id).wait(None).unwrap();

    unsafe {
        vulkano_taskgraph::execute(
            &vk.queue,
            &vk.resources,
            vk.flight_id,
            |_cbf, tcx| {
                for write in vertex_writes {
                    let end =
                        write.offset_bytes + (write.data.len() * size_of::<ChunkVertex>()) as u64;
                    tcx.try_write_buffer::<[ChunkVertex]>(
                        vertex_buffer_id,
                        write.offset_bytes..end,
                    )?
                    .copy_from_slice(&write.data);
                }
                for write in index_writes {
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

pub(super) fn setup_renderer(
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

    // UI instance buffer must exist before the task graph is built so we
    // can declare VERTEX_ATTRIBUTE_READ access on the node.
    let ui_gpu = UiPassGpu::new(&vk);
    let effect_arena_id = super::effects_gpu::create_effect_arena(&vk)
        .expect("failed to create mod-effect buffer arena");

    let shared = Arc::new(RwLock::new(RenderStateShared {
        pipeline: None,
        ui_pipeline: None,
        allocations: HashMap::new(),
        avatar: None,
        effects_gpu: None,
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

    task_graph.add_host_buffer_access(effect_arena_id, HostAccessType::Write);
    let compute_task = super::effects_gpu::EffectComputeTask {
        shared: Arc::clone(&shared),
        arena_id: effect_arena_id,
    };
    let mut compute_node = task_graph.create_task_node(
        "ModEffectsCompute",
        vulkano_taskgraph::QueueFamilyType::Graphics,
        compute_task,
    );
    let compute_node_id = compute_node
        .buffer_access(effect_arena_id, effect_storage_access())
        .build();

    let task = ChunkRenderTask {
        vertex_buffer_id,
        index_buffer_id,
        ui_buffer_id: ui_gpu.buffer_id,
        virtual_swapchain_id,
        virtual_depth_id,
        shared: Arc::clone(&shared),
    };

    let mut node =
        task_graph.create_task_node("Chunks", vulkano_taskgraph::QueueFamilyType::Graphics, task);
    let chunk_node_id = node
        .framebuffer(virtual_framebuffer_id)
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
        .buffer_access(ui_gpu.buffer_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
        .buffer_access(effect_arena_id, effect_storage_access())
        .build();

    // Both nodes access the effect arena, so make the compute-to-draw order
    // explicit. TaskGraph does not infer an edge from shared resource IDs.
    task_graph.add_edge(compute_node_id, chunk_node_id).unwrap();

    let task_graph = unsafe {
        task_graph
            .compile(&CompileInfo {
                queues: &[&vk.queue],
                present_queue: Some(&vk.queue),
                flight_id: vk.flight_id,
                ..Default::default()
            })
            .unwrap()
    };

    let subpass = task_graph
        .task_nodes()
        .find_map(|node| node.subpass().cloned())
        .unwrap();

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

    // UI pipeline shares the same subpass; alpha blend, no depth.
    let ui_pipeline = ui_pass::create_pipeline(&vk, &subpass);
    shared.write().unwrap().ui_pipeline = Some(ui_pipeline);
    shared.write().unwrap().effects_gpu = Some(super::effects_gpu::EffectGpuState::new(
        &vk,
        effect_arena_id,
        subpass,
    ));

    commands.insert_resource(RenderState {
        vertex_buffer_id,
        index_buffer_id,
        depth_image_id,
        virtual_swapchain_id,
        virtual_depth_id,
        task_graph: Mutex::new(task_graph),
        shared,
        allocator: MeshBufferAllocator::new(),
        ui_gpu,
        frame_index: 0,
    });
}

fn effect_storage_access() -> AccessTypes {
    AccessTypes::COMPUTE_SHADER_STORAGE_READ
        | AccessTypes::COMPUTE_SHADER_STORAGE_WRITE
        | AccessTypes::VERTEX_SHADER_STORAGE_READ
        | AccessTypes::VERTEX_SHADER_STORAGE_WRITE
        | AccessTypes::FRAGMENT_SHADER_STORAGE_READ
        | AccessTypes::FRAGMENT_SHADER_STORAGE_WRITE
}

pub(super) fn render_frame(
    mut vk: ResMut<VulkanContext>,
    mut render_state: ResMut<RenderState>,
    time: Res<Time>,
    mut mod_actions: MessageReader<super::effects::ModActionEvent>,
    ui_list: Option<Res<UiDrawList>>,
    local_player_id: Option<Res<LocalPlayerId>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Transform, &FlyCamera), With<LocalPlayer>>,
    remotes: Query<(&RemotePlayer, &Transform)>,
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
    let camera_transform = camera.iter().next().map(|(transform, _)| transform);
    let view_proj = camera
        .iter()
        .next()
        .map(|(transform, cam)| cam.view_projection(transform, aspect))
        .unwrap_or(Mat4::IDENTITY);

    let mut model_transforms: HashMap<u64, Mat4> = remotes
        .iter()
        .map(|(player, transform)| (player.0, transform.to_matrix()))
        .collect();
    if let (Some(player_id), Some((transform, _))) = (
        local_player_id.as_deref().and_then(|id| id.0),
        camera.iter().next(),
    ) {
        model_transforms.insert(player_id, transform.to_matrix());
    }
    let avatar_models = model_transforms.values().copied().collect();
    let mod_actions = mod_actions.read().cloned().collect();

    // Upload UI instances only when `UiDrawList.generation` changed.
    if let Some(list) = ui_list.as_deref() {
        ui_pass::sync(&vk, &mut render_state.ui_gpu, list);
    }
    let ui_count = render_state.ui_gpu.count;
    let ui_buffer_id = render_state.ui_gpu.buffer_id;

    let frame = FrameData {
        frame_index: render_state.frame_index,
        time_seconds: time.elapsed_secs(),
        delta_seconds: time.delta_secs(),
        view_proj,
        camera_position: camera_transform
            .map(|transform| transform.translation)
            .unwrap_or(Vec3::ZERO),
        viewport: vk.viewport.clone(),
        sampler_id: vk.block_sampler_id,
        avatar_models,
        model_transforms,
        mod_actions,
        ui_buffer_id,
        ui_count,
    };
    render_state.frame_index = render_state.frame_index.wrapping_add(1);

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
