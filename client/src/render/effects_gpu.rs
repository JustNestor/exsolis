//! Vulkano pipelines and task-graph execution for validated effect packages.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    sync::Arc,
};

use bevy_log::{info, warn};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    descriptor_set::{
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
    },
    device::Device,
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::{
        ComputePipeline, GraphicsPipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::VertexInputState,
            viewport::ViewportState,
        },
        layout::{PipelineLayoutCreateInfo, push_constant_ranges_from_stages},
    },
    render_pass::Subpass,
    shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
};
use vulkano_taskgraph::{
    Id, Task, TaskContext, TaskResult,
    command_buffer::{DependencyInfo, MemoryBarrier, RecordingCommandBuffer},
    resource::{AccessTypes, HostAccessType},
};

use crate::{
    assets::{AssetCacheChanged, AssetHandlerRegistry, WgslAsset},
    vulkan_prelude::VulkanContext,
};

use super::{
    effects::*,
    gpu::FrameData,
    world_data::{
        WORLD_CELL_COUNT, WORLD_TAG_BYTES, WORLD_WINDOW_CELLS, world_buffer_bytes,
        world_cell_bytes,
    },
};

pub(super) const EFFECT_ARENA_BYTES: u64 = 16 * 1024 * 1024;
const TRANSIENT_BYTES: u64 = 4 * 1024 * 1024;
/// Frame buffer layout. Everything after byte 240 was added later, so shaders
/// that declare only the classic 240-byte block keep working:
///   240 sampler_nearest: u32   244 sampler_linear: u32
///   248 texture_count:   u32   252 padding
///   256 textures: array<u32, MAX_EFFECT_TEXTURES>  (bindless image indices)
///   288 instance_age: f32      292 stop_age: f32 (-1 while running)
///   296 instance_slot: u32     300 padding
///   304 world_dims: vec4<u32>  (window chunks xyz, chunk edge in w)
///   320 world_anchor: vec4<i32> (window centre chunk xyz, w = usable flag)
const FRAME_TABLE_OFFSET: usize = 240;
const FRAME_INSTANCE_OFFSET: usize = FRAME_TABLE_OFFSET + 16 + MAX_EFFECT_TEXTURES * 4;
const FRAME_WORLD_OFFSET: usize = FRAME_INSTANCE_OFFSET + 16;
const FRAME_BYTES: usize = FRAME_WORLD_OFFSET + 32;
const _: () = assert!(FRAME_WORLD_OFFSET % 16 == 0);
/// Descriptor binding of the world block buffer (the `engine/world` import
/// declares it). Sits above the highest possible buffer binding (2 + 16).
pub(super) const WORLD_BINDING: u32 = 24;
const MIN_DESCRIPTOR_RANGE: u64 = 4;

#[derive(Clone, Copy)]
pub(super) struct BufferRange {
    pub offset: u64,
    pub size: u64,
    pub persistent: bool,
}

pub(super) enum EffectPipeline {
    Compute(Arc<ComputePipeline>),
    Graphics(Arc<GraphicsPipeline>),
}

pub(super) struct PreparedPass {
    pub id: String,
    pub kind: PassKind,
    pub pipeline: EffectPipeline,
    pub layout: Arc<PipelineLayout>,
    pub descriptor_layout: Arc<DescriptorSetLayout>,
    pub workgroups: [u32; 3],
    pub vertex_count: u32,
    pub instance_count: u32,
    pub phase: Option<RenderPhase>,
    /// (index into the effect's buffers, byte offset) of the GPU-written
    /// dispatch/draw command, when the manifest asks for indirect execution.
    pub indirect: Option<(usize, u64)>,
}

pub(super) struct PreparedEffect {
    pub manifest: EffectManifest,
    instances: Vec<EffectInstance>,
    pub passes: Vec<PreparedPass>,
}

struct EffectInstance {
    buffers: Vec<BufferRange>,
    state: Option<EffectInstanceState>,
}

#[derive(Clone)]
struct EffectInstanceState {
    payload: Vec<u8>,
    model: Option<glam::Mat4>,
    /// Player whose transform keeps `model` current. `None` for ordinary
    /// one-shot instances, which stay where they spawned.
    follow_player: Option<u64>,
    key: Option<String>,
    expires_at: f32,
    started_at: f32,
    stopped_at: Option<f32>,
}

/// Per-instance values shaders read from the frame buffer.
#[derive(Clone, Copy)]
struct InstanceInfo {
    age: f32,
    stop_age: f32,
    slot: u32,
}

impl InstanceInfo {
    fn new(state: &EffectInstanceState, slot: usize, now: f32) -> Self {
        Self {
            age: (now - state.started_at).max(0.0),
            stop_age: state.stopped_at.map_or(-1.0, |at| (now - at).max(0.0)),
            slot: slot as u32,
        }
    }
}

pub(crate) struct EffectGpuState {
    pub arena_id: Id<Buffer>,
    arena: Arc<Buffer>,
    world_id: Id<Buffer>,
    world: Arc<Buffer>,
    capacity: u64,
    storage_alignment: u64,
    persistent_end: u64,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    subpass: Subpass,
    pub effects: HashMap<String, PreparedEffect>,
    initialized_effects: HashSet<String>,
    rejected_effects: HashSet<String>,
    graphics_frame: Option<u64>,
    graphics_cursor: u64,
}

impl EffectGpuState {
    pub fn new(
        vk: &VulkanContext,
        arena_id: Id<Buffer>,
        world_id: Id<Buffer>,
        subpass: Subpass,
    ) -> Self {
        Self {
            arena_id,
            arena: vk.resources.buffer(arena_id).buffer().clone(),
            world_id,
            world: vk.resources.buffer(world_id).buffer().clone(),
            capacity: EFFECT_ARENA_BYTES,
            storage_alignment: storage_alignment(vk),
            persistent_end: 0,
            descriptor_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                &vk.device,
                &Default::default(),
            )),
            subpass,
            effects: HashMap::new(),
            initialized_effects: HashSet::new(),
            rejected_effects: HashSet::new(),
            graphics_frame: None,
            graphics_cursor: EFFECT_ARENA_BYTES - TRANSIENT_BYTES,
        }
    }

    pub fn reset(&mut self) {
        self.persistent_end = 0;
        self.effects.clear();
        self.initialized_effects.clear();
        self.rejected_effects.clear();
        self.graphics_frame = None;
        self.graphics_cursor = self.capacity - TRANSIENT_BYTES;
    }

    pub fn prepare_loaded(&mut self, vk: &VulkanContext, assets: &AssetHandlerRegistry) {
        for (path, manifest) in assets.iter_loaded::<EffectManifest>() {
            if self.effects.contains_key(&manifest.id) {
                continue;
            }
            if !self.rejected_effects.insert(manifest.id.clone()) {
                continue;
            }
            match self.prepare_effect(vk, assets, manifest) {
                Ok(effect) => {
                    info!("loaded GPU effect '{}' from {path}", manifest.id);
                    self.rejected_effects.remove(&manifest.id);
                    self.effects.insert(manifest.id.clone(), effect);
                }
                Err(error) => warn!("GPU effect '{}' from {path} rejected: {error}", manifest.id),
            }
        }
    }

    fn prepare_effect(
        &mut self,
        vk: &VulkanContext,
        assets: &AssetHandlerRegistry,
        manifest: &EffectManifest,
    ) -> Result<PreparedEffect, String> {
        if !manifest.images.is_empty() {
            return Err("manifest images are not supported by the current GPU executor".into());
        }
        let mut cursor = align_up(self.persistent_end, self.storage_alignment);
        let mut instances = Vec::with_capacity(manifest.max_instances as usize);
        for _ in 0..manifest.max_instances {
            let mut buffers = Vec::with_capacity(manifest.buffers.len());
            for buffer in &manifest.buffers {
                cursor = align_up(cursor, self.storage_alignment);
                let size = u64::from(buffer.bytes);
                let end = cursor
                    .checked_add(size)
                    .ok_or_else(|| "effect arena allocation overflow".to_string())?;
                if end > self.capacity.saturating_sub(TRANSIENT_BYTES) {
                    return Err("effect GPU arena is full".into());
                }
                buffers.push(BufferRange {
                    offset: cursor,
                    size,
                    persistent: buffer.persistent,
                });
                cursor = end;
            }
            instances.push(EffectInstance {
                buffers,
                state: None,
            });
        }
        let descriptor_layout =
            create_effect_descriptor_layout(&vk.device, manifest.buffers.len())?;
        let mut passes = Vec::with_capacity(manifest.passes.len());
        for pass in &manifest.passes {
            let indirect = pass.indirect.as_ref().and_then(|args| {
                manifest
                    .buffers
                    .iter()
                    .position(|buffer| buffer.id == args.buffer)
                    .map(|index| (index, u64::from(args.offset)))
            });
            let prepared = match pass.kind {
                PassKind::Compute => {
                    let shader = load_entry(
                        vk,
                        assets,
                        pass.shader.as_deref().unwrap(),
                        pass.entry_point.as_deref().unwrap(),
                        naga::ShaderStage::Compute,
                    )?;
                    let stage = PipelineShaderStageCreateInfo::new(&shader.entry_point);
                    let layout = create_effect_pipeline_layout(
                        vk,
                        &descriptor_layout,
                        std::slice::from_ref(&stage),
                    )?;
                    let pipeline = ComputePipeline::new(
                        &vk.device,
                        None,
                        &ComputePipelineCreateInfo::new(stage, &layout),
                    )
                    .map_err(|e| format!("compute pipeline '{}': {e}", pass.id))?;
                    PreparedPass {
                        id: pass.id.clone(),
                        kind: pass.kind,
                        pipeline: EffectPipeline::Compute(pipeline),
                        layout,
                        descriptor_layout: descriptor_layout.clone(),
                        workgroups: pass.workgroups,
                        vertex_count: 0,
                        instance_count: 0,
                        phase: None,
                        indirect,
                    }
                }
                PassKind::Graphics => {
                    let vertex = pass.vertex.as_ref().unwrap();
                    let fragment = pass.fragment.as_ref().unwrap();
                    let vs = load_entry(
                        vk,
                        assets,
                        &vertex.shader,
                        &vertex.entry_point,
                        naga::ShaderStage::Vertex,
                    )?;
                    let fs = load_entry(
                        vk,
                        assets,
                        &fragment.shader,
                        &fragment.entry_point,
                        naga::ShaderStage::Fragment,
                    )?;
                    let stages = [
                        PipelineShaderStageCreateInfo::new(&vs.entry_point),
                        PipelineShaderStageCreateInfo::new(&fs.entry_point),
                    ];
                    let layout = create_effect_pipeline_layout(vk, &descriptor_layout, &stages)?;
                    let blend = match pass.blend.unwrap() {
                        BlendMode::Opaque => None,
                        BlendMode::Alpha => Some(AttachmentBlend::alpha()),
                        // `AttachmentBlend::additive()` is One/One, which ignores the
                        // fragment's alpha: every quad would add its full colour as a
                        // hard square. Weight by alpha so soft edges and fade-out work.
                        BlendMode::Additive => Some(AttachmentBlend {
                            src_color_blend_factor: BlendFactor::SrcAlpha,
                            dst_color_blend_factor: BlendFactor::One,
                            ..AttachmentBlend::additive()
                        }),
                    };
                    let depth = match pass.depth.unwrap() {
                        DepthMode::Disabled => DepthStencilState::default(),
                        DepthMode::Test => DepthStencilState {
                            depth: Some(DepthState {
                                write_enable: false,
                                compare_op: CompareOp::Less,
                            }),
                            ..Default::default()
                        },
                    };
                    let pipeline = GraphicsPipeline::new(
                        &vk.device,
                        None,
                        &GraphicsPipelineCreateInfo {
                            stages: &stages,
                            // Effect geometry is generated from `vertex_index` in WGSL;
                            // no vertex buffers or attributes are required.
                            vertex_input_state: Some(&VertexInputState::default()),
                            input_assembly_state: Some(&InputAssemblyState::default()),
                            viewport_state: Some(&ViewportState::default()),
                            rasterization_state: Some(&RasterizationState {
                                cull_mode: CullMode::None,
                                ..Default::default()
                            }),
                            multisample_state: Some(&MultisampleState::default()),
                            color_blend_state: Some(&ColorBlendState {
                                attachments: &[ColorBlendAttachmentState {
                                    blend,
                                    ..Default::default()
                                }],
                                ..Default::default()
                            }),
                            depth_stencil_state: Some(&depth),
                            dynamic_state: &[vulkano::pipeline::DynamicState::Viewport],
                            subpass: Some((&self.subpass).into()),
                            ..GraphicsPipelineCreateInfo::new(&layout)
                        },
                    )
                    .map_err(|e| format!("graphics pipeline '{}': {e}", pass.id))?;
                    PreparedPass {
                        id: pass.id.clone(),
                        kind: pass.kind,
                        pipeline: EffectPipeline::Graphics(pipeline),
                        layout,
                        descriptor_layout: descriptor_layout.clone(),
                        workgroups: pass.workgroups,
                        vertex_count: pass.vertex_count,
                        instance_count: pass.instance_count,
                        phase: pass.phase,
                        indirect,
                    }
                }
            };
            passes.push(prepared);
        }
        self.persistent_end = cursor;
        Ok(PreparedEffect {
            manifest: manifest.clone(),
            instances,
            passes,
        })
    }

    /// Applies queued block-data uploads to the world buffer. Host writes
    /// land before the frame's commands run, and the previous frame's flight
    /// was waited on, so nothing in flight can still be reading these cells.
    fn write_world_uploads(&self, tcx: &mut TaskContext<'_>, frame: &FrameData) -> TaskResult {
        for upload in &frame.world_uploads {
            let tag_offset = u64::from(upload.cell) * WORLD_TAG_BYTES;
            tcx.try_write_buffer::<[i32]>(self.world_id, tag_offset..tag_offset + WORLD_TAG_BYTES)?
                .copy_from_slice(&upload.tag);
            if let Some(voxels) = &upload.voxels {
                let offset = WORLD_CELL_COUNT as u64 * WORLD_TAG_BYTES
                    + u64::from(upload.cell) * world_cell_bytes();
                tcx.try_write_buffer::<[u32]>(self.world_id, offset..offset + world_cell_bytes())?
                    .copy_from_slice(voxels);
            }
        }
        Ok(())
    }

    pub fn record_compute(
        &mut self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        frame: &FrameData,
    ) -> TaskResult {
        self.write_world_uploads(tcx, frame)?;

        let transient_start = align_up(self.persistent_end, self.storage_alignment);
        let mut transient = transient_start;
        for effect in self.effects.values_mut() {
            for instance in &mut effect.instances {
                if instance
                    .state
                    .as_ref()
                    .is_some_and(|state| state.expires_at <= frame.time_seconds)
                {
                    instance.state = None;
                }
                // Opt-in tracking (`follow_model`): instances that didn't ask
                // keep the matrix captured at spawn and stay where they were.
                if let Some(state) = instance.state.as_mut() {
                    if let Some(player) = state.follow_player {
                        if let Some(model) = frame.model_transforms.get(&player) {
                            state.model = Some(*model);
                        }
                    }
                }
            }
        }

        let mut initialized = Vec::new();
        for effect in self.effects.values() {
            if self.initialized_effects.contains(&effect.manifest.id) {
                continue;
            }
            for buffer in effect.instances.iter().flat_map(|slot| &slot.buffers) {
                tcx.try_write_buffer::<[u8]>(
                    self.arena_id,
                    buffer.offset..buffer.offset + buffer.size,
                )?
                .fill(0);
            }
            initialized.push(effect.manifest.id.clone());
        }
        self.initialized_effects.extend(initialized);

        for effect in self.effects.values() {
            if self.initialized_effects.contains(&effect.manifest.id) {
                for instance in &effect.instances {
                    if instance.state.is_none() {
                        continue;
                    }
                    for buffer in instance.buffers.iter().filter(|buffer| !buffer.persistent) {
                        tcx.try_write_buffer::<[u8]>(
                            self.arena_id,
                            buffer.offset..buffer.offset + buffer.size,
                        )?
                        .fill(0);
                    }
                }
            }
        }

        for event in &frame.mod_actions {
            match event.mode {
                ActionMode::Spawn => self.spawn_instance(cbf, tcx, frame, event, &mut transient)?,
                ActionMode::Update | ActionMode::Stop => {
                    self.address_instance(cbf, tcx, frame, event, &mut transient)?
                }
            }
        }

        let mut per_frame_passes = Vec::new();
        for (effect_id, effect) in &self.effects {
            for (instance_index, instance) in effect.instances.iter().enumerate() {
                if instance.state.is_none() {
                    continue;
                }
                for (pass_index, pass) in effect.passes.iter().enumerate() {
                    if pass.kind == PassKind::Compute
                        && effect
                            .manifest
                            .passes
                            .iter()
                            .any(|decl| decl.id == pass.id && decl.run_each_frame)
                    {
                        per_frame_passes.push((effect_id.clone(), instance_index, pass_index));
                    }
                }
            }
        }
        for (effect_id, instance_index, pass_index) in per_frame_passes {
            let effect = &self.effects[&effect_id];
            let instance = &effect.instances[instance_index];
            let Some(state) = instance.state.as_ref() else {
                continue;
            };
            let payload = state.payload.clone();
            let buffers = instance.buffers.clone();
            let pass = &effect.passes[pass_index];
            let frame_bytes = build_frame_bytes(
                frame,
                &effect.manifest.frame_inputs,
                &effect.manifest.textures,
                state.model,
                InstanceInfo::new(state, instance_index, frame.time_seconds),
            );
            self.dispatch(
                cbf,
                tcx,
                &buffers,
                pass,
                &payload,
                &frame_bytes,
                &mut transient,
            )?;
        }
        Ok(())
    }

    /// `spawn`: claim a slot (the live instance with the same key if there is
    /// one, else a free slot, else the oldest), reset its buffers and run the
    /// spawn pass once.
    fn spawn_instance(
        &mut self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        frame: &FrameData,
        event: &ModActionEvent,
        transient: &mut u64,
    ) -> TaskResult {
        let Some(pass_name) = event.compute_pass.as_deref() else {
            return Ok(());
        };
        let model = event
            .model_player_id
            .and_then(|id| frame.model_transforms.get(&id).copied());
        if event.model_player_id.is_some() && model.is_none() {
            warn!(
                "effect '{}' action '{}': no transform for player {:?}; skipped",
                event.effect_id, event.id, event.model_player_id
            );
            return Ok(());
        }
        let Some(effect) = self.effects.get_mut(&event.effect_id) else {
            return Ok(());
        };
        let Some(pass_index) = effect
            .passes
            .iter()
            .position(|pass| pass.id == pass_name && pass.kind == PassKind::Compute)
        else {
            return Ok(());
        };
        let now = frame.time_seconds;
        let keyed = event.instance_key.as_ref().and_then(|key| {
            effect.instances.iter().position(|instance| {
                instance
                    .state
                    .as_ref()
                    .is_some_and(|state| state.key.as_ref() == Some(key))
            })
        });
        let instance_index = keyed
            .or_else(|| {
                effect
                    .instances
                    .iter()
                    .position(|instance| instance.state.is_none())
            })
            .or_else(|| {
                effect
                    .instances
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.state
                            .as_ref()
                            .unwrap()
                            .started_at
                            .total_cmp(&b.state.as_ref().unwrap().started_at)
                    })
                    .map(|(index, _)| index)
            });
        let Some(instance_index) = instance_index else {
            return Ok(());
        };
        let buffers = effect.instances[instance_index].buffers.clone();
        for buffer in &buffers {
            tcx.try_write_buffer::<[u8]>(self.arena_id, buffer.offset..buffer.offset + buffer.size)?
                .fill(0);
        }
        let lifetime = event
            .lifetime_seconds
            .unwrap_or(effect.manifest.instance_lifetime_seconds);
        let state = EffectInstanceState {
            payload: event.payload.clone(),
            model,
            follow_player: event
                .follow_model
                .then_some(event.model_player_id)
                .flatten(),
            key: event.instance_key.clone(),
            expires_at: now + lifetime,
            started_at: now,
            stopped_at: None,
        };
        let info = InstanceInfo::new(&state, instance_index, now);
        effect.instances[instance_index].state = Some(state);
        let frame_bytes = build_frame_bytes(
            frame,
            &effect.manifest.frame_inputs,
            &effect.manifest.textures,
            model,
            info,
        );
        let effect = self.effects.get(&event.effect_id).unwrap();
        let pass = &effect.passes[pass_index];
        self.dispatch(
            cbf,
            tcx,
            &buffers,
            pass,
            &event.payload,
            &frame_bytes,
            transient,
        )
    }

    /// `update` / `stop`: find the live instance by key, adjust its lifetime
    /// and run the route's optional one-shot pass with the event's payload.
    /// The instance's own spawn payload (what per-frame passes and draws read)
    /// is never overwritten; mods carry changing data in persistent buffers.
    fn address_instance(
        &mut self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        frame: &FrameData,
        event: &ModActionEvent,
        transient: &mut u64,
    ) -> TaskResult {
        let Some(key) = event.instance_key.as_ref() else {
            return Ok(());
        };
        let Some(effect) = self.effects.get_mut(&event.effect_id) else {
            return Ok(());
        };
        let Some(slot) = effect.instances.iter().position(|instance| {
            instance
                .state
                .as_ref()
                .is_some_and(|state| state.key.as_ref() == Some(key) && state.stopped_at.is_none())
        }) else {
            // Nothing live under this key: the instance already expired or
            // the action raced its spawn. Dropping it is the safe reading.
            return Ok(());
        };
        let pass_index = event.compute_pass.as_deref().and_then(|name| {
            effect
                .passes
                .iter()
                .position(|pass| pass.id == name && pass.kind == PassKind::Compute)
        });
        let now = frame.time_seconds;
        let default_lifetime = effect.manifest.instance_lifetime_seconds;
        let buffers = effect.instances[slot].buffers.clone();
        let state = effect.instances[slot].state.as_mut().unwrap();
        match event.mode {
            ActionMode::Update => {
                state.expires_at = now + event.lifetime_seconds.unwrap_or(default_lifetime);
            }
            ActionMode::Stop => {
                state.stopped_at = Some(now);
                state.expires_at = now + event.linger_seconds;
            }
            ActionMode::Spawn => return Ok(()),
        }
        let info = InstanceInfo::new(state, slot, now);
        let model = state.model;
        if let Some(pass_index) = pass_index {
            let frame_bytes = build_frame_bytes(
                frame,
                &effect.manifest.frame_inputs,
                &effect.manifest.textures,
                model,
                info,
            );
            let effect = self.effects.get(&event.effect_id).unwrap();
            let pass = &effect.passes[pass_index];
            self.dispatch(
                cbf,
                tcx,
                &buffers,
                pass,
                &event.payload,
                &frame_bytes,
                transient,
            )?;
        }
        if event.mode == ActionMode::Stop && event.linger_seconds <= 0.0 {
            if let Some(effect) = self.effects.get_mut(&event.effect_id) {
                effect.instances[slot].state = None;
            }
        }
        Ok(())
    }

    fn dispatch(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        buffers: &[BufferRange],
        pass: &PreparedPass,
        payload: &[u8],
        frame_bytes: &[u8],
        transient: &mut u64,
    ) -> TaskResult {
        // WGSL rounds struct size up to its alignment (16 with a vec3 member);
        // binding a smaller range than the declared struct is invalid.
        let payload_size =
            align_up(payload.len().max(MIN_DESCRIPTOR_RANGE as usize) as u64, 16);
        let action_offset = align_up(*transient, self.storage_alignment);
        let frame_offset = align_up(action_offset + payload_size, self.storage_alignment);
        let end = frame_offset + FRAME_BYTES as u64;
        if end > self.capacity - TRANSIENT_BYTES {
            return Ok(());
        }
        let action_write = tcx
            .try_write_buffer::<[u8]>(self.arena_id, action_offset..action_offset + payload_size)?;
        action_write.fill(0);
        action_write[..payload.len()].copy_from_slice(payload);
        tcx.try_write_buffer::<[u8]>(self.arena_id, frame_offset..end)?
            .copy_from_slice(frame_bytes);

        let descriptor_set = match self.create_descriptor_set(
            pass,
            buffers,
            action_offset,
            payload_size,
            frame_offset,
        ) {
            Ok(set) => set,
            Err(error) => {
                warn!("could not allocate effect descriptor set: {error}");
                return Ok(());
            }
        };
        let EffectPipeline::Compute(pipeline) = &pass.pipeline else {
            return Ok(());
        };
        unsafe { cbf.bind_pipeline(pipeline) };
        unsafe {
            cbf.as_raw().bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout(),
                1,
                &[descriptor_set.as_raw()],
                &[],
            );
            match pass.indirect {
                Some((buffer_index, offset)) => {
                    cbf.dispatch_indirect(self.arena_id, buffers[buffer_index].offset + offset);
                }
                None => {
                    cbf.dispatch(pass.workgroups);
                }
            }
        }
        // Later passes may read what this one wrote, including as GPU-written
        // dispatch arguments (`indirect`), which are fetched at DRAW_INDIRECT.
        let compute_barrier = MemoryBarrier {
            src_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
            src_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
            dst_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER
                | vulkano::sync::PipelineStages::DRAW_INDIRECT,
            dst_access: vulkano::sync::AccessFlags::SHADER_STORAGE_READ
                | vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE
                | vulkano::sync::AccessFlags::INDIRECT_COMMAND_READ,
            ..Default::default()
        };
        unsafe {
            cbf.pipeline_barrier(&DependencyInfo {
                memory_barriers: std::slice::from_ref(&compute_barrier),
                ..Default::default()
            });
        }
        cbf.destroy_object(descriptor_set);
        *transient = end;
        Ok(())
    }

    pub fn record_graphics(
        &mut self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        frame: &FrameData,
        phase: RenderPhase,
    ) -> TaskResult {
        if self.graphics_frame != Some(frame.frame_index) {
            self.graphics_frame = Some(frame.frame_index);
            self.graphics_cursor = self.capacity - TRANSIENT_BYTES;
        }
        let draws: Vec<_> = self
            .effects
            .iter()
            .flat_map(|(effect_id, effect)| {
                effect
                    .instances
                    .iter()
                    .enumerate()
                    .flat_map(move |(instance_index, instance)| {
                        if instance.state.is_none() {
                            return Vec::new();
                        }
                        effect
                            .passes
                            .iter()
                            .enumerate()
                            .filter(|(_, pass)| {
                                pass.kind == PassKind::Graphics && pass.phase == Some(phase)
                            })
                            .map(|(pass_index, _)| (effect_id.clone(), instance_index, pass_index))
                            .collect()
                    })
            })
            .collect();
        for (effect_id, instance_index, pass_index) in draws {
            let effect = &self.effects[&effect_id];
            let instance = &effect.instances[instance_index];
            let Some(state) = instance.state.as_ref() else {
                continue;
            };
            let payload = state.payload.clone();
            let model = state.model;
            let info = InstanceInfo::new(state, instance_index, frame.time_seconds);
            let buffers = instance.buffers.clone();
            let pass = &effect.passes[pass_index];
            // Round up to the WGSL struct size (see `dispatch`).
            let payload_size =
                align_up(payload.len().max(MIN_DESCRIPTOR_RANGE as usize) as u64, 16);
            let action_offset = align_up(self.graphics_cursor, self.storage_alignment);
            let frame_offset = align_up(action_offset + payload_size, self.storage_alignment);
            let end = frame_offset + FRAME_BYTES as u64;
            if end > self.capacity {
                continue;
            }
            let action_write = tcx.try_write_buffer::<[u8]>(
                self.arena_id,
                action_offset..action_offset + payload_size,
            )?;
            action_write.fill(0);
            action_write[..payload.len()].copy_from_slice(&payload);
            tcx.try_write_buffer::<[u8]>(self.arena_id, frame_offset..end)?
                .copy_from_slice(&build_frame_bytes(
                    frame,
                    &effect.manifest.frame_inputs,
                    &effect.manifest.textures,
                    model,
                    info,
                ));
            let descriptor_set = match self.create_descriptor_set(
                pass,
                &buffers,
                action_offset,
                payload_size,
                frame_offset,
            ) {
                Ok(set) => set,
                Err(error) => {
                    warn!("could not allocate effect descriptor set: {error}");
                    continue;
                }
            };
            let EffectPipeline::Graphics(pipeline) = &pass.pipeline else {
                continue;
            };
            unsafe { cbf.bind_pipeline(pipeline) };
            unsafe {
                cbf.as_raw().bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout(),
                    1,
                    &[descriptor_set.as_raw()],
                    &[],
                );
                match pass.indirect {
                    // Counts come from the GPU (e.g. a compacted alive list).
                    Some((buffer_index, offset)) => {
                        cbf.draw_indirect(
                            self.arena_id,
                            buffers[buffer_index].offset + offset,
                            1,
                            16,
                        );
                    }
                    None => {
                        cbf.draw(pass.vertex_count, pass.instance_count, 0, 0);
                    }
                }
            }
            let graphics_barrier = MemoryBarrier {
                src_stages: vulkano::sync::PipelineStages::FRAGMENT_SHADER,
                src_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
                dst_stages: vulkano::sync::PipelineStages::VERTEX_SHADER
                    | vulkano::sync::PipelineStages::FRAGMENT_SHADER,
                dst_access: vulkano::sync::AccessFlags::SHADER_STORAGE_READ
                    | vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
                ..Default::default()
            };
            unsafe {
                cbf.pipeline_barrier(&DependencyInfo {
                    memory_barriers: std::slice::from_ref(&graphics_barrier),
                    ..Default::default()
                });
            }
            cbf.destroy_object(descriptor_set);
            self.graphics_cursor = end;
        }
        Ok(())
    }

    fn create_descriptor_set(
        &self,
        pass: &PreparedPass,
        buffers: &[BufferRange],
        action_offset: u64,
        action_size: u64,
        frame_offset: u64,
    ) -> Result<Arc<DescriptorSet>, vulkano::Validated<vulkano::VulkanError>> {
        let frame_info = DescriptorBufferInfo {
            buffer: Some(&self.arena),
            offset: frame_offset,
            range: Some(FRAME_BYTES as u64),
        };
        let action_info = DescriptorBufferInfo {
            buffer: Some(&self.arena),
            offset: action_offset,
            range: Some(action_size),
        };
        let world_info = DescriptorBufferInfo {
            buffer: Some(&self.world),
            offset: 0,
            range: Some(world_buffer_bytes()),
        };
        let mut writes = vec![
            WriteDescriptorSet::buffer(0, &action_info),
            WriteDescriptorSet::buffer(1, &frame_info),
            WriteDescriptorSet::buffer(WORLD_BINDING, &world_info),
        ];
        let infos: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| {
                (
                    i as u32 + 2,
                    DescriptorBufferInfo {
                        buffer: Some(&self.arena),
                        offset: buffer.offset,
                        range: Some(buffer.size),
                    },
                )
            })
            .collect();
        for (binding, info) in &infos {
            writes.push(WriteDescriptorSet::buffer(*binding, info));
        }
        DescriptorSet::new(
            &self.descriptor_allocator,
            &pass.descriptor_layout,
            &writes,
            &[],
        )
    }
}

pub(super) fn create_effect_arena(vk: &VulkanContext) -> Result<Id<Buffer>, String> {
    let layout = DeviceLayout::new(
        std::num::NonZero::new(EFFECT_ARENA_BYTES).unwrap(),
        DeviceAlignment::new(storage_alignment(vk)).unwrap(),
    )
    .ok_or_else(|| "effect arena layout exceeds device limits".to_string())?;
    vk.resources
        .create_buffer(
            &BufferCreateInfo {
                // INDIRECT_BUFFER: draw/dispatch arguments may live in mod buffers.
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            layout,
        )
        .map_err(|e| format!("could not allocate effect buffer arena: {e}"))
}

/// Block data window shared by every effect (see `world_data`): chunk tags
/// followed by packed voxels, filled incrementally through host writes.
pub(super) fn create_world_buffer(vk: &VulkanContext) -> Result<Id<Buffer>, String> {
    let layout = DeviceLayout::new(
        std::num::NonZero::new(world_buffer_bytes()).unwrap(),
        DeviceAlignment::new(storage_alignment(vk)).unwrap(),
    )
    .ok_or_else(|| "world buffer layout exceeds device limits".to_string())?;
    vk.resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            layout,
        )
        .map_err(|e| format!("could not allocate world data buffer: {e}"))
}

pub(super) fn prepare_gpu_effects(
    vk: Option<bevy_ecs::prelude::Res<VulkanContext>>,
    assets: Option<bevy_ecs::prelude::Res<AssetHandlerRegistry>>,
    render_state: Option<bevy_ecs::prelude::Res<super::gpu::RenderState>>,
    mut changed: bevy_ecs::prelude::MessageReader<AssetCacheChanged>,
) {
    let (Some(vk), Some(assets), Some(render_state)) = (vk, assets, render_state) else {
        return;
    };
    let reload = changed.read().any(|event| {
        event.assets.iter().any(|asset| {
            Path::new(&asset.path)
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(extension.to_ascii_lowercase().as_str(), "fxm" | "wgsl")
                })
        })
    });
    if reload {
        // Pipeline and arena ranges may still be referenced by a previous
        // frame, so retire that flight before replacing the package set.
        vk.resources.flight(vk.flight_id).wait(None).unwrap();
    }
    let mut shared = render_state.shared.write().unwrap();
    if let Some(effects) = shared.effects_gpu.as_mut() {
        if reload {
            effects.reset();
        }
        effects.prepare_loaded(&vk, &assets);
    }
}

pub(super) struct EffectComputeTask {
    pub shared: Arc<std::sync::RwLock<super::gpu::RenderStateShared>>,
    pub arena_id: Id<Buffer>,
}

impl Task for EffectComputeTask {
    type World = FrameData;
    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        let mut shared = self.shared.write().unwrap();
        if let Some(effects) = shared.effects_gpu.as_mut() {
            effects.record_compute(cbf, tcx, world)?;
        }
        Ok(())
    }
}

fn create_effect_descriptor_layout(
    device: &Arc<Device>,
    buffer_count: usize,
) -> Result<Arc<DescriptorSetLayout>, String> {
    let stages = ShaderStages::COMPUTE | ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    let bindings: Vec<_> = (0..buffer_count as u32 + 2)
        .chain(std::iter::once(WORLD_BINDING))
        .map(|binding| DescriptorSetLayoutBinding {
            binding,
            descriptor_count: 1,
            stages,
            ..DescriptorSetLayoutBinding::new(
                vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
            )
        })
        .collect();
    DescriptorSetLayout::new(
        device,
        &DescriptorSetLayoutCreateInfo {
            bindings: &bindings,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

fn create_effect_pipeline_layout(
    vk: &VulkanContext,
    local: &Arc<DescriptorSetLayout>,
    stages: &[PipelineShaderStageCreateInfo<'_>],
) -> Result<Arc<PipelineLayout>, String> {
    let bindless = vk
        .resources
        .bindless_context()
        .ok_or("bindless context unavailable")?;
    let layouts = [bindless.global_set_layout(), local];
    PipelineLayout::new(
        &vk.device,
        &PipelineLayoutCreateInfo {
            set_layouts: &layouts,
            push_constant_ranges: &push_constant_ranges_from_stages(stages),
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())
}

struct LoadedEntry {
    _module: Arc<ShaderModule>,
    entry_point: vulkano::shader::EntryPoint,
}

fn load_entry(
    vk: &VulkanContext,
    assets: &AssetHandlerRegistry,
    shader_path: &str,
    entry_name: &str,
    stage: naga::ShaderStage,
) -> Result<LoadedEntry, String> {
    let shader = assets
        .get::<WgslAsset>(shader_path)
        .ok_or_else(|| format!("shader '{shader_path}' is not loaded"))?;
    let compiled = shader
        .entry_points
        .iter()
        .find(|entry| entry.name == entry_name && entry.stage == stage)
        .ok_or_else(|| {
            format!("shader '{shader_path}' has no {stage:?} entry point '{entry_name}'")
        })?;
    let module =
        unsafe { ShaderModule::new(&vk.device, &ShaderModuleCreateInfo::new(&compiled.spirv)) }
            .map_err(|e| e.to_string())?;
    let entry_point = module
        .entry_point(entry_name)
        .ok_or_else(|| format!("compiled shader is missing entry point '{entry_name}'"))?;
    Ok(LoadedEntry {
        _module: module,
        entry_point,
    })
}

/// View-projection as effect shaders expect it: standard WebGPU convention
/// (Y up in NDC).
///
/// `FlyCamera::view_projection` bakes a Y flip into the projection because
/// the hand-written chunk shaders output Vulkan clip space directly. Effect
/// shaders are WGSL, and naga's SPIR-V writer (default
/// `ADJUST_COORDINATE_SPACE`) negates `@builtin(position).y` on output.
/// Handing mods the Vulkan-flipped matrix therefore flips them twice, which
/// mirrors every effect vertically about the screen centre: it only lines up
/// with the world when the effect is at the vertical centre of the view.
/// Undo the engine's flip here so mod authors can use plain WebGPU maths.
fn webgpu_view_proj(frame: &FrameData) -> glam::Mat4 {
    glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0)) * frame.view_proj
}

fn build_frame_bytes(
    frame: &FrameData,
    inputs: &[FrameInput],
    textures: &[TextureDecl],
    model: Option<glam::Mat4>,
    instance: InstanceInfo,
) -> Vec<u8> {
    let mut bytes = vec![0; FRAME_BYTES];
    let has = |input| inputs.contains(&input);
    let view_proj = webgpu_view_proj(frame);
    // Identity (not an all-zero matrix) when no model is bound.
    if has(FrameInput::Model) {
        write_mat4(&mut bytes, 112, glam::Mat4::IDENTITY);
    }
    if has(FrameInput::ModelViewProjection) {
        write_mat4(&mut bytes, 176, view_proj);
    }
    if has(FrameInput::FrameIndex) {
        write_u32(&mut bytes, 0, frame.frame_index as u32);
    }
    if has(FrameInput::Time) {
        write_f32(&mut bytes, 4, frame.time_seconds);
    }
    if has(FrameInput::DeltaTime) {
        write_f32(&mut bytes, 8, frame.delta_seconds);
    }
    if has(FrameInput::Viewport) {
        write_f32(&mut bytes, 16, frame.viewport.extent[0]);
        write_f32(&mut bytes, 20, frame.viewport.extent[1]);
    }
    if has(FrameInput::CameraPosition) {
        write_f32(&mut bytes, 32, frame.camera_position.x);
        write_f32(&mut bytes, 36, frame.camera_position.y);
        write_f32(&mut bytes, 40, frame.camera_position.z);
    }
    if has(FrameInput::ViewProjection) {
        write_mat4(&mut bytes, 48, view_proj);
    }
    if let Some(model) = model {
        if has(FrameInput::Model) {
            write_mat4(&mut bytes, 112, model);
        }
        if has(FrameInput::ModelViewProjection) {
            write_mat4(&mut bytes, 176, view_proj * model);
        }
    }
    write_texture_table(&mut bytes, frame, textures);
    write_f32(&mut bytes, FRAME_INSTANCE_OFFSET, instance.age);
    write_f32(&mut bytes, FRAME_INSTANCE_OFFSET + 4, instance.stop_age);
    write_u32(&mut bytes, FRAME_INSTANCE_OFFSET + 8, instance.slot);
    write_world_info(&mut bytes, frame);
    bytes
}

/// Describes the block-data window. All zeros (`world_anchor.w == 0`) until
/// the first uploads have been applied, so shaders never read cells that
/// haven't been initialised.
fn write_world_info(bytes: &mut [u8], frame: &FrameData) {
    if !frame.world.enabled {
        return;
    }
    for (axis, cells) in WORLD_WINDOW_CELLS.iter().enumerate() {
        write_u32(bytes, FRAME_WORLD_OFFSET + axis * 4, *cells);
    }
    write_u32(
        bytes,
        FRAME_WORLD_OFFSET + 12,
        crate::chunk::CHUNK_SIZE as u32,
    );
    for axis in 0..3 {
        write_u32(
            bytes,
            FRAME_WORLD_OFFSET + 16 + axis * 4,
            frame.world.anchor[axis] as u32,
        );
    }
    write_u32(bytes, FRAME_WORLD_OFFSET + 28, 1);
}

/// Fills the bindless texture table that follows the classic frame block.
///
/// Indices are resolved every frame from the live `TextureManager`, so a
/// texture hot-reload (which registers new bindless ids) is picked up with no
/// effect-side invalidation. A path that isn't loaded resolves to the
/// placeholder texture; before any texture exists the slot stays 0.
fn write_texture_table(bytes: &mut [u8], frame: &FrameData, textures: &[TextureDecl]) {
    write_u32(
        bytes,
        FRAME_TABLE_OFFSET,
        crate::texture::sampler_index(frame.sampler_id),
    );
    write_u32(
        bytes,
        FRAME_TABLE_OFFSET + 4,
        crate::texture::sampler_index(frame.linear_sampler_id),
    );
    write_u32(
        bytes,
        FRAME_TABLE_OFFSET + 8,
        textures.len().min(MAX_EFFECT_TEXTURES) as u32,
    );
    let Some(manager) = frame.textures.as_ref() else {
        return;
    };
    for (slot, decl) in textures.iter().take(MAX_EFFECT_TEXTURES).enumerate() {
        if let Some(id) = manager.get_asset(&decl.path) {
            write_u32(
                bytes,
                FRAME_TABLE_OFFSET + 16 + slot * 4,
                crate::texture::sampled_image_index(id),
            );
        }
    }
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn write_f32(bytes: &mut [u8], offset: usize, value: f32) {
    write_u32(bytes, offset, value.to_bits());
}
fn write_mat4(bytes: &mut [u8], offset: usize, mat: glam::Mat4) {
    for (index, value) in mat.to_cols_array().into_iter().enumerate() {
        write_f32(bytes, offset + index * 4, value);
    }
}
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn storage_alignment(vk: &VulkanContext) -> u64 {
    vk.device
        .physical_device()
        .properties()
        .min_storage_buffer_offset_alignment
        .max(DeviceAlignment::new(16).unwrap())
        .into()
}
