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
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
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

use super::{effects::*, gpu::FrameData};

pub(super) const EFFECT_ARENA_BYTES: u64 = 16 * 1024 * 1024;
const TRANSIENT_BYTES: u64 = 4 * 1024 * 1024;
const FRAME_BYTES: usize = 240;
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
}

pub(super) struct PreparedEffect {
    pub manifest: EffectManifest,
    pub buffers: Vec<BufferRange>,
    pub passes: Vec<PreparedPass>,
}

pub(super) struct EffectGpuState {
    pub arena_id: Id<Buffer>,
    arena: Arc<Buffer>,
    capacity: u64,
    storage_alignment: u64,
    persistent_end: u64,
    descriptor_allocator: Arc<StandardDescriptorSetAllocator>,
    subpass: Subpass,
    pub effects: HashMap<String, PreparedEffect>,
    latest_actions: HashMap<String, ModActionEvent>,
    initialized_effects: HashSet<String>,
    rejected_effects: HashSet<String>,
    graphics_frame: Option<u64>,
    graphics_cursor: u64,
}

impl EffectGpuState {
    pub fn new(vk: &VulkanContext, arena_id: Id<Buffer>, subpass: Subpass) -> Self {
        Self {
            arena_id,
            arena: vk.resources.buffer(arena_id).buffer().clone(),
            capacity: EFFECT_ARENA_BYTES,
            storage_alignment: storage_alignment(vk),
            persistent_end: 0,
            descriptor_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                &vk.device,
                &Default::default(),
            )),
            subpass,
            effects: HashMap::new(),
            latest_actions: HashMap::new(),
            initialized_effects: HashSet::new(),
            rejected_effects: HashSet::new(),
            graphics_frame: None,
            graphics_cursor: EFFECT_ARENA_BYTES - TRANSIENT_BYTES,
        }
    }

    pub fn reset(&mut self) {
        self.persistent_end = 0;
        self.effects.clear();
        self.latest_actions.clear();
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
        let descriptor_layout = create_effect_descriptor_layout(&vk.device, buffers.len())?;
        let mut passes = Vec::with_capacity(manifest.passes.len());
        for pass in &manifest.passes {
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
                        BlendMode::Additive => Some(AttachmentBlend::additive()),
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
                    }
                }
            };
            passes.push(prepared);
        }
        self.persistent_end = cursor;
        Ok(PreparedEffect {
            manifest: manifest.clone(),
            buffers,
            passes,
        })
    }

    pub fn record_compute(
        &mut self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        frame: &FrameData,
    ) -> TaskResult {
        let transient_start = align_up(self.persistent_end, self.storage_alignment);
        let mut transient = transient_start;
        let packages: Vec<_> = self.effects.values().collect();
        for effect in &packages {
            if !self.initialized_effects.contains(&effect.manifest.id) {
                for buffer in &effect.buffers {
                    let bytes = tcx.try_write_buffer::<[u8]>(
                        self.arena_id,
                        buffer.offset..buffer.offset + buffer.size,
                    )?;
                    bytes.fill(0);
                }
                self.initialized_effects.insert(effect.manifest.id.clone());
            } else {
                for buffer in &effect.buffers {
                    if !buffer.persistent {
                        let bytes = tcx.try_write_buffer::<[u8]>(
                            self.arena_id,
                            buffer.offset..buffer.offset + buffer.size,
                        )?;
                        bytes.fill(0);
                    }
                }
            }
        }

        for event in &frame.mod_actions {
            let Some(effect) = self.effects.get(&event.effect_id) else {
                continue;
            };
            let Some(pass) = effect
                .passes
                .iter()
                .find(|pass| pass.id == event.compute_pass && pass.kind == PassKind::Compute)
            else {
                continue;
            };
            self.latest_actions
                .insert(effect.manifest.id.clone(), event.clone());
            let model = event.model_player_id;
            let frame_bytes = build_frame_bytes(frame, &effect.manifest.frame_inputs, model);
            self.dispatch(
                cbf,
                tcx,
                effect,
                pass,
                &event.payload,
                &frame_bytes,
                &mut transient,
            )?;
        }

        for effect in &packages {
            for pass in effect
                .passes
                .iter()
                .filter(|pass| pass.kind == PassKind::Compute)
            {
                let declared_each_frame = effect
                    .manifest
                    .passes
                    .iter()
                    .any(|decl| decl.id == pass.id && decl.run_each_frame);
                if !declared_each_frame {
                    continue;
                }
                let frame_bytes = build_frame_bytes(frame, &effect.manifest.frame_inputs, None);
                self.dispatch(cbf, tcx, effect, pass, &[], &frame_bytes, &mut transient)?;
            }
        }
        Ok(())
    }

    fn dispatch(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        effect: &PreparedEffect,
        pass: &PreparedPass,
        payload: &[u8],
        frame_bytes: &[u8],
        transient: &mut u64,
    ) -> TaskResult {
        let payload_size = payload.len().max(MIN_DESCRIPTOR_RANGE as usize) as u64;
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
            effect,
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
            cbf.dispatch(pass.workgroups);
        }
        let compute_barrier = MemoryBarrier {
            src_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
            src_access: vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
            dst_stages: vulkano::sync::PipelineStages::COMPUTE_SHADER,
            dst_access: vulkano::sync::AccessFlags::SHADER_STORAGE_READ
                | vulkano::sync::AccessFlags::SHADER_STORAGE_WRITE,
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
        for effect in self.effects.values() {
            let latest = self.latest_actions.get(&effect.manifest.id);
            for pass in effect
                .passes
                .iter()
                .filter(|pass| pass.kind == PassKind::Graphics && pass.phase == Some(phase))
            {
                let (payload, model) = latest
                    .map(|action| (action.payload.as_slice(), action.model_player_id))
                    .unwrap_or((&[], None));
                let payload_size = payload.len().max(MIN_DESCRIPTOR_RANGE as usize) as u64;
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
                action_write[..payload.len()].copy_from_slice(payload);
                tcx.try_write_buffer::<[u8]>(self.arena_id, frame_offset..end)?
                    .copy_from_slice(&build_frame_bytes(
                        frame,
                        &effect.manifest.frame_inputs,
                        model,
                    ));
                let descriptor_set = match self.create_descriptor_set(
                    pass,
                    effect,
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
                    cbf.draw(pass.vertex_count, pass.instance_count, 0, 0);
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
        }
        Ok(())
    }

    fn create_descriptor_set(
        &self,
        pass: &PreparedPass,
        effect: &PreparedEffect,
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
        let mut writes = vec![
            WriteDescriptorSet::buffer(0, &action_info),
            WriteDescriptorSet::buffer(1, &frame_info),
        ];
        let infos: Vec<_> = effect
            .buffers
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
        .map_err(|e| format!("could not allocate effect buffer arena: {e}"))
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
    let bindings: Vec<_> = (0..buffer_count + 2)
        .map(|binding| DescriptorSetLayoutBinding {
            binding: binding as u32,
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

fn build_frame_bytes(frame: &FrameData, inputs: &[FrameInput], player_id: Option<u64>) -> Vec<u8> {
    let mut bytes = vec![0; FRAME_BYTES];
    let has = |input| inputs.contains(&input);
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
        write_mat4(&mut bytes, 48, frame.view_proj);
    }
    if let Some(model) = player_id.and_then(|id| frame.model_transforms.get(&id).copied()) {
        if has(FrameInput::Model) {
            write_mat4(&mut bytes, 112, model);
        }
        if has(FrameInput::ModelViewProjection) {
            write_mat4(&mut bytes, 176, frame.view_proj * model);
        }
    }
    bytes
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
