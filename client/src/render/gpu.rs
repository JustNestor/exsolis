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
use glam::{Mat4, Quat, Vec3};
use vulkano::{
    DeviceSize, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, BufferContents},
    descriptor_set::{
        DescriptorBufferInfo, DescriptorSet, WriteDescriptorSet,
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo},
    },
    format::{ClearValue, Format},
    image::{Image, ImageCreateInfo, ImageUsage},
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineShaderStageCreateInfo,
        layout::{PipelineLayoutCreateInfo, push_constant_ranges_from_stages},
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
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
use vulkano::shader::ShaderStages;
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

// GLTF shader modules (inline like mesh.rs)
pub mod gltf_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        root_path_env: "CARGO_MANIFEST_DIR",
        src: r"
        #version 450
        #include <vulkano.glsl>

        struct GltfMaterial {
            vec4 base_color_factor;
            vec4 emissive;
            uint base_color_texture;
            uint normal_texture;
            uint occlusion_texture;
            uint emissive_texture;
            float metallic;
            float roughness;
            uint alpha_mode;
            float alpha_cutoff;
            uint double_sided;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        };

        layout(push_constant) uniform PushConstants {
            mat4 view_proj;
            SamplerId sampler_id;
            SamplerId linear_sampler_id;
            vec4 camera_position;
            uint material_index;
            float time_seconds;
            vec2 _pad;
        } push_constants;

        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec2 uv;
        layout(location = 3) in vec4 color;
        layout(location = 4) in vec4 tangent;

        layout(set = 1, binding = 0, std430) readonly buffer InstanceData {
            mat4 model_matrices[];
        } instances;

        layout(set = 1, binding = 1, std430) readonly buffer MaterialData {
            GltfMaterial materials[];
        } materials;

        layout(location = 0) out vec3 v_world_pos;
        layout(location = 1) out vec3 v_world_normal;
        layout(location = 2) out vec2 v_uv;
        layout(location = 3) out vec4 v_color;
        layout(location = 4) out vec4 v_tangent;

        void main() {
            uint instance_id = gl_InstanceIndex;
            mat4 model = instances.model_matrices[instance_id];
            mat3 normal_matrix = transpose(inverse(mat3(model)));

            vec4 world_pos = model * vec4(position, 1.0);
            v_world_pos = world_pos.xyz;
            v_world_normal = normalize(normal_matrix * normal);
            v_uv = uv;
            v_color = color;
            v_tangent = tangent;

            gl_Position = push_constants.view_proj * world_pos;
        }
        ",
    }
}

pub mod gltf_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        root_path_env: "CARGO_MANIFEST_DIR",
        src: r"
        #version 450
        #include <vulkano.glsl>

        struct GltfMaterial {
            vec4 base_color_factor;
            vec4 emissive;
            uint base_color_texture;
            uint normal_texture;
            uint occlusion_texture;
            uint emissive_texture;
            float metallic;
            float roughness;
            uint alpha_mode;
            float alpha_cutoff;
            uint double_sided;
            uint _pad0;
            uint _pad1;
            uint _pad2;
        };

        layout(push_constant) uniform PushConstants {
            mat4 view_proj;
            SamplerId sampler_id;
            SamplerId linear_sampler_id;
            vec4 camera_position;
            uint material_index;
            float time_seconds;
            vec2 _pad;
        } push_constants;

        layout(set = 1, binding = 1, std430) readonly buffer MaterialData {
            GltfMaterial materials[];
        } materials;

        layout(location = 0) in vec3 v_world_pos;
        layout(location = 1) in vec3 v_world_normal;
        layout(location = 2) in vec2 v_uv;
        layout(location = 3) in vec4 v_color;
        layout(location = 4) in vec4 v_tangent;

        layout(location = 0) out vec4 f_color;

        const float PI = 3.14159265359;

        vec4 sample_texture(uint bindless_index, vec2 uv, float mip_level) {
            if (bindless_index == 0xFFFFFFFFu) {
                return vec4(1.0);
            }
            SampledImageId tex_id = SampledImageId(bindless_index, 256);
            return textureLod(vko_sampler2D(tex_id, push_constants.linear_sampler_id), uv, mip_level);
        }

        float distribution_ggx(vec3 N, vec3 H, float roughness) {
            float a = roughness * roughness;
            float a2 = a * a;
            float NdotH = max(dot(N, H), 0.0);
            float NdotH2 = NdotH * NdotH;
            float denom = (NdotH2 * (a2 - 1.0) + 1.0);
            denom = PI * denom * denom;
            return a2 / denom;
        }

        float geometry_schlick_ggx(float NdotV, float roughness) {
            float r = roughness + 1.0;
            float k = (r * r) / 8.0;
            float denom = NdotV * (1.0 - k) + k;
            return NdotV / denom;
        }

        float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
            float NdotV = max(dot(N, V), 0.0);
            float NdotL = max(dot(N, L), 0.0);
            float ggx2 = geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
            return ggx2;
        }

        vec3 fresnel_schlick(float cosTheta, vec3 F0) {
            return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }

        vec3 aces_tonemap(vec3 color) {
            vec3 a = color * (2.51 * color + 0.03);
            vec3 b = color * (2.43 * color + 0.59) + 0.14;
            return clamp(a / b, 0.0, 1.0);
        }

        vec3 srgb_to_linear(vec3 c) {
            return pow(c, vec3(2.2));
        }

        vec3 linear_to_srgb(vec3 c) {
            return pow(c, vec3(1.0 / 2.2));
        }

        void main() {
            // One draw per primitive, each with its own material in push.
            GltfMaterial mat = materials.materials[push_constants.material_index];

            vec3 N = normalize(v_world_normal);
            if (mat.normal_texture != 0xFFFFFFFFu) {
                vec3 tangent_normal = sample_texture(mat.normal_texture, v_uv, 0.0).xyz * 2.0 - 1.0;
                vec3 T = normalize(v_tangent.xyz);
                vec3 B = cross(N, T) * v_tangent.w;
                mat3 TBN = mat3(T, B, N);
                N = normalize(TBN * tangent_normal);
            }

            vec3 base_color = mat.base_color_factor.rgb * v_color.rgb;
            float alpha = mat.base_color_factor.a * v_color.a;

            if (mat.base_color_texture != 0xFFFFFFFFu) {
                vec4 tex_color = sample_texture(mat.base_color_texture, v_uv, 0.0);
                base_color *= tex_color.rgb;
                alpha *= tex_color.a;
            }

            base_color = srgb_to_linear(base_color);

            float metallic = mat.metallic;
            float roughness = clamp(mat.roughness, 0.04, 1.0);

            if (mat.alpha_mode == 1u) {
                if (alpha < mat.alpha_cutoff) {
                    discard;
                }
            } else if (alpha < 0.004) {
                // Fully-transparent texels contribute nothing under any
                // blending; discard so cutout textures (wing membranes,
                // tails) don't render their black RGB as opaque geometry.
                discard;
            }

            vec3 V = normalize(push_constants.camera_position.xyz - v_world_pos);
            vec3 F0 = mix(vec3(0.04), base_color, metallic);

            vec3 ambient = vec3(0.03) * mix(vec3(1.0), F0, metallic);
            vec3 Lo = ambient * base_color;

            vec3 emissive = mat.emissive.rgb;
            if (mat.emissive_texture != 0xFFFFFFFFu) {
                emissive *= sample_texture(mat.emissive_texture, v_uv, 0.0).rgb;
            }
            emissive = srgb_to_linear(emissive);
            Lo += emissive;

            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            vec3 light_color = vec3(1.0, 0.95, 0.9) * 3.0;

            vec3 L = light_dir;
            vec3 H = normalize(V + L);

            float NdotL = max(dot(N, L), 0.0);
            float NdotV = max(dot(N, V), 0.0);

            if (NdotL > 0.0) {
                float D = distribution_ggx(N, H, roughness);
                float G = geometry_smith(N, V, L, roughness);
                vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

                vec3 nominator = D * G * F;
                float denominator = 4.0 * NdotV * NdotL + 0.001;
                vec3 specular = nominator / denominator;

                vec3 kS = F;
                vec3 kD = vec3(1.0) - kS;
                kD *= 1.0 - metallic;

                Lo += (kD * base_color / PI + specular) * light_color * NdotL;
            }

            if (mat.occlusion_texture != 0xFFFFFFFFu) {
                float ao = sample_texture(mat.occlusion_texture, v_uv, 0.0).r;
                Lo *= ao;
            }

            Lo = aces_tonemap(Lo);
            Lo = linear_to_srgb(Lo);

            f_color = vec4(Lo, alpha);
        }
        ",
    }
}

/// GLTF vertex format for pipeline creation (canonical type in crate::gltf).
pub(crate) use crate::gltf::GltfGpuVertex;

// 32^3 chunks pack more geometry per chunk than the old 16x16x256 columns
// did on average once boundary culling removes most internal faces, and
// with no textures/AO there's less per-vertex bandwidth pressure, so this
// keeps the same generous headroom as the established project.
pub(super) const MAX_VERTICES: u64 = 40_000_000;
pub(super) const MAX_INDICES: u64 = 60_000_000;

pub(crate) struct RenderStateShared {
    pub pipeline: Option<Arc<GraphicsPipeline>>,
    /// UI quads (boxes + glyphs). Drawn after world/avatars in the same
    /// subpass with alpha blending and depth test/write disabled.
    pub ui_pipeline: Option<Arc<GraphicsPipeline>>,
    /// Chunk meshes only. Avatars share one [`avatar`] slice drawn with
    /// per-player model matrices.
    pub allocations: HashMap<Entity, GpuMeshAllocation>,
    pub avatar: Option<GpuMeshAllocation>,
    pub effects_gpu: Option<super::effects_gpu::EffectGpuState>,
    /// GLTF model pipeline
    pub gltf_pipeline: Option<Arc<GraphicsPipeline>>,
    /// GLTF pipeline variant with face culling disabled, for double-sided
    /// materials (thin membranes like wings would otherwise show gaps).
    pub gltf_pipeline_nocull: Option<Arc<GraphicsPipeline>>,
    /// Blend variants (alpha_mode == Blend): standard alpha blending, no
    /// depth write. Drawn after all opaque draws (see execute).
    pub gltf_pipeline_blend: Option<Arc<GraphicsPipeline>>,
    pub gltf_pipeline_blend_nocull: Option<Arc<GraphicsPipeline>>,
    /// GLTF material storage buffer
    pub gltf_material_buffer_id: Option<Id<Buffer>>,
    /// GLTF instance buffer (model matrices)
    pub gltf_instance_buffer_id: Option<Id<Buffer>>,
    /// Number of GLTF instances to draw
    pub gltf_instance_count: u32,
    /// Per-primitive GLTF draws for the current frame (built by sync).
    pub gltf_draws: Vec<GltfDrawCall>,
    /// Descriptor set for GLTF instance/material storage buffers (set 1;
    /// set 0 is the bindless globals). Buffers are created once, so one
    /// persistent set covers uploads and hot-reload rewrites.
    pub gltf_descriptors: Option<GltfDescriptors>,
}

/// GLTF storage-buffer bindings for set 1.
pub(crate) struct GltfDescriptors {
    pub set: Arc<DescriptorSet>,
    #[allow(dead_code)]
    pub allocator: Arc<StandardDescriptorSetAllocator>,
}

pub(crate) struct FrameData {
    pub frame_index: u64,
    pub time_seconds: f32,
    pub delta_seconds: f32,
    pub view_proj: Mat4,
    pub camera_position: Vec3,
    pub viewport: vulkano::pipeline::graphics::viewport::Viewport,
    /// Bindless sampler used to read every block texture. Same sampler for
    /// all chunks; per-texture state lives in the image, not here.
    pub sampler_id: SamplerId,
    /// Bindless sampler with linear magnification, exposed to effect shaders
    /// next to `sampler_id` (nearest) in the frame buffer.
    pub linear_sampler_id: SamplerId,
    /// Snapshot of loaded textures so effects can resolve their declared
    /// PNGs each frame (an `Arc` clone, and it survives texture hot-reload).
    pub textures: Option<crate::texture::TextureManager>,
    /// One model matrix per remote player. Mesh is local-space in [`avatar`].
    pub avatar_models: Vec<Mat4>,
    /// Player ID -> model transform, for actions that explicitly request it.
    pub model_transforms: HashMap<u64, Mat4>,
    /// Validated actions received since the preceding rendered frame.
    pub mod_actions: Vec<super::effects::ModActionEvent>,
    /// UI instance buffer + live count (synced from `UiDrawList` each frame).
    pub ui_buffer_id: Id<Buffer>,
    pub ui_count: u32,
    /// GLTF instance data
    pub gltf_instance_buffer_id: Id<Buffer>,
    pub gltf_instance_count: u32,
    pub gltf_material_buffer_id: Id<Buffer>,
    /// GLTF primitive draws for this frame
    pub gltf_draws: Vec<GltfDrawCall>,
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

// One GLTF primitive draw: indexes into the dedicated GLTF buffers.
#[derive(Clone, Copy)]
pub(crate) struct GltfDrawCall {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub material_index: u32,
    pub first_instance: u32,
    pub instance_count: u32,
    pub double_sided: bool,
    pub alpha_blend: bool,
}

// GLTF render task for drawing instanced GLTF models
struct GltfRenderTask {
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    virtual_swapchain_id: Id<Swapchain>,
    virtual_depth_id: Id<Image>,
    shared: Arc<RwLock<RenderStateShared>>,
}

impl Task for GltfRenderTask {
    type World = FrameData;

    fn clear_values(&self, _clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        // No clears - we draw on top of the existing framebuffer
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        world: &Self::World,
    ) -> TaskResult {
        if world.gltf_draws.is_empty() || world.gltf_instance_count == 0 {
            return Ok(());
        }

        let shared = self.shared.read().unwrap();
        let (Some(pipeline), Some(pipeline_nocull), Some(pipeline_blend), Some(pipeline_blend_nocull)) = (
            shared.gltf_pipeline.as_ref(),
            shared.gltf_pipeline_nocull.as_ref(),
            shared.gltf_pipeline_blend.as_ref(),
            shared.gltf_pipeline_blend_nocull.as_ref(),
        ) else {
            return Ok(());
        };
        {
            cbf.set_viewport(0, std::slice::from_ref(&world.viewport));

            cbf.bind_index_buffer(
                self.index_buffer_id,
                0,
                None,
                vulkano::buffer::IndexType::U32,
            );
            cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[]);

            // Storage buffers (instances, materials) live at set 1.
            // NOTE: without gltf_descriptors the draws below read garbage;
            if let Some(descriptors) = shared.gltf_descriptors.as_ref() {
                unsafe {
                    cbf.as_raw().bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout(),
                        1,
                        &[descriptors.set.as_raw()],
                        &[],
                    );
                }
            }

            // Four groups so each binds its pipeline once: opaque first
            // (culled, then double-sided), blend materials last.
            for (blend, double_sided) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let pipe = match (blend, double_sided) {
                    (false, false) => pipeline,
                    (false, true) => pipeline_nocull,
                    (true, false) => pipeline_blend,
                    (true, true) => pipeline_blend_nocull,
                };
                cbf.bind_pipeline(pipe);
                for draw in world.gltf_draws.iter().filter(|d| {
                    d.double_sided == double_sided && d.alpha_blend == blend
                }) {
                    cbf.push_constants(
                        pipe.layout(),
                        0,
                        &gltf_vs::PushConstants {
                            view_proj: world.view_proj.to_cols_array_2d(),
                            sampler_id: world.sampler_id,
                            linear_sampler_id: world.linear_sampler_id,
                            camera_position: [world.camera_position.x, world.camera_position.y, world.camera_position.z, 1.0],
                            material_index: draw.material_index,
                            time_seconds: world.time_seconds,
                            _pad: [0.0, 0.0],
                        },
                    );
                    unsafe {
                        cbf.draw_indexed(
                            draw.index_count,
                            draw.instance_count,
                            draw.first_index,
                            draw.vertex_offset,
                            draw.first_instance,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Resource)]
pub struct RenderState {
    pub(crate) vertex_buffer_id: Id<Buffer>,
    pub(crate) index_buffer_id: Id<Buffer>,
    pub(crate) gltf_vertex_buffer_id: Id<Buffer>,
    pub(crate) gltf_index_buffer_id: Id<Buffer>,
    pub(crate) depth_image_id: Id<Image>,
    pub(crate) virtual_swapchain_id: Id<Swapchain>,
    pub(crate) virtual_depth_id: Id<Image>,
    pub(crate) task_graph: Mutex<ExecutableTaskGraph<FrameData>>,
    pub(crate) shared: Arc<RwLock<RenderStateShared>>,
    pub(crate) allocator: MeshBufferAllocator,
    pub(crate) gltf_allocator: MeshBufferAllocator,
    pub(crate) ui_gpu: UiPassGpu,
    pub(crate) frame_index: u64,
}

/// Max GLTF vertices/indices in dedicated buffers (GltfGpuVertex = 64B).
pub(super) const MAX_GLTF_VERTICES: u64 = 2_000_000;
pub(super) const MAX_GLTF_INDICES: u64 = 4_000_000;

pub(crate) struct PendingWrite<T> {
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

/// One wait + one batched host write for GLTF vertex/index/material/instance data.
pub(crate) fn upload_gltf_writes(
    vk: &VulkanContext,
    vertex_buffer_id: Id<Buffer>,
    index_buffer_id: Id<Buffer>,
    material_buffer_id: Id<Buffer>,
    instance_buffer_id: Id<Buffer>,
    vertex_writes: &[PendingWrite<GltfGpuVertex>],
    index_writes: &[PendingWrite<u32>],
    material_writes: &[PendingWrite<crate::gltf::GltfGpuMaterial>],
    instance_writes: &[PendingWrite<Mat4>],
) {
    if vertex_writes.is_empty()
        && index_writes.is_empty()
        && material_writes.is_empty()
        && instance_writes.is_empty()
    {
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
                        write.offset_bytes + (write.data.len() * size_of::<GltfGpuVertex>()) as u64;
                    tcx.try_write_buffer::<[GltfGpuVertex]>(
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
                for write in material_writes {
                    let end = write.offset_bytes
                        + (write.data.len() * size_of::<crate::gltf::GltfGpuMaterial>()) as u64;
                    tcx.try_write_buffer::<[crate::gltf::GltfGpuMaterial]>(
                        material_buffer_id,
                        write.offset_bytes..end,
                    )?
                    .copy_from_slice(&write.data);
                }
                for write in instance_writes {
                    let end =
                        write.offset_bytes + (write.data.len() * size_of::<Mat4>()) as u64;
                    tcx.try_write_buffer::<[Mat4]>(
                        instance_buffer_id,
                        write.offset_bytes..end,
                    )?
                    .copy_from_slice(&write.data);
                }
                Ok(())
            },
            [
                (vertex_buffer_id, HostAccessType::Write),
                (index_buffer_id, HostAccessType::Write),
                (material_buffer_id, HostAccessType::Write),
                (instance_buffer_id, HostAccessType::Write),
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

    // Create GLTF material storage buffer (max 256 materials)
    const MAX_GLTF_MATERIALS: u32 = 256;
    let gltf_material_buffer_size = size_of::<crate::gltf::GltfGpuMaterial>() as u64 * MAX_GLTF_MATERIALS as u64;
    let gltf_material_layout = DeviceLayout::new(
        NonZero::new(gltf_material_buffer_size).unwrap(),
        DeviceAlignment::new(align_of::<crate::gltf::GltfGpuMaterial>() as u64).unwrap(),
    ).expect("gltf material buffer layout exceeds device limits");

    let gltf_material_buffer_id = vk
        .resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            gltf_material_layout,
        )
        .unwrap();

    // Create GLTF instance buffer (max 1024 instances)
    const MAX_GLTF_INSTANCES: u32 = 1024;
    let gltf_instance_buffer_size = size_of::<Mat4>() as u64 * MAX_GLTF_INSTANCES as u64;
    let gltf_instance_layout = DeviceLayout::new(
        NonZero::new(gltf_instance_buffer_size).unwrap(),
        DeviceAlignment::new(align_of::<Mat4>() as u64).unwrap(),
    ).expect("gltf instance buffer layout exceeds device limits");

    let gltf_instance_buffer_id = vk
        .resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            gltf_instance_layout,
        )
        .unwrap();

    // Dedicated GLTF vertex/index buffers (GltfGpuVertex stride differs from ChunkVertex)
    let gltf_vertex_buffer_size: DeviceSize = MAX_GLTF_VERTICES * size_of::<GltfGpuVertex>() as u64;
    let gltf_index_buffer_size: DeviceSize = MAX_GLTF_INDICES * size_of::<u32>() as u64;

    let gltf_vertex_layout = DeviceLayout::new(
        NonZero::new(gltf_vertex_buffer_size).expect("gltf vertex buffer size must be non-zero"),
        DeviceAlignment::new(align_of::<GltfGpuVertex>() as u64).unwrap(),
    )
    .expect("gltf vertex layout exceeds device limits");

    let gltf_index_layout = DeviceLayout::new(
        NonZero::new(gltf_index_buffer_size).expect("gltf index buffer size must be non-zero"),
        DeviceAlignment::new(align_of::<u32>() as u64).unwrap(),
    )
    .expect("gltf index layout exceeds device limits");

    let gltf_vertex_buffer_id = vk
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
            gltf_vertex_layout,
        )
        .unwrap();

    let gltf_index_buffer_id = vk
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
            gltf_index_layout,
        )
        .unwrap();

    // Host writes for GLTF buffers + material/instance storage buffers
    // (declared as host accesses once the task graph exists below).

    let shared = Arc::new(RwLock::new(RenderStateShared {
        pipeline: None,
        ui_pipeline: None,
        allocations: HashMap::new(),
        avatar: None,
        effects_gpu: None,
        gltf_pipeline: None,
        gltf_pipeline_nocull: None,
        gltf_pipeline_blend: None,
        gltf_pipeline_blend_nocull: None,
        gltf_material_buffer_id: Some(gltf_material_buffer_id),
        gltf_instance_buffer_id: Some(gltf_instance_buffer_id),
        gltf_instance_count: 0,
        gltf_draws: Vec::new(),
        gltf_descriptors: None,
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
    task_graph.add_host_buffer_access(gltf_vertex_buffer_id, HostAccessType::Write);
    task_graph.add_host_buffer_access(gltf_index_buffer_id, HostAccessType::Write);
    task_graph.add_host_buffer_access(gltf_material_buffer_id, HostAccessType::Write);
    task_graph.add_host_buffer_access(gltf_instance_buffer_id, HostAccessType::Write);
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

    // GLTF render task - draws after chunks
    let gltf_task = GltfRenderTask {
        vertex_buffer_id: gltf_vertex_buffer_id,
        index_buffer_id: gltf_index_buffer_id,
        virtual_swapchain_id,
        virtual_depth_id,
        shared: Arc::clone(&shared),
    };
    let mut gltf_node = task_graph.create_task_node("GltfModels", vulkano_taskgraph::QueueFamilyType::Graphics, gltf_task);
    let gltf_node_id = gltf_node
        .framebuffer(virtual_framebuffer_id)
        .color_attachment(
            virtual_swapchain_id.current_image_id(),
            AccessTypes::COLOR_ATTACHMENT_WRITE,
            Optimal,
            &AttachmentInfo {
                clear: false, // Don't clear - draw on top of chunks
                ..Default::default()
            },
        )
        .depth_stencil_attachment(
            virtual_depth_id,
            AccessTypes::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Optimal,
            &AttachmentInfo {
                clear: false,
                ..Default::default()
            },
        )
        .buffer_access(gltf_vertex_buffer_id, AccessTypes::VERTEX_ATTRIBUTE_READ)
        .buffer_access(gltf_index_buffer_id, AccessTypes::INDEX_READ)
        .buffer_access(gltf_material_buffer_id, effect_storage_access())
        .buffer_access(gltf_instance_buffer_id, effect_storage_access())
        .build();

    // GLTF draws after chunks
    task_graph.add_edge(chunk_node_id, gltf_node_id).unwrap();

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

    // GLTF pipeline (create before moving subpass into effects_gpu)
    let gltf_vs = unsafe { gltf_vs::load(&vk.device).unwrap().entry_point("main").unwrap() };
    let gltf_fs = unsafe { gltf_fs::load(&vk.device).unwrap().entry_point("main").unwrap() };

    let gltf_vertex_input_state = GltfGpuVertex::per_vertex().definition(&gltf_vs).unwrap();
    let gltf_stages = [
        PipelineShaderStageCreateInfo::new(&gltf_vs),
        PipelineShaderStageCreateInfo::new(&gltf_fs),
    ];

    // Explicit layout: set 0 = bindless globals, set 1 = our storage
    // buffers. `pipeline_layout_from_stages` only carries the bindless set,
    // which is what dropped our bindings (same shape as effect pipelines).
    let gltf_local_layout = DescriptorSetLayout::new(
        &vk.device,
        &DescriptorSetLayoutCreateInfo {
            bindings: &[
                DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 1,
                    stages: ShaderStages::VERTEX,
                    ..DescriptorSetLayoutBinding::new(
                        vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
                    )
                },
                DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_count: 1,
                    stages: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ..DescriptorSetLayoutBinding::new(
                        vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
                    )
                },
            ],
            ..Default::default()
        },
    )
    .unwrap();
    let bcx = vk.resources.bindless_context().unwrap();
    let gltf_layout = vulkano::pipeline::layout::PipelineLayout::new(
        &vk.device,
        &PipelineLayoutCreateInfo {
            set_layouts: &[bcx.global_set_layout(), &gltf_local_layout],
            push_constant_ranges: &push_constant_ranges_from_stages(&gltf_stages),
            ..Default::default()
        },
    )
    .unwrap();

    let gltf_raster_cull = RasterizationState {
        cull_mode: CullMode::Back,
        // Same as the chunk pipeline: FlyCamera::view_projection
        // negates Y (a reflection), so CCW-from-outside faces in
        // world/model space — which is what both mesh.rs and glTF
        // author — rasterize as clockwise in NDC.
        front_face: FrontFace::Clockwise,
        ..Default::default()
    };
    let gltf_raster_nocull = RasterizationState {
        cull_mode: CullMode::None,
        front_face: FrontFace::Clockwise,
        ..Default::default()
    };
    let gltf_blend_off = ColorBlendState {
        attachments: &[ColorBlendAttachmentState::default()],
        ..Default::default()
    };
    let gltf_blend_on = ColorBlendState {
        attachments: &[ColorBlendAttachmentState {
            blend: Some(AttachmentBlend::alpha()),
            ..Default::default()
        }],
        ..Default::default()
    };
    let gltf_depth_write = DepthStencilState {
        depth: Some(DepthState {
            write_enable: true,
            compare_op: CompareOp::Less,
        }),
        ..Default::default()
    };
    // Transparent draws must not poison the depth buffer for what follows.
    let gltf_depth_readonly = DepthStencilState {
        depth: Some(DepthState {
            write_enable: false,
            compare_op: CompareOp::Less,
        }),
        ..Default::default()
    };

    // Four variants: {cull, nocull} x {opaque, blend}. Same shaders/layout,
    // so one descriptor set serves all four.
    let build_gltf_pipeline = |raster: &RasterizationState,
                               blend: &ColorBlendState,
                               depth: &DepthStencilState| {
        GraphicsPipeline::new(
            &vk.device,
            None,
            &GraphicsPipelineCreateInfo {
                stages: &gltf_stages,
                vertex_input_state: Some(&gltf_vertex_input_state),
                input_assembly_state: Some(&InputAssemblyState::default()),
                viewport_state: Some(&ViewportState::default()),
                rasterization_state: Some(raster),
                multisample_state: Some(&MultisampleState::default()),
                color_blend_state: Some(blend),
                depth_stencil_state: Some(depth),
                dynamic_state: &[DynamicState::Viewport],
                subpass: Some((&subpass).into()),
                ..GraphicsPipelineCreateInfo::new(&gltf_layout)
            },
        )
        .unwrap()
    };

    let gltf_pipeline = build_gltf_pipeline(&gltf_raster_cull, &gltf_blend_off, &gltf_depth_write);
    let gltf_pipeline_nocull =
        build_gltf_pipeline(&gltf_raster_nocull, &gltf_blend_off, &gltf_depth_write);
    let gltf_pipeline_blend =
        build_gltf_pipeline(&gltf_raster_cull, &gltf_blend_on, &gltf_depth_readonly);
    let gltf_pipeline_blend_nocull =
        build_gltf_pipeline(&gltf_raster_nocull, &gltf_blend_on, &gltf_depth_readonly);

    shared.write().unwrap().gltf_pipeline = Some(gltf_pipeline);
    shared.write().unwrap().gltf_pipeline_nocull = Some(gltf_pipeline_nocull);
    shared.write().unwrap().gltf_pipeline_blend = Some(gltf_pipeline_blend);
    shared.write().unwrap().gltf_pipeline_blend_nocull = Some(gltf_pipeline_blend_nocull);

    // Descriptor set for GLTF storage buffers at set 1 (set 0 = bindless).
    // Buffers are whole-range bound once; uploads rewrite contents in place.
    {
        let allocator = Arc::new(StandardDescriptorSetAllocator::new(
            &vk.device,
            &Default::default(),
        ));
        let instance_buf = vk.resources.buffer(gltf_instance_buffer_id).buffer().clone();
        let material_buf = vk.resources.buffer(gltf_material_buffer_id).buffer().clone();
        let instance_info = DescriptorBufferInfo {
            buffer: Some(&instance_buf),
            offset: 0,
            range: None,
        };
        let material_info = DescriptorBufferInfo {
            buffer: Some(&material_buf),
            offset: 0,
            range: None,
        };
        match DescriptorSet::new(
            &allocator,
            &gltf_local_layout,
            &[
                WriteDescriptorSet::buffer(0, &instance_info),
                WriteDescriptorSet::buffer(1, &material_info),
            ],
            &[],
        ) {
            Ok(set) => {
                shared.write().unwrap().gltf_descriptors =
                    Some(GltfDescriptors { set, allocator });
            }
            Err(e) => {
                bevy_log::warn!("gltf descriptor set failed: {e:?}");
            }
        }
    }

    shared.write().unwrap().effects_gpu = Some(super::effects_gpu::EffectGpuState::new(
        &vk,
        effect_arena_id,
        subpass,
    ));

    commands.insert_resource(RenderState {
        vertex_buffer_id,
        index_buffer_id,
        gltf_vertex_buffer_id,
        gltf_index_buffer_id,
        depth_image_id,
        virtual_swapchain_id,
        virtual_depth_id,
        task_graph: Mutex::new(task_graph),
        shared,
        allocator: MeshBufferAllocator::new(),
        gltf_allocator: MeshBufferAllocator::new(),
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
    textures: Option<Res<crate::texture::TextureManager>>,
    local_player_id: Option<Res<LocalPlayerId>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Transform, &FlyCamera), With<LocalPlayer>>,
    remotes: Query<(&RemotePlayer, &Transform)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    // Swapchain/viewport are in physical pixels; logical size is wrong on HiDPI.
    let width = window.resolution.physical_width();
    let height = window.resolution.physical_height();
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
    let local_player_id = local_player_id.as_deref().and_then(|id| id.0);
    if let (Some(player_id), Some((transform, camera))) = (local_player_id, camera.iter().next()) {
        // The camera pitch belongs to the view, not the character model.
        // The avatar mesh faces local +Z while the camera looks along -Z, so
        // turn its yaw by PI to make the avatar face the direction of travel.
        let body_rotation = Quat::from_rotation_y(camera.yaw + std::f32::consts::PI);
        let model = Mat4::from_scale_rotation_translation(
            transform.scale,
            body_rotation,
            transform.translation,
        );
        model_transforms.insert(player_id, model);
    }
    // This is a first-person camera: drawing its own head from a matrix that
    // encloses the camera exposes the inside of the cube and blocks the view.
    // Keep the local matrix in `model_transforms` for shader effects, but only
    // draw remote avatars here.
    let avatar_models = model_transforms
        .iter()
        .filter(|(player_id, _)| Some(**player_id) != local_player_id)
        .map(|(_, model)| *model)
        .collect();
    let mod_actions = mod_actions.read().cloned().collect();

    // Upload UI instances only when `UiDrawList.generation` changed.
    if let Some(list) = ui_list.as_deref() {
        ui_pass::sync(&vk, &mut render_state.ui_gpu, list);
    }
    let ui_count = render_state.ui_gpu.count;
    let ui_buffer_id = render_state.ui_gpu.buffer_id;

    // Get GLTF buffers, staged draws, and instance count from shared state
    // (staged by sync_gltf_instances earlier in the chain).
    let (
        gltf_instance_buffer_id,
        gltf_instance_count,
        gltf_material_buffer_id,
        gltf_draws,
    ) = {
        let shared = render_state.shared.read().unwrap();
        (
            shared.gltf_instance_buffer_id.unwrap(),
            shared.gltf_instance_count,
            shared.gltf_material_buffer_id.unwrap(),
            shared.gltf_draws.clone(),
        )
    };

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
        linear_sampler_id: vk.linear_sampler_id,
        textures: textures.map(|textures| (*textures).clone()),
        avatar_models,
        model_transforms,
        mod_actions,
        ui_buffer_id,
        ui_count,
        gltf_instance_buffer_id,
        gltf_instance_count,
        gltf_material_buffer_id,
        gltf_draws,
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
