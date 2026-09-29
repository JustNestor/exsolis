//! UI pass: draws `bevy_ui`'s (real, `bevy_text`-computed) layout with our
//! own Vulkan pipeline, since this project has no `bevy_render` renderer
//! for `bevy_ui` to draw itself through.
//!
//! Data flow:
//!   `ui::render_bridge::extract_ui` --> [`UiDrawList`] (CPU, back-to-front)
//!   --> [`sync`] uploads it to one host-visible instance buffer, only
//!       when the list actually changed content (cheap no-op otherwise)
//!   --> [`record_draw`], called from `ChunkRenderTask::execute` after the
//!       world/avatars, same subpass, alpha blended, no depth test/write.
//!
//! One instanced draw covers every box and every glyph: each instance is a
//! screen-space quad (4 vertices, triangle strip, no index buffer).
//!   * KIND_BOX   rounded rect with per-side border, SDF'd in the fragment shader
//!   * KIND_GLYPH samples one glyph rect out of a `bevy_text` font atlas
//!     (uploaded bindless by `ui::render_bridge::sync_glyph_atlases`) —
//!     this pass does no text shaping or rasterization of its own.
//!   * KIND_IMAGE samples a whole `<img>` texture (uploaded bindless by
//!     `ui::render_bridge::sync_ui_images`), tinted, with rounded corners.
//!
//! Instance attribute NAMES in the shader must equal the Rust field names
//! (vulkano maps them by name, same as `ChunkVertex` in mesh.rs).

use std::{
    mem::{align_of, size_of},
    num::NonZero,
    sync::Arc,
};

use bevy_ecs::prelude::*;
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    format::NumericFormat,
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::{
        DynamicState, GraphicsPipeline, Pipeline, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
        },
    },
    render_pass::Subpass,
};
use vulkano_taskgraph::{
    Id, command_buffer::RecordingCommandBuffer, descriptor_set::SamplerId, resource::HostAccessType,
};

use crate::vulkan_prelude::VulkanContext;

/// Hard cap: 16k quads (~2.4 MB) covers any sane screen — a full page of
/// dense text is on the order of a few thousand glyphs.
pub const MAX_UI_INSTANCES: usize = 16_384;

pub const KIND_BOX: u32 = 0;
pub const KIND_GLYPH: u32 = 1;
pub const KIND_IMAGE: u32 = 2;

/// One screen-space quad. All coordinates are PHYSICAL pixels, origin top-left
/// — same space `ComputedNode`/`UiGlobalTransform` report in.
#[derive(Clone, Copy, Default, PartialEq, BufferContents, Vertex)]
#[repr(C)]
pub struct UiInstance {
    /// x, y, w, h.
    #[format(R32G32B32A32_SFLOAT)]
    pub rect: [f32; 4],
    /// Atlas rect as u0, v0, u1, v1 in PIXELS (unused for boxes). Divided
    /// by the atlas size in the vertex shader, since `GlyphAtlasInfo::rect`
    /// from `bevy_text` is already in atlas pixels, not normalized.
    #[format(R32G32B32A32_SFLOAT)]
    pub uvs: [f32; 4],
    /// Box fill, or glyph color. Pre-encoded for the swapchain format by
    /// the extractor (see `render_bridge::to_instance_color`).
    #[format(R32G32B32A32_SFLOAT)]
    pub fill: [f32; 4],
    #[format(R32G32B32A32_SFLOAT)]
    pub border_col: [f32; 4],
    /// Border thickness: left, top, right, bottom.
    #[format(R32G32B32A32_SFLOAT)]
    pub border_w: [f32; 4],
    /// Clip rect min.x, min.y, max.x, max.y (screen pixels).
    #[format(R32G32B32A32_SFLOAT)]
    pub clip_rect: [f32; 4],
    /// Atlas width, height in texels (for UV normalization).
    #[format(R32G32_SFLOAT)]
    pub atlas_size: [f32; 2],
    #[format(R32_SFLOAT)]
    pub radius: f32,
    #[format(R32_UINT)]
    pub kind: u32,
    /// Bindless sampled-image index of the glyph atlas (glyphs only).
    #[format(R32_UINT)]
    pub tex_id: u32,
}

impl UiInstance {
    pub const NO_CLIP: [f32; 4] = [-1.0e9, -1.0e9, 1.0e9, 1.0e9];

    pub fn boxed(
        rect: [f32; 4],
        fill: [f32; 4],
        border_col: [f32; 4],
        border_w: [f32; 4],
        radius: f32,
        clip_rect: [f32; 4],
    ) -> Self {
        Self {
            rect,
            fill,
            border_col,
            border_w,
            radius,
            clip_rect,
            kind: KIND_BOX,
            ..Default::default()
        }
    }

    /// `uv_rect_px` is the glyph's rect in the atlas, in TEXELS (as
    /// `GlyphAtlasInfo::rect` gives it) — this normalizes it against
    /// `atlas_size` on the GPU rather than the CPU so the extractor
    /// doesn't need to divide per glyph.
    pub fn glyph(
        rect: [f32; 4],
        uv_rect_px: [f32; 4],
        color: [f32; 4],
        atlas_size: [f32; 2],
        tex_id: u32,
        clip_rect: [f32; 4],
    ) -> Self {
        Self {
            rect,
            uvs: uv_rect_px,
            fill: color,
            atlas_size,
            clip_rect,
            kind: KIND_GLYPH,
            tex_id,
            ..Default::default()
        }
    }

    /// A whole texture stretched over `rect`. `tint` multiplies the texel
    /// color (white = untouched); `radius` rounds the corners.
    pub fn image(
        rect: [f32; 4],
        tex_id: u32,
        tint: [f32; 4],
        radius: f32,
        clip_rect: [f32; 4],
    ) -> Self {
        Self {
            rect,
            uvs: [0.0, 0.0, 1.0, 1.0],
            fill: tint,
            atlas_size: [1.0, 1.0],
            radius,
            clip_rect,
            kind: KIND_IMAGE,
            tex_id,
            ..Default::default()
        }
    }
}

/// Back-to-front draw list, rebuilt every frame by `ui::render_bridge::extract_ui`.
/// `generation` only bumps when content actually changed, so the GPU
/// upload (which waits for the frame in flight) is skipped for static UI.
#[derive(Resource, Default)]
pub struct UiDrawList {
    pub instances: Vec<UiInstance>,
    pub generation: u64,
}

/// Render-side state for the UI pass.
pub(super) struct UiPassGpu {
    pub buffer_id: Id<Buffer>,
    pub count: u32,
    generation: u64,
}

impl UiPassGpu {
    pub fn new(vk: &VulkanContext) -> Self {
        let size: DeviceSize = (MAX_UI_INSTANCES * size_of::<UiInstance>()) as u64;
        let layout = DeviceLayout::new(
            NonZero::new(size).unwrap(),
            DeviceAlignment::new(align_of::<UiInstance>() as u64).unwrap(),
        )
        .expect("ui instance buffer layout exceeds device limits");

        let buffer_id = vk
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
                layout,
            )
            .unwrap();

        Self {
            buffer_id,
            count: 0,
            // Different from `UiDrawList::default().generation` (0) so an
            // empty first list is still "seen" and `count` gets set.
            generation: u64::MAX,
        }
    }
}

/// True when the swapchain does the linear->sRGB encode on write, so the
/// extractor must hand this pass LINEAR colors instead of pre-encoding them.
pub fn swapchain_is_srgb(vk: &VulkanContext) -> bool {
    vk.swapchain_format.numeric_format_color() == Some(NumericFormat::SRGB)
}

/// Upload the draw list if it changed. Waits for the frame in flight first
/// — same ordering `upload_writes` in gpu.rs uses for chunk meshes.
pub(super) fn sync(vk: &VulkanContext, gpu: &mut UiPassGpu, list: &UiDrawList) {
    if gpu.generation == list.generation {
        return;
    }
    gpu.generation = list.generation;

    let n = list.instances.len().min(MAX_UI_INSTANCES);
    gpu.count = n as u32;
    if n == 0 {
        return;
    }

    let data = &list.instances[..n];
    let buffer_id = gpu.buffer_id;
    let bytes = (n * size_of::<UiInstance>()) as u64;

    vk.resources.flight(vk.flight_id).wait(None).unwrap();
    unsafe {
        vulkano_taskgraph::execute(
            &vk.queue,
            &vk.resources,
            vk.flight_id,
            |_cbf, tcx| {
                tcx.try_write_buffer::<[UiInstance]>(buffer_id, 0..bytes)?
                    .copy_from_slice(data);
                Ok(())
            },
            [(buffer_id, HostAccessType::Write)],
            [],
            [],
        )
        .unwrap();
    }
}

pub(super) fn create_pipeline(vk: &VulkanContext, subpass: &Subpass) -> Arc<GraphicsPipeline> {
    let vs = unsafe { vs::load(&vk.device).unwrap().entry_point("main").unwrap() };
    let fs = unsafe { fs::load(&vk.device).unwrap().entry_point("main").unwrap() };

    let vertex_input_state = UiInstance::per_instance().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(&vs),
        PipelineShaderStageCreateInfo::new(&fs),
    ];
    let bcx = vk.resources.bindless_context().unwrap();
    let layout = bcx.pipeline_layout_from_stages(&stages).unwrap();

    GraphicsPipeline::new(
        &vk.device,
        None,
        &GraphicsPipelineCreateInfo {
            stages: &stages,
            vertex_input_state: Some(&vertex_input_state),
            input_assembly_state: Some(&InputAssemblyState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            }),
            viewport_state: Some(&ViewportState::default()),
            rasterization_state: Some(&RasterizationState {
                cull_mode: CullMode::None,
                ..Default::default()
            }),
            multisample_state: Some(&MultisampleState::default()),
            color_blend_state: Some(&ColorBlendState {
                attachments: &[ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend::alpha()),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            // The subpass carries a depth attachment (the world pass); UI
            // neither tests nor writes it, so it always draws on top.
            depth_stencil_state: Some(&DepthStencilState::default()),
            dynamic_state: &[DynamicState::Viewport],
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::new(&layout)
        },
    )
    .unwrap()
}

/// Called from inside `ChunkRenderTask::execute`, after world+avatar draws,
/// viewport already set.
pub(super) fn record_draw(
    cbf: &mut RecordingCommandBuffer<'_>,
    pipeline: &Arc<GraphicsPipeline>,
    buffer_id: Id<Buffer>,
    count: u32,
    viewport_px: [f32; 2],
    sampler_id: SamplerId,
) {
    if count == 0 {
        return;
    }
    unsafe {
        cbf.bind_pipeline(pipeline);
        cbf.push_constants(
            pipeline.layout(),
            0,
            &vs::PushConstants {
                viewport: viewport_px,
                sampler_id,
            },
        );
        cbf.bind_vertex_buffers(0, &[buffer_id], &[0], &[], &[]);
    }
    unsafe {
        cbf.draw(4, count, 0, 0);
    }
}

// Push-constant block declared identically in both stages, same reasoning
// as mesh.rs. `SampledImageId(v_tex, 256)` mirrors mesh.rs's fragment
// shader so glyph atlases resolve exactly like block textures do.
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        root_path_env: "CARGO_MANIFEST_DIR",
        src: r"
        #version 450
        #include <vulkano.glsl>

        layout(push_constant) uniform PushConstants {
            vec2 viewport;
            SamplerId sampler_id;
        } pc;

        layout(location = 0) in vec4 rect;
        layout(location = 1) in vec4 uvs;
        layout(location = 2) in vec4 fill;
        layout(location = 3) in vec4 border_col;
        layout(location = 4) in vec4 border_w;
        layout(location = 5) in vec4 clip_rect;
        layout(location = 6) in vec2 atlas_size;
        layout(location = 7) in float radius;
        layout(location = 8) in uint kind;
        layout(location = 9) in uint tex_id;

        layout(location = 0) out vec2 v_local;
        layout(location = 1) out vec2 v_uv;
        layout(location = 2) out flat vec4 v_fill;
        layout(location = 3) out flat vec4 v_border_col;
        layout(location = 4) out flat vec4 v_border_w;
        layout(location = 5) out flat vec4 v_clip;
        layout(location = 6) out flat vec3 v_half_radius;
        layout(location = 7) out flat uint v_kind;
        layout(location = 8) out flat uint v_tex;

        void main() {
            // Triangle strip corners: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
            vec2 c = vec2(float(gl_VertexIndex & 1), float(gl_VertexIndex >> 1));
            vec2 px = rect.xy + c * rect.zw;
            // Vulkan NDC: y = -1 is the TOP of the framebuffer, matching y-down pixels.
            gl_Position = vec4(px / pc.viewport * 2.0 - 1.0, 0.0, 1.0);

            v_local = (c - 0.5) * rect.zw;
            // `uvs` arrives in atlas texels; normalize here so the CPU
            // extractor never has to know the atlas's current size.
            vec2 inv_atlas = 1.0 / max(atlas_size, vec2(1.0));
            v_uv = mix(uvs.xy, uvs.zw, c) * inv_atlas;
            v_fill = fill;
            v_border_col = border_col;
            v_border_w = border_w;
            v_clip = clip_rect;
            v_half_radius = vec3(rect.zw * 0.5, radius);
            v_kind = kind;
            v_tex = tex_id;
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
            vec2 viewport;
            SamplerId sampler_id;
        } pc;

        layout(location = 0) in vec2 v_local;
        layout(location = 1) in vec2 v_uv;
        layout(location = 2) in flat vec4 v_fill;
        layout(location = 3) in flat vec4 v_border_col;
        layout(location = 4) in flat vec4 v_border_w;
        layout(location = 5) in flat vec4 v_clip;
        layout(location = 6) in flat vec3 v_half_radius;
        layout(location = 7) in flat uint v_kind;
        layout(location = 8) in flat uint v_tex;

        layout(location = 0) out vec4 f_color;

        // Signed distance to a rounded box centered at the origin (negative inside).
        float sd_round_box(vec2 p, vec2 half_size, float r) {
            vec2 q = abs(p) - half_size + r;
            return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
        }

        void main() {
            if (gl_FragCoord.x < v_clip.x || gl_FragCoord.y < v_clip.y ||
                gl_FragCoord.x > v_clip.z || gl_FragCoord.y > v_clip.w) {
                discard;
            }

            vec4 c;
            if (v_kind == 1u) {
                // Real glyph pixels from a bevy_text-rasterized atlas —
                // this pass just samples them; RGBA8 (color emoji, etc.)
                // or an alpha mask (bevy_text encodes mask glyphs as
                // white RGB + coverage alpha), either works unmodified.
                SampledImageId g = SampledImageId(v_tex, 256);
                vec4 s = texture(vko_sampler2D(g, pc.sampler_id), v_uv);
                c = vec4(v_fill.rgb, v_fill.a * s.a);
            } else if (v_kind == 2u) {
                // <img>: full RGBA texel * tint, rounded-corner mask.
                SampledImageId g = SampledImageId(v_tex, 256);
                vec4 s = texture(vko_sampler2D(g, pc.sampler_id), v_uv);
                vec2 half_size = v_half_radius.xy;
                float r = min(v_half_radius.z, min(half_size.x, half_size.y));
                float d_outer = sd_round_box(v_local, half_size, r);
                c = s * v_fill;
                c.a *= clamp(0.5 - d_outer, 0.0, 1.0);
            } else {
                vec2 half_size = v_half_radius.xy;
                float r = min(v_half_radius.z, min(half_size.x, half_size.y));
                float d_outer = sd_round_box(v_local, half_size, r);

                vec2 in_min = -half_size + v_border_w.xy;
                vec2 in_max =  half_size - v_border_w.zw;
                vec2 in_c = 0.5 * (in_min + in_max);
                vec2 in_h = max(0.5 * (in_max - in_min), vec2(0.0));
                float max_b = max(max(v_border_w.x, v_border_w.y), max(v_border_w.z, v_border_w.w));
                float d_inner = sd_round_box(v_local - in_c, in_h, max(r - max_b, 0.0));

                float outer_a = clamp(0.5 - d_outer, 0.0, 1.0);
                float inner_a = clamp(0.5 - d_inner, 0.0, 1.0);
                c = mix(v_border_col, v_fill, inner_a);
                c.a *= outer_a;
            }

            if (c.a < 0.003) {
                discard;
            }
            f_color = c;
        }
        ",
    }
}
