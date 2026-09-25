//! Bridges `bevy_ui`'s own layout/text output to the Vulkan renderer.
//!
//! This does NOT shape, rasterize, or lay out anything — `bevy_text`
//! (parley + swash, `bevy_ui`'s own `text_system`) already did that and
//! left the result on every text entity as `TextLayoutInfo` /
//! `PositionedGlyph`, with each glyph's pixels living in an `Assets<Image>`
//! atlas that `bevy_text::FontAtlasSet` manages. This module's only two
//! jobs:
//!
//!  1. [`sync_glyph_atlases`] — keep those atlas `Image`s uploaded as
//!     bindless sampled images, the same discover/stage/upload/register
//!     shape as `texture.rs`, just driven by `AssetEvent<Image>` instead
//!     of a directory scan, and re-uploading a growing atlas is done by
//!     freeing its old bindless slot and re-registering (atlases are
//!     append-only within a run, so this only fires a handful of times as
//!     new glyphs are first seen, not every frame).
//!  2. [`extract_ui`] — walk `Node`/`ComputedNode`/`BackgroundColor`/
//!     `BorderColor`/`TextLayoutInfo` (and this crate's `UiRoot`/
//!     `UiElement` to know which are ours) and turn them into
//!     [`UiInstance`] quads for `ui_pass.rs`, using `UiGlobalTransform`
//!     for position exactly like `bevy_ui`'s own renderer would.
//!
//! Ordering: both run in `PostUpdate`, after `bevy_ui::UiSystems::PostLayout`
//! (where `ComputedNode`/`UiGlobalTransform`/`TextLayoutInfo` are finalized
//! for this frame) — see `UiRenderBridgePlugin`.

use std::collections::HashMap;
use std::num::NonZero;

use bevy_app::prelude::*;
use bevy_asset::{AssetEvent, AssetId, Assets};
use bevy_color::Color;
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_ui::prelude::*;
use bevy_ui::{BorderColor, ComputedNode, Outline, OverflowClipMargin, UiGlobalTransform};
use vulkano::buffer::{BufferCreateInfo, BufferUsage};
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{
    ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageType, ImageUsage,
};
use vulkano::memory::DeviceAlignment;
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter};
use vulkano_taskgraph::Id;
use vulkano_taskgraph::command_buffer::CopyBufferToImageInfo;
use vulkano_taskgraph::descriptor_set::SampledImageId;
use vulkano_taskgraph::resource::HostAccessType;

use crate::render::ui_pass::{UiDrawList, UiInstance, swapchain_is_srgb};
use crate::texture::sampled_image_index;
use crate::vulkan_prelude::VulkanContext;

use super::spawn::{UiElement, UiRoot};

// ---------------------------------------------------------------- atlases

/// Bindless registration for one `bevy_text` glyph-atlas `Image`, plus the
/// generation of its `data` we last uploaded (atlases are resized/redrawn
/// in place, so `AssetEvent::Modified` can mean "same handle, new pixels").
struct AtlasEntry {
    bindless_id: SampledImageId,
    tex_id: u32,
    size: [f32; 2],
    uploaded_len: usize,
}

/// `AssetId<Image>` -> bindless registration, for every atlas image
/// `bevy_text` has handed us. Persists across frames; only touched when
/// `AssetEvent<Image>` fires for an id already seen as a glyph atlas.
#[derive(Resource, Default)]
struct GlyphAtlasRegistry {
    entries: HashMap<AssetId<Image>, AtlasEntry>,
}

/// Upload/refresh bindless copies of whichever `Image` assets this frame's
/// glyphs point at. Mirrors `texture.rs::setup_textures`'s batched
/// stage-then-copy shape, but keyed by asset id instead of a filesystem
/// scan, and running every frame (cheap: usually zero work once fonts
/// settle, since a `String`-keyed `HashMap` lookup catches "already
/// uploaded, unchanged" before any GPU call).
fn sync_glyph_atlases(
    vk: Option<Res<VulkanContext>>,
    images: Res<Assets<Image>>,
    mut events: MessageReader<AssetEvent<Image>>,
    mut registry: ResMut<GlyphAtlasRegistry>,
    text_query: Query<&bevy_text::TextLayoutInfo, With<UiElement>>,
) {
    let Some(vk) = vk else {
        return;
    };

    // Which atlas ids are actually referenced by our UI text this frame.
    // `bevy_text::FontAtlasSet` can hold atlases for Text2D/other UI too;
    // we only want to spend bindless slots on ones we draw.
    let mut wanted: Vec<AssetId<Image>> = Vec::new();
    for info in &text_query {
        for glyph in &info.glyphs {
            let id = glyph.atlas_info.texture;
            if !wanted.contains(&id) {
                wanted.push(id);
            }
        }
    }

    // Drop bindless registrations for atlases nobody references and that
    // were removed/replaced (frees the slot rather than leaking it).
    for ev in events.read() {
        if let AssetEvent::Removed { id } | AssetEvent::Unused { id } = ev {
            if let Some(entry) = registry.entries.remove(id) {
                let global_set = vk.resources.bindless_context().unwrap().global_set();
                let _ = global_set.remove_sampled_image(entry.bindless_id);
            }
        }
    }

    for id in wanted {
        let Some(image) = images.get(id) else {
            continue;
        };
        let Some(data) = image.data.as_ref() else {
            continue;
        };

        if let Some(existing) = registry.entries.get(&id) {
            if existing.uploaded_len == data.len() {
                continue; // Same handle, same size => assume unchanged (see below).
            }
            // Atlas grew or was redrawn: free the old slot before re-upload.
            let global_set = vk.resources.bindless_context().unwrap().global_set();
            let _ = global_set.remove_sampled_image(existing.bindless_id);
            registry.entries.remove(&id);
        }

        let Some(entry) = upload_atlas_image(&vk, image, data) else {
            continue;
        };
        registry.entries.insert(id, entry);
    }
}

/// One image's worth of `texture.rs`'s stage->copy->register, no mip chain
/// (glyph atlases are drawn 1:1 in screen pixels, never minified).
fn upload_atlas_image(vk: &VulkanContext, image: &Image, data: &[u8]) -> Option<AtlasEntry> {
    let width = image.width();
    let height = image.height();
    let format = image.texture_descriptor.format;

    let image_id = vk
        .resources
        .create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format,
                extent: [width, height, 1],
                mip_levels: 1,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        )
        .ok()?;

    let layout = DeviceLayout::new(
        NonZero::new(data.len() as u64)?,
        DeviceAlignment::new(4).unwrap(),
    )?;
    let staging_id: Id<vulkano::buffer::Buffer> = vk
        .resources
        .create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            layout,
        )
        .ok()?;

    unsafe {
        vulkano_taskgraph::execute(
            &vk.queue,
            &vk.resources,
            vk.flight_id,
            |cbf, tcx| {
                tcx.try_write_buffer::<[u8]>(staging_id, 0..data.len() as u64)?
                    .copy_from_slice(data);
                cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                    src_buffer: staging_id,
                    dst_image: image_id,
                    ..Default::default()
                });
                Ok(())
            },
            [(staging_id, HostAccessType::Write)],
            [],
            [],
        )
        .ok()?;
    }

    let global_set = vk.resources.bindless_context().unwrap().global_set();
    let bindless_id = global_set
        .create_sampled_image(
            image_id,
            &ImageViewCreateInfo {
                format,
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    base_mip_level: 0,
                    level_count: Some(1),
                    base_array_layer: 0,
                    layer_count: Some(1),
                },
                ..Default::default()
            },
            ImageLayout::ShaderReadOnlyOptimal,
        )
        .ok()?;

    Some(AtlasEntry {
        bindless_id,
        tex_id: sampled_image_index(bindless_id),
        size: [width as f32, height as f32],
        uploaded_len: data.len(),
    })
}

// ------------------------------------------------------------------ extract

/// Walk this frame's UI tree (ours only — gated by [`UiElement`], so a
/// future non-SDUI HUD using `bevy_ui` directly isn't drawn twice) and
/// build the flat instance list `ui_pass.rs` uploads and draws.
///
/// Runs after `UiSystems::PostLayout`, so `ComputedNode`/
/// `UiGlobalTransform`/`TextLayoutInfo` are this frame's final values —
/// same dependency `bevy_ui`'s own extraction has on those systems.
#[allow(clippy::too_many_arguments)]
fn extract_ui(
    vk: Option<Res<VulkanContext>>,
    registry: Res<GlyphAtlasRegistry>,
    mut draw_list: ResMut<UiDrawList>,
    roots: Query<Entity, With<UiRoot>>,
    nodes: Query<(
        &ComputedNode,
        &UiGlobalTransform,
        &ComputedNodeTarget,
        Option<&BackgroundColor>,
        Option<&BorderColor>,
        Option<&Outline>,
        Option<&OverflowClipMargin>,
        Option<&bevy_text::TextLayoutInfo>,
        Option<&bevy_text::TextColor>,
        Option<&Children>,
    )>,
) {
    let Some(vk) = vk else {
        return;
    };
    let to_linear = swapchain_is_srgb(&vk);

    let mut instances = Vec::new();
    for root in &roots {
        walk(
            root,
            UiInstance::NO_CLIP,
            &nodes,
            &registry,
            to_linear,
            &mut instances,
        );
    }

    // Cheap content check avoids the GPU wait+upload in `ui_pass::sync`
    // on frames where nothing changed (static menus, idle HUD).
    if instances.len() != draw_list.instances.len() || instances != draw_list.instances {
        draw_list.instances = instances;
        draw_list.generation = draw_list.generation.wrapping_add(1);
    }
}

fn walk(
    entity: Entity,
    inherited_clip: [f32; 4],
    nodes: &Query<(
        &ComputedNode,
        &UiGlobalTransform,
        &ComputedNodeTarget,
        Option<&BackgroundColor>,
        Option<&BorderColor>,
        Option<&Outline>,
        Option<&OverflowClipMargin>,
        Option<&bevy_text::TextLayoutInfo>,
        Option<&bevy_text::TextColor>,
        Option<&Children>,
    )>,
    registry: &GlyphAtlasRegistry,
    to_linear: bool,
    out: &mut Vec<UiInstance>,
) {
    let Ok((
        computed,
        transform,
        _target,
        bg,
        border,
        _outline,
        _clip_margin,
        text,
        text_color,
        children,
    )) = nodes.get(entity)
    else {
        return;
    };

    let size = computed.size();
    let half = 0.5 * size;
    // `UiGlobalTransform` is centered on the node (see bevy_ui's own
    // migration notes); our quad shader wants a top-left rect.
    let top_left = transform.translation - half;
    let rect = [top_left.x, top_left.y, size.x, size.y];

    if let Some(bg) = bg {
        if bg.0.alpha() > 0.0 {
            let fill = to_instance_color(bg.0, to_linear);
            let border_col = border
                .map(|b| to_instance_color(b.top, to_linear))
                .unwrap_or([0.0; 4]);
            let b = computed.border();
            out.push(UiInstance::boxed(
                rect,
                fill,
                border_col,
                [b.left, b.top, b.right, b.bottom],
                computed.border_radius().top_left,
                inherited_clip,
            ));
        }
    }

    let node_clip = if computed.overflow().is_visible() {
        inherited_clip
    } else {
        intersect_clip(
            inherited_clip,
            [
                top_left.x,
                top_left.y,
                top_left.x + size.x,
                top_left.y + size.y,
            ],
        )
    };

    if let Some(info) = text {
        let color = text_color.map(|c| c.0).unwrap_or(Color::WHITE);
        let color = to_instance_color(color, to_linear);
        // `TextLayoutInfo::glyphs` positions are relative to the text
        // block's own top-left, exactly like `bevy_ui`'s own glyph
        // extraction consumes them.
        let text_top_left = transform.translation - 0.5 * info.size;
        for glyph in &info.glyphs {
            let Some(entry) = registry.entries.get(&glyph.atlas_info.texture) else {
                continue; // Not uploaded yet; picked up again next frame.
            };
            let atlas = &glyph.atlas_info;
            let px = text_top_left + glyph.position;
            let uv_rect = [
                atlas.rect.min.x,
                atlas.rect.min.y,
                atlas.rect.max.x,
                atlas.rect.max.y,
            ];
            out.push(UiInstance::glyph(
                [px.x, px.y, atlas.rect.width(), atlas.rect.height()],
                uv_rect,
                color,
                entry.size,
                entry.tex_id,
                node_clip,
            ));
        }
    }

    if let Some(children) = children {
        for &child in children {
            walk(child, node_clip, nodes, registry, to_linear, out);
        }
    }
}

fn intersect_clip(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

/// The swapchain does the sRGB encode on write only when its format is an
/// `_SRGB` variant; otherwise we must pre-encode here ourselves, same
/// reasoning as any other UI-over-swapchain-without-a-render-graph setup.
fn to_instance_color(c: Color, to_linear: bool) -> [f32; 4] {
    let srgba = c.to_srgba();
    if to_linear {
        let l = c.to_linear();
        [l.red, l.green, l.blue, srgba.alpha]
    } else {
        [srgba.red, srgba.green, srgba.blue, srgba.alpha]
    }
}

pub struct UiRenderBridgePlugin;

impl Plugin for UiRenderBridgePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GlyphAtlasRegistry>()
            .init_resource::<UiDrawList>()
            .add_systems(
                PostUpdate,
                (sync_glyph_atlases, extract_ui)
                    .chain()
                    .after(bevy_ui::UiSystems::PostLayout),
            );
    }
}
