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
//!     of a directory scan. Old bindless slots are intentionally never
//!     freed (see the NOTE inside the function) — same as `texture.rs`,
//!     which doesn't free block textures either — so a re-uploaded atlas
//!     just stops being referenced rather than reclaiming its slot. This
//!     only fires a handful of times per run as new glyphs are first
//!     seen, not every frame.
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
use bevy_color::{Alpha, Color};
use bevy_ecs::prelude::*;
use bevy_image::Image;
use bevy_log::info;
use bevy_ui::prelude::*;
use bevy_ui::{BorderColor, CalculatedClip, ComputedNode, Outline, UiGlobalTransform};
use bevy_window::{PrimaryWindow, Window};
use glam::Vec2;
// NOTE on API history in this file (checked against docs.rs/bevy_ui/0.19.1):
// `ComputedNodeTarget` doesn't exist. `OverflowClipMargin` exists but is an
// *input* (the margin the person configures on `Node`), not a `Component`
// you query for a result — it's not even listed as deriving `Component` on
// docs.rs. What bevy_ui's OWN renderer actually reads is `CalculatedClip`
// (a real `Component`, `{ clip: Rect }`), which `update_clipping` already
// computes for us every frame — walking `Overflow` and re-deriving clip
// rects by hand was reinventing something bevy_ui hands us for free. Using
// the real component now instead.
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

use vulkano::format::Format;
use wgpu_types::TextureFormat;

use crate::render::ui_pass::{UiDrawList, UiInstance, swapchain_is_srgb};
use crate::texture::sampled_image_index;
use crate::vulkan_prelude::VulkanContext;

use super::spawn::{UiElement, UiRoot};

fn wgpu_to_vulkano_format(format: TextureFormat) -> Option<Format> {
    match format {
        TextureFormat::R8Unorm => Some(Format::R8_UNORM),
        TextureFormat::R8Snorm => Some(Format::R8_SNORM),
        TextureFormat::R8Uint => Some(Format::R8_UINT),
        TextureFormat::R8Sint => Some(Format::R8_SINT),
        TextureFormat::Rg8Unorm => Some(Format::R8G8_UNORM),
        TextureFormat::Rg8Snorm => Some(Format::R8G8_SNORM),
        TextureFormat::Rg8Uint => Some(Format::R8G8_UINT),
        TextureFormat::Rg8Sint => Some(Format::R8G8_SINT),
        TextureFormat::Rgba8Unorm => Some(Format::R8G8B8A8_UNORM),
        TextureFormat::Rgba8UnormSrgb => Some(Format::R8G8B8A8_SRGB),
        TextureFormat::Rgba8Snorm => Some(Format::R8G8B8A8_SNORM),
        TextureFormat::Rgba8Uint => Some(Format::R8G8B8A8_UINT),
        TextureFormat::Rgba8Sint => Some(Format::R8G8B8A8_SINT),
        TextureFormat::Bgra8Unorm => Some(Format::B8G8R8A8_UNORM),
        TextureFormat::Bgra8UnormSrgb => Some(Format::B8G8R8A8_SRGB),
        TextureFormat::R16Unorm => Some(Format::R16_UNORM),
        TextureFormat::R16Snorm => Some(Format::R16_SNORM),
        TextureFormat::R16Uint => Some(Format::R16_UINT),
        TextureFormat::R16Sint => Some(Format::R16_SINT),
        TextureFormat::R16Float => Some(Format::R16_SFLOAT),
        TextureFormat::Rg16Unorm => Some(Format::R16G16_UNORM),
        TextureFormat::Rg16Snorm => Some(Format::R16G16_SNORM),
        TextureFormat::Rg16Uint => Some(Format::R16G16_UINT),
        TextureFormat::Rg16Sint => Some(Format::R16G16_SINT),
        TextureFormat::Rg16Float => Some(Format::R16G16_SFLOAT),
        TextureFormat::Rgba16Unorm => Some(Format::R16G16B16A16_UNORM),
        TextureFormat::Rgba16Snorm => Some(Format::R16G16B16A16_SNORM),
        TextureFormat::Rgba16Uint => Some(Format::R16G16B16A16_UINT),
        TextureFormat::Rgba16Sint => Some(Format::R16G16B16A16_SINT),
        TextureFormat::Rgba16Float => Some(Format::R16G16B16A16_SFLOAT),
        TextureFormat::R32Uint => Some(Format::R32_UINT),
        TextureFormat::R32Sint => Some(Format::R32_SINT),
        TextureFormat::R32Float => Some(Format::R32_SFLOAT),
        TextureFormat::Rg32Uint => Some(Format::R32G32_UINT),
        TextureFormat::Rg32Sint => Some(Format::R32G32_SINT),
        TextureFormat::Rg32Float => Some(Format::R32G32_SFLOAT),
        TextureFormat::Rgba32Uint => Some(Format::R32G32B32A32_UINT),
        TextureFormat::Rgba32Sint => Some(Format::R32G32B32A32_SINT),
        TextureFormat::Rgba32Float => Some(Format::R32G32B32A32_SFLOAT),
        _ => None,
    }
}

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

    // NOTE: `GlobalDescriptorSet` has no `remove_sampled_image` — rustc's
    // suggestion was the `sampled_image` getter, not a removal method, and
    // I don't actually know the real free/deregister call for this
    // vulkano-taskgraph version. Rather than guess another wrong name,
    // bindless slots are deliberately never freed here — same as
    // `texture.rs`, which never frees its block-texture registrations
    // either. Glyph atlases are few (one per distinct font/size actually
    // used) and effectively permanent for a run, so this leaks at most a
    // handful of slots total, not per-frame. If you hit real slot
    // exhaustion, grep your local vulkano-taskgraph source under
    // descriptor_set/ for the actual removal API and wire it in — both
    // call sites below marked (*).
    let removed_ids: Vec<AssetId<Image>> = events
        .read()
        .filter_map(|ev| match ev {
            AssetEvent::Removed { id } | AssetEvent::Unused { id } => Some(*id),
            _ => None,
        })
        .collect();
    for id in removed_ids {
        registry.entries.remove(&id); // (*) slot itself is intentionally leaked, not freed.
    }
    // bevy_log::info!(
    //     "glyph sync: {} TextLayoutInfo, {} wanted atlases",
    //     text_query.iter().count(),
    //     wanted.len()
    // );

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
            // Atlas grew or was redrawn. (*) Old slot is intentionally
            // left registered/leaked (see NOTE above) rather than freed;
            // we just stop using it and register a new one below.
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
    let format = wgpu_to_vulkano_format(image.texture_descriptor.format).unwrap();

    vk.resources.flight(vk.flight_id).wait(None).unwrap();

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
        .map_err(|e| bevy_log::warn!("atlas img: {e}"))
        .ok()?;
    }
    // vk.resources.flight(vk.flight_id).wait(None).unwrap();
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
        Option<&CalculatedClip>,
        Option<&BackgroundColor>,
        Option<&BorderColor>,
        Option<&Outline>,
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
        walk(root, &nodes, &registry, to_linear, &mut instances);
    }

    // bevy_log::info!(
    //     "UI extract: {} instances (gen {})",
    //     instances.len(),
    //     draw_list.generation
    // );

    // bevy_log::info!(
    //     "text: node={:?} info_size={:?} computed={:?} tt={:?} first5={:?}",
    //     entity,
    //     info.size,
    //     computed.size(),
    //     transform.translation,
    //     info.glyphs
    //         .iter()
    //         .take(5)
    //         .map(|g| (g.position, g.atlas_info.rect))
    //         .collect::<Vec<_>>(),
    // );

    // bevy_log::info!(
    //     "label {:?}: {} glyphs, size {:?}, tt {:?}",
    //     entity,
    //     info.glyphs.len(),
    //     info.size,
    //     transform.translation
    // );

    // Cheap content check avoids the GPU wait+upload in `ui_pass::sync`
    // on frames where nothing changed (static menus, idle HUD).
    if instances.len() != draw_list.instances.len() || instances != draw_list.instances {
        draw_list.instances = instances;
        draw_list.generation = draw_list.generation.wrapping_add(1);
    }
}

fn walk(
    entity: Entity,
    nodes: &Query<(
        &ComputedNode,
        &UiGlobalTransform,
        Option<&CalculatedClip>,
        Option<&BackgroundColor>,
        Option<&BorderColor>,
        Option<&Outline>,
        Option<&bevy_text::TextLayoutInfo>,
        Option<&bevy_text::TextColor>,
        Option<&Children>,
    )>,
    registry: &GlyphAtlasRegistry,
    to_linear: bool,
    out: &mut Vec<UiInstance>,
) {
    let Ok((computed, transform, clip, bg, border, _outline, text, text_color, children)) =
        nodes.get(entity)
    else {
        return;
    };

    // `bevy_ui`'s own `update_clipping` system already computed this node's
    // final, ancestor-intersected clip rect (honoring `Overflow`,
    // `OverflowClipMargin`, and `OverrideClip` for us) — no clip present
    // means unclipped, same as bevy_ui's own convention.
    let node_clip = clip
        .map(|c| [c.clip.min.x, c.clip.min.y, c.clip.max.x, c.clip.max.y])
        .unwrap_or(UiInstance::NO_CLIP);

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
            // `BorderRect` is `{ min_inset: Vec2, max_inset: Vec2 }`, not
            // named left/top/right/bottom fields — same min/max-corner
            // convention as `Rect` (docs.rs/bevy_ui: "no field `left` on
            // `BorderRect`; available fields are `min_inset`, `max_inset`").
            let b = computed.border();
            out.push(UiInstance::boxed(
                rect,
                fill,
                border_col,
                [b.min_inset.x, b.min_inset.y, b.max_inset.x, b.max_inset.y],
                computed.border_radius().top_left,
                node_clip,
            ));
        }
    }

    let node_size = computed.size();
    let node_top_left = transform.translation - 0.5 * node_size;

    if let Some(info) = text {
        let color = text_color.map(|c| c.0).unwrap_or(Color::WHITE);
        let color = to_instance_color(color, to_linear);

        for glyph in &info.glyphs {
            let Some(entry) = registry.entries.get(&glyph.atlas_info.texture) else {
                continue;
            };
            let atlas = &glyph.atlas_info;
            let glyph_size = Vec2::new(atlas.rect.width(), atlas.rect.height());

            // position is already the center, relative to node top-left
            let glyph_top_left = node_top_left + glyph.position - 0.5 * glyph_size;

            let uv_rect = [
                atlas.rect.min.x,
                atlas.rect.min.y,
                atlas.rect.max.x,
                atlas.rect.max.y,
            ];
            out.push(UiInstance::glyph(
                [
                    glyph_top_left.x,
                    glyph_top_left.y,
                    glyph_size.x,
                    glyph_size.y,
                ],
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
            walk(child, nodes, registry, to_linear, out);
        }
    }
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
                (
                    sync_glyph_atlases, //.run_if(resource_exists::<Assets<Image>>),
                    extract_ui,
                    // force_ui_geometry,
                    // debug_ui_layout,
                )
                    .chain()
                    .after(bevy_ui::UiSystems::PostLayout),
            );
    }
}

fn debug_ui_layout(
    q: Query<
        (
            Entity,
            &ComputedNode,
            &UiGlobalTransform,
            Option<&BackgroundColor>,
        ),
        With<UiElement>,
    >,
) {
    for (e, c, t, bg) in &q {
        bevy_log::info!(
            "ui node {e:?} size={:?} translation={:?} bg={:?}",
            c.size(),
            t.translation,
            bg.map(|b| b.0)
        );
    }
}

fn force_ui_geometry(
    windows: Query<&Window, With<PrimaryWindow>>,
    roots: Query<Entity, With<UiRoot>>,
    mut nodes: Query<(
        &Node,
        &mut ComputedNode,
        &mut UiGlobalTransform,
        Option<&Children>,
        Option<&BackgroundColor>,
    )>,
) {
    let Ok(window) = windows.single() else { return };
    let screen = Vec2::new(
        window.resolution.physical_width() as f32,
        window.resolution.physical_height() as f32,
    );

    for root in &roots {
        // Root = full window, centered content handled by flex-like walk
        if let Ok((_, mut computed, mut transform, children, _)) = nodes.get_mut(root) {
            computed.size = screen;
            *transform = UiGlobalTransform::from_scale(Vec2::new(screen.x * 0.5, screen.y * 0.5));
            // then walk children with a tiny layout if needed
            let _ = children;
        }
    }
}
