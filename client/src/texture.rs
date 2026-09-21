//! Bindless texture loading. Ported from the old project's `texture.rs`:
//! same discover-scan-batch-upload-register shape, adapted from `&App`
//! to `VulkanContext` and turned into a resource + Bevy system instead of
//! a method called out of `resumed()`.
//!
//! One difference worth calling out: the old version panicked
//! (`.unwrap_or_else(|_| panic!(...))`) on a missing/bad file. This one
//! skips it and falls back to whichever texture loaded first for any
//! block that wanted it — a missing asset should look wrong, not crash
//! the game.

use std::collections::HashMap;
use std::fs;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use vulkano::buffer::{BufferCreateInfo, BufferUsage};
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{
    ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageType, ImageUsage,
};
use vulkano::memory::DeviceAlignment;
use vulkano::memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter};
use vulkano_taskgraph::Id;
use vulkano_taskgraph::command_buffer::{BlitImageInfo, CopyBufferToImageInfo, ImageBlit};
use vulkano_taskgraph::descriptor_set::SampledImageId;
use vulkano_taskgraph::resource::{HostAccessType, ImageLayoutType};

use crate::vulkan_prelude::VulkanContext;

/// How a `BlockType` maps onto loaded texture keys (see `discover_pngs`
/// for how a file path becomes a key).
#[derive(Clone, Copy, Debug)]
pub enum TextureMapping {
    Single(&'static str),
    TopBottomSide {
        top: &'static str,
        bottom: &'static str,
        side: &'static str,
    },
}

/// Cloned into every async meshing task (up to several per frame), so the
/// map lives behind an `Arc`: `clone()` is a refcount bump.
#[derive(Resource, Debug, Default, Clone)]
pub struct TextureManager {
    textures: Arc<HashMap<String, SampledImageId>>,
    /// First texture that loaded successfully. Used for any key that
    /// isn't found so a missing asset renders as *a* texture, not a
    /// crash or an unmeshed hole.
    fallback: Option<SampledImageId>,
}

impl TextureManager {
    pub fn get(&self, name: &str) -> Option<SampledImageId> {
        self.textures.get(name).copied().or(self.fallback)
    }

    /// True once at least one texture is loaded. Meshing must wait for
    /// this: `setup_textures` inserts an EMPTY manager when `assets/` has
    /// no PNGs (to stop rescanning every frame), and meshing against that
    /// would permanently bake "no faces" into every chunk.
    pub fn has_textures(&self) -> bool {
        self.fallback.is_some()
    }

    fn discover_pngs(dir: &Path, base_dir: &Path, out: &mut Vec<(String, PathBuf)>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                Self::discover_pngs(&path, base_dir, out);
            } else if path.extension().is_some_and(|ext| ext == "png") {
                if let Ok(relative) = path.strip_prefix(base_dir) {
                    let key = relative
                        .with_extension("")
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push((key, path));
                }
            }
        }
    }
}

pub struct TexturePlugin;

impl Plugin for TexturePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, setup_textures);
    }
}

struct PendingTexture {
    key: String,
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    image_id: Id<vulkano::image::Image>,
    staging_id: Id<vulkano::buffer::Buffer>,
    mip_levels: u32,
}

/// Scans `assets/` for every `.png`, uploads the whole batch (mip-chain
/// blits included) in one command-buffer submission, then registers each
/// as a bindless sampled image. Runs every frame until `VulkanContext`
/// exists and `TextureManager` doesn't yet — same early-return-until-
/// ready pattern as `setup_renderer`, so ordering relative to it doesn't
/// matter (`upload_dirty_chunk_meshes` separately waits on this resource
/// before meshing anything).
fn setup_textures(
    mut commands: Commands,
    vk: Option<Res<VulkanContext>>,
    textures: Option<Res<TextureManager>>,
) {
    let Some(vk) = vk else {
        return;
    };
    if textures.is_some() {
        return;
    }

    let base_dir = Path::new("assets");
    let mut discovered = Vec::new();
    TextureManager::discover_pngs(base_dir, base_dir, &mut discovered);

    let mut manager = TextureManager::default();
    let mut loaded: HashMap<String, SampledImageId> = HashMap::new();

    if discovered.is_empty() {
        // Nothing under assets/ yet. Insert the (empty) manager anyway so
        // this system stops re-scanning the filesystem every frame —
        // chunks just won't get any faces meshed until PNGs show up and
        // the entity's `ChunkMeshUploaded` marker gets removed to retry.
        commands.insert_resource(manager);
        return;
    }

    let mut pending = Vec::new();

    for (key, path) in discovered {
        let Ok(dynamic_img) = image::open(&path) else {
            continue;
        };
        let dynamic_img = dynamic_img.to_rgba8();
        let (width, height) = dynamic_img.dimensions();
        let image_data = dynamic_img.into_raw();
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let Ok(image_id) = vk.resources.create_image(
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                extent: [width, height, 1],
                mip_levels,
                usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            &AllocationCreateInfo::default(),
        ) else {
            continue;
        };

        let Some(layout) = DeviceLayout::new(
            NonZero::new(image_data.len() as u64).unwrap(),
            DeviceAlignment::new(4).unwrap(),
        ) else {
            continue;
        };

        let Ok(staging_id) = vk.resources.create_buffer(
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            layout,
        ) else {
            continue;
        };

        pending.push(PendingTexture {
            key,
            image_data,
            width,
            height,
            image_id,
            staging_id,
            mip_levels,
        });
    }

    if pending.is_empty() {
        commands.insert_resource(manager);
        return;
    }

    let host_accesses: Vec<_> = pending
        .iter()
        .map(|p| (p.staging_id, HostAccessType::Write))
        .collect();

    unsafe {
        vulkano_taskgraph::execute(
            &vk.queue,
            &vk.resources,
            vk.flight_id,
            |cbf, tcx| {
                for p in &pending {
                    tcx.try_write_buffer::<[u8]>(p.staging_id, 0..p.image_data.len() as u64)?
                        .copy_from_slice(&p.image_data);

                    cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: p.staging_id,
                        dst_image: p.image_id,
                        ..Default::default()
                    });

                    let mut mip_w = p.width as i32;
                    let mut mip_h = p.height as i32;
                    for mip in 1..p.mip_levels {
                        let next_w = (mip_w / 2).max(1);
                        let next_h = (mip_h / 2).max(1);

                        cbf.blit_image(&BlitImageInfo {
                            src_image: p.image_id,
                            src_image_layout: ImageLayoutType::Optimal,
                            dst_image: p.image_id,
                            dst_image_layout: ImageLayoutType::Optimal,
                            regions: &[ImageBlit {
                                src_subresource: vulkano::image::ImageSubresourceLayers {
                                    aspects: ImageAspects::COLOR,
                                    mip_level: mip - 1,
                                    base_array_layer: 0,
                                    layer_count: Some(1),
                                },
                                src_offsets: [[0, 0, 0], [mip_w as u32, mip_h as u32, 1]],
                                dst_subresource: vulkano::image::ImageSubresourceLayers {
                                    aspects: ImageAspects::COLOR,
                                    mip_level: mip,
                                    base_array_layer: 0,
                                    layer_count: Some(1),
                                },
                                dst_offsets: [[0, 0, 0], [next_w as u32, next_h as u32, 1]],
                                ..Default::default()
                            }],
                            filter: vulkano::image::sampler::Filter::Linear,
                            ..Default::default()
                        });

                        mip_w = next_w;
                        mip_h = next_h;
                    }
                }
                Ok(())
            },
            host_accesses,
            [],
            [],
        )
        .unwrap();
    }

    let global_set = vk.resources.bindless_context().unwrap().global_set();

    for p in pending {
        let Ok(bindless_id) = global_set.create_sampled_image(
            p.image_id,
            &ImageViewCreateInfo {
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    base_mip_level: 0,
                    level_count: Some(p.mip_levels),
                    base_array_layer: 0,
                    layer_count: Some(1),
                },
                ..Default::default()
            },
            ImageLayout::ShaderReadOnlyOptimal,
        ) else {
            continue;
        };

        if manager.fallback.is_none() {
            manager.fallback = Some(bindless_id);
        }
        loaded.insert(p.key, bindless_id);
    }

    manager.textures = Arc::new(loaded);
    commands.insert_resource(manager);
}

/// Extracts the raw bindless index from a `SampledImageId` for embedding
/// in a vertex as a plain `u32` (vertex formats need POD data, not the
/// full descriptor handle). Mirrors the old project's `get_texture_id`
/// exactly — this relies on `SampledImageId`'s layout being two `u32`s,
/// which is an internal detail of `vulkano_taskgraph`, not a stable
/// public guarantee, so if a `vulkano-taskgraph` update ever changes it,
/// this is the one place that would need to follow.
pub fn sampled_image_index(id: SampledImageId) -> u32 {
    let raw: [u32; 2] = unsafe { std::mem::transmute(id) };
    raw[0]
}
