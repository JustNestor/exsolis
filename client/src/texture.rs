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
use vulkano_taskgraph::descriptor_set::SamplerId;

pub use vulkano_taskgraph::descriptor_set::SampledImageId;
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
    /// Bumped on every (re)build. GLTF uploads stamp the generation they
    /// were built against; a mismatch means `setup_textures` replaced this
    /// manager (wiping GLTF bindless ids) and textures must be re-uploaded.
    pub generation: u64,
}

#[derive(Resource, Default)]
struct TextureReload(bool);

impl TextureManager {
    pub fn get(&self, name: &str) -> Option<SampledImageId> {
        self.textures.get(name).copied().or(self.fallback)
    }

    /// Resolves a mod-declared asset path such as `mods/fly/effects/spark.png`
    /// to its bindless texture. Keys are asset paths with the `.png` stripped
    /// (see `discover_pngs`). Unlike [`get`], an unknown path falls back to
    /// the same placeholder texture used for blocks, so a missing effect
    /// sprite looks wrong instead of crashing; `None` only before any
    /// texture has loaded.
    pub fn get_asset(&self, path: &str) -> Option<SampledImageId> {
        match path.strip_suffix(".png") {
            Some(key) => self.get(key),
            None => self.fallback,
        }
    }

    /// True once at least one texture is loaded. Meshing must wait for
    /// this: `setup_textures` inserts an EMPTY manager when `assets/` has
    /// no PNGs (to stop rescanning every frame), and meshing against that
    /// would permanently bake "no faces" into every chunk.
    pub fn has_textures(&self) -> bool {
        self.fallback.is_some()
    }

    /// Load GLTF embedded textures from memory and upload to GPU.
    /// Returns a vector of bindless texture indices (or None if failed).
    pub fn load_gltf_textures(
        &mut self,
        vk: &VulkanContext,
        gltf_textures: &[crate::gltf::GltfTexture],
    ) -> Vec<Option<SampledImageId>> {
        let mut results = Vec::with_capacity(gltf_textures.len());
        let mut pending = Vec::new();

        for (idx, tex) in gltf_textures.iter().enumerate() {
            // GltfTexture.image_data is already decoded RGBA8 pixels.
            if tex.width == 0 || tex.height == 0 || tex.image_data.is_empty() {
                results.push(None);
                continue;
            }
            let width = tex.width;
            let height = tex.height;
            let image_data = tex.image_data.clone();
            let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

            let Ok(image_id) = vk.resources.create_image(
                &ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: vulkano::format::Format::R8G8B8A8_UNORM,
                    extent: [width, height, 1],
                    mip_levels,
                    usage: ImageUsage::TRANSFER_SRC
                        | ImageUsage::TRANSFER_DST
                        | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                &AllocationCreateInfo::default(),
            ) else {
                results.push(None);
                continue;
            };

            let Some(layout) = DeviceLayout::new(
                NonZero::new(image_data.len() as u64).unwrap(),
                DeviceAlignment::new(4).unwrap(),
            ) else {
                results.push(None);
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
                results.push(None);
                continue;
            };

            pending.push(PendingGltfTexture {
                key: format!("gltf_embedded_{}", idx),
                image_data,
                width,
                height,
                image_id,
                staging_id,
                mip_levels,
                result_idx: idx,
            });
            results.push(None); // placeholder
        }

        if pending.is_empty() {
            return results;
        }

        let host_accesses: Vec<_> = pending
            .iter()
            .map(|p| (p.staging_id, HostAccessType::Write))
            .collect();
        vk.resources.flight(vk.flight_id).wait(None).unwrap();

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

        let mut loaded = (*self.textures).clone();
        let mut fallback = self.fallback;

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
                bevy_log::warn!("create_sampled_image failed for GLTF texture {}", p.key);
                continue;
            };

            if fallback.is_none() {
                fallback = Some(bindless_id);
            }
            loaded.insert(p.key.clone(), bindless_id);
            results[p.result_idx] = Some(bindless_id);
        }

        self.textures = Arc::new(loaded);
        self.fallback = fallback;
        self.generation += 1;

        results
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
        app.init_resource::<TextureReload>()
            .init_resource::<BlockAnims>()
            .add_systems(Update, (request_texture_reload, setup_textures).chain())
            .add_systems(Update, tick_block_anims);
    }
}

fn request_texture_reload(
    mut changes: MessageReader<crate::assets::AssetCacheChanged>,
    mut reload: ResMut<TextureReload>,
) {
    if changes.read().any(|change| {
        change
            .assets
            .iter()
            .any(|asset| asset.path.ends_with(".png"))
    }) {
        reload.0 = true;
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
    /// Pre-split `.mcmeta` animation frames (ordered RGBA frame buffers).
    /// The uploaded base image is frame 0 at frame size, not the strip.
    anim: Option<AnimFrames>,
}

struct AnimFrames {
    frames: Vec<Vec<u8>>,
    /// MC ticks per frame (20 ticks = 1 second).
    frametime: u32,
}

/// Runtime state for one cycling `.mcmeta` texture. The GPU image holds a
/// single frame; `tick_block_anims` uploads the next frame in place, so the
/// bindless id baked into chunk vertices never changes.
struct AnimEntry {
    key: String,
    image_id: Id<vulkano::image::Image>,
    staging_id: Id<vulkano::buffer::Buffer>,
    frames: Vec<Vec<u8>>,
    frametime: u32,
    current: usize,
    elapsed_ticks: f32,
}

#[derive(Resource, Default)]
pub struct BlockAnims {
    entries: Vec<AnimEntry>,
    logged: bool,
}

struct PendingGltfTexture {
    key: String,
    image_data: Vec<u8>,
    width: u32,
    height: u32,
    image_id: Id<vulkano::image::Image>,
    staging_id: Id<vulkano::buffer::Buffer>,
    mip_levels: u32,
    result_idx: usize,
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
    mut reload: ResMut<TextureReload>,
    mc: Option<Res<crate::loader::McAssets>>,
    mut anims: ResMut<BlockAnims>,
) {
    let Some(vk) = vk else {
        return;
    };
    if textures.is_some() && !reload.0 {
        return;
    }
    reload.0 = false;

    let base_dir = crate::assets::asset_directory();
    let mut discovered = Vec::new();
    TextureManager::discover_pngs(&base_dir, &base_dir, &mut discovered);
    // MC pack textures (`minecraft/block/stone.png` -> key
    // `"minecraft/block/stone"`, matching baked quad texture keys).
    if let Some(mc) = crate::loader::mc_pack_dir() {
        let tex_base = mc.join("assets").join("minecraft").join("textures");
        let mut mc_found = Vec::new();
        TextureManager::discover_pngs(&tex_base, &tex_base, &mut mc_found);
        let n = mc_found.len();
        for (key, path) in mc_found {
            discovered.push((format!("minecraft/{key}"), path));
        }
        if n > 0 {
            bevy_log::info!("TextureManager: +{n} MC pack textures from {}", tex_base.display());
        }
    }

    let mut manager = TextureManager::default();
    // A rebuild replaces the whole resource, invalidating every bindless id
    // handed out before (including GLTF textures) — stamp it so uploaders
    // can detect the wipe and re-upload.
    manager.generation = textures.map(|t| t.generation).unwrap_or(0) + 1;
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
        let (full_w, full_h) = dynamic_img.dimensions();
        let full = dynamic_img.into_raw();

        // `.mcmeta` split: the base upload is frame 0 at frame size, and the
        // ordered frames ride along for the cycler. Animated images get no
        // mip chain (the cycler rewrites level 0 in place).
        let mut anim: Option<AnimFrames> = None;
        let mut image_data = full;
        let (mut width, mut height) = (full_w, full_h);
        if let Some(meta) = mc.as_ref().and_then(|m| m.anims.iter().find(|a| a.texture == key))
            && meta.frame_count > 1
            && full_h >= full_w
            && full_h % meta.frame_count as u32 == 0
        {
            let frame_h = full_h / meta.frame_count as u32;
            let row_bytes = (full_w * 4) as usize;
            let order: Vec<usize> = if meta.frames.is_empty() {
                (0..meta.frame_count).collect()
            } else {
                meta.frames
                    .iter()
                    .map(|&i| (i as usize) % meta.frame_count)
                    .collect()
            };
            let mut frames = Vec::with_capacity(order.len());
            for strip_idx in order {
                let mut buf = vec![0u8; (row_bytes * frame_h as usize)];
                for row in 0..frame_h as usize {
                    let src = ((strip_idx * frame_h as usize + row) * row_bytes)..;
                    let dst = (row * row_bytes)..((row + 1) * row_bytes);
                    buf[dst].copy_from_slice(&image_data[src.start..src.start + row_bytes]);
                }
                frames.push(buf);
            }
            if let Some(first) = frames.first().cloned() {
                image_data = first;
                width = full_w;
                height = frame_h;
                anim = Some(AnimFrames {
                    frames,
                    frametime: meta.frametime.max(1),
                });
            }
        }
        // Grayscale tintindex textures (grass, foliage) ship gray; the
        // plains biome tint makes them read as MC grass/leaves.
        if mc
            .as_ref()
            .is_some_and(|m| m.tinted_textures.contains(&key))
        {
            crate::loader::apply_biome_tint(&mut image_data);
            if let Some(anim) = anim.as_mut() {
                for frame in anim.frames.iter_mut() {
                    crate::loader::apply_biome_tint(frame);
                }
            }
        }
        let mip_levels = if anim.is_some() {
            1
        } else {
            (width.max(height) as f32).log2().floor() as u32 + 1
        };

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
            anim,
        });
    }

    // Embedded bbmodel textures (Thaumcraft props): already RGBA, keyed
    // `thaumcraft/<model>/<tex>` to match the retargeted bakes.
    if let Some(mc) = mc.as_ref() {
        for (key, (w, h, rgba)) in mc.embedded.iter() {
            let (width, height) = (*w, *h);
            if width == 0 || height == 0 || rgba.is_empty() {
                continue;
            }
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
                NonZero::new(rgba.len() as u64).unwrap(),
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
                key: key.clone(),
                image_data: rgba.clone(),
                width,
                height,
                image_id,
                staging_id,
                mip_levels,
                anim: None,
            });
        }
    }

    if pending.is_empty() {
        commands.insert_resource(manager);
        return;
    }

    let host_accesses: Vec<_> = pending
        .iter()
        .map(|p| (p.staging_id, HostAccessType::Write))
        .collect();
    vk.resources.flight(vk.flight_id).wait(None).unwrap();

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

    // A rebuild replaces every image, so the cycler is rebuilt too.
    anims.entries.clear();
    anims.logged = false;
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
        loaded.insert(p.key.clone(), bindless_id);
        // Keep the staging buffer: the cycler reuses it for frame uploads.
        if let Some(anim) = p.anim {
            anims.entries.push(AnimEntry {
                key: p.key,
                image_id: p.image_id,
                staging_id: p.staging_id,
                frames: anim.frames,
                frametime: anim.frametime,
                current: 0,
                elapsed_ticks: 0.0,
            });
        }
    }

    manager.textures = Arc::new(loaded);
    commands.insert_resource(manager);
}

/// Advance `.mcmeta` textures: when an entry's frame is due, upload the
/// next frame into its GPU image in place (bindless ids stay valid).
/// Bounded per frame so 100+ animated textures can't spike.
fn tick_block_anims(
    vk: Option<Res<VulkanContext>>,
    time: Res<bevy_time::Time>,
    mut anims: ResMut<BlockAnims>,
) {
    if anims.entries.is_empty() {
        return;
    }
    if !anims.logged {
        let frames: usize = anims.entries.iter().map(|e| e.frames.len()).sum();
        bevy_log::info!(
            "BlockAnims: cycling {} animated textures ({frames} frames)",
            anims.entries.len()
        );
        anims.logged = true;
    }
    let Some(vk) = vk else {
        return;
    };
    const MAX_UPLOADS_PER_FRAME: usize = 4;
    let mut due = Vec::new();
    for (i, entry) in anims.entries.iter_mut().enumerate() {
        if entry.frames.len() < 2 {
            continue;
        }
        entry.elapsed_ticks += time.delta_secs() * 20.0;
        if entry.elapsed_ticks >= entry.frametime as f32 {
            entry.elapsed_ticks = 0.0;
            entry.current = (entry.current + 1) % entry.frames.len();
            due.push(i);
            if due.len() >= MAX_UPLOADS_PER_FRAME {
                break;
            }
        }
    }
    if due.is_empty() {
        return;
    }
    let host_accesses: Vec<_> = due
        .iter()
        .map(|&i| (anims.entries[i].staging_id, HostAccessType::Write))
        .collect();
    vk.resources.flight(vk.flight_id).wait(None).unwrap();
    unsafe {
        vulkano_taskgraph::execute(
            &vk.queue,
            &vk.resources,
            vk.flight_id,
            |cbf, tcx| {
                for &i in &due {
                    let entry = &anims.entries[i];
                    let bytes = &entry.frames[entry.current];
                    tcx.try_write_buffer::<[u8]>(
                        entry.staging_id,
                        0..bytes.len() as u64,
                    )?
                    .copy_from_slice(bytes);
                    cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                        src_buffer: entry.staging_id,
                        dst_image: entry.image_id,
                        ..Default::default()
                    });
                }
                Ok(())
            },
            host_accesses,
            [],
            [],
        )
        .unwrap();
    }
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

/// Same as [`sampled_image_index`] but for samplers. `SamplerId` is generated
/// by the same `declare_key_with_null!` macro, so it has the identical
/// `[u32; 2]` layout (`transmute` fails to compile if that ever changes size).
pub fn sampler_index(id: SamplerId) -> u32 {
    let raw: [u32; 2] = unsafe { std::mem::transmute(id) };
    raw[0]
}
