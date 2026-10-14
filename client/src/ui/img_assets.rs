//! Disk side of `<img>`: where files live and how they are decoded.
//!
//! `src` values are validated by `parse::is_valid_img_src` (relative, no
//! `..`, png/jpg/jpeg only) before they ever reach this module. The
//! synthetic `block#namespace:name` form never touches disk: `probe_size`
//! reports the fixed 48x48 preview size and [`decode_block`] rasterizes the
//! block's baked model on the CPU (see `ui::block_preview` for the resolver
//! mapping `minecraft/block/stone` ->
//! `<pack>/assets/minecraft/textures/block/stone.png`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::loader::baked::BakedModel;

/// Sanity cap so a hostile/huge file can't eat VRAM.
pub const MAX_IMAGE_DIM: u32 = 4096;

pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8, straight (non-premultiplied) alpha.
    pub rgba: Vec<u8>,
}

/// UI images are read from the client cache populated by server downloads.
pub fn path_for(src: &str) -> PathBuf {
    crate::assets::asset_directory().join("ui").join(src)
}

/// Header-only read: cheap enough to call while spawning a screen.
/// `block#` previews are a fixed 48x48 and never touch disk.
pub fn probe_size(src: &str) -> Option<(u32, u32)> {
    if crate::ui::parse::is_block_preview_src(src) {
        return Some((
            crate::ui::block_preview::PREVIEW_SIZE,
            crate::ui::block_preview::PREVIEW_SIZE,
        ));
    }
    let (w, h) = ::image::image_dimensions(path_for(src)).ok()?;
    (w > 0 && h > 0 && w <= MAX_IMAGE_DIM && h <= MAX_IMAGE_DIM).then_some((w, h))
}

/// Full decode to RGBA8.
pub fn decode(src: &str) -> Result<DecodedImage, String> {
    let img = ::image::open(path_for(src)).map_err(|e| e.to_string())?;
    let (width, height) = (img.width(), img.height());
    if width == 0 || height == 0 || width > MAX_IMAGE_DIM || height > MAX_IMAGE_DIM {
        return Err(format!("{width}x{height} is outside 1..={MAX_IMAGE_DIM}"));
    }
    Ok(DecodedImage {
        width,
        height,
        rgba: img.into_rgba8().into_raw(),
    })
}

/// Rasterize a `block#<name>` preview (see [`probe_size`]). Thin wrapper so
/// callers only touch `img_assets`; the rasterizer lives in `block_preview`.
pub fn decode_block(
    name: &str,
    baked: &HashMap<String, BakedModel>,
    pack_dir: &Path,
    embedded: &HashMap<String, (u32, u32, Vec<u8>)>,
) -> Result<DecodedImage, String> {
    super::block_preview::decode_block(name, baked, pack_dir, embedded)
}
