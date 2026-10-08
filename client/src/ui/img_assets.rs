//! Disk side of `<img>`: where files live and how they are decoded.
//!
//! `src` values are validated by `parse::is_valid_img_src` (relative, no
//! `..`, png/jpg/jpeg only) before they ever reach this module.

use std::path::PathBuf;

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
pub fn probe_size(src: &str) -> Option<(u32, u32)> {
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
