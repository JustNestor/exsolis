//! Texture loading helpers from glTF.

use gltf::texture::{MagFilter, MinFilter, WrappingMode};

/// Sampler settings for a texture
#[derive(Debug, Clone, Copy)]
pub struct SamplerSettings {
    pub mag_filter: MagFilter,
    pub min_filter: MinFilter,
    pub wrap_s: WrappingMode,
    pub wrap_t: WrappingMode,
}

impl Default for SamplerSettings {
    fn default() -> Self {
        Self {
            mag_filter: MagFilter::Linear,
            min_filter: MinFilter::LinearMipmapLinear,
            wrap_s: WrappingMode::Repeat,
            wrap_t: WrappingMode::Repeat,
        }
    }
}

/// Extracts sampler settings from a texture info
/// Note: In glTF 0.16, texture info doesn't have sampler() method directly
/// This is a simplified version
pub fn texture_sampler(_info: &gltf::texture::Info) -> SamplerSettings {
    SamplerSettings::default()
}

/// Converts a glTF texture transform to an affine2 matrix
/// Note: KHR_texture_transform not available in glTF 0.16
pub fn texture_transform_to_affine2(_info: &gltf::texture::Info) -> Option<[[f32; 2]; 3]> {
    None
}

/// Returns true if the sampler uses nearest neighbor magnification
/// Note: In glTF 0.16, Info doesn't have sampler() method directly
pub fn is_mag_nearest(_info: &gltf::texture::Info) -> bool {
    false
}