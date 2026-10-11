//! Material loading helpers from glTF.

use crate::gltf_utils::gltf_ext::texture::{texture_transform_to_affine2, SamplerSettings};
use gltf::Material;

/// Alpha mode for materials
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    Opaque,
    Mask(f32),
    Blend,
}

/// Returns the alpha mode for a material.
pub fn alpha_mode(material: &Material) -> AlphaMode {
    match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask(material.alpha_cutoff().unwrap_or(0.5)),
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    }
}

/// Returns the UV channel for a texture.
pub fn uv_channel(material: &Material, texture_kind: &str, tex_coord: u32) -> u32 {
    match tex_coord {
        0 => 0,
        1 => 1,
        _ => {
            let material_name = material
                .name()
                .map(|n| format!("the material \"{n}\""))
                .unwrap_or_else(|| "an unnamed material".to_string());
            let material_index = material
                .index()
                .map(|i| format!("index {i}"))
                .unwrap_or_else(|| "default".to_string());
            bevy_log::warn!(
                "Only 2 UV Channels are supported, but {material_name} ({material_index}) \
                has the TEXCOORD attribute {} on texture kind {texture_kind}, which will fallback to 0.",
                tex_coord,
            );
            0
        }
    }
}

/// Returns true if the material needs mesh tangents in order to be successfully rendered.
/// We generate them if this function returns true.
pub fn needs_tangents(material: &Material) -> bool {
    material.normal_texture().is_some()
}

/// UV transform for texture coordinates
#[derive(Debug, Clone, Copy, Default)]
pub struct UvTransform {
    pub offset: [f32; 2],
    pub scale: [f32; 2],
    pub rotation: f32,
}

/// Parses the texture transform from a glTF texture info.
/// Note: KHR_texture_transform is not available in glTF 0.16, returns default.
pub fn parse_texture_transform(_info: &gltf::texture::Info) -> UvTransform {
    UvTransform::default()
}

/// Warns if a material has differing texture transforms on non-base-color textures.
/// Note: KHR_texture_transform is not available in glTF 0.16, no-op.
pub fn warn_on_differing_texture_transforms(
    _material: &Material,
    _info: &gltf::texture::Info,
    _texture_kind: &str,
) {
    // KHR_texture_transform not available in glTF 0.16
}