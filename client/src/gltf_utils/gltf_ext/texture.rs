//! Texture loading helpers from glTF.

use gltf::texture::{Info, MagFilter, MinFilter, WrappingMode};

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

/// Extracts sampler settings from a glTF texture info.
///
/// The route is `info.texture().sampler()`, not `info.sampler()` — the
/// sampler lives on the `texture` object the *info* references, and several
/// materials may share one texture through separate infos.
///
/// When the texture carries no sampler, glTF specifies these defaults. Note
/// that `magFilter`/`minFilter` genuinely default (they are `Option`s) while
/// the wrapping modes already fall back inside the gltf crate.
pub fn texture_sampler(info: &Info) -> SamplerSettings {
    let sampler = info.texture().sampler();
    SamplerSettings {
        mag_filter: sampler.mag_filter().unwrap_or(MagFilter::Linear),
        min_filter: sampler
            .min_filter()
            .unwrap_or(MinFilter::LinearMipmapLinear),
        wrap_s: sampler.wrap_s(),
        wrap_t: sampler.wrap_t(),
    }
}

/// Converts a glTF texture transform to an affine2 matrix.
///
/// Always `None` on gltf 0.16: `KHR_texture_transform` is not exposed by that
/// version of the crate, so honouring it needs a newer `gltf`. Returning
/// `None` keeps call sites honest about that rather than silently pretending
/// the UVs are untransformed.
pub fn texture_transform_to_affine2(_info: &Info) -> Option<[[f32; 2]; 3]> {
    None
}

/// Returns true when the sampler uses nearest-neighbour magnification.
///
/// glTF's `magFilter` defaults to Linear, so a texture with no sampler is
/// *not* nearest. The first generation of this helper assumed the opposite;
/// only an explicit `Nearest` opts in.
pub fn is_mag_nearest(info: &Info) -> bool {
    matches!(
        info.texture().sampler().mag_filter(),
        Some(MagFilter::Nearest)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_gltf_spec() {
        // glTF 2.0 §5.27: a missing sampler has exactly these values.
        let d = SamplerSettings::default();
        assert_eq!(d.mag_filter, MagFilter::Linear);
        assert_eq!(d.min_filter, MinFilter::LinearMipmapLinear);
        assert_eq!(d.wrap_s, WrappingMode::Repeat);
        assert_eq!(d.wrap_t, WrappingMode::Repeat);
    }
}
