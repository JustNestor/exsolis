//! Client-side texture knowledge about blocks.
//!
//! Block ids are numeric [`BlockId`]s into the shared [`BlockRegistry`]
//! (see `exsolis_common::blocks`); the string keys below are asset paths
//! relative to `assets/`, without the `.png` extension, using `/`
//! separators (see `TextureManager::discover_pngs`).
//! e.g. `assets/blocks/stone.png` -> key `"blocks/stone"`.
//!
//! TODO(nestor): these keys are my best guess — I haven't seen your `assets/`
//! tree. Adjust the strings to match your real filenames. A wrong key does
//! NOT crash: `TextureManager::get` falls back to the first loaded texture,
//! so it renders as the wrong texture (which is easy to spot).

use exsolis_common::{AIR_ID, BlockId, BlockRegistry};

use crate::texture::TextureMapping;

/// `None` for blocks that are never drawn (air, unknown ids without art).
pub fn texture_mapping(registry: &BlockRegistry, id: BlockId) -> Option<TextureMapping> {
    if id == AIR_ID {
        return None;
    }
    let def = registry.def(id)?;
    Some(match def.name.as_str() {
        "exsolis:stone" => TextureMapping::Single("blocks/stone"),
        "exsolis:dirt" => TextureMapping::Single("blocks/dirt"),
        "exsolis:grass" => TextureMapping::TopBottomSide {
            top: "blocks/grass_top",
            bottom: "blocks/dirt",
            side: "blocks/grass_side",
        },
        "exsolis:sand" => TextureMapping::Single("blocks/sand"),
        "exsolis:water" => TextureMapping::Single("blocks/water"),
        "exsolis:snow" => TextureMapping::Single("blocks/snow"),
        "exsolis:glass" => TextureMapping::Single("blocks/glass"),
        "exsolis:log" => TextureMapping::TopBottomSide {
            top: "blocks/log_top",
            bottom: "blocks/log_top",
            side: "blocks/log_side",
        },
        "exsolis:leaves" => TextureMapping::Single("blocks/leaves"),
        // No dedicated chest art yet; planks read as a wooden crate and
        // the block entity's container is what actually distinguishes it.
        "exsolis:chest" => TextureMapping::Single("blocks/cherry_planks"),
        _ => return None,
    })
}

/// Needs a face-visibility check against neighbours: true for anything
/// you can see through (air, water, glass, leaves).
pub fn is_transparent(registry: &BlockRegistry, id: BlockId) -> bool {
    registry.is_transparent(id)
}
