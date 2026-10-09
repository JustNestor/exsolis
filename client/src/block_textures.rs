//! Client-side texture knowledge about blocks.
//!
//! `BlockType` lives in `exsolis_common` (no GPU/asset deps, shared with the
//! server), so it can't grow an inherent `texture_mapping()` like the old
//! single-crate project had. An extension trait gives the same call syntax
//! (`block.texture_mapping()`) without polluting the common crate.
//!
//! The string keys are asset paths relative to `assets/`, without the `.png`
//! extension, using `/` separators (see `TextureManager::discover_pngs`).
//! e.g. `assets/blocks/stone.png` -> key `"blocks/stone"`.
//!
//! TODO(nestor): these keys are my best guess — I haven't seen your `assets/`
//! tree. Adjust the strings to match your real filenames. A wrong key does
//! NOT crash: `TextureManager::get` falls back to the first loaded texture,
//! so it renders as the wrong texture (which is easy to spot).

use exsolis_common::BlockType;

use crate::texture::TextureMapping;

pub trait BlockTextures {
    /// `None` for blocks that are never drawn (air).
    fn texture_mapping(self) -> Option<TextureMapping>;

    /// Needs a face-visibility check against neighbours: true for anything
    /// you can see through (air, water, glass, leaves).
    fn is_transparent(self) -> bool;
}

impl BlockTextures for BlockType {
    fn texture_mapping(self) -> Option<TextureMapping> {
        Some(match self {
            BlockType::Air => return None,
            BlockType::Stone => TextureMapping::Single("blocks/stone"),
            BlockType::Dirt => TextureMapping::Single("blocks/dirt"),
            BlockType::Grass => TextureMapping::TopBottomSide {
                top: "blocks/grass_top",
                bottom: "blocks/dirt",
                side: "blocks/grass_side",
            },
            BlockType::Sand => TextureMapping::Single("blocks/sand"),
            BlockType::Water => TextureMapping::Single("blocks/water"),
            BlockType::Snow => TextureMapping::Single("blocks/snow"),
            BlockType::Glass => TextureMapping::Single("blocks/glass"),
            BlockType::Log => TextureMapping::TopBottomSide {
                top: "blocks/log_top",
                bottom: "blocks/log_top",
                side: "blocks/log_side",
            },
            BlockType::Leaves => TextureMapping::Single("blocks/leaves"),
            // No dedicated chest art yet; planks read as a wooden crate and
            // the block entity's container is what actually distinguishes it.
            BlockType::Chest => TextureMapping::Single("blocks/cherry_planks"),
        })
    }

    fn is_transparent(self) -> bool {
        matches!(
            self,
            BlockType::Air | BlockType::Water | BlockType::Glass | BlockType::Leaves
        )
    }
}
