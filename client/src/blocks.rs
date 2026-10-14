//! Client-side block registry resource.
//!
//! The canonical [`BlockRegistry`] lives in `exsolis_common`; this module
//! only owns the Bevy resource wrapper around it so every client system
//! resolves names <-> ids against the same table.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use exsolis_common::{BlockId, BlockRegistry};

/// The client's block registry. Starts from the frozen M0 ids and extends
/// with `minecraft:*` names scanned from the MC asset tree (same scan as
/// the server, same sort => same numeric ids on both sides).
#[derive(Resource, Clone)]
pub struct ClientBlocks(pub BlockRegistry);

/// Solidity/transparency heuristic for MC block names. Mirrors
/// `server/src/blocks.rs::mc_flags` — keep the two in sync (a registry-sync
/// message replaces both once mod blocks go dynamic).
pub fn mc_flags(name: &str) -> (bool, bool) {
    let b = name.trim_start_matches("minecraft:");
    if matches!(b, "air" | "cave_air" | "void_air" | "barrier" | "light") {
        return (false, true);
    }
    if b.contains("water") || b.contains("bubble_column") {
        return (false, true);
    }
    for key in [
        "glass", "leaves", "ice", "slime", "honey", "sapling", "flower", "crop", "cross",
        "torch", "rail", "ladder", "vine", "fern", "mushroom", "cactus", "sugar_cane",
        "lily", "scaffolding", "lantern", "chain", "bars", "pane", "door", "trapdoor",
        "fence", "wall", "sign", "banner", "grate", "bulb", "azalea", "allium", "tulip",
        "daisy", "poppy", "dandelion", "orchid", "bluet", "cornflower", "roots", "sprouts",
        "fungus", "wart", "melon_stem", "pumpkin_stem", "candle", "cake", "carpet",
        "button", "lever", "hook", "plate", "snow", "carpet", "moss",
    ] {
        if b.contains(key) {
            return (true, true);
        }
    }
    (true, false)
}

impl ClientBlocks {
    /// Numeric id for a block name, if registered.
    pub fn id(&self, name: &str) -> Option<BlockId> {
        self.0.id(name)
    }

    /// Block name for a numeric id, if known.
    pub fn name(&self, id: BlockId) -> Option<&str> {
        self.0.def(id).map(|d| d.name.as_str())
    }
}

pub struct BlocksPlugin;

impl Plugin for BlocksPlugin {
    fn build(&self, app: &mut App) {
        let mut registry = BlockRegistry::default_registry();
        match crate::loader::mc_pack_dir() {
            Some(dir) => {
                let names = crate::loader::scan_mc_block_names(&dir);
                let mut added = 0;
                for name in names {
                    if registry.id(&name).is_none() {
                        let (solid, transparent) = mc_flags(&name);
                        let be = name.ends_with("chest")
                            || name.ends_with("barrel")
                            || name.ends_with("shulker_box");
                        registry.register_mod(name, solid, transparent, be);
                        added += 1;
                    }
                }
                bevy_log::info!(
                    "ClientBlocks: registry holds {} blocks ({} from {})",
                    registry.len(),
                    added,
                    dir.display()
                );
            }
            None => {
                bevy_log::info!("ClientBlocks: no MC pack; 11 built-in blocks only");
            }
        }
        app.insert_resource(ClientBlocks(registry));
    }
}
