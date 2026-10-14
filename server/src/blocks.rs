//! Server-side block registry resource.
//!
//! The canonical [`BlockRegistry`] lives in `exsolis_common`; this module
//! only owns the Bevy resource wrapper around it so every server system
//! resolves names <-> ids against the same table.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use exsolis_common::{BlockId, BlockRegistry};

/// The server's block registry. Inserted once at startup with the frozen
/// M0 ids (`BlockRegistry::default_registry()`); mod blocks register here
/// later.
#[derive(Resource, Clone)]
pub struct ServerBlocks(pub BlockRegistry);

impl ServerBlocks {
    /// Numeric id for a block name, if registered.
    pub fn id(&self, name: &str) -> Option<BlockId> {
        self.0.id(name)
    }

    /// Block name for a numeric id, if known.
    pub fn name(&self, id: BlockId) -> Option<&str> {
        self.0.def(id).map(|d| d.name.as_str())
    }

    pub fn is_air(&self, id: BlockId) -> bool {
        self.0.is_air(id)
    }

    pub fn is_block_entity(&self, id: BlockId) -> bool {
        self.0.is_block_entity(id)
    }
}

pub struct BlocksPlugin;

impl Plugin for BlocksPlugin {
    fn build(&self, app: &mut App) {
        let mut registry = BlockRegistry::default_registry();
        match mc_pack_dir() {
            Some(dir) => {
                let names = scan_mc_block_names(&dir);
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
                    "ServerBlocks: registry holds {} blocks ({} from {})",
                    registry.len(),
                    added,
                    dir.display()
                );
            }
            None => {
                bevy_log::info!("ServerBlocks: no MC pack; 11 built-in blocks only");
            }
        }
        app.insert_resource(ServerBlocks(registry));
    }
}

/// Same scan shape as the client (`client/src/loader`): same dir + same
/// sort => same numeric ids. Replaced by a registry-sync message later.
fn mc_pack_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("EXSOLIS_MC_ASSETS").map(std::path::PathBuf::from) {
        return dir.is_dir().then_some(dir);
    }
    let dev = std::path::PathBuf::from("temp/minecraft");
    dev.is_dir().then_some(dev)
}

fn scan_mc_block_names(pack_dir: &std::path::Path) -> Vec<String> {
    let mut names = Vec::new();
    let dir = pack_dir.join("assets").join("minecraft").join("blockstates");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            names.push(format!("minecraft:{stem}"));
        }
    }
    names.sort();
    names
}

/// Solidity/transparency heuristic. MUST match `client/src/blocks.rs::mc_flags`.
fn mc_flags(name: &str) -> (bool, bool) {
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
