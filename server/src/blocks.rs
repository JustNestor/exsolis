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
        // Thaumcraft bbmodel props: mirrors the client scan exactly
        // (same dir shape, same sort => same ids). Will move to a
        // registry-sync message with dynamic mod blocks.
        if let Some(tc) = tc_pack_dir() {
            let mut added = 0;
            for name in scan_tc_block_names(&tc) {
                if registry.id(&name).is_none() {
                    registry.register_mod(name, true, false, false);
                    added += 1;
                }
            }
            bevy_log::info!("ServerBlocks: +{added} thaumcraft props from {}", tc.display());
        }
        // Generic multiblock parts + layouts and items. Same generation
        // rule and same order as the client => same ids.
        let known: Vec<String> = registry.iter().map(|(_, d)| d.name.clone()).collect();
        let mut parts = 0;
        for (master, part, offset) in multiblock_parts(&known) {
            let flags = registry
                .def(registry.id(&master).unwrap_or(0))
                .map(|d| (d.solid, d.transparent))
                .unwrap_or((true, false));
            registry.register_part(part.clone(), flags.0, flags.1);
            registry.set_multiblock(
                &master,
                exsolis_common::MultiblockLayout {
                    parts: vec![exsolis_common::MultiblockPart { offset, part }],
                },
            );
            parts += 1;
        }
        let mut items = 0;
        if let Some(dir) = mc_pack_dir() {
            for name in scan_item_names(&dir) {
                if registry.id(&name).is_none() {
                    registry.register_item(name);
                    items += 1;
                }
            }
        }
        if parts + items > 0 {
            bevy_log::info!("ServerBlocks: +{parts} multiblock parts, +{items} items");
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

/// Thaumcraft prop dir: `EXSOLIS_TC_ASSETS` or `assets/models/thaumcraft`.
/// Same mirror rule as the MC pack (client reads the same relative dir).
fn tc_pack_dir() -> Option<std::path::PathBuf> {
    if let Some(dir) = std::env::var_os("EXSOLIS_TC_ASSETS").map(std::path::PathBuf::from) {
        return dir.is_dir().then_some(dir);
    }
    let dev = std::path::PathBuf::from("assets/models/thaumcraft");
    dev.is_dir().then_some(dev)
}

fn scan_tc_block_names(pack_dir: &std::path::Path) -> Vec<String> {
    let mut names = Vec::new();
    let Ok(entries) = std::fs::read_dir(pack_dir) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bbmodel") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            names.push(format!("thaumcraft:{stem}"));
        }
    }
    names.sort();
    names
}

fn scan_item_names(pack_dir: &std::path::Path) -> Vec<String> {
    let mut names = Vec::new();
    let dir = pack_dir
        .join("assets")
        .join("minecraft")
        .join("models")
        .join("item");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            // Same template rule as the client icon scan => same ids.
            if exsolis_common::mc_item_template(stem) {
                continue;
            }
            names.push(format!("minecraft:{stem}"));
        }
    }
    names.sort();
    names
}

fn multiblock_parts(names: &[String]) -> Vec<(String, String, [i32; 3])> {
    let mut out = Vec::new();
    for name in names {
        let Some(base) = name.strip_prefix("minecraft:") else {
            continue;
        };
        if base.ends_with("_door") {
            out.push((name.clone(), format!("{name}_top"), [0, 1, 0]));
        } else if base.ends_with("_bed") {
            out.push((name.clone(), format!("{name}_head"), [0, 0, -1]));
        }
    }
    out
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
