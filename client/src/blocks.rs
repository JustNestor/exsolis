//! Client-side block registry resource.
//!
//! The canonical [`BlockRegistry`] lives in `exsolis_common`; this module
//! only owns the Bevy resource wrapper around it so every client system
//! resolves names <-> ids against the same table.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use exsolis_common::{BlockAnimCatalog, BlockId, BlockRegistry};

/// The client's block registry. Starts from the frozen M0 ids and extends
/// with `minecraft:*` names scanned from the MC asset tree (same scan as
/// the server, same sort => same numeric ids on both sides).
#[derive(Resource, Clone)]
pub struct ClientBlocks(pub BlockRegistry);

/// The client's block animation catalog, parsed from
/// `assets/animations/blocks.json` (same file the server reads; see
/// [`BlockAnimCatalog::from_json`]).
#[derive(Resource, Debug, Clone, Default)]
pub struct ClientAnimCatalog {
    pub sets: BlockAnimCatalog,
}

impl ClientAnimCatalog {
    /// The animation set for a block, if the catalog declares one.
    pub fn get(&self, block: &str) -> Option<&exsolis_common::BlockAnimSet> {
        self.sets.get(block)
    }
}

/// Dev asset dir for the block animation catalog.
pub fn block_animations_path() -> std::path::PathBuf {
    std::path::PathBuf::from("assets/animations/blocks.json")
}

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
        // Thaumcraft bbmodel props (static meshes for now): same scan as
        // the server, same sort => same ids.
        if let Some(tc) = crate::loader::tc_pack_dir() {
            let mut added = 0;
            for name in crate::loader::scan_tc_block_names(&tc) {
                if registry.id(&name).is_none() {
                    registry.register_mod(name, true, false, false);
                    added += 1;
                }
            }
            bevy_log::info!("ClientBlocks: +{added} thaumcraft props from {}", tc.display());
        }
        // Generic multiblock parts + layouts (doors, beds) and items.
        // Same generation rule as the server => same ids.
        let known: Vec<String> = registry.iter().map(|(_, d)| d.name.clone()).collect();
        let mut parts = 0;
        for (master, part, offset) in crate::loader::multiblock_parts(&known) {
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
        if let Some(dir) = crate::loader::mc_pack_dir() {
            for name in crate::loader::scan_item_names(&dir) {
                if registry.id(&name).is_none() {
                    registry.register_item(name);
                    items += 1;
                }
            }
        }
        // Orientation parts (stairs facings, slab halves, pillar axes…):
        // hidden, drop their master. Same rule + same order both sides.
        let known2: Vec<String> = registry.iter().map(|(_, d)| d.name.clone()).collect();
        let mut oparts = 0;
        for master in &known2 {
            for (part, _, _) in exsolis_common::orientation_parts(master) {
                if registry.id(&part).is_some() {
                    continue;
                }
                let flags = registry
                    .def(registry.id(master).unwrap_or(0))
                    .map(|d| (d.solid, d.transparent))
                    .unwrap_or((true, false));
                registry.register_opart(part, master.clone(), flags.0, flags.1);
                oparts += 1;
            }
        }
        if parts + items + oparts > 0 {
            bevy_log::info!(
                "ClientBlocks: +{parts} multiblock parts, +{oparts} orientation parts, +{items} items"
            );
        }
        app.insert_resource(ClientBlocks(registry));
        // Block animation sets (doors, beds, pistons). Same file the server
        // reads; absent file => empty catalog and no block animations.
        let anims = std::fs::read_to_string(block_animations_path())
            .ok()
            .and_then(|text| BlockAnimCatalog::from_json(&text).ok())
            .unwrap_or_default();
        bevy_log::info!("ClientAnimCatalog: {} block anim sets", anims.sets.len());
        app.insert_resource(ClientAnimCatalog { sets: anims });
    }
}
