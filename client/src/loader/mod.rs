//! Client-side voxel model loading (modular per-game packs).
//!
//! The engine mesher only sees [`baked::BakedModel`]. Each source game
//! implements [`baked::BlockPack`]: [`mcmodel::McPack`] for latest-MC JSON
//! (blockstates + models + mcmeta) plus the minimal OBJ loader in [`obj`].

pub mod baked;
pub mod mcmodel;
pub mod obj;

pub use baked::{BakedModel, BakedQuad, BakedTri, BlockPack, RenderType, face_quad};
pub use mcmodel::{AnimInfo, LoadReport, McPack};
pub use obj::{ObjMesh, parse_mtl, parse_obj};

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

/// Env var pointing at an extracted MC asset tree
/// (`<root>/assets/minecraft/...`). Defaults to `<cwd>/temp/minecraft`
/// when that directory exists (dev workflow).
pub fn mc_pack_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("EXSOLIS_MC_ASSETS").map(PathBuf::from) {
        return dir.is_dir().then_some(dir);
    }
    let dev = PathBuf::from("temp/minecraft");
    dev.is_dir().then_some(dev)
}

/// Scan `<pack>/assets/minecraft/blockstates/*.json` stems, sorted.
/// Shared scan shape with the server (`server/src/blocks.rs`): same dir +
/// same sort => same ids on both sides. Any divergence is logged loudly.
pub fn scan_mc_block_names(pack_dir: &std::path::Path) -> Vec<String> {
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

/// Pre-baked MC models, shared with meshing tasks behind an `Arc`.
#[derive(Resource, Clone, Default)]
pub struct McAssets {
    pub baked: Arc<HashMap<String, BakedModel>>,
    pub pack_dir: Option<PathBuf>,
    pub animated: usize,
}

impl McAssets {
    pub fn model_for<'a>(
        &'a self,
        name: &'a str,
    ) -> Option<(&'a BakedModel, &'a str)> {
        Self::model_for_baked(&self.baked, name).map(|m| (m, name))
    }

    /// Map lookup with legacy `exsolis:*` -> `minecraft:*` fallback.
    /// Separate from `model_for` so the mesher (which only holds the
    /// baked map) can use it without the full resource.
    pub fn model_for_baked<'a>(
        baked: &'a HashMap<String, BakedModel>,
        name: &str,
    ) -> Option<&'a BakedModel> {
        if let Some(m) = baked.get(name) {
            return Some(m);
        }
        legacy_mc_name(name).and_then(|mc| baked.get(mc))
    }
}

/// `exsolis:*` terrain blocks -> their MC look-alikes.
fn legacy_mc_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "exsolis:stone" => "minecraft:stone",
        "exsolis:dirt" => "minecraft:dirt",
        "exsolis:grass" => "minecraft:grass_block",
        "exsolis:sand" => "minecraft:sand",
        "exsolis:water" => "minecraft:water",
        "exsolis:snow" => "minecraft:snow_block",
        "exsolis:glass" => "minecraft:glass",
        "exsolis:log" => "minecraft:oak_log",
        "exsolis:leaves" => "minecraft:oak_leaves",
        "exsolis:chest" => "minecraft:chest",
        _ => return None,
    })
}

pub struct McAssetsPlugin;

impl Plugin for McAssetsPlugin {
    fn build(&self, app: &mut App) {
        let mut assets = McAssets::default();
        match mc_pack_dir() {
            Some(dir) => {
                match McPack::load(&dir, "minecraft") {
                    Ok(pack) => {
                        let (models, report) = pack.bake_all();
                        bevy_log::info!(
                            "McPack: baked {}/{} blocks ({} stubs, {} animated; opaque={} cutout={} translucent={}) from {}",
                            report.blocks_baked,
                            report.blocks_total,
                            report.blocks_fallback,
                            report.animated_textures,
                            report.opaque,
                            report.cutout,
                            report.translucent,
                            dir.display()
                        );
                        for w in report.warnings.iter().take(12) {
                            bevy_log::warn!("McPack: {w}");
                        }
                        if report.warnings.len() > 12 {
                            bevy_log::warn!(
                                "McPack: ... and {} more warnings",
                                report.warnings.len() - 12
                            );
                        }
                        // Legacy look-alike coverage: the terrain still emits
                        // exsolis:* ids, so every legacy name must resolve.
                        for legacy in [
                            "exsolis:stone",
                            "exsolis:dirt",
                            "exsolis:grass",
                            "exsolis:sand",
                            "exsolis:water",
                            "exsolis:snow",
                            "exsolis:glass",
                            "exsolis:log",
                            "exsolis:leaves",
                            "exsolis:chest",
                        ] {
                            let mc = legacy_mc_name(legacy).unwrap();
                            if !assets.baked.contains_key(mc) {
                                bevy_log::warn!(
                                    "McPack: legacy {legacy} has no baked {mc}"
                                );
                            }
                        }
                        assets.animated = report.animated_textures;
                        // `bake_all` keys are bare stems; the mesher looks up
                        // full `namespace:name` ids.
                        assets.baked = Arc::new(
                            models
                                .into_iter()
                                .map(|(k, v)| (format!("minecraft:{k}"), v))
                                .collect(),
                        );
                        assets.pack_dir = Some(dir);
                    }
                    Err(e) => {
                        bevy_log::warn!("McPack: load failed ({e}); vanilla cubes only");
                    }
                }
            }
            None => {
                bevy_log::warn!(
                    "McPack: no asset tree (set EXSOLIS_MC_ASSETS or keep temp/minecraft/); vanilla cubes only"
                );
            }
        }
        app.insert_resource(assets);
    }
}
