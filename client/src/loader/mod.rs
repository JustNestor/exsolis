//! Client-side voxel model loading (modular per-game packs).
//!
//! The engine mesher only sees [`baked::BakedModel`]. Each source game
//! implements [`baked::BlockPack`]: [`mcmodel::McPack`] for latest-MC JSON
//! (blockstates + models + mcmeta) plus the minimal OBJ loader in [`obj`].

pub mod baked;
pub mod bbmodel;
pub mod mcmodel;
pub mod obj;

pub use baked::{BakedModel, BakedQuad, BakedTri, BlockPack, RenderType, face_quad, rotate_point};
pub use bbmodel::{BbAnimInfo, BbLoadReport, BbModel, BbTextureImage, load_bbmodel, parse_bbmodel};
pub use mcmodel::{AnimInfo, LoadReport, McPack};
pub use obj::{ObjMesh, parse_mtl, parse_obj};

use std::collections::HashMap;
use std::collections::HashSet;
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

/// Env var pointing at Blockbench `.bbmodel` props
/// (`<root>/*.bbmodel`). Defaults to `assets/models/thaumcraft` (dev).
pub fn tc_pack_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("EXSOLIS_TC_ASSETS").map(PathBuf::from) {
        return dir.is_dir().then_some(dir);
    }
    let dev = PathBuf::from("assets/models/thaumcraft");
    dev.is_dir().then_some(dev)
}

/// Scan `<dir>/*.bbmodel` stems, sorted (registry ids; mirrors server).
pub fn scan_tc_block_names(pack_dir: &std::path::Path) -> Vec<String> {
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

/// Item names from `<pack>/assets/minecraft/models/item/*.json`, sorted.
/// Unknown ones register as items (unified id space, unplaceable).
pub fn scan_item_names(pack_dir: &std::path::Path) -> Vec<String> {
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
            if exsolis_common::mc_item_template(stem) {
                continue;
            }
            names.push(format!("minecraft:{stem}"));
        }
    }
    names.sort();
    names
}

/// Generic multiblock masters present in `names`: doors (upper half above)
/// and beds (head north of foot). Returns `(master, part, north-offset)`.
/// MUST stay in sync with `McPack::bake_parts` part stems.
pub fn multiblock_parts(names: &[String]) -> Vec<(String, String, [i32; 3])> {
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
#[derive(Resource, Clone, Default)]
pub struct McAssets {
    pub baked: Arc<HashMap<String, BakedModel>>,
    pub pack_dir: Option<PathBuf>,
    pub animated: usize,
    /// `.mcmeta` animation metadata (texture key -> frames). The texture
    /// cycler (`texture.rs::BlockAnims`) consumes this after upload.
    pub anims: Vec<AnimInfo>,
    /// Texture keys used by at least one `tintindex` quad (grass, foliage).
    /// MC ships these grayscale; both the world upload and the UI previews
    /// multiply them by the plains biome tint. Single tint for all foliage
    /// (oak/birch/spruce differ slightly in vanilla) — documented M-limitation.
    pub tinted_textures: HashSet<String>,
    /// Embedded bbmodel textures (`thaumcraft/<model>/<tex>` -> pixels),
    /// uploaded by `setup_textures` and served to `block#` previews.
    pub embedded: Arc<HashMap<String, (u32, u32, Vec<u8>)>>,
    /// Item icons (`minecraft:stick` -> `minecraft/item/stick` texture key).
    /// Served to `item#` previews as flat blits.
    pub item_icons: Arc<HashMap<String, String>>,
}

/// Plains biome tint (MC grass colormap): #91BD59.
pub const GRASS_TINT: [u8; 3] = [145, 189, 89];

/// Multiply straight-alpha RGBA pixels by the biome tint (alpha untouched).
pub fn apply_biome_tint(px: &mut [u8]) {
    for p in px.chunks_exact_mut(4) {
        for (c, t) in p.iter_mut().take(3).zip(GRASS_TINT.iter()) {
            *c = ((*c as u16 * *t as u16) / 255) as u8;
        }
    }
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

/// `.bbmodel` files in a prop dir, sorted for deterministic logs.
fn sorted_bbmodels(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("bbmodel") {
            out.push(path);
        }
    }
    out.sort();
    out
}

/// Rewrite a bbmodel bake's texture keys into `thaumcraft/<stem>/<tex>`.
fn retarget_bb_textures(mut model: BakedModel, stem: &str) -> BakedModel {
    for q in &mut model.quads {
        q.texture = format!("thaumcraft/{stem}/{}", q.texture);
    }
    for t in &mut model.tris {
        t.texture = format!("thaumcraft/{stem}/{}", t.texture);
    }
    model
}

pub struct McAssetsPlugin;

impl Plugin for McAssetsPlugin {
    fn build(&self, app: &mut App) {
        let mut assets = McAssets::default();
        match mc_pack_dir() {
            Some(dir) => {
                match McPack::load(&dir, "minecraft") {
                    Ok(pack) => {
                        let (models, mut report) = pack.bake_all();
                        // `bake_all` keys are bare stems; the mesher looks up
                        // full `namespace:name` ids.
                        let mut baked: HashMap<String, BakedModel> = models
                            .into_iter()
                            .map(|(k, v)| (format!("minecraft:{k}"), v))
                            .collect();
                        // Hidden multiblock parts (door tops, bed heads).
                        let mut part_count = 0;
                        for (stem, model) in pack.bake_parts(&mut report) {
                            baked.insert(format!("minecraft:{stem}"), model);
                            part_count += 1;
                        }
                        // Hidden orientation parts (stairs, slabs, pillars…).
                        let mut oparts = 0;
                        for (name, model) in pack.bake_orientation(&mut report) {
                            baked.insert(name, model);
                            oparts += 1;
                        }
                        if part_count + oparts > 0 {
                            bevy_log::info!(
                                "McPack: +{part_count} multiblock parts, +{oparts} orientation parts"
                            );
                        }
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
                            if !baked.contains_key(mc) {
                                bevy_log::warn!(
                                    "McPack: legacy {legacy} has no baked {mc}"
                                );
                            }
                        }
                        assets.animated = report.animated_textures;
                        assets.anims = pack.anim_textures().cloned().collect();
                        // Tinted texture set: any key behind a tintindex quad.
                        let mut tinted = HashSet::new();
                        for model in baked.values() {
                            for q in &model.quads {
                                if q.tinted {
                                    tinted.insert(q.texture.clone());
                                }
                            }
                        }
                        if !tinted.is_empty() {
                            bevy_log::info!(
                                "McPack: {} grayscale textures use biome tint",
                                tinted.len()
                            );
                        }
                        assets.tinted_textures = tinted;
                        // Thaumcraft bbmodel props: bake + merge under
                        // `thaumcraft:<stem>`, rewriting texture keys to
                        // `thaumcraft/<stem>/<tex>` (per-model namespace:
                        // several models call their texture "texture").
                        if let Some(tc) = tc_pack_dir() {
                            let mut count = 0;
                            for path in sorted_bbmodels(&tc) {
                                let stem = path
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("prop")
                                    .to_string();
                                match load_bbmodel(&path) {
                                    Ok((mut model, report)) => {
                                        let key = format!("thaumcraft:{stem}");
                                        let baked_model = std::mem::take(&mut model.baked);
                                        let baked_model =
                                            retarget_bb_textures(baked_model, &stem);
                                        for img in &model.textures {
                                            let tkey =
                                                format!("thaumcraft/{stem}/{}", img.name);
                                            Arc::get_mut(&mut assets.embedded)
                                                .expect("sole owner at startup")
                                                .insert(
                                                    tkey,
                                                    (img.width, img.height, img.rgba.clone()),
                                                );
                                        }
                                        baked.insert(key, baked_model);
                                        count += 1;
                                        bevy_log::info!(
                                            "TcPack: {stem}: {} quads, {} tris, {} embedded tex, {} anims",
                                            report.quads,
                                            report.tris,
                                            model.textures.len(),
                                            report.animations.len()
                                        );
                                        for w in report.warnings.iter().take(4) {
                                            bevy_log::warn!("TcPack {stem}: {w}");
                                        }
                                    }
                                    Err(e) => {
                                        bevy_log::warn!("TcPack: {}: {e}", path.display());
                                    }
                                }
                            }
                            if count > 0 {
                                bevy_log::info!("TcPack: {count} prop models from {}", tc.display());
                            }
                        }
                        assets.baked = Arc::new(baked);
                        assets.pack_dir = Some(dir.clone());
                        assets.item_icons = Arc::new(pack.item_icons());
                        if !assets.item_icons.is_empty() {
                            bevy_log::info!(
                                "McPack: {} item icons",
                                assets.item_icons.len()
                            );
                        }
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
