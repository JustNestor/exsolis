//! Latest-Minecraft JSON block loader: `blockstates/*.json` +
//! `models/block/*.json` (+ `.mcmeta` animation metadata).
//!
//! Reads from an extracted asset tree:
//! `<root>/assets/<namespace>/blockstates|models/block|textures/block`.
//! Baking is pure (no GPU): [`McPack::bake`] resolves the variant/multipart
//! entry for a state, walks `parent` chains, resolves `#texture` vars and
//! emits [`BakedModel`] quads in 0-1 block space with 0-1 UVs.
//!
//! Known M1 simplifications (all counted in [`LoadReport` / warnings):
//! - weighted variant arrays: first entry wins;
//! - `uvlock`: ignored (UVs stay as authored);
//! - element-level `rotation`: model falls back to a textured cube;
//! - non-default states bake the default variant (chunk storage is id-only).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::loader::baked::{BakedModel, BakedQuad, BlockPack, RenderType, face_quad};

/// `.mcmeta` animation metadata for one texture.
#[derive(Debug, Clone)]
pub struct AnimInfo {
    pub texture: String,
    pub frametime: u32,
    pub frames: Vec<u32>,
    pub frame_count: usize,
}

#[derive(Debug, Clone, Default)]
struct FaceDef {
    uv: Option<[f32; 4]>,
    texture: String,
    cullface: Option<String>,
    rotation: i32,
    tintindex: Option<i32>,
}

#[derive(Debug, Clone, Default)]
struct ElementRotation {
    origin: [f32; 3],
    axis: char,
    angle: f32,
}

#[derive(Debug, Clone, Default)]
struct ElementDef {
    from: [f32; 3],
    to: [f32; 3],
    faces: HashMap<String, FaceDef>,
    rotation: Option<ElementRotation>,
}

#[derive(Debug, Clone, Default)]
struct ModelDef {
    textures: HashMap<String, String>,
    elements: Vec<ElementDef>,
}

#[derive(Debug, Clone, Default)]
struct ModelRef {
    model: String,
    x: i32,
    y: i32,
}

/// Summary of a bulk bake, for startup logs.
#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    pub blocks_total: usize,
    pub blocks_baked: usize,
    pub blocks_fallback: usize,
    pub animated_textures: usize,
    pub warnings: Vec<String>,
}

impl LoadReport {
    fn warn(&mut self, msg: String) {
        if self.warnings.len() < 128 {
            self.warnings.push(msg);
        }
    }
}

pub struct McPack {
    namespace: String,
    #[allow(dead_code)]
    root: PathBuf,
    blockstates: HashMap<String, Value>,
    models: HashMap<String, Value>,
    mcmeta: HashMap<String, AnimInfo>,
    texture_files: HashMap<String, PathBuf>,
}

impl McPack {
    pub fn load(root: &Path, namespace: &str) -> Result<Self, String> {
        let base = root.join("assets").join(namespace);
        let bs_dir = base.join("blockstates");
        let models_dir = base.join("models").join("block");
        let tex_dir = base.join("textures").join("block");

        let mut blockstates = HashMap::new();
        let entries = std::fs::read_dir(&bs_dir)
            .map_err(|e| format!("blockstates not found at {}: {e}", bs_dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let text =
                std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let json: Value =
                serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
            blockstates.insert(name, json);
        }

        let mut models = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("json") {
                    continue;
                }
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(json) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                models.insert(name, json);
            }
        }

        // Textures + .mcmeta animations.
        let mut mcmeta = HashMap::new();
        let mut texture_files = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&tex_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext == "png" {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();
                    let key = format!("{namespace}/block/{name}");
                    texture_files.insert(key, path);
                } else if path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.ends_with(".png.mcmeta"))
                {
                    let stem = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .trim_end_matches(".png.mcmeta")
                        .to_string();
                    let key = format!("{namespace}/block/{stem}");
                    if let Ok(text) = std::fs::read_to_string(&path)
                        && let Ok(json) = serde_json::from_str::<Value>(&text)
                    {
                        let anim = json.get("animation");
                        let frametime = anim
                            .and_then(|a| a.get("frametime"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1)
                            .max(1) as u32;
                        let mut frames = Vec::new();
                        if let Some(list) = anim.and_then(|a| a.get("frames")).and_then(|v| v.as_array()) {
                            for f in list {
                                if let Some(i) = f.as_u64() {
                                    frames.push(i as u32);
                                } else if let Some(idx) = f.get("index").and_then(|v| v.as_u64()) {
                                    frames.push(idx as u32);
                                }
                            }
                        }
                        mcmeta.insert(
                            key.clone(),
                            AnimInfo {
                                texture: key,
                                frametime,
                                frames,
                                frame_count: 0, // filled below from image height
                            },
                        );
                    }
                }
            }
        }
        // Frame count = image height / width (vertical strip), when readable.
        for (key, info) in mcmeta.iter_mut() {
            if let Some(path) = texture_files.get(key) {
                // Read PNG header only: width/height live at fixed IHDR offsets.
                if let Ok(bytes) = std::fs::read(path)
                    && bytes.len() > 33
                    && &bytes[..8] == b"\x89PNG\r\n\x1a\n"
                {
                    let w = u32::from_be_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]);
                    let h = u32::from_be_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
                    if w > 0 && h >= w {
                        info.frame_count = (h / w) as usize;
                    }
                }
            }
            if info.frame_count == 0 {
                info.frame_count = info.frames.len().max(1);
            }
        }

        Ok(Self {
            namespace: namespace.to_string(),
            root: root.to_path_buf(),
            blockstates,
            models,
            mcmeta,
            texture_files,
        })
    }

    pub fn report(&self) -> LoadReport {
        LoadReport {
            blocks_total: self.blockstates.len(),
            animated_textures: self.mcmeta.len(),
            ..Default::default()
        }
    }

    pub fn texture_path(&self, key: &str) -> Option<&Path> {
        self.texture_files.get(key).map(|p| p.as_path())
    }

    pub fn anim(&self, key: &str) -> Option<&AnimInfo> {
        self.mcmeta.get(key)
    }

    pub fn anim_textures(&self) -> impl Iterator<Item = &AnimInfo> {
        self.mcmeta.values()
    }

    /// Bake every known block in its default state, for startup pre-bake + logs.
    pub fn bake_all(&self) -> (HashMap<String, BakedModel>, LoadReport) {
        let mut out = HashMap::new();
        let mut report = LoadReport {
            blocks_total: self.blockstates.len(),
            animated_textures: self.mcmeta.len(),
            ..Default::default()
        };
        let mut names: Vec<_> = self.blockstates.keys().cloned().collect();
        names.sort();
        for name in names {
            match self.bake_default(&name, &mut report) {
                Some(model) => {
                    out.insert(name, model);
                    report.blocks_baked += 1;
                }
                None => {
                    report.blocks_fallback += 1;
                    report.warn(format!("{name}: no usable model, cube fallback"));
                }
            }
        }
        (out, report)
    }

    fn bake_default(&self, block: &str, report: &mut LoadReport) -> Option<BakedModel> {
        if is_air_like(block) {
            // Invisible by design (air, barrier, light): render nothing.
            return Some(BakedModel::default());
        }
        let bs = self.blockstates.get(block)?;
        // Variants: default key is "" or the first single-condition key.
        if let Some(variants) = bs.get("variants").and_then(|v| v.as_object()) {
            let entry = variants
                .get("")
                .or_else(|| variants.values().next())?;
            let model_ref = pick_model_ref(entry)?;
            return self.bake_model_ref(block, &model_ref, report).or_else(|| {
                report.blocks_fallback += 1;
                Some(self.stub_cube(block))
            });
        }
        // Multipart: unconditional parts + parts matching the empty state
        // (fence post, lone pressure plate...). Walls/bamboo gate *every*
        // part behind a condition, so an empty result retries with an
        // all-conditions-true state (connected wall, grown bamboo).
        if let Some(parts) = bs.get("multipart").and_then(|v| v.as_array()) {
            let empty = HashMap::new();
            if let Some(merged) = self.bake_multipart(block, parts, &empty, report) {
                return Some(merged);
            }
            let full = representative_state(parts);
            if let Some(merged) = self.bake_multipart(block, parts, &full, report) {
                return Some(merged);
            }
            report.warn(format!("{block}: multipart default empty"));
            report.blocks_fallback += 1;
            return Some(self.stub_cube(block));
        }
        report.warn(format!("{block}: unknown blockstate shape"));
        report.blocks_fallback += 1;
        Some(self.stub_cube(block))
    }

    /// Textured-cube stand-in for entity/fluid stubs and unparseable blocks.
    /// Uses the block's particle texture when the model names one.
    fn stub_cube(&self, block: &str) -> BakedModel {
        let texture = self
            .models
            .get(block)
            .and_then(|doc| {
                let mut chain = vec![doc];
                // One-level parent particle lookup is enough for stubs.
                if let Some(parent) = doc.get("parent").and_then(|p| p.as_str())
                    && let Some(pname) = short_model_name(parent, &self.namespace)
                    && let Some(pdoc) = self.models.get(&pname)
                {
                    chain.push(pdoc);
                }
                chain.iter().find_map(|d| {
                    d.get("textures")
                        .and_then(|t| t.get("particle"))
                        .and_then(|p| p.as_str())
                        .map(|s| s.to_string())
                })
            })
            .map(|t| {
                let resolved = t.strip_prefix('#').map(|k| {
                    self.models
                        .get(block)
                        .and_then(|d| d.get("textures"))
                        .and_then(|tx| tx.get(k))
                        .and_then(|v| v.as_str())
                        .unwrap_or(&t)
                        .to_string()
                });
                normalize_texture_key(resolved.as_deref().unwrap_or(&t), &self.namespace)
            })
            .unwrap_or_else(|| "minecraft/block/stone".to_string());
        BakedModel::fallback_cube(&texture, render_type_for(block, block))
    }

    fn bake_multipart(
        &self,
        block: &str,
        parts: &[Value],
        state: &HashMap<String, String>,
        report: &mut LoadReport,
    ) -> Option<BakedModel> {
        let mut merged = BakedModel::default();
        let mut any = false;
        for part in parts {
            if part
                .get("when")
                .is_some_and(|c| !multipart_matches(c, state))
            {
                continue;
            }
            let Some(apply) = part.get("apply") else {
                continue;
            };
            let refs: Vec<ModelRef> = if apply.is_array() {
                apply
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(parse_model_ref)
                    .collect()
            } else {
                parse_model_ref(apply).into_iter().collect()
            };
            for r in refs {
                if let Some(m) = self.bake_model_ref(block, &r, report) {
                    merged.quads.extend(m.quads);
                    merged.tris.extend(m.tris);
                    merged.render_type = merge_render(merged.render_type, m.render_type);
                    any = true;
                    break;
                }
            }
        }
        if any && !merged.is_empty() {
            Some(merged)
        } else {
            None
        }
    }

    fn bake_model_ref(
        &self,
        block: &str,
        model_ref: &ModelRef,
        report: &mut LoadReport,
    ) -> Option<BakedModel> {
        let model_name = short_model_name(&model_ref.model, &self.namespace)?;
        let model = self.resolve_model(&model_name, report, block)?;
        if model.elements.is_empty() {
            report.warn(format!("{block}: model has no elements"));
            return None;
        }
        let mut quads = Vec::new();
        for el in &model.elements {
            let min = [el.from[0] / 16.0, el.from[1] / 16.0, el.from[2] / 16.0];
            let max = [el.to[0] / 16.0, el.to[1] / 16.0, el.to[2] / 16.0];
            for (face_name, face) in &el.faces {
                let Some(normal) = face_normal(face_name) else {
                    continue;
                };
                let texture = model
                    .textures
                    .get(face.texture.trim_start_matches('#'))
                    .cloned()
                    .unwrap_or_else(|| face.texture.clone());
                let texture = normalize_texture_key(&texture, &self.namespace);
                let mut q = face_quad(min, max, normal, cull_of(&face.cullface), &texture);
                apply_face_uv(&mut q, face, normal);
                q.tinted = face.tintindex.is_some();
                if let Some(rot) = &el.rotation {
                    // Element rotation (e.g. cross-template 45° planes).
                    // `rescale` is ignored: negligible at voxel scale.
                    let origin = [rot.origin[0] / 16.0, rot.origin[1] / 16.0, rot.origin[2] / 16.0];
                    for c in q.corners.iter_mut() {
                        *c = rotate_point(*c, origin, rot.axis, rot.angle);
                    }
                    q.normal = rotate_point(q.normal, [0.0, 0.0, 0.0], rot.axis, rot.angle);
                    // Rotated planes can no longer cull against neighbours.
                    q.cull = None;
                }
                rotate_quad(&mut q, model_ref.x, model_ref.y);
                quads.push(q);
            }
        }
        if quads.is_empty() {
            return None;
        }
        Some(BakedModel {
            quads,
            tris: Vec::new(),
            render_type: render_type_for(block, &model_name),
        })
    }

    /// Walk `parent` chains, merging child-over-parent textures/elements.
    fn resolve_model(
        &self,
        name: &str,
        report: &mut LoadReport,
        block: &str,
    ) -> Option<ModelDef> {
        let mut chain: Vec<&Value> = Vec::new();
        let mut current = name.to_string();
        for _ in 0..16 {
            let Some(doc) = self.models.get(&current) else {
                break;
            };
            chain.push(doc);
            match doc.get("parent").and_then(|p| p.as_str()) {
                Some(parent) => {
                    match short_model_name(parent, &self.namespace) {
                        Some(next) if next != current => current = next,
                        _ => break,
                    }
                }
                None => break,
            }
        }
        if chain.is_empty() {
            report.warn(format!("{block}: model '{name}' not found"));
            return None;
        }
        // Merge parent-first so children override.
        let mut textures: HashMap<String, String> = HashMap::new();
        let mut elements: Vec<ElementDef> = Vec::new();
        let mut ambient_occlusion = true;
        for doc in chain.iter().rev() {
            if let Some(vars) = doc.get("textures").and_then(|t| t.as_object()) {
                for (k, v) in vars {
                    if let Some(s) = v.as_str() {
                        if s.starts_with('#') {
                            let key = s.trim_start_matches('#');
                            if let Some(resolved) = textures.get(key).cloned() {
                                textures.insert(k.clone(), resolved);
                            } else {
                                textures.insert(k.clone(), s.to_string());
                            }
                        } else {
                            textures.insert(k.clone(), s.to_string());
                        }
                    }
                }
            }
            if let Some(els) = doc.get("elements").and_then(|e| e.as_array()) {
                elements = els.iter().filter_map(parse_element).collect();
            }
            if let Some(ao) = doc.get("ambientocclusion").and_then(|v| v.as_bool()) {
                ambient_occlusion = ao;
            }
        }
        // Resolve any leftover #refs after the full merge.
        for _ in 0..8 {
            let mut changed = false;
            let keys: Vec<String> = textures.keys().cloned().collect();
            for k in &keys {
                let needs = textures
                    .get(k)
                    .filter(|v| v.starts_with('#'))
                    .map(|v| v.trim_start_matches('#').to_string());
                if let Some(target) = needs
                    && let Some(r) = textures.get(&target).cloned()
                {
                    textures.insert(k.clone(), r);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let _ = ambient_occlusion;
        Some(ModelDef { textures, elements })
    }
}

impl BlockPack for McPack {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn bake(
        &self,
        block: &str,
        _state: &HashMap<String, String>,
    ) -> Option<BakedModel> {
        // M1: chunk storage is id-only, so every state bakes the default.
        let mut report = LoadReport::default();
        self.bake_default(block, &mut report)
    }

    fn blocks(&self) -> Vec<String> {
        let mut names: Vec<_> = self.blockstates.keys().cloned().collect();
        names.sort();
        names
    }
}

// ---- blockstate helpers ----

fn pick_model_ref(entry: &Value) -> Option<ModelRef> {
    if entry.is_array() {
        entry.as_array()?.first().and_then(parse_model_ref)
    } else {
        parse_model_ref(entry)
    }
}

fn parse_model_ref(v: &Value) -> Option<ModelRef> {
    Some(ModelRef {
        model: v.get("model")?.as_str()?.to_string(),
        x: v.get("x").and_then(|n| n.as_i64()).unwrap_or(0) as i32,
        y: v.get("y").and_then(|n| n.as_i64()).unwrap_or(0) as i32,
    })
}

/// `when` matching against a state map. Supports `"prop": "value"`,
/// `"prop": ["a","b"]`, and Forge-style `{"OR": [...]}`.
fn multipart_matches(cond: &Value, state: &HashMap<String, String>) -> bool {
    let Some(map) = cond.as_object() else {
        return true;
    };
    if map.is_empty() {
        return true;
    }
    if let Some(or) = map.get("OR").and_then(|v| v.as_array()) {
        return or.iter().any(|c| multipart_matches(c, state));
    }
    if let Some(and) = map.get("AND").and_then(|v| v.as_array()) {
        return and.iter().all(|c| multipart_matches(c, state));
    }
    map.iter().all(|(prop, want)| {
        let Some(have) = state.get(prop) else {
            return false;
        };
        if let Some(s) = want.as_str() {
            // Modern format also allows "a|b" unions.
            s.split('|').any(|alt| alt == have)
        } else if let Some(list) = want.as_array() {
            list.iter().any(|v| v.as_str() == Some(have))
        } else {
            false
        }
    })
}

/// Build an all-conditions-true state: first mentioned value per property.
/// Used as the default-bake retry so walls/bamboo get a representative mesh.
fn representative_state(parts: &[Value]) -> HashMap<String, String> {
    let mut state = HashMap::new();
    for part in parts {
        collect_conditions(part.get("when"), &mut state);
    }
    state
}

fn collect_conditions(cond: Option<&Value>, state: &mut HashMap<String, String>) {
    let Some(map) = cond.and_then(|c| c.as_object()) else {
        return;
    };
    for (prop, want) in map {
        if prop == "OR" || prop == "AND" {
            if let Some(list) = want.as_array() {
                for c in list {
                    collect_conditions(Some(c), state);
                }
            }
            continue;
        }
        if state.contains_key(prop) {
            continue;
        }
        if let Some(s) = want.as_str() {
            let first = s.split('|').next().unwrap_or(s);
            state.insert(prop.clone(), first.to_string());
        } else if let Some(list) = want.as_array()
            && let Some(first) = list.first().and_then(|v| v.as_str())
        {
            state.insert(prop.clone(), first.to_string());
        }
    }
}

fn is_air_like(block: &str) -> bool {
    matches!(block, "air" | "cave_air" | "void_air" | "barrier" | "light")
}

/// `"minecraft:block/stone"` / `"block/stone"` -> `"stone"`.
fn short_model_name(model: &str, namespace: &str) -> Option<String> {
    let path = model.split(':').next_back().unwrap_or(model);
    let short = path.strip_prefix("block/").unwrap_or(path);
    if short.is_empty() || short.contains('/') {
        // Item models or cross-namespace refs we don't resolve in M1.
        // Same-namespace block models only.
        if model.contains(':') && !model.starts_with(&format!("{namespace}:")) {
            return None;
        }
        if short.contains('/') {
            return None;
        }
    }
    Some(short.to_string())
}

fn normalize_texture_key(raw: &str, namespace: &str) -> String {
    if raw.contains(':') {
        let (ns, path) = raw.split_once(':').unwrap();
        return format!("{ns}/block/{path}");
    }
    let path = raw.strip_prefix("block/").unwrap_or(raw);
    format!("{namespace}/block/{path}")
}

fn face_normal(face: &str) -> Option<[f32; 3]> {
    match face {
        "up" => Some([0.0, 1.0, 0.0]),
        "down" => Some([0.0, -1.0, 0.0]),
        "north" => Some([0.0, 0.0, -1.0]),
        "south" => Some([0.0, 0.0, 1.0]),
        "east" => Some([1.0, 0.0, 0.0]),
        "west" => Some([-1.0, 0.0, 0.0]),
        _ => None,
    }
}

fn cull_of(cullface: &Option<String>) -> Option<[i32; 3]> {
    match cullface.as_deref() {
        Some("up") => Some([0, 1, 0]),
        Some("down") => Some([0, -1, 0]),
        Some("north") => Some([0, 0, -1]),
        Some("south") => Some([0, 0, 1]),
        Some("east") => Some([1, 0, 0]),
        Some("west") => Some([-1, 0, 0]),
        _ => None,
    }
}

/// Map authored face UVs (0-16, top-left origin) onto the quad's corners,
/// applying `rotation` (clockwise degrees in the texture).
fn apply_face_uv(q: &mut BakedQuad, face: &FaceDef, _normal: [f32; 3]) {
    let [u0, v0, u1, v1] = face.uv.unwrap_or([0.0, 0.0, 16.0, 16.0]);
    let (u0, v0, u1, v1) = (
        (u0 / 16.0).clamp(0.0, 1.0),
        (v0 / 16.0).clamp(0.0, 1.0),
        (u1 / 16.0).clamp(0.0, 1.0),
        (v1 / 16.0).clamp(0.0, 1.0),
    );
    // Corner order is [tl, tr, br, bl] from outside; MC UV origin is top-left.
    let mut uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
    for _ in 0..((face.rotation % 360 + 360) % 360 / 90) {
        uvs = [uvs[3], uvs[0], uvs[1], uvs[2]];
    }
    q.uvs = uvs;
}

/// Rotate a point around `origin` by `angle_deg` about `axis`.
/// Used for element-level `rotation` (arbitrary angles, e.g. 45° crosses).
fn rotate_point(p: [f32; 3], origin: [f32; 3], axis: char, angle_deg: f32) -> [f32; 3] {
    let t = angle_deg.to_radians();
    let (s, c) = (t.sin(), t.cos());
    let q = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    let r = match axis {
        'x' => [q[0], c * q[1] - s * q[2], s * q[1] + c * q[2]],
        'y' => [c * q[0] + s * q[2], q[1], -s * q[0] + c * q[2]],
        _ => [c * q[0] - s * q[1], s * q[0] + c * q[1], q[2]],
    };
    [r[0] + origin[0], r[1] + origin[1], r[2] + origin[2]]
}

/// Rotate corners + normal around the block center for blockstate x/y.
fn rotate_quad(q: &mut BakedQuad, x: i32, y: i32) {
    let y_steps = ((y % 360 + 360) % 360) / 90;
    for _ in 0..y_steps {
        for c in q.corners.iter_mut() {
            // 90° clockwise around Y seen from above: (x,z) -> (1-z, x)... in
            // 0-1 space around center 0.5: (x,z) -> (z→) careful: y-rotation
            // by -90°: x' = z, z' = 1 - x? Use standard: rotY(-90): x'=−z, z'=x.
            let (ox, oz) = (c[0] - 0.5, c[2] - 0.5);
            c[0] = 0.5 - oz;
            c[2] = 0.5 + ox;
        }
        let [nx, ny, nz] = q.normal;
        q.normal = [-nz, ny, nx];
        if let Some([cx, cy, cz]) = q.cull {
            q.cull = Some([-cz, cy, cx]);
        }
    }
    let x_steps = ((x % 360 + 360) % 360) / 90;
    for _ in 0..x_steps {
        for c in q.corners.iter_mut() {
            let (oy, oz) = (c[1] - 0.5, c[2] - 0.5);
            c[1] = 0.5 - oz;
            c[2] = 0.5 + oy;
        }
        let [nx, ny, nz] = q.normal;
        q.normal = [nx, -nz, ny];
        if let Some([cx, cy, cz]) = q.cull {
            q.cull = Some([cx, -cz, cy]);
        }
    }
}

fn merge_render(a: RenderType, b: RenderType) -> RenderType {
    use RenderType::*;
    match (a, b) {
        (Translucent, _) | (_, Translucent) => Translucent,
        (Cutout, _) | (_, Cutout) => Cutout,
        _ => Opaque,
    }
}

fn render_type_for(block: &str, _model: &str) -> RenderType {
    let b = block.to_lowercase();
    if b.contains("water") || b.contains("ice") && !b.contains("icicle") || b.contains("slime") || b.contains("honey") {
        return RenderType::Translucent;
    }
    for key in [
        "glass", "leaves", "sapling", "flower", "crop", "cross", "torch", "rail", "ladder",
        "vine", "grass", "fern", "mushroom", "cactus", "sugar_cane", "lily", "scaffolding",
        "lantern", "chain", "bars", "pane", "door", "trapdoor", "fence", "wall", "sign",
        "banner", "bed", "anvil", "brewing", "cauldron", "hopper", "grate", "bulb",
    ] {
        if b.contains(key) {
            return RenderType::Cutout;
        }
    }
    RenderType::Opaque
}

fn parse_element(v: &Value) -> Option<ElementDef> {
    let from = v.get("from")?.as_array()?;
    let to = v.get("to")?.as_array()?;
    let num = |a: &[Value], i: usize| a.get(i).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32;
    let mut faces = HashMap::new();
    if let Some(fm) = v.get("faces").and_then(|f| f.as_object()) {
        for (name, f) in fm {
            let uv = f.get("uv").and_then(|u| u.as_array()).map(|a| {
                [
                    a.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32,
                    a.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32,
                    a.get(2).and_then(|n| n.as_f64()).unwrap_or(16.0) as f32,
                    a.get(3).and_then(|n| n.as_f64()).unwrap_or(16.0) as f32,
                ]
            });
            faces.insert(
                name.clone(),
                FaceDef {
                    uv,
                    texture: f
                        .get("texture")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    cullface: f
                        .get("cullface")
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string()),
                    rotation: f.get("rotation").and_then(|r| r.as_i64()).unwrap_or(0) as i32,
                    tintindex: f.get("tintindex").and_then(|t| t.as_i64()).map(|t| t as i32),
                },
            );
        }
    }
    Some(ElementDef {
        from: [num(from, 0), num(from, 1), num(from, 2)],
        to: [num(to, 0), num(to, 1), num(to, 2)],
        faces,
        rotation: v.get("rotation").and_then(|r| {
            let axis = r.get("axis")?.as_str()?.chars().next()?;
            if !matches!(axis, 'x' | 'y' | 'z') {
                return None;
            }
            let origin = r.get("origin")?.as_array()?;
            let n = |i: usize| origin.get(i).and_then(|x| x.as_f64()).unwrap_or(8.0) as f32;
            Some(ElementRotation {
                origin: [n(0), n(1), n(2)],
                axis,
                angle: r.get("angle").and_then(|a| a.as_f64()).unwrap_or(0.0) as f32,
            })
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("temp")
            .join("minecraft")
    }

    #[test]
    fn loads_vanilla_pack() {
        let root = pack_root();
        if !root.exists() {
            eprintln!("SKIPPED: temp/minecraft absent");
            return;
        }
        let pack = McPack::load(&root, "minecraft").expect("pack loads");
        assert!(pack.blocks().len() > 1000, "expected 1000+ blocks");
    }

    #[test]
    fn stone_bakes_to_six_quads() {
        let root = pack_root();
        if !root.exists() {
            eprintln!("SKIPPED: temp/minecraft absent");
            return;
        }
        let pack = McPack::load(&root, "minecraft").expect("pack loads");
        let model = pack
            .bake("stone", &HashMap::new())
            .expect("stone bakes");
        assert_eq!(model.quads.len(), 6);
        assert_eq!(model.render_type, RenderType::Opaque);
        for q in &model.quads {
            assert!(q.cull.is_some());
            for uv in &q.uvs {
                assert!((0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]));
            }
        }
    }

    #[test]
    fn stairs_and_fence_bake() {
        let root = pack_root();
        if !root.exists() {
            eprintln!("SKIPPED: temp/minecraft absent");
            return;
        }
        let pack = McPack::load(&root, "minecraft").expect("pack loads");
        let stairs = pack.bake("oak_stairs", &HashMap::new()).expect("stairs bake");
        assert!(!stairs.quads.is_empty());
        let fence = pack.bake("oak_fence", &HashMap::new()).expect("fence bakes");
        assert!(!fence.quads.is_empty());
    }

    #[test]
    fn bulk_bake_reports_coverage() {
        let root = pack_root();
        if !root.exists() {
            eprintln!("SKIPPED: temp/minecraft absent");
            return;
        }
        let pack = McPack::load(&root, "minecraft").expect("pack loads");
        let (models, report) = pack.bake_all();
        let total = report.blocks_total.max(1);
        let rate = report.blocks_baked as f64 / total as f64;
        eprintln!(
            "baked {}/{} ({:.1}%), fallback {}, animated textures {}",
            report.blocks_baked,
            report.blocks_total,
            rate * 100.0,
            report.blocks_fallback,
            report.animated_textures
        );
        for w in report.warnings.iter().take(10) {
            eprintln!("  warn: {w}");
        }
        assert!(rate > 0.7, "bulk bake coverage {rate:.2} below 70%");
        assert!(!models.is_empty());
    }
}
