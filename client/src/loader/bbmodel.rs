//! Blockbench `.bbmodel` loader (cubes + meshes + groups + textures).
//!
//! Reads the generic Blockbench project JSON (any `model_format`: bedrock
//! entities, `animated_entity_model` GeckoLib-style blocks like the
//! Thaumcraft-rewrite ones, `free` meshes). Bakes the rest pose to a
//! [`BakedModel`]: cubes become quads (per-face pixel UVs, cube and group
//! rotations applied), meshes become triangles.
//!
//! Units are Blockbench space (arbitrary per project); the bake normalizes
//! by the bounding box into 0-1, like the OBJ loader. Animations are parsed
//! to summaries only — our runtime plays glTF clips, so bedrock bone
//! sampling is deferred (see `BbAnim`).
//!
//! Reference: jasonjgardner/blockbench-mcp-project skills
//! (`docs/blockbench/`); the format facts used here (outliner hierarchy
//! with pivots, mesh-local vertices, per-face UV rects, animator keyframes)
//! match those docs.

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use super::baked::{BakedModel, BakedQuad, BakedTri, RenderType, rotate_point};

/// One decoded texture: embedded `data:` sources decode to pixels,
/// path/external ones keep the name for later resolution.
#[derive(Debug, Clone)]
pub struct BbTextureImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Animation summary (no sampling yet).
#[derive(Debug, Clone)]
pub struct BbAnimInfo {
    pub name: String,
    pub loop_mode: String,
    pub length: f32,
    /// bone -> (channels, keyframe count)
    pub bones: HashMap<String, (Vec<String>, usize)>,
}

#[derive(Debug, Clone, Default)]
pub struct BbLoadReport {
    pub cubes: usize,
    pub meshes: usize,
    pub tris: usize,
    pub quads: usize,
    pub textures_embedded: usize,
    pub textures_external: usize,
    pub animations: Vec<BbAnimInfo>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct BbModel {
    pub name: String,
    pub format: String,
    pub baked: BakedModel,
    pub textures: Vec<BbTextureImage>,
    /// Texture names referenced but not embedded (resolve via pack dir).
    pub external_textures: Vec<String>,
}

pub fn load_bbmodel(path: &Path) -> Result<(BbModel, BbLoadReport), String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_bbmodel(&text, path)
}

pub fn parse_bbmodel(text: &str, path: &Path) -> Result<(BbModel, BbLoadReport), String> {
    let doc: Value =
        serde_json::from_str(text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let mut report = BbLoadReport::default();

    let name = doc
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed")
        .to_string();
    let format = doc
        .get("meta")
        .and_then(|m| m.get("model_format"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // Element index by uuid.
    let mut by_uuid: HashMap<String, &Value> = HashMap::new();
    if let Some(els) = doc.get("elements").and_then(|v| v.as_array()) {
        for el in els {
            if let Some(uuid) = el.get("uuid").and_then(|v| v.as_str()) {
                by_uuid.insert(uuid.to_string(), el);
            }
        }
    }

    // Textures.
    let mut tex_names: Vec<String> = Vec::new();
    let mut tex_dims: Vec<(u32, u32)> = Vec::new();
    let mut textures = Vec::new();
    let mut external = Vec::new();
    if let Some(texs) = doc.get("textures").and_then(|v| v.as_array()) {
        for (i, t) in texs.iter().enumerate() {
            let tname = t
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let w = t.get("width").and_then(|v| v.as_u64()).unwrap_or(16) as u32;
            let h = t.get("height").and_then(|v| v.as_u64()).unwrap_or(16) as u32;
            let key = if tname.is_empty() {
                format!("texture_{i}")
            } else {
                tname.clone()
            };
            tex_names.push(key.clone());
            tex_dims.push((w.max(1), h.max(1)));
            match t.get("source").and_then(|v| v.as_str()).unwrap_or("") {
                s if s.starts_with("data:") => {
                    match decode_data_url(s) {
                        Some((rw, rh, rgba)) => {
                            report.textures_embedded += 1;
                            textures.push(BbTextureImage {
                                name: key,
                                width: rw,
                                height: rh,
                                rgba,
                            });
                        }
                        None => {
                            report.textures_external += 1;
                            external.push(key);
                        }
                    }
                }
                _ => {
                    report.textures_external += 1;
                    external.push(key);
                }
            }
        }
    }

    // Walk the outliner, accumulating group transforms.
    let mut quads = Vec::new();
    let mut tris = Vec::new();
    if let Some(outliner) = doc.get("outliner").and_then(|v| v.as_array()) {
        for node in outliner {
            walk_node(
                node,
                &by_uuid,
                &Transform::identity(),
                &tex_names,
                &tex_dims,
                &mut quads,
                &mut tris,
                &mut report,
            );
        }
    } else {
        // No outliner: bake every element at identity.
        for el in by_uuid.values() {
            bake_element(
                el,
                &Transform::identity(),
                &tex_names,
                &tex_dims,
                &mut quads,
                &mut tris,
                &mut report,
            );
        }
    }

    // Normalize by bounding box into 0-1.
    normalize_bake(&mut quads, &mut tris);

    // Animations: summaries only.
    if let Some(anims) = doc.get("animations").and_then(|v| v.as_array()) {
        for a in anims {
            let mut bones: HashMap<String, (Vec<String>, usize)> = HashMap::new();
            if let Some(map) = a.get("animators").and_then(|v| v.as_object()) {
                for anim in map.values() {
                    let bname = anim
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let mut channels = Vec::new();
                    let mut keys = 0;
                    if let Some(kfs) = anim.get("keyframes").and_then(|v| v.as_array()) {
                        for kf in kfs {
                            if let Some(ch) = kf.get("channel").and_then(|v| v.as_str()) {
                                if !channels.contains(&ch.to_string()) {
                                    channels.push(ch.to_string());
                                }
                                keys += 1;
                            }
                        }
                    }
                    bones.insert(bname, (channels, keys));
                }
            }
            // Bedrock `bones` shape (some exporters inline it).
            if let Some(map) = a.get("bones").and_then(|v| v.as_object()) {
                for (bname, channels) in map {
                    let chs: Vec<String> = channels
                        .as_object()
                        .map(|m| m.keys().cloned().collect())
                        .unwrap_or_default();
                    bones.insert(bname.clone(), (chs, 0));
                }
            }
            report.animations.push(BbAnimInfo {
                name: a
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string(),
                loop_mode: a
                    .get("loop")
                    .and_then(|v| v.as_str())
                    .unwrap_or("once")
                    .to_string(),
                length: a.get("length").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
                bones,
            });
        }
    }

    let baked = BakedModel {
        quads,
        tris,
        render_type: RenderType::Opaque,
    };
    report.quads = baked.quads.len();
    report.tris = baked.tris.len();
    Ok((
        BbModel {
            name,
            format,
            baked,
            textures,
            external_textures: external,
        },
        report,
    ))
}

// ---- transforms ----

#[derive(Debug, Clone)]
struct Transform {
    /// Accumulated euler rotation (degrees, XYZ order) + pivot handling is
    /// done by composing points through each level (see apply()).
    stack: Vec<(char, f32, [f32; 3])>,
}

impl Transform {
    fn identity() -> Self {
        Self { stack: Vec::new() }
    }

    fn is_identity(&self) -> bool {
        self.stack.is_empty()
    }

    fn rotated(&self, euler: [f32; 3], origin: [f32; 3]) -> Self {
        let mut next = self.clone();
        // Blockbench euler XYZ: apply X, then Y, then Z (intrinsic readout
        // order). Rest-pose groups are usually unrotated; exact order only
        // matters for posed previews.
        for (axis, angle) in [('x', euler[0]), ('y', euler[1]), ('z', euler[2])] {
            if angle.abs() > 1e-6 {
                next.stack.push((axis, angle, origin));
            }
        }
        next
    }

    fn apply(&self, mut p: [f32; 3]) -> [f32; 3] {
        for (axis, angle, origin) in &self.stack {
            p = rotate_point(p, *origin, *axis, *angle);
        }
        p
    }

    fn apply_normal(&self, mut n: [f32; 3]) -> [f32; 3] {
        for (axis, angle, _) in &self.stack {
            n = rotate_point(n, [0.0, 0.0, 0.0], *axis, *angle);
        }
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

fn euler_of(v: &Value) -> [f32; 3] {
    if let Some(a) = v.get("rotation").and_then(|r| r.as_array())
        && a.len() >= 3
    {
        return [
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
        ];
    }
    [0.0, 0.0, 0.0]
}

fn origin_of(v: &Value) -> [f32; 3] {
    if let Some(a) = v.get("origin").and_then(|r| r.as_array())
        && a.len() >= 3
    {
        return [
            a[0].as_f64().unwrap_or(0.0) as f32,
            a[1].as_f64().unwrap_or(0.0) as f32,
            a[2].as_f64().unwrap_or(0.0) as f32,
        ];
    }
    [0.0, 0.0, 0.0]
}

fn walk_node(
    node: &Value,
    by_uuid: &HashMap<String, &Value>,
    parent: &Transform,
    tex_names: &[String],
    tex_dims: &[(u32, u32)],
    quads: &mut Vec<BakedQuad>,
    tris: &mut Vec<BakedTri>,
    report: &mut BbLoadReport,
) {
    if let Some(s) = node.as_str() {
        // Root-level element reference.
        if let Some(el) = by_uuid.get(s) {
            bake_element(el, parent, tex_names, tex_dims, quads, tris, report);
        }
        return;
    }
    let local = parent.rotated(euler_of(node), origin_of(node));
    if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
        for child in children {
            walk_node(
                child, by_uuid, &local, tex_names, tex_dims, quads, tris, report,
            );
        }
    }
}

// ---- elements ----

fn bake_element(
    el: &Value,
    parent: &Transform,
    tex_names: &[String],
    tex_dims: &[(u32, u32)],
    quads: &mut Vec<BakedQuad>,
    tris: &mut Vec<BakedTri>,
    report: &mut BbLoadReport,
) {
    if el.get("vertices").is_some() {
        bake_mesh(el, parent, tex_names, tex_dims, tris, report);
    } else if el.get("from").is_some() {
        bake_cube(el, parent, tex_names, tex_dims, quads, report);
    }
}

fn tex_key(tex_names: &[String], idx: Option<i64>) -> String {
    idx.and_then(|i| tex_names.get(i as usize))
        .cloned()
        .unwrap_or_else(|| "bbmodel:missing".to_string())
}

fn bake_cube(
    el: &Value,
    parent: &Transform,
    tex_names: &[String],
    tex_dims: &[(u32, u32)],
    quads: &mut Vec<BakedQuad>,
    report: &mut BbLoadReport,
) {
    let num = |v: &Value, k: &str, i: usize| {
        v.get(k)
            .and_then(|a| a.as_array())
            .and_then(|a| a.get(i))
            .and_then(|n| n.as_f64())
            .unwrap_or(0.0) as f32
    };
    let from = [num(el, "from", 0), num(el, "from", 1), num(el, "from", 2)];
    let to = [num(el, "to", 0), num(el, "to", 1), num(el, "to", 2)];
    let local = parent.rotated(euler_of(el), origin_of(el));

    let Some(faces) = el.get("faces").and_then(|v| v.as_object()) else {
        report
            .warnings
            .push("cube without faces, skipped".to_string());
        return;
    };
    report.cubes += 1;
    // Unrotated boxes keep axis cull faces (props collide); anything the
    // hierarchy rotated gets none (a wrong cull hides the block).
    let keep_cull = local.is_identity();
    for (dir, face) in faces {
        let Some(normal) = bb_face_normal(dir) else {
            continue;
        };
        let tidx = face.get("texture").and_then(|v| v.as_i64());
        let texture = tex_key(tex_names, tidx);
        let (tw, th) = tidx
            .and_then(|i| tex_dims.get(i as usize).copied())
            .unwrap_or((16, 16));
        // Full-face corners in the cube's rest space.
        let corners = cube_face_corners(from, to, normal);
        let mut world = [[0.0; 3]; 4];
        for (i, c) in corners.iter().enumerate() {
            world[i] = local.apply(*c);
        }
        let wnormal = local.apply_normal(normal);
        // UV rect in pixels (flips preserved); per-face uv rotation applied.
        let uv = face.get("uv").and_then(|u| u.as_array());
        let u = |i: usize, dflt: f32| {
            uv.and_then(|a| a.get(i))
                .and_then(|n| n.as_f64())
                .unwrap_or(dflt as f64) as f32
        };
        let (mut u0, mut v0, mut u1, mut v1) = (u(0, 0.0), u(1, 0.0), u(2, 16.0), u(3, 16.0));
        (u0, v0, u1, v1) = (
            u0 / tw as f32,
            v0 / th as f32,
            u1 / tw as f32,
            v1 / th as f32,
        );
        let mut uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        let rot = face.get("rotation").and_then(|r| r.as_i64()).unwrap_or(0);
        for _ in 0..((rot % 360 + 360) % 360 / 90) {
            uvs = [uvs[3], uvs[0], uvs[1], uvs[2]];
        }
        quads.push(BakedQuad {
            corners: world,
            normal: wnormal,
            uvs,
            texture,
            cull: keep_cull.then(|| normal_dir(dir)).flatten(),
            tinted: false,
        });
    }
}

fn normal_dir(dir: &str) -> Option<[i32; 3]> {
    match dir {
        "up" => Some([0, 1, 0]),
        "down" => Some([0, -1, 0]),
        "north" => Some([0, 0, -1]),
        "south" => Some([0, 0, 1]),
        "east" => Some([1, 0, 0]),
        "west" => Some([-1, 0, 0]),
        _ => None,
    }
}

fn bb_face_normal(dir: &str) -> Option<[f32; 3]> {
    match dir {
        "up" => Some([0.0, 1.0, 0.0]),
        "down" => Some([0.0, -1.0, 0.0]),
        "north" => Some([0.0, 0.0, -1.0]),
        "south" => Some([0.0, 0.0, 1.0]),
        "east" => Some([1.0, 0.0, 0.0]),
        "west" => Some([-1.0, 0.0, 0.0]),
        _ => None,
    }
}

fn cube_face_corners(from: [f32; 3], to: [f32; 3], normal: [f32; 3]) -> [[f32; 3]; 4] {
    // CCW-from-outside, same layout as baked::face_quad.
    match normal {
        [0.0, 1.0, 0.0] => [
            [from[0], to[1], from[2]],
            [to[0], to[1], from[2]],
            [to[0], to[1], to[2]],
            [from[0], to[1], to[2]],
        ],
        [0.0, -1.0, 0.0] => [
            [from[0], from[1], to[2]],
            [to[0], from[1], to[2]],
            [to[0], from[1], from[2]],
            [from[0], from[1], from[2]],
        ],
        [0.0, 0.0, -1.0] => [
            [to[0], to[1], from[2]],
            [from[0], to[1], from[2]],
            [from[0], from[1], from[2]],
            [to[0], from[1], from[2]],
        ],
        [1.0, 0.0, 0.0] => [
            [to[0], to[1], to[2]],
            [to[0], to[1], from[2]],
            [to[0], from[1], from[2]],
            [to[0], from[1], to[2]],
        ],
        [0.0, 0.0, 1.0] => [
            [from[0], to[1], to[2]],
            [to[0], to[1], to[2]],
            [to[0], from[1], to[2]],
            [from[0], from[1], to[2]],
        ],
        _ => [
            [from[0], to[1], from[2]],
            [from[0], to[1], to[2]],
            [from[0], from[1], to[2]],
            [from[0], from[1], from[2]],
        ],
    }
}

fn bake_mesh(
    el: &Value,
    parent: &Transform,
    tex_names: &[String],
    tex_dims: &[(u32, u32)],
    tris: &mut Vec<super::baked::BakedTri>,
    report: &mut BbLoadReport,
) {
    let verts = el.get("vertices").and_then(|v| v.as_object());
    let faces = el.get("faces").and_then(|v| v.as_object());
    let (Some(verts), Some(faces)) = (verts, faces) else {
        return;
    };
    let local = parent.rotated(euler_of(el), origin_of(el));
    report.meshes += 1;
    for face in faces.values() {
        let Some(idx) = face.get("vertices").and_then(|v| v.as_array()) else {
            continue;
        };
        if idx.len() < 3 {
            continue;
        }
        let pos = |key: &str| -> [f32; 3] {
            verts
                .get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    [
                        a.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32,
                        a.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32,
                        a.get(2).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32,
                    ]
                })
                .unwrap_or([0.0; 3])
        };
        let tidx = face.get("texture").and_then(|v| v.as_i64());
        let texture = tex_key(tex_names, tidx);
        let (tw, th) = tidx
            .and_then(|i| tex_dims.get(i as usize).copied())
            .unwrap_or((16, 16));
        let uvmap = face.get("uv").and_then(|v| v.as_object());
        let uv = |key: &str| -> [f32; 2] {
            uvmap
                .and_then(|m| m.get(key))
                .and_then(|v| v.as_array())
                .map(|a| {
                    [
                        a.first().and_then(|n| n.as_f64()).unwrap_or(0.0) as f32 / tw as f32,
                        a.get(1).and_then(|n| n.as_f64()).unwrap_or(0.0) as f32 / th as f32,
                    ]
                })
                .unwrap_or([0.0, 0.0])
        };
        let keys: Vec<&str> = idx.iter().filter_map(|v| v.as_str()).collect();
        // Fan triangulation.
        for i in 1..keys.len().saturating_sub(1) {
            let ks = [keys[0], keys[i], keys[i + 1]];
            let mut v = [[0.0; 3]; 3];
            let mut u = [[0.0; 2]; 3];
            for (j, k) in ks.iter().enumerate() {
                v[j] = local.apply(pos(k));
                u[j] = uv(k);
            }
            tris.push(super::baked::BakedTri {
                verts: v,
                normal: tri_normal(v[0], v[1], v[2]),
                uvs: u,
                texture: texture.clone(),
            });
        }
    }
}

fn tri_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
    if len <= 1e-9 {
        return [0.0, 1.0, 0.0];
    }
    [n[0] / len, n[1] / len, n[2] / len]
}

fn normalize_bake(quads: &mut [BakedQuad], tris: &mut [super::baked::BakedTri]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut touch = |p: &[f32; 3]| {
        for a in 0..3 {
            min[a] = min[a].min(p[a]);
            max[a] = max[a].max(p[a]);
        }
    };
    for q in quads.iter() {
        for c in &q.corners {
            touch(c);
        }
    }
    for t in tris.iter() {
        for v in &t.verts {
            touch(v);
        }
    }
    if min[0] == f32::INFINITY {
        return;
    }
    let span = [
        (max[0] - min[0]).max(1e-6),
        (max[1] - min[1]).max(1e-6),
        (max[2] - min[2]).max(1e-6),
    ];
    let biggest = span[0].max(span[1]).max(span[2]);
    let mut norm = |p: &mut [f32; 3]| {
        for a in 0..3 {
            p[a] = (p[a] - min[a]) / biggest;
        }
    };
    for q in quads.iter_mut() {
        for c in q.corners.iter_mut() {
            norm(c);
        }
    }
    for t in tris.iter_mut() {
        for v in t.verts.iter_mut() {
            norm(v);
        }
    }
}

fn decode_data_url(url: &str) -> Option<(u32, u32, Vec<u8>)> {
    let (_, b64) = url.split_once(',')?;
    // Minimal base64 decoder (standard alphabet, padding-aware).
    const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [255u8; 256];
    for (i, c) in ALPHA.iter().enumerate() {
        table[*c as usize] = i as u8;
    }
    let clean: Vec<u8> = b64.bytes().filter(|b| *b != b'=').collect();
    let mut raw = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let mut n = 0u32;
        let mut bits = 0;
        for b in chunk {
            let v = table[*b as usize];
            if v == 255 {
                return None;
            }
            n = (n << 6) | v as u32;
            bits += 6;
        }
        while bits >= 8 {
            bits -= 8;
            raw.push((n >> bits) as u8);
        }
    }
    let img = image::load_from_memory(&raw).ok()?.into_rgba8();
    let (w, h) = (img.width(), img.height());
    Some((w, h, img.into_raw()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI_BBMODEL: &str = r#"{
        "meta": {"format_version": "4.10", "model_format": "bedrock"},
        "name": "mini",
        "resolution": {"width": 64, "height": 64},
        "elements": [{
            "name": "cube", "type": "cube", "uuid": "c1",
            "from": [0, 0, 0], "to": [16, 16, 16],
            "faces": {
                "up": {"uv": [0, 0, 16, 16], "texture": 0},
                "down": {"uv": [0, 0, 16, 16], "texture": 0},
                "north": {"uv": [0, 0, 16, 16], "texture": 0},
                "south": {"uv": [0, 0, 16, 16], "texture": 0},
                "east": {"uv": [0, 0, 16, 16], "texture": 0},
                "west": {"uv": [0, 0, 16, 16], "texture": 0}
            }
        }],
        "outliner": ["c1"],
        "textures": [{"name": "t", "width": 64, "height": 64, "source": ""}]
    }"#;

    #[test]
    fn mini_cube_bakes() {
        let path = Path::new("mini.bbmodel");
        let (model, report) = parse_bbmodel(MINI_BBMODEL, path).expect("parses");
        assert_eq!(report.cubes, 1);
        assert_eq!(model.baked.quads.len(), 6);
        for q in &model.baked.quads {
            for c in &q.corners {
                assert!((0.0..=1.0).contains(&c[0]));
            }
        }
    }

    #[test]
    fn thaumcraft_rewrite_blocks_bake() {
        let dir = Path::new("/home/nestor/Downloads");
        if !dir.is_dir() {
            eprintln!("SKIPPED: ~/Downloads absent");
            return;
        }
        let mut files = vec![
            "arcane_workbench.bbmodel",
            "infusion_matrix.bbmodel",
            "infusion_pedestal.bbmodel",
            "infusion_pillar.bbmodel",
            "model.bbmodel",
        ];
        files.retain(|f| dir.join(f).exists());
        if files.is_empty() {
            eprintln!("SKIPPED: no bbmodels in ~/Downloads");
            return;
        }
        for f in files {
            let (model, report) = load_bbmodel(&dir.join(f)).expect("loads");
            eprintln!(
                "{}: format={} cubes={} meshes={} quads={} tris={} tex_emb={} anims={:?}",
                f,
                model.format,
                report.cubes,
                report.meshes,
                report.quads,
                report.tris,
                report.textures_embedded,
                report.animations.iter().map(|a| (a.name.clone(), a.bones.len())).collect::<Vec<_>>(),
            );
            assert!(
                !model.baked.is_empty(),
                "{f} baked nothing (quads={} tris={})",
                report.quads,
                report.tris
            );
        }
    }

    #[test]
    fn girl_bbmodel_meshes_bake() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("assets")
            .join("models")
            .join("girl.bbmodel");
        if !path.exists() {
            eprintln!("SKIPPED: girl.bbmodel absent");
            return;
        }
        // 10MB project: parse + bake meshes only (no texture decode needed
        // for the geometry assertion).
        let text = std::fs::read_to_string(&path).expect("reads");
        let (model, report) = parse_bbmodel(&text, &path).expect("parses");
        eprintln!(
            "girl: meshes={} tris={} anims={}",
            report.meshes,
            report.tris,
            report.animations.len()
        );
        assert!(report.tris > 1000, "girl should bake thousands of tris");
    }
}
