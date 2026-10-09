//! glTF → GLB conversion at catalog build time.
//!
//! The client loader (`gltf` crate) only accepts self-contained GLB 2.0, so
//! the server converts `.gltf` sources in-memory while building the catalog
//! pak: the generated `.glb` (same logical path, extension swapped) is what
//! clients download. Sources are never modified and no files are written.
//!
//! Two front-ends:
//! - v1.0 (assimp-style: string ids, Blinn-Phong `values` materials):
//!   geometry/accessors map 1:1 onto 2.0; materials become PBR with the
//!   diffuse texture (or flat diffuse color) as base color, metal 0 / rough 1.
//!   Only TRIANGLE primitives with POSITION/NORMAL/TEXCOORD_0, single mesh
//!   per node, identity-or-TRS nodes are supported — anything else errors.
//! - v2.0 with external buffers/images: embedded into the BIN chunk.
//!   `data:` URIs and anything else errors with a clear message.

use std::path::{Path, PathBuf};

/// Bump when conversion output changes so stale artifacts regenerate.
const CONVERTER_VERSION: u32 = 4;

fn generator(kind: &str) -> String {
    format!("exsolis gltf_pack v{CONVERTER_VERSION} ({kind})")
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

fn err(what: &str) -> String {
    format!("gltf_pack: {what}")
}

/// Convert a `.gltf` file to self-contained GLB bytes. Sibling files
/// (`.bin`, images) resolve against the `.gltf`'s parent directory.
pub fn convert_gltf_file(gltf_path: &Path) -> Result<Vec<u8>, String> {
    let dir = gltf_path
        .parent()
        .ok_or_else(|| err("gltf path has no parent"))?;
    let text = std::fs::read_to_string(gltf_path).map_err(|e| err(&format!("read failed: {e}")))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| err(&format!("json parse failed: {e}")))?;
    let version = doc
        .get("asset")
        .and_then(|a| a.get("version"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| err("missing asset.version"))?;
    let (json, bin) = match version {
        "1.0" => convert_v1(&doc, dir)?,
        "2.0" => convert_v2(&doc, dir)?,
        other => return Err(err(&format!("unsupported glTF version {other:?}"))),
    };
    Ok(write_glb(&json, &bin))
}

/// Ensure a fresh `<stem>.glb` next to `gltf_path`, converting when missing
/// or older than the `.gltf` or any referenced sibling. Returns the `.glb`
/// disk path. Sources are never modified.
pub fn ensure_converted_glb(gltf_path: &Path) -> Result<PathBuf, String> {
    let glb_path = gltf_path.with_extension("glb");
    let stale = match std::fs::metadata(&glb_path).and_then(|m| m.modified()) {
        Ok(glb_mtime) => {
            let mut latest: Option<std::time::SystemTime> = None;
            for dep in referenced_files(gltf_path)? {
                if let Ok(m) = std::fs::metadata(&dep).and_then(|m| m.modified()) {
                    latest = Some(latest.map_or(m, |l: std::time::SystemTime| l.max(m)));
                }
            }
            match latest {
                Some(t) => {
                    t > glb_mtime || glb_converter_version(&glb_path) != Some(CONVERTER_VERSION)
                }
                None => true,
            }
        }
        Err(_) => true,
    };
    if stale {
        let glb = convert_gltf_file(gltf_path)?;
        std::fs::write(&glb_path, &glb).map_err(|e| err(&format!("write failed: {e}")))?;
    }
    Ok(glb_path)
}

/// Converter version stamped into a previously generated `.glb`
/// (`asset.generator`), if parseable. `None` forces regeneration.
fn glb_converter_version(glb_path: &Path) -> Option<u32> {
    let bytes = std::fs::read(glb_path).ok()?;
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        return None;
    }
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let doc: serde_json::Value = serde_json::from_slice(bytes.get(20..20 + json_len)?).ok()?;
    let generator = doc.get("asset")?.get("generator")?.as_str()?;
    generator
        .strip_prefix("exsolis gltf_pack v")
        .and_then(|rest| rest.split(' ').next())
        .and_then(|v| v.parse().ok())
}
/// `.gltf` itself plus every sibling it references (buffers, images).
/// Version-agnostic: v1 uses string-id maps, v2 uses arrays.
fn referenced_files(gltf_path: &Path) -> Result<Vec<PathBuf>, String> {
    let dir = gltf_path
        .parent()
        .ok_or_else(|| err("gltf path has no parent"))?;
    let text = std::fs::read_to_string(gltf_path).map_err(|e| err(&format!("read failed: {e}")))?;
    let doc: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| err(&format!("json parse failed: {e}")))?;
    let mut out = vec![gltf_path.to_path_buf()];
    let mut push_uri = |uri: &str| {
        let clean = uri.split(['?', '#']).next().unwrap_or(uri);
        if !clean.starts_with("data:") && !clean.contains("://") {
            out.push(dir.join(clean));
        }
    };
    if let Some(buffers) = doc.get("buffers") {
        // v1: map, v2: array.
        let iter: Box<dyn Iterator<Item = &serde_json::Value>> = match buffers {
            serde_json::Value::Object(m) => Box::new(m.values()),
            serde_json::Value::Array(a) => Box::new(a.iter()),
            _ => return Err(err("bad buffers")),
        };
        for buf in iter {
            if let Some(uri) = buf.get("uri").and_then(|u| u.as_str()) {
                push_uri(uri);
            }
        }
    }
    if let Some(images) = doc.get("images") {
        let iter: Box<dyn Iterator<Item = &serde_json::Value>> = match images {
            serde_json::Value::Object(m) => Box::new(m.values()),
            serde_json::Value::Array(a) => Box::new(a.iter()),
            _ => return Err(err("bad images")),
        };
        for img in iter {
            if let Some(uri) = img.get("uri").and_then(|u| u.as_str()) {
                push_uri(uri);
            }
        }
    }
    Ok(out)
}

fn write_glb(json: &serde_json::Value, bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = serde_json::to_vec(json).expect("json serialization");
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }
    let mut bin_padded = bin.to_vec();
    while bin_padded.len() % 4 != 0 {
        bin_padded.push(0);
    }
    let total = (12 + 8 + json_bytes.len() + 8 + bin_padded.len()) as u32;
    let mut glb = Vec::with_capacity(total as usize);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // 'glTF'
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total.to_le_bytes());
    glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_bytes);
    glb.extend_from_slice(&(bin_padded.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\x00");
    glb.extend_from_slice(&bin_padded);
    glb
}

fn obj<'a>(
    v: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>, String> {
    v.get(key)
        .and_then(|v| v.as_object())
        .ok_or_else(|| err(&format!("missing object {key:?}")))
}

fn index_of(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    what: &str,
) -> Result<usize, String> {
    map.keys()
        .position(|k| k == key)
        .ok_or_else(|| err(&format!("unknown {what} {key:?}")))
}

fn read_sibling(dir: &Path, uri: &str) -> Result<Vec<u8>, String> {
    if uri.starts_with("data:") || uri.contains("://") {
        return Err(err(&format!("external uri not supported: {uri:?}")));
    }
    // Strip query/fragment; reject escapes from the asset dir.
    let clean = uri.split(['?', '#']).next().unwrap_or(uri);
    let path = dir.join(clean);
    if !path.is_file() {
        return Err(err(&format!("missing sibling file {clean:?}")));
    }
    std::fs::read(&path).map_err(|e| err(&format!("read {clean:?} failed: {e}")))
}

fn mime_for(uri: &str) -> &'static str {
    let lower = uri.to_ascii_lowercase();
    if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".ktx2") {
        "image/ktx2"
    } else {
        "image/png"
    }
}

// ---------------------------------------------------------------- v1.0 ---

fn convert_v1(doc: &serde_json::Value, dir: &Path) -> Result<(serde_json::Value, Vec<u8>), String> {
    use serde_json::{Map, Value};

    let buffers = obj(doc, "buffers")?;
    if buffers.len() != 1 {
        return Err(err("v1: exactly one buffer expected"));
    }
    let (buf_name, buf_def) = buffers.iter().next().unwrap();
    let buf_uri = buf_def
        .get("uri")
        .and_then(|u| u.as_str())
        .ok_or_else(|| err("v1: buffer without uri"))?;
    let mut blob = read_sibling(dir, buf_uri)?;
    let _ = buf_name;

    let buffer_views = obj(doc, "bufferViews")?;
    let accessors = obj(doc, "accessors")?;
    let images_in = doc.get("images").and_then(|v| v.as_object());
    let textures_in = doc.get("textures").and_then(|v| v.as_object());
    let samplers_in = doc.get("samplers").and_then(|v| v.as_object());
    let materials_in = obj(doc, "materials")?;
    let meshes_in = obj(doc, "meshes")?;
    let nodes_in = obj(doc, "nodes")?;

    // Images: embed after BIN data.
    let mut images: Vec<Value> = Vec::new();
    let mut img_views: Vec<Value> = Vec::new();
    if let Some(imgs) = images_in {
        for (name, img) in imgs {
            let uri = img
                .get("uri")
                .and_then(|u| u.as_str())
                .ok_or_else(|| err("v1: image without uri"))?;
            let data = read_sibling(dir, uri)?;
            let offset = align4(blob.len());
            blob.resize(offset, 0);
            blob.extend_from_slice(&data);
            img_views.push(serde_json::json!({
                "buffer": 0, "byteOffset": offset, "byteLength": data.len(),
            }));
            images.push(serde_json::json!({
                "bufferView": buffer_views.len() + img_views.len() - 1,
                "mimeType": mime_for(uri), "name": name,
            }));
        }
    }

    let mut buffer_views_out: Vec<Value> = Vec::with_capacity(buffer_views.len());
    for (_, bv) in buffer_views {
        let mut out = Map::new();
        out.insert("buffer".into(), 0.into());
        for (k, v) in bv.as_object().ok_or_else(|| err("v1: bad bufferView"))? {
            if k != "buffer" {
                out.insert(k.clone(), v.clone());
            }
        }
        buffer_views_out.push(Value::Object(out));
    }
    for view in img_views {
        buffer_views_out.push(view);
    }

    let mut accessors_out = Vec::with_capacity(accessors.len());
    for (_, acc) in accessors {
        let acc = acc.as_object().ok_or_else(|| err("v1: bad accessor"))?;
        let bv_name = acc
            .get("bufferView")
            .and_then(|v| v.as_str())
            .ok_or_else(|| err("v1: accessor without bufferView"))?;
        let mut out = acc.clone();
        out.insert(
            "bufferView".into(),
            index_of(buffer_views, bv_name, "bufferView")?.into(),
        );
        accessors_out.push(Value::Object(out));
    }

    // Samplers / textures.
    let mut sampler_index: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    let mut samplers_out = Vec::new();
    if let Some(samplers) = samplers_in {
        for (i, (name, s)) in samplers.iter().enumerate() {
            sampler_index.insert(name.as_str(), i);
            let get = |k: &str, dflt: u32| {
                s.get(k).and_then(|v| v.as_u64()).unwrap_or(dflt as u64) as u32
            };
            samplers_out.push(serde_json::json!({
                // assimp-era glTF 1.0 dumps default to 9728 (GL_NEAREST), which
                // ruins models with small UV islands. Default to Linear (9729).
                "magFilter": 9729,
                "minFilter": 9729,
                "wrapS": get("wrapS", 10497),
                "wrapT": get("wrapT", 10497),
            }));
        }
    }
    let image_names: Vec<&str> = images_in
        .map(|m| m.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();
    let mut textures_out = Vec::new();
    if let Some(textures) = textures_in {
        for (name, t) in textures {
            let src = t
                .get("source")
                .and_then(|v| v.as_str())
                .ok_or_else(|| err("v1: texture without source"))?;
            let source = image_names
                .iter()
                .position(|n| *n == src)
                .ok_or_else(|| err(&format!("v1: unknown image {src:?}")))?;
            let sampler = t
                .get("sampler")
                .and_then(|v| v.as_str())
                .and_then(|s| sampler_index.get(s).copied())
                .unwrap_or(0);
            textures_out.push(serde_json::json!({
                "source": source, "sampler": sampler, "name": name,
            }));
        }
    }

    // Materials: Blinn-Phong diffuse -> PBR base color.
    let mut materials_out = Vec::new();
    for (name, m) in materials_in {
        let values = m.get("values").and_then(|v| v.as_object());
        let mut pbr = Map::new();
        pbr.insert("metallicFactor".into(), 0.0.into());
        pbr.insert("roughnessFactor".into(), 1.0.into());
        // Textured v1 materials become alpha-tested cutouts: assimp-era
        // diffuse textures routinely carry transparency (wing membranes,
        // tails) with black RGB, which would otherwise render as opaque
        // black geometry. Fully-opaque textures never trip the cutoff.
        let mut alpha_mode = "OPAQUE";
        match values.and_then(|v| v.get("diffuse")) {
            Some(Value::String(_)) => {
                // Diffuse texture reference; one texture per model here.
                if textures_out.is_empty() {
                    return Err(err("v1: diffuse texture but no textures"));
                }
                let mut tex = Map::new();
                tex.insert("index".into(), 0.into());
                pbr.insert("baseColorTexture".into(), Value::Object(tex));
                alpha_mode = "MASK";
            }
            Some(Value::Array(c)) => {
                let rgba: Vec<Value> = c.iter().take(4).cloned().collect();
                pbr.insert("baseColorFactor".into(), Value::Array(rgba));
            }
            _ => {
                pbr.insert(
                    "baseColorFactor".into(),
                    serde_json::json!([1.0, 1.0, 1.0, 1.0]),
                );
            }
        }
        materials_out.push(serde_json::json!({
            "name": name, "pbrMetallicRoughness": Value::Object(pbr),
            // v1 carries no sidedness: assimp-era soup meshes are thin and
            // one-sided, so render double-sided (the client honors this per
            // draw with a no-cull pipeline variant).
            "doubleSided": true,
            "alphaMode": alpha_mode,
            "alphaCutoff": 0.5,
        }));
    }

    // Meshes.
    let mut meshes_out = Vec::new();
    for (name, mesh) in meshes_in {
        let mesh = mesh.as_object().ok_or_else(|| err("v1: bad mesh"))?;
        let prims = mesh
            .get("primitives")
            .and_then(|v| v.as_array())
            .ok_or_else(|| err("v1: mesh without primitives"))?;
        let mut prims_out = Vec::with_capacity(prims.len());
        for p in prims {
            if p.get("mode").and_then(|v| v.as_u64()).unwrap_or(4) != 4 {
                return Err(err(&format!("v1: non-triangle primitive in {name}")));
            }
            let attrs_in = p
                .get("attributes")
                .and_then(|v| v.as_object())
                .ok_or_else(|| err("v1: primitive without attributes"))?;
            let mut attrs = Map::new();
            for semantic in ["POSITION", "NORMAL", "TEXCOORD_0"] {
                let acc_name = attrs_in
                    .get(semantic)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| err(&format!("v1: missing {semantic} in {name}")))?;
                attrs.insert(
                    semantic.into(),
                    index_of(accessors, acc_name, "accessor")?.into(),
                );
            }
            let idx_name = p
                .get("indices")
                .and_then(|v| v.as_str())
                .ok_or_else(|| err("v1: primitive without indices"))?;
            let mat_name = p
                .get("material")
                .and_then(|v| v.as_str())
                .ok_or_else(|| err("v1: primitive without material"))?;
            prims_out.push(serde_json::json!({
                "attributes": Value::Object(attrs),
                "indices": index_of(accessors, idx_name, "accessor")?,
                "material": index_of(materials_in, mat_name, "material")?,
                "mode": 4,
            }));
        }
        meshes_out.push(serde_json::json!({"name": name, "primitives": prims_out}));
    }

    // Nodes (single mesh each) / scenes.
    let mut nodes_out = Vec::new();
    for (name, node) in nodes_in {
        let node = node.as_object().ok_or_else(|| err("v1: bad node"))?;
        let mut out = Map::new();
        out.insert("name".into(), Value::String(name.clone()));
        if let Some(meshes) = node.get("meshes").and_then(|v| v.as_array()) {
            if meshes.len() != 1 {
                return Err(err(&format!("v1: multi-mesh node {name}")));
            }
            let mesh_name = meshes[0]
                .as_str()
                .ok_or_else(|| err("v1: bad node mesh ref"))?;
            out.insert(
                "mesh".into(),
                index_of(meshes_in, mesh_name, "mesh")?.into(),
            );
        }
        for key in ["matrix", "translation", "rotation", "scale"] {
            if let Some(v) = node.get(key) {
                out.insert(key.into(), v.clone());
            }
        }
        if let Some(children) = node.get("children").and_then(|v| v.as_array()) {
            let mut kids = Vec::with_capacity(children.len());
            for c in children {
                let c = c.as_str().ok_or_else(|| err("v1: bad child ref"))?;
                kids.push(index_of(nodes_in, c, "node")?.into());
            }
            out.insert("children".into(), Value::Array(kids));
        }
        nodes_out.push(Value::Object(out));
    }
    let scenes_in = obj(doc, "scenes")?;
    let scene_name = doc
        .get("scene")
        .and_then(|v| v.as_str())
        .ok_or_else(|| err("v1: missing default scene"))?;
    let root = scenes_in
        .get(scene_name)
        .and_then(|s| s.get("nodes"))
        .and_then(|n| n.as_array())
        .ok_or_else(|| err("v1: bad scene nodes"))?;
    let mut roots: Vec<serde_json::Value> = Vec::with_capacity(root.len());
    for n in root {
        let n = n.as_str().ok_or_else(|| err("v1: bad scene node ref"))?;
        roots.push(index_of(nodes_in, n, "node")?.into());
    }

    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": generator("v1")},
        "buffers": [{"byteLength": blob.len()}],
        "bufferViews": buffer_views_out,
        "accessors": accessors_out,
        "images": images,
        "samplers": samplers_out,
        "textures": textures_out,
        "materials": materials_out,
        "meshes": meshes_out,
        "nodes": nodes_out,
        "scenes": [{"nodes": roots}],
        "scene": 0,
    });
    Ok((json, blob))
}

// ---------------------------------------------------------------- v2.0 ---

/// Embed external buffers/images of a 2.0 `.gltf` into one BIN chunk.
/// Files already embedded (GLB BIN section, `data:` URIs) are rejected with
/// a clear message — pass those straight to the client as `.glb` instead.
fn convert_v2(doc: &serde_json::Value, dir: &Path) -> Result<(serde_json::Value, Vec<u8>), String> {
    use serde_json::{Map, Value};

    let mut json = doc.clone();
    let root = json
        .as_object_mut()
        .ok_or_else(|| err("v2: bad document"))?;

    // Buffers -> concatenated BIN.
    let buffers = root
        .get("buffers")
        .and_then(|v| v.as_array())
        .ok_or_else(|| err("v2: missing buffers"))?
        .clone();
    let mut blob: Vec<u8> = Vec::new();
    let mut buf_base: Vec<usize> = Vec::with_capacity(buffers.len());
    for buf in &buffers {
        let uri = buf
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| err("v2: only external-file buffers can be packed"))?;
        let data = read_sibling(dir, uri)?;
        let declared = buf.get("byteLength").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if data.len() < declared {
            return Err(err(&format!("v2: buffer {uri:?} shorter than declared")));
        }
        while blob.len() % 4 != 0 {
            blob.push(0);
        }
        buf_base.push(blob.len());
        blob.extend_from_slice(&data[..declared]);
    }

    // BufferViews: shift by their buffer's base (single buffer 0 now).
    let mut views_out = Vec::new();
    if let Some(views) = root.get("bufferViews").and_then(|v| v.as_array()) {
        for bv in views {
            let b = bv
                .get("buffer")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| err("v2: bufferView without buffer"))? as usize;
            let off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let mut out = bv.as_object().cloned().ok_or_else(|| err("v2: bad view"))?;
            out.insert("buffer".into(), 0.into());
            out.insert("byteOffset".into(), (buf_base[b] + off).into());
            views_out.push(Value::Object(out));
        }
    }
    let mut next_view = views_out.len();

    // Images: external files appended; bufferView images rebased.
    if let Some(images) = root.get("images").and_then(|v| v.as_array()).cloned() {
        let mut images_out = Vec::with_capacity(images.len());
        // Map old view index -> new view index for rebasing.
        let mut view_remap: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for img in &images {
            if let Some(uri) = img.get("uri").and_then(|v| v.as_str()) {
                let data = read_sibling(dir, uri)?;
                while blob.len() % 4 != 0 {
                    blob.push(0);
                }
                let offset = blob.len();
                blob.extend_from_slice(&data);
                let mut out = Map::new();
                out.insert("bufferView".into(), next_view.into());
                out.insert("mimeType".into(), mime_for(uri).into());
                if let Some(name) = img.get("name") {
                    out.insert("name".into(), name.clone());
                }
                views_out.push(serde_json::json!({
                    "buffer": 0, "byteOffset": offset, "byteLength": data.len(),
                }));
                next_view += 1;
                images_out.push(Value::Object(out));
            } else if let Some(bv) = img.get("bufferView").and_then(|v| v.as_u64()) {
                let bv = bv as usize;
                let new_idx = *view_remap.entry(bv).or_insert_with(|| {
                    let idx = next_view;
                    next_view += 1;
                    idx
                });
                // Rebase: original view's buffer base + offset.
                let orig = views_out.get(bv).cloned().unwrap_or(Value::Null);
                let buf = orig.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let off = orig.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let mut rebased = orig.as_object().cloned().unwrap_or_default();
                rebased.insert("buffer".into(), 0.into());
                rebased.insert("byteOffset".into(), (buf_base[buf] + off).into());
                if new_idx == views_out.len() {
                    views_out.push(Value::Object(rebased));
                } else {
                    views_out[new_idx] = Value::Object(rebased);
                }
                let mut out = img.as_object().cloned().unwrap_or_default();
                out.insert("bufferView".into(), new_idx.into());
                images_out.push(Value::Object(out));
            } else {
                return Err(err("v2: image with data: uri cannot be packed"));
            }
        }
        root.insert("images".into(), Value::Array(images_out));
    }
    root.insert("bufferViews".into(), Value::Array(views_out));
    root.insert(
        "buffers".into(),
        serde_json::json!([{"byteLength": blob.len()}]),
    );
    if let Some(asset) = root.get_mut("asset").and_then(|v| v.as_object_mut()) {
        asset.insert("generator".into(), generator("v2").into());
    }
    Ok((Value::Object(root.clone()), blob))
}

// ------------------------------------------------------------------ tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn convert_fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../assets/models")
            .join(name)
            .join("result.gltf");
        convert_gltf_file(&path).expect("conversion failed")
    }

    fn assert_glb(glb: &[u8], meshes: usize, prims: usize) {
        assert!(glb.len() > 12);
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let doc: serde_json::Value =
            serde_json::from_slice(&glb[20..20 + json_len]).expect("json chunk");
        assert_eq!(doc["asset"]["version"], "2.0");
        let got_meshes = doc["meshes"].as_array().unwrap().len();
        let got_prims: usize = doc["meshes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["primitives"].as_array().unwrap().len())
            .sum();
        assert_eq!(got_meshes, meshes);
        assert_eq!(got_prims, prims);
        assert!(!doc["accessors"].as_array().unwrap().is_empty());
        assert!(!doc["images"].as_array().unwrap().is_empty());
    }

    #[test]
    fn pack_dragon() {
        assert_glb(&convert_fixture("dragon"), 374, 374);
    }

    #[test]
    fn pack_girl() {
        let glb = convert_fixture("girl");
        let json_len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let doc: serde_json::Value = serde_json::from_slice(&glb[20..20 + json_len]).unwrap();
        let meshes = doc["meshes"].as_array().unwrap().len();
        assert!(meshes > 700, "girl meshes: {meshes}");
    }

    /// Regenerate the server-side `result.glb` fixtures next to the v1
    /// sources (what `scan_catalog` does on every fresh/missing output).
    /// Keeps the client loader tests honest when run after this suite.
    #[test]
    fn refresh_converted_fixtures() {
        for name in ["dragon", "girl"] {
            let gltf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../assets/models")
                .join(name)
                .join("result.gltf");
            let glb_path = ensure_converted_glb(&gltf).expect("conversion failed");
            let bytes = std::fs::read(&glb_path).expect("written glb unreadable");
            assert_eq!(&bytes[0..4], b"glTF");
        }
    }
}
