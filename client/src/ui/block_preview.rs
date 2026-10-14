//! CPU isometric previews for `<img src="block#namespace:name">`.
//!
//! A 48x48 RGBA rasterization of the block's [`BakedModel`] quads (tris from
//! non-quad sources are ignored): each needed texture PNG is loaded from the
//! MC pack, corners are projected with classic 2:1 isometric, quads are
//! painter-sorted by depth and rasterized as two triangles with
//! nearest-neighbor texture sampling. Faces pointing away from the viewer
//! are culled; quads whose texture file is missing are skipped.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy_log::debug;

use crate::loader::McAssets;
use crate::loader::baked::{BakedModel, BakedQuad};
use crate::ui::img_assets::DecodedImage;

/// Fixed preview size advertised by `probe_size` and produced here.
pub const PREVIEW_SIZE: u32 = 48;

/// Viewer direction in block space (camera sits at +x/+y/+z looking back at
/// the origin). Faces with `dot(normal, VIEW) <= 0` are culled, except quads
/// with no cull direction (crosses, plants), which always draw.
const VIEW: [f32; 3] = [1.0, 1.0, 1.0];
const COS30: f32 = 0.8660254;
const SIN30: f32 = 0.5;
/// Source texels with alpha below this are treated as cutout holes.
const ALPHA_CUTOFF: u8 = 128;

/// Resolver mapping: a baked texture key (`minecraft/block/stone`) is
/// `<namespace>/<rest>`, loaded from
/// `<pack_dir>/assets/<namespace>/textures/<rest>.png`.
///
/// Quirk: some baked keys double the `block/` segment (stone's model names
/// its texture `minecraft:block/stone`, and normalization prepends another
/// `block/` → `minecraft/block/block/stone`). When the literal path misses,
/// one doubled `block/block/` is collapsed and retried, so previews resolve
/// the same file the key names.
pub fn texture_path_for(pack_dir: &Path, key: &str) -> Option<PathBuf> {
    let (namespace, rest) = key.split_once('/')?;
    if namespace.is_empty() || rest.is_empty() {
        return None;
    }
    let base = pack_dir
        .join("assets")
        .join(namespace)
        .join("textures");
    for candidate in [
        base.join(format!("{rest}.png")),
        base.join(format!("{}.png", rest.replace("block/block/", "block/"))),
    ] {
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Rasterize `name`'s baked model to a 48x48 RGBA preview.
///
/// `baked` is the full baked-model map (legacy `exsolis:*` names fall back
/// to their MC look-alikes, same as the mesher). Errors when the model is
/// missing, has no quads, or no pixel could be drawn.
pub fn decode_block(
    name: &str,
    baked: &HashMap<String, BakedModel>,
    pack_dir: &Path,
) -> Result<DecodedImage, String> {
    let Some(model) = McAssets::model_for_baked(baked, name) else {
        return Err(format!("no baked model for `{name}`"));
    };
    if model.quads.is_empty() {
        return Err(format!("`{name}` has no quads to preview"));
    }
    // Cache decoded PNGs for the duration of this call; a cube reuses one
    // texture across all six faces.
    let mut tex_cache: HashMap<String, (u32, u32, Vec<u8>)> = HashMap::new();
    let mut missing_logged = false;

    // Project every kept corner so the whole model fits the 48x48 box.
    let mut kept: Vec<&BakedQuad> = Vec::new();
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for quad in &model.quads {
        if !is_visible(quad) {
            continue;
        }
        let tex = texture_path_for(pack_dir, &quad.texture)
            .and_then(|p| load_texture(&mut tex_cache, &quad.texture, &p));
        let Some((tw, th, _)) = tex else {
            if !missing_logged {
                debug!("block preview `{name}`: missing texture, skipping quads");
                missing_logged = true;
            }
            continue;
        };
        let _ = (tw, th);
        kept.push(quad);
        for c in &quad.corners {
            let (sx, sy) = project(c);
            min[0] = min[0].min(sx);
            min[1] = min[1].min(sy);
            max[0] = max[0].max(sx);
            max[1] = max[1].max(sy);
        }
    }
    if kept.is_empty() {
        return Err(format!("nothing to preview for `{name}`"));
    }

    let span_x = (max[0] - min[0]).max(1e-6);
    let span_y = (max[1] - min[1]).max(1e-6);
    let pad = 3.0f32;
    let scale = ((PREVIEW_SIZE as f32 - 2.0 * pad) / span_x)
        .min((PREVIEW_SIZE as f32 - 2.0 * pad) / span_y);
    let map = |sx: f32, sy: f32| -> (f32, f32) {
        // Screen y grows downwards; iso y grows upwards, hence the flip.
        (
            pad + (sx - min[0]) * scale,
            pad + (max[1] - sy) * scale,
        )
    };

    // Painter order: far quads first, near quads overdraw them. The z-buffer
    // below makes the order mostly redundant, but it keeps coplanar seams
    // deterministic.
    kept.sort_by(|a, b| {
        depth_of(a)
            .partial_cmp(&depth_of(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let size = PREVIEW_SIZE as usize;
    let mut rgba = vec![0u8; size * size * 4];
    let mut zbuf = vec![f32::NEG_INFINITY; size * size];
    for quad in kept {
        let shade = shade_for(&quad.normal);
        let path = texture_path_for(pack_dir, &quad.texture).expect("cached above");
        let Some((tw, th, px)) = load_texture(&mut tex_cache, &quad.texture, &path) else {
            continue;
        };
        let pts: Vec<(f32, f32, f32, f32, f32)> = quad
            .corners
            .iter()
            .zip(quad.uvs.iter())
            .map(|(c, uv)| {
                let (sx, sy) = map(project(c).0, project(c).1);
                (sx, sy, (c[0] + c[2]) - c[1], uv[0], uv[1])
            })
            .collect();
        for tri in [(0, 1, 2), (0, 2, 3)] {
            draw_tri(
                &pts[tri.0],
                &pts[tri.1],
                &pts[tri.2],
                tw,
                th,
                &px,
                shade,
                &mut rgba,
                &mut zbuf,
            );
        }
    }

    let opaque = rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
    if opaque == 0 {
        return Err(format!("nothing to preview for `{name}`"));
    }
    Ok(DecodedImage {
        width: PREVIEW_SIZE,
        height: PREVIEW_SIZE,
        rgba,
    })
}

fn is_visible(quad: &BakedQuad) -> bool {
    if quad.cull.is_none() {
        return true;
    }
    quad.normal[0] * VIEW[0] + quad.normal[1] * VIEW[1] + quad.normal[2] * VIEW[2] > 0.0
}

/// Larger = nearer the viewer (front-top wins over back-bottom).
fn depth_of(quad: &BakedQuad) -> f32 {
    let (mut xyz, mut y) = (0.0f32, 0.0f32);
    for c in &quad.corners {
        xyz += c[0] + c[2];
        y += c[1];
    }
    xyz / 4.0 - y / 4.0
}

fn shade_for(normal: &[f32; 3]) -> f32 {
    if normal[1] > 0.5 {
        1.0
    } else if normal[1] < -0.5 {
        0.55
    } else if normal[0].abs() > 0.5 {
        0.8
    } else if normal[2].abs() > 0.5 {
        0.7
    } else {
        0.85
    }
}

fn project(c: &[f32; 3]) -> (f32, f32) {
    ((c[0] - c[2]) * COS30, (c[0] + c[2]) * SIN30 - c[1])
}

fn load_texture(
    cache: &mut HashMap<String, (u32, u32, Vec<u8>)>,
    key: &str,
    path: &Path,
) -> Option<(u32, u32, Vec<u8>)> {
    if let Some(hit) = cache.get(key) {
        return Some(hit.clone());
    }
    let img = image::open(path).ok()?.into_rgba8();
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return None;
    }
    let entry = (w, h, img.into_raw());
    cache.insert(key.to_owned(), entry.clone());
    Some(entry)
}

type Vert = (f32, f32, f32, f32, f32); // (x, y, depth, u, v)

#[allow(clippy::too_many_arguments)]
fn draw_tri(
    a: &Vert,
    b: &Vert,
    c: &Vert,
    tw: u32,
    th: u32,
    px: &[u8],
    shade: f32,
    rgba: &mut [u8],
    zbuf: &mut [f32],
) {
    let size = PREVIEW_SIZE as i32;
    let min_x = (a.0.min(b.0).min(c.0).floor() as i32).clamp(0, size - 1);
    let max_x = (a.0.max(b.0).max(c.0).ceil() as i32).clamp(0, size - 1);
    let min_y = (a.1.min(b.1).min(c.1).floor() as i32).clamp(0, size - 1);
    let max_y = (a.1.max(b.1).max(c.1).ceil() as i32).clamp(0, size - 1);
    let denom = (b.1 - c.1) * (a.0 - c.0) + (c.0 - b.0) * (a.1 - c.1);
    if denom.abs() < 1e-9 {
        return;
    }
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px_ = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let w1 = ((b.1 - c.1) * (px_ - c.0) + (c.0 - b.0) * (py - c.1)) / denom;
            let w2 = ((c.1 - a.1) * (px_ - c.0) + (a.0 - c.0) * (py - c.1)) / denom;
            let w3 = 1.0 - w1 - w2;
            if w1 < -0.001 || w2 < -0.001 || w3 < -0.001 {
                continue;
            }
            let depth = w1 * a.2 + w2 * b.2 + w3 * c.2;
            let idx = (y as usize) * (size as usize) + (x as usize);
            if depth <= zbuf[idx] {
                continue;
            }
            let u = (w1 * a.3 + w2 * b.3 + w3 * c.3).clamp(0.0, 1.0);
            let v = (w1 * a.4 + w2 * b.4 + w3 * c.4).clamp(0.0, 1.0);
            // Baked UVs have v=0 at the top edge, matching top-first rows.
            let tx = ((u * tw as f32) as u32).min(tw - 1) as usize;
            let ty = ((v * th as f32) as u32).min(th - 1) as usize;
            let s = &px[(ty * tw as usize + tx) * 4..][..4];
            if s[3] < ALPHA_CUTOFF {
                continue;
            }
            let o = idx * 4;
            rgba[o] = (s[0] as f32 * shade) as u8;
            rgba[o + 1] = (s[1] as f32 * shade) as u8;
            rgba[o + 2] = (s[2] as f32 * shade) as u8;
            rgba[o + 3] = s[3];
            zbuf[idx] = depth;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn rasterizes_stone_preview() {
        let Some(pack_dir) = crate::loader::mc_pack_dir() else {
            return;
        };
        let pack = crate::loader::McPack::load(&pack_dir, "minecraft")
            .expect("pack loads in dev");
        let (models, _) = pack.bake_all();
        let baked: HashMap<String, BakedModel> = models
            .into_iter()
            .map(|(k, v)| (format!("minecraft:{k}"), v))
            .collect();
        let img = decode_block("minecraft:stone", &baked, &pack_dir).expect("stone previews");
        assert_eq!((img.width, img.height), (PREVIEW_SIZE, PREVIEW_SIZE));
        let opaque = img.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
        let total = (PREVIEW_SIZE * PREVIEW_SIZE) as usize;
        assert!(
            opaque * 100 / total > 10,
            "stone preview is mostly empty ({opaque}/{total} opaque)"
        );
    }
}
