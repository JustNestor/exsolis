//! Minimal OBJ loader (positions + texcoords + normals + materials).
//!
//! Covers Forge-style static mod models (and future mod machines) without a
//! new dependency. Triangulates n-gons as fans; quads pass through as two
//! triangles with matching UVs. Coordinates are normalized by the file's
//! bounding box into 0-1 space unless the file already fits.

use std::collections::HashMap;

use crate::loader::baked::BakedTri;

#[derive(Debug, Clone, Default)]
pub struct ObjMesh {
    pub tris: Vec<BakedTri>,
    /// `mtl name -> diffuse texture path` from the `.mtl` file, if any.
    pub materials: HashMap<String, String>,
}

pub fn parse_obj(text: &str, default_texture: &str) -> Result<ObjMesh, String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut texcoords: Vec<[f32; 2]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut mesh = ObjMesh::default();
    let mut current_mtl = default_texture.to_string();
    // mtl name -> texture, resolved at the end (mtllib may come after use).
    let mut pending_mtls: Vec<String> = Vec::new();

    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(kind) = parts.next() else { continue };
        match kind {
            "v" => {
                let n: Vec<f32> = parts.map(|p| p.parse().unwrap_or(0.0)).collect();
                if n.len() >= 3 {
                    positions.push([n[0], n[1], n[2]]);
                }
            }
            "vt" => {
                let n: Vec<f32> = parts.map(|p| p.parse().unwrap_or(0.0)).collect();
                if n.len() >= 2 {
                    texcoords.push([n[0], n[1]]);
                }
            }
            "vn" => {
                let n: Vec<f32> = parts.map(|p| p.parse().unwrap_or(0.0)).collect();
                if n.len() >= 3 {
                    normals.push([n[0], n[1], n[2]]);
                }
            }
            "usemtl" => {
                current_mtl = parts.next().unwrap_or(default_texture).to_string();
                if !pending_mtls.contains(&current_mtl) {
                    pending_mtls.push(current_mtl.clone());
                }
            }
            "f" => {
                let verts: Vec<(usize, Option<usize>, Option<usize>)> = parts
                    .map(|spec| {
                        let mut idx = spec.split('/');
                        let p = idx.next().and_then(|s| s.parse::<usize>().ok());
                        let t = idx.next().and_then(|s| {
                            if s.is_empty() {
                                None
                            } else {
                                s.parse::<usize>().ok()
                            }
                        });
                        let n = idx.next().and_then(|s| {
                            if s.is_empty() {
                                None
                            } else {
                                s.parse::<usize>().ok()
                            }
                        });
                        (p.unwrap_or(0), t, n)
                    })
                    .collect();
                if verts.len() < 3 {
                    return Err(format!("line {}: face has <3 verts", lineno + 1));
                }
                // Fan triangulation.
                for i in 1..verts.len() - 1 {
                    let tri = [&verts[0], &verts[i], &verts[i + 1]];
                    let mut vs = [[0.0; 3]; 3];
                    let mut uv = [[0.0; 2]; 3];
                    let mut nm: Option<[f32; 3]> = None;
                    for (j, (p, t, n)) in tri.iter().enumerate() {
                        vs[j] = positions.get(p.saturating_sub(1)).copied().unwrap_or([0.0; 3]);
                        if let Some(ti) = t {
                            let tc = texcoords.get(ti.saturating_sub(1)).copied().unwrap_or([0.0, 0.0]);
                            uv[j] = [tc[0].clamp(0.0, 1.0), 1.0 - tc[1].clamp(0.0, 1.0)];
                        }
                        if nm.is_none()
                            && let Some(ni) = n
                        {
                            nm = normals.get(ni.saturating_sub(1)).copied();
                        }
                    }
                    let normal = nm.unwrap_or_else(|| tri_normal(vs[0], vs[1], vs[2]));
                    mesh.tris.push(BakedTri {
                        verts: vs,
                        normal,
                        uvs: uv,
                        texture: current_mtl.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    // Normalize into 0-1 space when the file uses modelling units.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for tri in &mesh.tris {
        for v in &tri.verts {
            for a in 0..3 {
                min[a] = min[a].min(v[a]);
                max[a] = max[a].max(v[a]);
            }
        }
    }
    let span = [
        (max[0] - min[0]).max(1e-6),
        (max[1] - min[1]).max(1e-6),
        (max[2] - min[2]).max(1e-6),
    ];
    let biggest = span[0].max(span[1]).max(span[2]);
    if biggest > 1.5 || min != [0.0, 0.0, 0.0] {
        for tri in &mut mesh.tris {
            for v in tri.verts.iter_mut() {
                for a in 0..3 {
                    v[a] = (v[a] - min[a]) / biggest;
                }
            }
        }
    }
    for m in pending_mtls {
        mesh.materials.entry(m).or_insert_with(|| default_texture.to_string());
    }
    Ok(mesh)
}

/// Parse a `.mtl` companion: returns `material name -> map_Kd path`.
pub fn parse_mtl(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("newmtl") => current = parts.next().map(|s| s.to_string()),
            Some("map_Kd") => {
                if let (Some(name), Some(path)) = (current.clone(), parts.next()) {
                    out.insert(name, path.to_string());
                }
            }
            _ => {}
        }
    }
    out
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
    [n[0] / len, n[1] / len, n[2] / len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quad_and_fan() {
        let text = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 1.5 0\nf 1 2 3 4\nf 2 3 5\n";
        let mesh = parse_obj(text, "tex").expect("parses");
        assert_eq!(mesh.tris.len(), 3); // quad->2 + tri->1
    }

    #[test]
    fn parses_mtl() {
        let m = parse_mtl("newmtl wood\nmap_Kd wood.png\n");
        assert_eq!(m.get("wood").map(|s| s.as_str()), Some("wood.png"));
    }
}
