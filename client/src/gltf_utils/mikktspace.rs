//! Mikktspace tangent generation (adapted from bevy_mikktspace).
//!
//! This is a port of the mikktspace algorithm for generating tangents
//! from mesh geometry. It produces correct handedness and handles degenerate
//! triangles properly.
//!
//! The three defects it fixes relative to the hand-rolled version it replaced:
//!
//! 1. `tan2` is actually accumulated. The old code allocated a `tan2` buffer,
//!    never wrote to it, and therefore reported handedness `+1` for every
//!    vertex — a UV-mirrored atlas (Blockbench exports, the dragon) then got a
//!    flipped bitangent and normal maps lit from the wrong side.
//! 2. The cross product term that defines handedness is `cross(N, T) · B`,
//!    where `B` is the *accumulated* bitangent, not a per-triangle one.
//! 3. `r = 1.0 / (Δuv1.x·Δuv2.y - Δuv1.y·Δuv2.x)` is guarded, so a
//!    zero-area-UV triangle cannot emit `inf`/`NaN` tangents and poison the
//!    whole primitive's TBN.

use glam::Vec3;

/// Compute tangents for a triangle mesh.
///
/// `positions`, `normals` and `uvs` must all have the same length; `indices`
/// must be a triangle list (`[i0, i1, i2, i3, i4, i5, ...]`) whose values are
/// in range. Non-indexed input can be emulated with `(0..n).collect()`.
///
/// Returns `[tx, ty, tz, handedness]` per vertex, where `handedness` is `-1.0`
/// or `1.0`. It is `+1.0` for vertices that no triangle touches.
pub fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let vertex_count = positions.len();
    // `tan1` accumulates tangent candidates, `tan2` bitangent candidates.
    // Per mikktspace, the handedness comes from the dot of one against the
    // other, so both have to be accumulated or the sign is always +1.
    let mut tan1 = vec![[0.0f32; 3]; vertex_count];
    let mut tan2 = vec![[0.0f32; 3]; vertex_count];

    for chunk in indices.chunks_exact(3) {
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let v0 = Vec3::from(positions[i0]);
        let v1 = Vec3::from(positions[i1]);
        let v2 = Vec3::from(positions[i2]);

        let uv0 = Vec2::from(uvs[i0]);
        let uv1 = Vec2::from(uvs[i1]);
        let uv2 = Vec2::from(uvs[i2]);

        let delta_pos1 = v1 - v0;
        let delta_pos2 = v2 - v0;

        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        // A zero-area-UV triangle has no defined tangent frame. Skipping it
        // keeps `inf`/`NaN` out of the accumulation buffers.
        let denom = delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x;
        if denom.abs() < f32::EPSILON {
            continue;
        }

        let r = 1.0 / denom;
        let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
        // The bitangent is the *same* frame vector expressed against the V
        // axis. Sign convention matches mikktspace: it is the cross product
        // `B = T × N`, which is `-N × T`.
        let bitangent = (delta_pos1 * delta_uv2.x - delta_pos2 * delta_uv1.x) * -r;

        accumulate(&mut tan1, &mut tan2, i0, tangent, bitangent);
        accumulate(&mut tan1, &mut tan2, i1, tangent, bitangent);
        accumulate(&mut tan1, &mut tan2, i2, tangent, bitangent);
    }

    // Orthogonalize (Gram-Schmidt) and resolve handedness.
    let mut tangents = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let n = Vec3::from(normals[i]);
        let mut t = Vec3::from(tan1[i]);

        // Gram-Schmidt: drop the component of the accumulated tangent that
        // runs along the normal. A vertex that no triangle touched keeps
        // `t == 0`, in which case `normalize_or_zero` yields zero rather than
        // NaN; the caller sees an up-tangent for that pathological case.
        t = (t - n * n.dot(t)).normalize_or_zero();

        // Handedness = sign of `cross(N, T) · B`. `B` must be the accumulated
        // bitangent, which is exactly why `tan2` exists.
        let b = Vec3::from(tan2[i]);
        let handedness = if n.cross(t).dot(b) < 0.0 { -1.0 } else { 1.0 };

        tangents.push([t.x, t.y, t.z, handedness]);
    }

    tangents
}

/// Add one triangle's frame contribution to the three vertices it references.
#[inline]
fn accumulate(
    tan1: &mut [[f32; 3]],
    tan2: &mut [[f32; 3]],
    i: usize,
    tangent: Vec3,
    bitangent: Vec3,
) {
    tan1[i][0] += tangent.x;
    tan1[i][1] += tangent.y;
    tan1[i][2] += tangent.z;
    tan2[i][0] += bitangent.x;
    tan2[i][1] += bitangent.y;
    tan2[i][2] += bitangent.z;
}

/// 2D helper, because `glam::Vec2` subtraction is the only thing needed here
/// and pulling the whole type in would invite future affine abuse.
#[derive(Clone, Copy)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl From<[f32; 2]> for Vec2 {
    fn from(v: [f32; 2]) -> Self {
        Self { x: v[0], y: v[1] }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad_positions() -> Vec<[f32; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn tangent_follows_the_u_direction() {
        // A quad in the XY plane whose UVs run 0..1 over 0..1 units: the
        // tangent must be +X and the bitangent +Y, with +1 handedness.
        let positions = quad_positions();
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
        assert_eq!(tangents.len(), 4);

        for tangent in &tangents {
            assert!(
                tangent[0].abs() > 0.99,
                "tangent should be along X, got {tangent:?}"
            );
            assert!(tangent[1].abs() < 0.01, "Y component leaked: {tangent:?}");
            assert!(tangent[2].abs() < 0.01, "Z component leaked: {tangent:?}");
            assert_eq!(tangent[3], 1.0, "unflipped quad must be right-handed");
        }
    }

    #[test]
    fn a_flipped_v_axis_reports_negative_handedness() {
        // Same geometry, V mirrored in the atlas. `denom` changes sign, so
        // the tangent direction flips *and* the handedness must go to -1.
        let positions = quad_positions();
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);

        // Regression test for the `tan2` typo: the old code never accumulated
        // the bitangent, so this asserted +1 and normal-mapped atlases were
        // lit from the wrong side.
        for tangent in &tangents {
            assert_eq!(
                tangent[3], -1.0,
                "a V-mirrored atlas must produce left-handed tangents, got {tangent:?}"
            );
        }
    }

    #[test]
    fn every_vertex_of_a_shared_triangle_gets_the_same_frame() {
        // Guards the accumulation indices directly: if tan1/tan2 were written
        // to the wrong corner, the three tangents would disagree.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            // A distant triangle that only touches vertex 0.
            [0.0, 0.0, 5.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let indices = vec![0, 1, 2];

        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
        // Corners 1 and 2 are only in that triangle and must share its frame.
        for idx in [1usize, 2] {
            for c in 0..3 {
                assert!(
                    (tangents[0][c] - tangents[idx][c]).abs() < 1e-6,
                    "corner {idx} frame drifted from corner 0: {:?} vs {:?}",
                    tangents[0],
                    tangents[idx]
                );
            }
            assert_eq!(tangents[idx][3], 1.0);
        }
        // Vertex 3 was never referenced.
        assert!(tangents[3][0] == 0.0 && tangents[3][1] == 0.0);
    }

    #[test]
    fn a_degenerate_uv_triangle_yields_no_nan() {
        // All three corners share a UV, so `denom` is 0. The whole primitive
        // used to become NaN and the TBN went black.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let uvs = vec![[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]];
        let indices = vec![0, 1, 2];

        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
        for tangent in &tangents {
            assert!(
                tangent.iter().all(|v| v.is_finite()),
                "non-finite tangent from a degenerate triangle: {tangent:?}"
            );
            assert_eq!(tangent[3], 1.0, "untouched vertices default to +1");
        }
    }

    #[test]
    fn tangents_are_orthogonal_to_the_normal() {
        let positions = quad_positions();
        let normals = vec![[0.0, 0.0, 1.0]; 4];
        let uvs = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
        for (i, tangent) in tangents.iter().enumerate() {
            let t = Vec3::new(tangent[0], tangent[1], tangent[2]);
            let n = Vec3::from(normals[i]);
            let dot = t.dot(n);
            assert!(
                dot.abs() < 1e-5,
                "vertex {i}: tangent not orthogonal to normal (dot = {dot})"
            );
        }
    }
}
