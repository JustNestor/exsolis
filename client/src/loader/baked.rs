//! Baked block geometry shared by the mesher and all loader packs.
//!
//! The engine mesher only sees [`BakedModel`] (quads/tris in 0-1 block
//! space, 0-1 UVs with v=0 at the top edge, texture keys like
//! `"minecraft:block/stone"`). Each source game implements [`BlockPack`]:
//! [`crate::mcmodel::McPack`] (latest-MC JSON) or the minimal OBJ loader
//! in [`crate::obj`]. Future games add another pack, never mesher changes.

use std::collections::HashMap;

/// How the GPU should draw a baked model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderType {
    #[default]
    Opaque,
    /// Alpha-tested (leaves, glass, flowers, crops). No blending.
    Cutout,
    /// Blended, drawn after opaque (water, ice).
    Translucent,
}

/// One axis-aligned face. Corners are in 0-1 block space, ordered
/// counter-clockwise seen from outside (matches `client/mesh.rs` winding:
/// indices `[0,1,2, 0,2,3]`).
#[derive(Debug, Clone)]
pub struct BakedQuad {
    pub corners: [[f32; 3]; 4],
    pub normal: [f32; 3],
    pub uvs: [[f32; 2]; 4],
    /// Texture key, e.g. `"minecraft:block/stone"`.
    pub texture: String,
    /// Face-culling direction, e.g. `[0,1,0]` for up.
    /// `None` means never culled (crosses, plants).
    pub cull: Option<[i32; 3]>,
    pub tinted: bool,
}

/// One triangle from non-quad sources (OBJ). Same conventions as [`BakedQuad`].
#[derive(Debug, Clone)]
pub struct BakedTri {
    pub verts: [[f32; 3]; 3],
    pub normal: [f32; 3],
    pub uvs: [[f32; 2]; 3],
    pub texture: String,
}

#[derive(Debug, Clone, Default)]
pub struct BakedModel {
    pub quads: Vec<BakedQuad>,
    pub tris: Vec<BakedTri>,
    pub render_type: RenderType,
}

impl BakedModel {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.tris.is_empty()
    }

    /// Fallback full cube with one texture (missing model or unsupported
    /// feature). Keeps the world hole-free.
    pub fn fallback_cube(texture: &str, render_type: RenderType) -> Self {
        let mut quads = Vec::with_capacity(6);
        for (normal, cull) in [
            ([0.0, 1.0, 0.0], [0, 1, 0]),
            ([0.0, -1.0, 0.0], [0, -1, 0]),
            ([-1.0, 0.0, 0.0], [-1, 0, 0]),
            ([1.0, 0.0, 0.0], [1, 0, 0]),
            ([0.0, 0.0, 1.0], [0, 0, 1]),
            ([0.0, 0.0, -1.0], [0, 0, -1]),
        ] {
            quads.push(face_quad(
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                normal,
                Some(cull),
                texture,
            ));
        }
        Self {
            quads,
            tris: Vec::new(),
            render_type,
        }
    }
}

/// One axis-aligned face of the box `[min,max]` (0-1 space) with outward
/// normal `normal`. Corner order matches the chunk mesher's
/// `face_positions` layout so winding stays CCW-from-outside.
/// Rotate a point around `origin` by `angle_deg` about `axis` (shared by
/// the MC and Blockbench bakers: element/cube rotations).
pub fn rotate_point(p: [f32; 3], origin: [f32; 3], axis: char, angle_deg: f32) -> [f32; 3] {
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

pub fn face_quad(
    min: [f32; 3],
    max: [f32; 3],
    normal: [f32; 3],
    cull: Option<[i32; 3]>,
    texture: &str,
) -> BakedQuad {
    let corners = match normal {
        [0.0, 1.0, 0.0] => [
            [min[0], max[1], min[2]],
            [max[0], max[1], min[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ],
        [0.0, -1.0, 0.0] => [
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], min[1], min[2]],
            [min[0], min[1], min[2]],
        ],
        [0.0, 0.0, -1.0] => [
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
        ],
        [1.0, 0.0, 0.0] => [
            [max[0], max[1], max[2]],
            [max[0], max[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], min[1], max[2]],
        ],
        [0.0, 0.0, 1.0] => [
            [min[0], max[1], max[2]],
            [max[0], max[1], max[2]],
            [max[0], min[1], max[2]],
            [min[0], min[1], max[2]],
        ],
        _ => [
            [min[0], max[1], min[2]],
            [min[0], max[1], max[2]],
            [min[0], min[1], max[2]],
            [min[0], min[1], min[2]],
        ],
    };
    BakedQuad {
        corners,
        normal,
        uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        texture: texture.to_string(),
        cull,
        tinted: false,
    }
}

/// A source of baked block models (one per voxel game / mod format).
pub trait BlockPack: Send + Sync {
    fn namespace(&self) -> &str;
    /// Bake `block` in `state` (blockstate properties). Empty = default.
    fn bake(&self, block: &str, state: &HashMap<String, String>) -> Option<BakedModel>;
    /// All block names this pack knows about.
    fn blocks(&self) -> Vec<String>;
}
