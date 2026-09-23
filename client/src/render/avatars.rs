//! Remote-player heads.
//!
//! One local-space mesh is uploaded once and drawn with each player's
//! `Transform` as a model matrix. Rebuilding world-space verts every
//! interpolated frame was fighting the chunk bump allocator and painting
//! every moving player onto the GPU every tick.
//!
//! Texture keys must match `TextureManager` (`"blocks/stone"`, not
//! `"stone"`). The old keys missed, `get` fell back to the first loaded
//! PNG (often leaves), and the whole front face was "glass" — which the
//! cutout shader discards — so heads looked like leaf cubes with no eyes.

use crate::mesh::{ChunkMeshData, ChunkVertex};
use crate::texture::{TextureManager, sampled_image_index};
use crate::vulkan_prelude::VulkanContext;

use bevy_ecs::prelude::*;

use super::gpu::{PendingWrite, RenderState, upload_writes};

const HALF: f32 = 0.25;
const HEAD_Y0: f32 = 1.35;
const HEAD_Y1: f32 = 1.85;
/// Sit the eyes just in front of the +Z face so they don't z-fight.
const EYE_Z: f32 = HALF + 0.004;
const PUPIL_Z: f32 = HALF + 0.006;

pub(super) fn ensure_avatar_mesh(
    vk: Res<VulkanContext>,
    textures: Option<Res<TextureManager>>,
    mut render_state: ResMut<RenderState>,
) {
    if render_state.shared.read().unwrap().avatar.is_some() {
        return;
    }
    let Some(textures) = textures else {
        return;
    };
    if !textures.has_textures() {
        return;
    }

    let Some(stone_id) = textures.get("blocks/stone") else {
        return;
    };
    let stone = sampled_image_index(stone_id);
    // Snow is opaque white — glass is a cutout and would vanish as "eyes".
    let whites = textures
        .get("blocks/snow")
        .map(sampled_image_index)
        .unwrap_or(stone);
    let pupils = textures
        .get("blocks/log_side")
        .or_else(|| textures.get("blocks/log_top"))
        .map(sampled_image_index)
        .unwrap_or(stone);

    let mesh = build_player_head_mesh(stone, whites, pupils);
    let Some(alloc) = render_state
        .allocator
        .allocate(mesh.vertices.len() as u32, mesh.indices.len() as u32)
    else {
        bevy_log::warn_once!("mesh buffer exhausted; cannot upload avatar head");
        return;
    };

    let vertex_writes = [PendingWrite {
        offset_bytes: alloc.vertex_offset_bytes(),
        data: mesh.vertices,
    }];
    let index_writes = [PendingWrite {
        offset_bytes: alloc.index_offset_bytes(),
        data: mesh.indices,
    }];
    upload_writes(
        &vk,
        render_state.vertex_buffer_id,
        render_state.index_buffer_id,
        &vertex_writes,
        &index_writes,
    );

    render_state.shared.write().unwrap().avatar = Some(alloc);
}

fn push_quad(
    vertices: &mut Vec<ChunkVertex>,
    indices: &mut Vec<u32>,
    positions: [[f32; 3]; 4],
    uv_rect: [[f32; 2]; 4],
    sampled_text: u32,
    shade: f32,
) {
    let start = vertices.len() as u32;
    for (position, uv) in positions.into_iter().zip(uv_rect) {
        vertices.push(ChunkVertex {
            position,
            uv,
            sampled_text,
            shade,
        });
    }
    indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
}

/// Head in local space, feet at the origin (same offset the old world-space
/// builder used relative to `Transform.translation`).
fn build_player_head_mesh(stone: u32, whites: u32, pupils: u32) -> ChunkMeshData {
    let mut vertices = Vec::with_capacity(48);
    let mut indices = Vec::with_capacity(72);

    let min_x = -HALF;
    let max_x = HALF;
    let min_y = HEAD_Y0;
    let max_y = HEAD_Y1;
    let min_z = -HALF;
    let max_z = HALF;

    const UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let cube_faces: [(f32, [[f32; 3]; 4]); 6] = [
        // Top
        (
            1.0,
            [
                [min_x, max_y, min_z],
                [max_x, max_y, min_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z],
            ],
        ),
        // Bottom
        (
            0.5,
            [
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, min_y, min_z],
                [min_x, min_y, min_z],
            ],
        ),
        // Back (−Z)
        (
            0.8,
            [
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
            ],
        ),
        // Front (+Z) — stone, not glass. Eyes are separate quads.
        (
            0.8,
            [
                [min_x, max_y, max_z],
                [max_x, max_y, max_z],
                [max_x, min_y, max_z],
                [min_x, min_y, max_z],
            ],
        ),
        // Left
        (
            0.65,
            [
                [min_x, max_y, min_z],
                [min_x, max_y, max_z],
                [min_x, min_y, max_z],
                [min_x, min_y, min_z],
            ],
        ),
        // Right
        (
            0.65,
            [
                [max_x, max_y, max_z],
                [max_x, max_y, min_z],
                [max_x, min_y, min_z],
                [max_x, min_y, max_z],
            ],
        ),
    ];

    for (shade, positions) in cube_faces {
        push_quad(&mut vertices, &mut indices, positions, UVS, stone, shade);
    }

    // Eyes sit in the upper half of the +Z face. Winding matches the front.
    let eye_y0 = 1.58;
    let eye_y1 = 1.70;
    let eyes = [
        // left
        [-0.15, -0.03],
        // right
        [0.03, 0.15],
    ];
    for [x0, x1] in eyes {
        push_quad(
            &mut vertices,
            &mut indices,
            [
                [x0, eye_y1, EYE_Z],
                [x1, eye_y1, EYE_Z],
                [x1, eye_y0, EYE_Z],
                [x0, eye_y0, EYE_Z],
            ],
            UVS,
            whites,
            1.0,
        );

        let px0 = x0 + 0.03;
        let px1 = x1 - 0.03;
        let py0 = eye_y0 + 0.03;
        let py1 = eye_y1 - 0.03;
        push_quad(
            &mut vertices,
            &mut indices,
            [
                [px0, py1, PUPIL_Z],
                [px1, py1, PUPIL_Z],
                [px1, py0, PUPIL_Z],
                [px0, py0, PUPIL_Z],
            ],
            UVS,
            pupils,
            0.55,
        );
    }

    ChunkMeshData { vertices, indices }
}
