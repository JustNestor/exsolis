//! `bbmodel_rs` — Blockbench `.bbmodel` model loading and animation
//! summaries, baked to the engine's voxel format.
//!
//! This crate is the home of the Blockbench loader that used to live in the
//! client crate (`client/src/loader/bbmodel.rs`) together with the shared
//! baked-geometry types it produces ([`baked`]).
//!
//! It is deliberately free of Bevy and of the game itself: the bake is a pure
//! function of a Blockbench project JSON document, so the geometry pipeline
//! can be unit-tested, reused by other consumers (the block-preview
//! rasterizer, mod tooling, the server-side Thaumcraft prop scan) and later
//! swapped for a full animation sampler without touching the mesher.
//!
//! # Pipeline
//!
//! ```text
//! Blockbench project JSON ──▶ outliner walk (groups, pivots, euler) ──▶ quads / tris
//!                                                            │
//!                                                            ▼
//!                                             bake normalizes by bounding box
//!                                             into 0-1 block space
//! ```
//!
//! Animations are parsed to summaries only — [`BbAnimInfo`] carries the bone
//! → channels → keyframe map — while the runtime keeps playing glTF clips
//! today, so Bedrock bone sampling stays deferred.

pub mod baked;
pub mod bbmodel;

pub use bbmodel::{
    BbAnimInfo, BbLoadReport, BbModel, BbTextureImage, load_bbmodel, parse_bbmodel,
};
pub use baked::{BakedModel, BakedQuad, BakedTri, BlockPack, RenderType, face_quad, rotate_point};
