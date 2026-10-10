//! Shared server-side asset tooling.
//!
//! `gltf_pack` lives here rather than in `main.rs` because the client
//! depends on it: `normalize_glb_textures` has to run in both places —
//! here while building the asset catalog, and in the client for models
//! loaded straight off disk.

pub mod gltf_pack;
