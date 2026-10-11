//! Production-ready GLTF utilities adapted from bevy_gltf and bevy_animation.
//!
//! This module provides robust GLTF parsing, vertex attribute conversion,
//! coordinate system handling, and animation curve sampling — without any
//! Bevy ECS or renderer coupling.

pub mod convert_coordinates;
pub mod vertex_attributes;
pub mod gltf_ext;
pub mod animation;
pub mod mikktspace;

// Re-exports for convenience
pub use convert_coordinates::*;
pub use vertex_attributes::*;
pub use gltf_ext::*;
pub use animation::*;

#[cfg(test)]
mod tests;