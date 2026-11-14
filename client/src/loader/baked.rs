//! Baked block geometry shared by the mesher and all loader packs.
//!
//! The types now live in the [`bbmodel_rs`] crate (`crates/bbmodel_rs`); this
//! module re-exports them so the rest of the loader (`mcmodel`, `obj`,
//! `bbmodel`) keeps using `crate::loader::baked::…`.

pub use bbmodel_rs::{
    BakedModel, BakedQuad, BakedTri, BlockPack, RenderType, face_quad, rotate_point,
};
