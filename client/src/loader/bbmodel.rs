//! Blockbench `.bbmodel` loader (cubes + meshes + groups + textures).
//!
//! The implementation lives in the [`bbmodel_rs`] crate (`crates/bbmodel_rs`);
//! this module re-exports it so gameplay code keeps using
//! `crate::loader::bbmodel::…`.

pub use bbmodel_rs::{
    BbAnimInfo, BbLoadReport, BbModel, BbTextureImage, load_bbmodel, parse_bbmodel,
};
