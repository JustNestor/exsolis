//! `tesjit-dwarf` — export Rust type layouts from DWARF into a loadable
//! database.
//!
//! The exporter is meant to run once per build (CI or a `just` recipe) against
//! a binary compiled with debug info, producing a `types.tdb` that the server
//! loads at startup:
//!
//! ```text
//! tesjit-dwarf target/debug/exsolis_server_bin -o types.tdb
//! ```

pub mod export;
pub mod schema;

pub use export::{ExportError, ExportStats, export};
pub use schema::{
    ArrayType, Discriminant, EnumType, FORMAT_VERSION, Field, Head, LeafField, OpaqueKind,
    Resolution, Resolved, StructType, TypeArgs, TypeDb, TypeDbDecodeError, TypeDbError, TypeDef,
    TypeName, Variant,
};
