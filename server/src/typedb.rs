//! Type database (`types.tdb`) loaded at server start.
//!
//! The database is produced once by `cargo run -p tesjit-dwarf -- <binary>`
//! out of the DWARF debug info of a debug build, and records every Rust type
//! layout (size, alignment, field offsets) the compiled mod/script layer may
//! ever touch. Keeping it outside the binary means:
//!
//! - the server starts from a plain `Arc<TypeDb>`, no DWARF parsing at runtime
//! - the format is versioned (`head.version`) and validated on load
//! - the game and recompiled script code agree on layouts even across builds
//!
//! Later phases use it to give JIT-compiled code exact offsets for Rust-side
//! payloads (see `docs/tesjit-abi.md`).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bevy_ecs::prelude::Resource;
use bevy_log::{error, info, warn};

pub use tesjit_dwarf::{
    Field, Head, LeafField, OpaqueKind, TypeArgs, TypeDb, TypeDbDecodeError, TypeDef, TypeName,
    Variant, FORMAT_VERSION,
};

/// Path the type database is read from.
///
/// Order: `EXSOLIS_TYPES_DB` env var, `./types.tdb`, then `types.tdb` next to
/// the workspace root.
pub fn types_db_path() -> PathBuf {
    if let Some(path) = std::env::var_os("EXSOLIS_TYPES_DB") {
        return PathBuf::from(path);
    }
    let cwd = PathBuf::from("types.tdb");
    if cwd.is_file() {
        return cwd;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../types.tdb")
}

/// Load the type database, if one is present and valid.
///
/// A missing file is a warning (not an error) so the game stays runnable
/// before types have been exported for the current build.
pub fn load_types_db() -> Option<Arc<TypeDb>> {
    let path = types_db_path();
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(_) => {
            warn!(
                "typedb: no type database at {} (run `cargo run -p tesjit-dwarf -- <server-binary> -o types.tdb` to generate it)",
                path.display()
            );
            return None;
        }
    };
    match TypeDb::from_bytes(&bytes) {
        Ok(db) => {
            info!(
                "typedb: loaded {} types from {} (format v{}, producer: {})",
                db.len(),
                path.display(),
                db.head.version,
                db.head.producer
            );
            Some(Arc::new(db))
        }
        Err(e) => {
            error!("typedb: failed to load {}: {e}", path.display());
            None
        }
    }
}

/// Resource holding the loaded type database for systems/plugins to consult.
#[derive(Resource, Clone)]
pub struct ServerTypes(pub Arc<TypeDb>);

impl ServerTypes {
    pub fn load() -> Option<Self> {
        load_types_db().map(Self)
    }
}

impl std::ops::Deref for ServerTypes {
    type Target = Arc<TypeDb>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
