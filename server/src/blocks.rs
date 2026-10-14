//! Server-side block registry resource.
//!
//! The canonical [`BlockRegistry`] lives in `exsolis_common`; this module
//! only owns the Bevy resource wrapper around it so every server system
//! resolves names <-> ids against the same table.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use exsolis_common::{BlockId, BlockRegistry};

/// The server's block registry. Inserted once at startup with the frozen
/// M0 ids (`BlockRegistry::default_registry()`); mod blocks register here
/// later.
#[derive(Resource, Clone)]
pub struct ServerBlocks(pub BlockRegistry);

impl ServerBlocks {
    /// Numeric id for a block name, if registered.
    pub fn id(&self, name: &str) -> Option<BlockId> {
        self.0.id(name)
    }

    /// Block name for a numeric id, if known.
    pub fn name(&self, id: BlockId) -> Option<&str> {
        self.0.def(id).map(|d| d.name.as_str())
    }

    pub fn is_air(&self, id: BlockId) -> bool {
        self.0.is_air(id)
    }

    pub fn is_block_entity(&self, id: BlockId) -> bool {
        self.0.is_block_entity(id)
    }
}

pub struct BlocksPlugin;

impl Plugin for BlocksPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ServerBlocks(BlockRegistry::default_registry()));
    }
}
