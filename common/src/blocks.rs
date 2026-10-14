//! Data-driven block registry. Replaces the old hardcoded `BlockType` enum.
//!
//! Numeric [`BlockId`]s are what go into `ChunkBlocks` palettes and the GPU
//! world buffer. String names (`"exsolis:stone"`) are what mods, Lua, and
//! `PlaceBlock`/`BlockUpdate` messages use. The server owns the id assignment;
//! for M0 both sides build the same [`BlockRegistry::default_registry()`] so
//! ids agree without a sync message (a registry-sync message comes later
//! with mod-defined blocks).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Numeric block handle. `0` is always air.
pub type BlockId = u16;

pub const AIR_ID: BlockId = 0;

/// Static definition of one block type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDef {
    pub name: String,
    pub solid: bool,
    pub transparent: bool,
    pub block_entity: bool,
}

impl BlockDef {
    pub fn is_air(&self) -> bool {
        self.name == "exsolis:air"
    }
}

/// Registry mapping names <-> ids. Ids are indices into `defs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    #[serde(skip)]
    by_name: HashMap<String, BlockId>,
}

impl BlockRegistry {
    pub fn new() -> Self {
        let mut r = Self {
            defs: Vec::new(),
            by_name: HashMap::new(),
        };
        r.register("exsolis:air", false, true, false);
        r
    }

    fn register(&mut self, name: &str, solid: bool, transparent: bool, block_entity: bool) -> BlockId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = self.defs.len() as BlockId;
        self.defs.push(BlockDef {
            name: name.to_string(),
            solid,
            transparent,
            block_entity,
        });
        self.by_name.insert(name.to_string(), id);
        id
    }

    /// M0 built-in set. Order is frozen (air = 0 .. chest = 10) so old
    /// numeric ids keep working while Lua/mods move to names.
    pub fn default_registry() -> Self {
        let mut r = Self::new();
        r.register("exsolis:stone", true, false, false); // 1
        r.register("exsolis:dirt", true, false, false); // 2
        r.register("exsolis:grass", true, false, false); // 3
        r.register("exsolis:sand", true, false, false); // 4
        r.register("exsolis:water", false, true, false); // 5
        r.register("exsolis:snow", true, false, false); // 6
        r.register("exsolis:glass", true, true, false); // 7
        r.register("exsolis:log", true, false, false); // 8
        r.register("exsolis:leaves", true, true, false); // 9
        r.register("exsolis:chest", true, false, true); // 10
        r
    }

    /// Register a mod block. Returns the assigned id.
    pub fn register_mod(
        &mut self,
        name: String,
        solid: bool,
        transparent: bool,
        block_entity: bool,
    ) -> BlockId {
        if let Some(&id) = self.by_name.get(&name) {
            return id;
        }
        let id = self.defs.len() as BlockId;
        self.defs.push(BlockDef {
            name: name.clone(),
            solid,
            transparent,
            block_entity,
        });
        self.by_name.insert(name, id);
        id
    }

    pub fn rebuild_index(&mut self) {
        self.by_name.clear();
        for (i, d) in self.defs.iter().enumerate() {
            self.by_name.insert(d.name.clone(), i as BlockId);
        }
    }

    pub fn id(&self, name: &str) -> Option<BlockId> {
        self.by_name.get(name).copied()
    }

    pub fn def(&self, id: BlockId) -> Option<&BlockDef> {
        self.defs.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &BlockDef)> {
        self.defs
            .iter()
            .enumerate()
            .map(|(i, d)| (i as BlockId, d))
    }

    pub fn is_air(&self, id: BlockId) -> bool {
        id == AIR_ID
    }

    pub fn is_solid(&self, id: BlockId) -> bool {
        self.def(id).is_some_and(|d| d.solid)
    }

    pub fn is_transparent(&self, id: BlockId) -> bool {
        // Unknown ids read as transparent so a missing def never culls
        // a face it shouldn't (holes to the void are worse than overdraw).
        self.def(id).is_none_or(|d| d.transparent)
    }

    pub fn is_block_entity(&self, id: BlockId) -> bool {
        self.def(id).is_some_and(|d| d.block_entity)
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}
