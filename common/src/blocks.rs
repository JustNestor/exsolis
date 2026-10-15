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
    /// False for items and multiblock parts: cannot be placed directly.
    #[serde(default = "default_true")]
    pub placeable: bool,
    /// True for multiblock parts: hidden from creative/`api.blocks()`.
    #[serde(default)]
    pub hidden: bool,
    /// Set on masters (beds, doors): placing/breaking resolves the parts.
    #[serde(default)]
    pub multiblock: Option<MultiblockLayout>,
}

fn default_true() -> bool {
    true
}

/// Generic N-cell structure. Offsets are defined for [`Facing::North`]
/// (-z) and rotated by placement facing. Every part names a hidden,
/// unplaceable registry entry baked from the base block with a fixed state
/// (e.g. `half=top`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiblockLayout {
    pub parts: Vec<MultiblockPart>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultiblockPart {
    /// Cell offset from the master (facing-north frame).
    pub offset: [i32; 3],
    /// Hidden part entry, e.g. `"minecraft:oak_door_top"`.
    pub part: String,
}

/// Placement facing. Yaw 0 looks along -z (see `FlyCamera::forward`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Facing {
    North,
    East,
    South,
    West,
}

impl Facing {
    pub fn from_yaw(yaw: f32) -> Self {
        use std::f32::consts::PI;
        let mut a = yaw % (2.0 * PI);
        if a > PI {
            a -= 2.0 * PI;
        }
        if a < -PI {
            a += 2.0 * PI;
        }
        if a.abs() <= PI / 4.0 {
            Facing::North
        } else if a > PI / 4.0 && a < 3.0 * PI / 4.0 {
            Facing::West
        } else if a < -PI / 4.0 && a > -3.0 * PI / 4.0 {
            Facing::East
        } else {
            Facing::South
        }
    }

    /// Rotate a facing-north offset into this facing.
    pub fn rotate_offset(&self, offset: [i32; 3]) -> [i32; 3] {
        let (x, y, z) = (offset[0], offset[1], offset[2]);
        match self {
            Facing::North => [x, y, z],
            Facing::East => [-z, y, x],
            Facing::South => [-x, y, -z],
            Facing::West => [z, y, -x],
        }
    }
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
        self.register_full(
            name,
            BlockDef {
                name: name.to_string(),
                solid,
                transparent,
                block_entity,
                placeable: true,
                hidden: false,
                multiblock: None,
            },
        )
    }

    fn register_full(&mut self, name: &str, def: BlockDef) -> BlockId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = self.defs.len() as BlockId;
        self.defs.push(def);
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
        self.register_full(
            &name.clone(),
            BlockDef {
                name,
                solid,
                transparent,
                block_entity,
                placeable: true,
                hidden: false,
                multiblock: None,
            },
        )
    }

    /// Register an item (unified id space with blocks): never solid,
    /// never placeable, never in chunks.
    pub fn register_item(&mut self, name: String) -> BlockId {
        self.register_full(
            &name.clone(),
            BlockDef {
                name,
                solid: false,
                transparent: true,
                block_entity: false,
                placeable: false,
                hidden: false,
                multiblock: None,
            },
        )
    }

    /// Register a hidden multiblock part (unplaceable directly, resolved
    /// through its master). Inherits collision/display flags from the base.
    pub fn register_part(&mut self, name: String, solid: bool, transparent: bool) -> BlockId {
        self.register_full(
            &name.clone(),
            BlockDef {
                name,
                solid,
                transparent,
                block_entity: false,
                placeable: false,
                hidden: true,
                multiblock: None,
            },
        )
    }

    /// Attach a multiblock layout to an already-registered master.
    pub fn set_multiblock(&mut self, master: &str, layout: MultiblockLayout) -> bool {
        match self.by_name.get(master).copied() {
            Some(id) => {
                if let Some(def) = self.defs.get_mut(id as usize) {
                    def.multiblock = Some(layout);
                    return true;
                }
                false
            }
            None => false,
        }
    }

    /// Master entry whose layout covers `part_name`, if any.
    pub fn master_for_part(&self, part_name: &str) -> Option<BlockId> {
        self.defs.iter().enumerate().find_map(|(i, d)| {
            d.multiblock.as_ref().and_then(|layout| {
                layout
                    .parts
                    .iter()
                    .any(|p| p.part == part_name)
                    .then_some(i as BlockId)
            })
        })
    }

    /// Creative-visible names (parts hidden), sorted.
    pub fn creative_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .defs
            .iter()
            .filter(|d| !d.hidden && d.name != "exsolis:air")
            .map(|d| d.name.clone())
            .collect();
        names.sort();
        names
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

    pub fn is_placeable(&self, id: BlockId) -> bool {
        self.def(id).is_some_and(|d| d.placeable)
    }

    pub fn is_item(&self, id: BlockId) -> bool {
        self.def(id).is_some_and(|d| !d.placeable && !d.hidden)
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::default_registry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facing_rotates_offsets() {
        let o = [0, 0, -1]; // head north of foot
        assert_eq!(Facing::North.rotate_offset(o), [0, 0, -1]);
        assert_eq!(Facing::East.rotate_offset(o), [1, 0, 0]);
        assert_eq!(Facing::South.rotate_offset(o), [0, 0, 1]);
        assert_eq!(Facing::West.rotate_offset(o), [-1, 0, 0]);
        // Vertical offsets never rotate.
        assert_eq!(Facing::East.rotate_offset([0, 1, 0]), [0, 1, 0]);
    }

    #[test]
    fn facing_from_yaw_cardinals() {
        use std::f32::consts::PI;
        assert_eq!(Facing::from_yaw(0.0), Facing::North);
        assert_eq!(Facing::from_yaw(-PI / 2.0), Facing::East);
        assert_eq!(Facing::from_yaw(PI / 2.0), Facing::West);
        assert_eq!(Facing::from_yaw(PI), Facing::South);
    }

    #[test]
    fn parts_hide_from_creative_but_resolve() {
        let mut r = BlockRegistry::default_registry();
        r.register_mod("minecraft:oak_door".into(), true, false, false);
        r.register_part("minecraft:oak_door_top".into(), true, false);
        r.set_multiblock(
            "minecraft:oak_door",
            MultiblockLayout {
                parts: vec![MultiblockPart {
                    offset: [0, 1, 0],
                    part: "minecraft:oak_door_top".into(),
                }],
            },
        );
        let master = r.id("minecraft:oak_door").unwrap();
        let part = r.id("minecraft:oak_door_top").unwrap();
        assert!(r.def(master).unwrap().placeable);
        assert!(!r.def(part).unwrap().placeable);
        assert_eq!(r.master_for_part("minecraft:oak_door_top"), Some(master));
        assert!(!r.creative_names().contains(&"minecraft:oak_door_top".to_string()));
        assert!(r.creative_names().contains(&"minecraft:oak_door".to_string()));
    }
}
