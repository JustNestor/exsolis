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

use crate::{BlockAnimState, Face};

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
    /// Set on orientation parts (`oak_stairs__…`, door tops): the base to
    /// drop on break and to show in creative. `None` everywhere else.
    #[serde(default)]
    pub part_of: Option<String>,
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
        let half = PI / 4.0;
        let three_quarters = 3.0 * half; // 135°
        // The camera looks along `forward = (-sin yaw, -cos yaw)` with yaw 0
        // down -Z (north) and east = +X, so *positive* yaw turns west. This
        // matches Minecraft's `facing` property ("the direction the player
        // faces"). Quadrant boundaries sit on the diagonals (±45°, ±135°), so
        // a placement resolves to the cardinal the player actually looked at.
        if a.abs() <= half {
            Facing::North
        } else if a > half && a <= three_quarters {
            Facing::West
        } else if a > -three_quarters && a < -half {
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
        // Air-family entries are never placeable and never listed.
        let air_like = is_air_like_name(name);
        self.register_full(
            name,
            BlockDef {
                name: name.to_string(),
                solid,
                transparent,
                block_entity,
                placeable: !air_like,
                hidden: air_like,
                multiblock: None,
                part_of: None,
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
        let air_like = is_air_like_name(&name);
        self.register_full(
            &name.clone(),
            BlockDef {
                name,
                solid,
                transparent,
                block_entity,
                placeable: !air_like,
                hidden: air_like,
                multiblock: None,
                part_of: None,
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
                part_of: None,
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
                part_of: None,
            },
        )
    }

    /// Register a hidden orientation part (`oak_stairs__facing=east…`):
    /// unplaceable directly, drops `master`, invisible in creative.
    pub fn register_opart(&mut self, name: String, master: String, solid: bool, transparent: bool) -> BlockId {
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
                part_of: Some(master),
            },
        )
    }

    /// Drop id for a broken cell: orientation parts drop their master,
    /// everything else drops itself.
    pub fn drop_for(&self, id: BlockId) -> BlockId {
        if let Some(master) = self.def(id).and_then(|d| d.part_of.clone())
            && let Some(mid) = self.id(&master)
        {
            return mid;
        }
        id
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

    /// Creative-visible names: parts hidden, air-family out, sorted.
    pub fn creative_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .defs
            .iter()
            .filter(|d| !d.hidden && !is_air_like_name(&d.name))
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

/// Item model files that are not items: parents, templates, handheld
/// poses and override states. They get no icon and register nothing.
/// Shared by the client icon scan and the server item scan (same rule =>
/// same ids).
pub fn mc_item_template(stem: &str) -> bool {
    if matches!(stem, "handheld" | "generated") {
        return true;
    }
    if stem.starts_with("template_") || stem.ends_with("_in_hand") {
        return true;
    }
    if stem.starts_with("clock_") {
        return true;
    }
    matches!(
        stem,
        "bow_pulling_0"
            | "bow_pulling_1"
            | "bow_pulling_2"
            | "crossbow_arrow"
            | "crossbow_fireworks"
            | "crossbow_standby"
            | "shield_blocking"
            | "trident_throwing"
            | "brush_brushing_0"
            | "brush_brushing_1"
            | "brush_brushing_2"
    )
}

/// Orientation variants for view-dependent placement. Returns
/// `(part_name, bake_base, state)`; part names are deterministic so the
/// baker, both registries and placement all agree without messaging.
///
/// - stairs `<x>_stairs` → 8 × (facing, half=bottom/top, shape=straight)
/// - slabs `<x>_slab` → 2 × (type=bottom/top)
/// - pillars `<x>_log|_stem|_pillar|hay_block|bone_block` → 3 × axis
/// - `torch` → floor default + 4 wall facings (baked from `wall_torch`)
pub fn orientation_parts(master: &str) -> Vec<(String, String, Vec<(String, String)>)> {
    let Some(base) = master.strip_prefix("minecraft:") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let part_name = |state: &[(String, String)]| {
        let mut suffix = String::new();
        for (k, v) in state {
            suffix.push_str("__");
            suffix.push_str(k);
            suffix.push('=');
            suffix.push_str(v);
        }
        format!("{master}{suffix}")
    };
    if base.ends_with("_stairs") {
        for f in ["north", "east", "south", "west"] {
            for h in ["bottom", "top"] {
                let state = vec![
                    ("facing".to_string(), f.to_string()),
                    ("half".to_string(), h.to_string()),
                    ("shape".to_string(), "straight".to_string()),
                ];
                out.push((part_name(&state), master.to_string(), state));
            }
        }
    } else if base.ends_with("_slab") {
        for t in ["bottom", "top"] {
            let state = vec![("type".to_string(), t.to_string())];
            out.push((part_name(&state), master.to_string(), state));
        }
    } else if base.ends_with("_log")
        || base.ends_with("_stem")
        || base.ends_with("_pillar")
        || matches!(base, "hay_block" | "bone_block" | "basalt" | "polished_basalt")
    {
        for a in ["x", "y", "z"] {
            let state = vec![("axis".to_string(), a.to_string())];
            out.push((part_name(&state), master.to_string(), state));
        }
    } else if base == "torch" {
        for f in ["north", "east", "south", "west"] {
            let state = vec![("facing".to_string(), f.to_string())];
            out.push((
                part_name(&state),
                "minecraft:wall_torch".to_string(),
                state,
            ));
        }
    }
    out
}

/// Pick the placed part (or master) from the clicked face, the player
/// facing and eye height. `eye_above` = eye is clearly above the target
/// cell top (side clicks then land on the upper half — an approximation
/// of the exact hit point MC uses).
pub fn placement_pick(
    master: &str,
    face: Face,
    facing: Facing,
    eye_above: bool,
) -> Option<String> {
    let Some(base) = master.strip_prefix("minecraft:") else {
        return None;
    };
    // Which half for slabs/stairs: underside clicks go top, top clicks go
    // bottom, side clicks follow the eye heuristic.
    let top_half = match face {
        Face::Bottom => true,
        Face::Top => false,
        _ => eye_above,
    };
    if base.ends_with("_stairs") {
        let f = match facing {
            Facing::North => "north",
            Facing::East => "east",
            Facing::South => "south",
            Facing::West => "west",
        };
        let h = if top_half { "top" } else { "bottom" };
        return Some(format!(
            "{master}__facing={f}__half={h}__shape=straight"
        ));
    }
    if base.ends_with("_slab") {
        let t = if top_half { "top" } else { "bottom" };
        return Some(format!("{master}__type={t}"));
    }
    if base.ends_with("_log")
        || base.ends_with("_stem")
        || base.ends_with("_pillar")
        || matches!(base, "hay_block" | "bone_block" | "basalt" | "polished_basalt")
    {
        let a = match face {
            Face::Top | Face::Bottom => "y",
            Face::Left | Face::Right => "x",
            Face::Front | Face::Back => "z",
        };
        return Some(format!("{master}__axis={a}"));
    }
    if base == "torch" {
        match face {
            Face::Top => return None, // floor torch = master default
            Face::Bottom => return None, // nothing hangs from ceilings v1
            _ => {
                // Wall torch facing = the clicked face direction.
                let f = match face {
                    Face::Left => "west",
                    Face::Right => "east",
                    Face::Front => "south",
                    Face::Back => "north",
                    _ => unreachable!(),
                };
                return Some(format!("minecraft:wall_torch__facing={f}"));
            }
        }
    }
    None
}

/// Air-family blocks render nothing; they stay out of creative even when
/// placeable (barrier/light keep working through commands).
pub fn is_air_like_name(name: &str) -> bool {
    let base = name.split(':').next_back().unwrap_or(name);
    matches!(
        base,
        "air" | "cave_air" | "void_air" | "barrier" | "light"
    )
}

// ---------------------------------------------------------------- block anim --

/// One clip that poses a block in a given animation state.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockAnimClip {
    /// Name of the animation clip as authored (glTF `animation.name`, or the
    /// bbmodel animator name). The client resolves it to a skeleton clip
    /// index when the model is loaded.
    pub name: String,
    /// Clip duration in seconds; cross-fades should not outlast it.
    pub duration: f32,
}

/// Which half of the rig a state transition blends, for models that pose
/// independent branches (a chest lid is upper body; a door swinging is
/// full body). Mirrors the client's `anim::Mask`; `Common` cannot name a
/// client type, so the driver maps between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlockAnimMask {
    /// Blend every joint.
    #[default]
    Full,
    /// Blend head/hair/upper-body joints only.
    Upper,
    /// Blend hips/legs only.
    Lower,
}

impl BlockAnimMask {
    /// Stable name used by the animation config files.
    pub fn as_str(self) -> &'static str {
        match self {
            BlockAnimMask::Full => "full",
            BlockAnimMask::Upper => "upper",
            BlockAnimMask::Lower => "lower",
        }
    }

    /// Inverse of [`BlockAnimMask::as_str`]; `None` for unknown masks.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" => Some(BlockAnimMask::Full),
            "upper" | "upper_body" => Some(BlockAnimMask::Upper),
            "lower" | "lower_body" => Some(BlockAnimMask::Lower),
            _ => None,
        }
    }
}

/// One directed transition between two animation states: how long the
/// cross-fade runs and which part of the rig it is allowed to move.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockAnimTransition {
    pub from: BlockAnimState,
    pub to: BlockAnimState,
    pub duration: f32,
    pub mask: BlockAnimMask,
}

/// Animation set for one block (a master, or the root of a multiblock such
/// as a door or bed). The config file describes *how* the block animates and
/// *between which states*; the runtime just plays the clip a state names and
/// cross-fades along the transition durations, optionally masking the blend
/// to one half of the rig.
#[derive(Debug, Clone, Default)]
pub struct BlockAnimSet {
    pub block: String,
    /// States this set knows about, in config order. The first state is the
    /// at-rest pose.
    pub states: Vec<BlockAnimState>,
    /// State -> clip.
    pub clips: HashMap<BlockAnimState, BlockAnimClip>,
    /// Directed transitions between states, e.g. closed -> open.
    pub transitions: Vec<BlockAnimTransition>,
}

impl BlockAnimSet {
    /// The clip that poses `state`, if the set defines one.
    pub fn clip_for(&self, state: BlockAnimState) -> Option<&BlockAnimClip> {
        self.clips.get(&state)
    }

    /// The state whose clip has the given authoring name, if any.
    pub fn state_for_clip_name(&self, name: &str) -> Option<BlockAnimState> {
        self.clips
            .iter()
            .find(|(_, clip)| clip.name == name)
            .map(|(state, _)| *state)
    }

    /// Duration of the transition `from -> to`, if the set defines one.
    pub fn transition_duration(&self, from: BlockAnimState, to: BlockAnimState) -> Option<f32> {
        self.transition(from, to).map(|t| t.duration)
    }

    /// The transition `from -> to`, if the set defines one.
    pub fn transition(
        &self,
        from: BlockAnimState,
        to: BlockAnimState,
    ) -> Option<&BlockAnimTransition> {
        self.transitions.iter().find(|t| t.from == from && t.to == to)
    }
}

/// Catalog of block animation sets, keyed by block name (master for
/// multiblock parts).
#[derive(Debug, Clone, Default)]
pub struct BlockAnimCatalog {
    pub sets: HashMap<String, BlockAnimSet>,
}

impl BlockAnimCatalog {
    /// Parse a block animation catalog from its JSON document:
    ///
    /// ```json
    /// {
    ///   "minecraft:oak_door": {
    ///     "states": ["closed", "open"],
    ///     "clips": {
    ///       "closed": { "name": "door_closed", "duration": 0.2 },
    ///       "open": { "name": "door_open", "duration": 0.4 }
    ///     },
    ///     "transitions": [
    ///       { "from": "closed", "to": "open", "duration": 0.25 },
    ///       { "from": "open", "to": "closed", "duration": 0.25, "mask": "lower" }
    ///     ]
    ///   }
    /// }
    /// ```
    pub fn from_json(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("parse block anims: {e}"))?;
        let mut sets = HashMap::new();
        let Some(obj) = doc.as_object() else {
            return Err("block anims: top level must be an object".to_string());
        };
        for (block, value) in obj {
            let Some(set) = value.as_object() else {
                continue; // skip malformed entries
            };
            let mut states = Vec::new();
            if let Some(arr) = set.get("states").and_then(|v| v.as_array()) {
                for s in arr {
                    if let Some(state) = BlockAnimState::parse(s.as_str().unwrap_or("")) {
                        states.push(state);
                    }
                }
            }
            let mut clips = HashMap::new();
            if let Some(clips_obj) = set.get("clips").and_then(|v| v.as_object()) {
                for (state, clip) in clips_obj {
                    let Some(clip) = clip.as_object() else { continue };
                    let name = clip.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                    let duration = clip
                        .get("duration")
                        .and_then(|d| d.as_f64())
                        .unwrap_or(0.0) as f32;
                    if let Some(state) = BlockAnimState::parse(state) {
                        clips.insert(state, BlockAnimClip { name, duration });
                    }
                }
            }
            let mut transitions = Vec::new();
            if let Some(trs) = set.get("transitions").and_then(|v| v.as_array()) {
                for t in trs {
                    let Some(t) = t.as_object() else { continue };
                    let Some(from) = BlockAnimState::parse(
                        t.get("from").and_then(|v| v.as_str()).unwrap_or(""),
                    ) else {
                        continue;
                    };
                    let Some(to) = BlockAnimState::parse(
                        t.get("to").and_then(|v| v.as_str()).unwrap_or(""),
                    ) else {
                        continue;
                    };
                    let duration = t.get("duration").and_then(|d| d.as_f64()).unwrap_or(0.0) as f32;
                    let mask = t
                        .get("mask")
                        .and_then(|m| m.as_str())
                        .and_then(BlockAnimMask::parse)
                        .unwrap_or_default();
                    transitions.push(BlockAnimTransition {
                        from,
                        to,
                        duration,
                        mask,
                    });
                }
            }
            if states.is_empty() && clips.is_empty() && transitions.is_empty() {
                continue; // nothing usable declared for this block
            }
            sets.insert(
                block.clone(),
                BlockAnimSet {
                    block: block.clone(),
                    states,
                    clips,
                    transitions,
                },
            );
        }
        Ok(Self { sets })
    }

    /// The animation set for a block, if the catalog declares one.
    pub fn get(&self, block: &str) -> Option<&BlockAnimSet> {
        self.sets.get(block)
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
        // Yaw 0 looks north; the camera's forward is (-sin yaw, -cos yaw),
        // so positive yaw (mouse left) turns west and negative yaw east.
        assert_eq!(Facing::from_yaw(0.0), Facing::North);
        assert_eq!(Facing::from_yaw(-PI / 2.0), Facing::East);
        assert_eq!(Facing::from_yaw(PI / 2.0), Facing::West);
        assert_eq!(Facing::from_yaw(PI), Facing::South);
    }

    #[test]
    fn facing_from_yaw_quadrants() {
        use std::f32::consts::PI;
        // Diagonals sit exactly on the ±45° boundaries; ties fall to north.
        assert_eq!(Facing::from_yaw(PI / 4.0), Facing::North);
        assert_eq!(Facing::from_yaw(-PI / 4.0), Facing::North);
        // ... and the ±135° boundaries tie to the north/south side.
        assert_eq!(Facing::from_yaw(3.0 * (PI / 4.0)), Facing::West);
        assert_eq!(Facing::from_yaw(-3.0 * (PI / 4.0)), Facing::South);
        // Mid-range values must agree with the cardinal neighbours.
        assert_eq!(Facing::from_yaw(0.0), Facing::North);
        assert_eq!(Facing::from_yaw(1.2), Facing::West);
        assert_eq!(Facing::from_yaw(2.5), Facing::South);
        assert_eq!(Facing::from_yaw(4.0), Facing::East);
    }

    #[test]
    fn facing_matches_the_camera_forward_vector() {
        // The invariant that matters for placement: `Facing` names the world
        // direction the player looks, and the camera forward is
        // `(-sin yaw, -cos yaw)` with north = -Z, east = +X. If the two ever
        // disagree, beds/doors land on the wrong side and stairs turn around.
        for deg in [
            -170.0f32, -130.0, -100.0, -80.0, -50.0, -20.0, 0.0, 20.0, 50.0, 80.0, 100.0,
            130.0, 170.0, 180.0, -180.0,
        ] {
            let yaw = deg.to_radians();
            let (fx, fz) = (-yaw.sin(), -yaw.cos());
            let expected = if fx.abs() > fz.abs() {
                if fx > 0.0 { Facing::East } else { Facing::West }
            } else if fz > 0.0 {
                Facing::South
            } else {
                Facing::North
            };
            assert_eq!(Facing::from_yaw(yaw), expected, "yaw {deg}deg");
        }
    }

    #[test]
    fn placement_pick_orients() {
        use crate::Face;
        // Stairs take the direction the player faces; slabs follow the
        // clicked face.
        assert_eq!(
            placement_pick("minecraft:oak_stairs", Face::Top, Facing::East, false),
            Some("minecraft:oak_stairs__facing=east__half=bottom__shape=straight".to_string())
        );
        assert_eq!(
            placement_pick("minecraft:oak_stairs", Face::Bottom, Facing::North, false),
            Some("minecraft:oak_stairs__facing=north__half=top__shape=straight".to_string())
        );
        assert_eq!(
            placement_pick("minecraft:stone_slab", Face::Bottom, Facing::North, false),
            Some("minecraft:stone_slab__type=top".to_string())
        );
        // Pillars follow the face axis; torches go wall-mounted on sides.
        assert_eq!(
            placement_pick("minecraft:oak_log", Face::Left, Facing::North, false),
            Some("minecraft:oak_log__axis=x".to_string())
        );
        assert_eq!(
            placement_pick("minecraft:torch", Face::Front, Facing::North, false),
            Some("minecraft:wall_torch__facing=south".to_string())
        );
        assert_eq!(placement_pick("minecraft:torch", Face::Top, Facing::North, false), None);
        assert_eq!(placement_pick("minecraft:stone", Face::Top, Facing::North, false), None);
    }

    #[test]
    fn block_anim_catalog_parses_state_clips_and_transitions() {
        let catalog = BlockAnimCatalog::from_json(
            r#"{
                "minecraft:oak_door": {
                    "states": ["closed", "open"],
                    "clips": {
                        "closed": { "name": "door_closed", "duration": 0.2 },
                        "open": { "name": "door_open", "duration": 0.4 }
                    },
                    "transitions": [
                        { "from": "closed", "to": "open", "duration": 0.25 },
                        { "from": "open", "to": "closed", "duration": 0.3, "mask": "lower" }
                    ]
                }
            }"#,
        )
        .expect("door animation catalog parses");
        let set = catalog.get("minecraft:oak_door").expect("door set present");
        assert_eq!(set.states.len(), 2);
        let open = set.clip_for(BlockAnimState::Open).expect("open clip");
        assert_eq!(open.name, "door_open");
        assert!((open.duration - 0.4).abs() < 1e-6);
        assert_eq!(
            set.transition_duration(BlockAnimState::Closed, BlockAnimState::Open),
            Some(0.25)
        );
        // An absent mask means a full-body blend; an explicit one is parsed.
        assert_eq!(
            set.transition(BlockAnimState::Closed, BlockAnimState::Open)
                .expect("closed -> open")
                .mask,
            BlockAnimMask::Full
        );
        assert_eq!(
            set.transition(BlockAnimState::Open, BlockAnimState::Closed)
                .expect("open -> closed")
                .mask,
            BlockAnimMask::Lower
        );
        assert_eq!(BlockAnimState::parse("sideways"), None);
        assert_eq!(BlockAnimMask::parse("upper"), Some(BlockAnimMask::Upper));
    }

    #[test]
    fn block_anim_catalog_tolerates_unusable_entries() {
        let catalog = BlockAnimCatalog::from_json(r#"{ "minecraft:not_an_anim_block": { } }"#)
            .expect("malformed entries are skipped, not fatal");
        assert!(catalog.get("minecraft:not_an_anim_block").is_none());
        assert!(catalog.get("minecraft:absent").is_none());
    }

    #[test]
    fn block_anim_catalog_rejects_bad_json() {
        assert!(BlockAnimCatalog::from_json("{ not json").is_err());
        assert!(BlockAnimCatalog::from_json("[1, 2, 3]").is_err());
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
