use serde::{Deserialize, Serialize};

mod chunk_blocks;
pub mod inventory;

pub use chunk_blocks::{CHUNK_SIZE, CHUNK_VOLUME, ChunkBlocks};

/// Bumped to 3 for the entity messages (`SpawnEntity`, `DespawnEntity`,
/// `EntityStateUpdate`). `EntityKind` gained variants, so an old client's
/// decode of a new server's stream would desync rather than fail cleanly.
pub const PROTOCOL_ID: u64 = 3;
pub const SERVER_ADDR: &str = "127.0.0.1:5000";
pub const TICK_RATE: f32 = 20.0;

/// Where players spawn, shared by server and client so they can't drift.
/// Terrain surface tops out around y=24, so this is just above ground.
pub const SPAWN_POS: [f32; 3] = [0.0, 28.0, 0.0];

pub type LevelId = u32;
pub type PlayerId = u64;

/// A stable asset name and the fastpak xxh3-64 checksum of its decoded bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetHash {
    pub path: String,
    pub checksum: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkCoord {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl BlockPos {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// `#[repr(u8)]` + explicit discriminants: the numeric id is what goes into
/// `ChunkBlocks` palettes on the wire. Never reorder/renumber existing
/// variants without bumping the protocol; append new ones and extend `ALL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum BlockType {
    Air = 0,
    Stone = 1,
    Dirt = 2,
    Grass = 3,
    Sand = 4,
    Water = 5,
    Snow = 6,
    Glass = 7,
    Log = 8,
    Leaves = 9,
    /// Carries a `Container`. Placing one creates a block entity; breaking it
    /// despawns that entity and spills its slots.
    Chest = 10,
}

impl BlockType {
    /// Every variant, indexed by its `u8` discriminant (checked by a test in
    /// chunk_blocks.rs).
    pub const ALL: [BlockType; 11] = [
        BlockType::Air,
        BlockType::Stone,
        BlockType::Dirt,
        BlockType::Grass,
        BlockType::Sand,
        BlockType::Water,
        BlockType::Snow,
        BlockType::Glass,
        BlockType::Log,
        BlockType::Leaves,
        BlockType::Chest,
    ];

    pub fn from_u8(id: u8) -> Option<Self> {
        Self::ALL.get(id as usize).copied()
    }

    pub fn is_air(self) -> bool {
        matches!(self, BlockType::Air)
    }

    pub fn is_solid(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Face {
    Top,
    Bottom,
    Left,
    Right,
    Front,
    Back,
}

impl Face {
    pub fn normal(self) -> (i32, i32, i32) {
        match self {
            Face::Top => (0, 1, 0),
            Face::Bottom => (0, -1, 0),
            Face::Left => (-1, 0, 0),
            Face::Right => (1, 0, 0),
            Face::Front => (0, 0, 1),
            Face::Back => (0, 0, -1),
        }
    }
}

/// Server-allocated handle for anything in the world that is not a block:
/// players, block entities, dropped items and mod-spawned props.
pub type EntityId = u64;

/// What an [`EntityId`] refers to. Serialized as a bincode enum index, so new
/// kinds MUST be appended or every client's decode shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    /// A connected player. `entity_id` is stable for the whole session, so a
    /// client can correlate it with the `PlayerId` from `YourPlayerId`.
    Player,
    /// Bound to a `BlockPos` (a chest, a furnace, ...). Spawned and despawned
    /// by block changes, never moved by physics.
    Block,
    /// A dropped item stack, simulated with simple gravity.
    Item,
    /// A mod-spawned decorative prop with a model but no behaviour.
    Prop,
}

/// Server-authoritative placement + facing of an entity.
///
/// `anim` is locomotion the client can use to pick a clip without inferring it
/// from velocity deltas: the server already knows the authoritative speed, and
/// deriving speed client-side from unreliable position updates is noisy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EntityState {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Blocks/second along the horizontal facing. `0.0` means standing still.
    pub speed: f32,
    /// `true` while airborne, so the client can pick a non-ground clip.
    pub airborne: bool,
}

impl Default for EntityState {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            speed: 0.0,
            airborne: false,
        }
    }
}

/// NOTE: no longer `Copy` — `UiAction` owns `String`s. Anything that copied a
/// `ClientMessage` implicitly now needs `.clone()` (or a move).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    MoveAndLook {
        x: f32,
        y: f32,
        z: f32,
        yaw: f32,
        pitch: f32,
    },
    Look {
        yaw: f32,
        pitch: f32,
    },
    PlaceBlock {
        pos: BlockPos,
        kind: BlockType,
    },
    BreakBlock {
        pos: BlockPos,
    },
    SelectHotbar(u8),
    SetRenderDistance(u8),
    /// A UI interaction. `ui_id` is the screen the server mounted with
    /// `UiShow`, `element_id` the widget's `id` attribute (or an auto
    /// path like `#0.2.1` when it had none), `action` the raw `on-click`
    /// string (`"namespace:action"`), `data` the widget's optional `data`
    /// attribute. The server MUST treat all four as untrusted input.
    /// Appended last so existing variant indices don't shift.
    UiAction {
        ui_id: String,
        element_id: String,
        action: String,
        data: Option<String>,
    },
    /// A widget marked `draggable="true"` was dropped on a widget marked
    /// `droppable="true"` within the same `ui_id`, and their `drag-group`s
    /// matched (or neither specified one). Both element ids are whatever
    /// `UiAction.element_id` would be for that widget (explicit `id`, or
    /// the auto structural path). Like `UiAction`, the server MUST treat
    /// `source_id`/`target_id`/`data` as untrusted: the client only
    /// reports geometry, it never moves inventory items itself. Appended
    /// last so existing variant indices don't shift.
    UiDrag {
        ui_id: String,
        source_id: String,
        target_id: String,
        data: Option<String>,
    },
    KeybindAction {
        id: String, // "namespace:action"
        state: KeybindState,
    },
    /// Assets the client does not have (or has at a different content hash).
    AssetRequest {
        assets: Vec<AssetHash>,
    },
    /// The client has persisted and verified the requested assets.
    AssetSyncComplete,
    /// A client asset handler completed a server-requested preload.
    AssetLoadComplete {
        paths: Vec<String>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServerMessage {
    PlayerJoined(PlayerId),
    PlayerLeft(PlayerId),
    PlayerMoved {
        player_id: PlayerId,
        x: f32,
        y: f32,
        z: f32,
        yaw: f32,
        pitch: f32,
    },
    ChunkData {
        coord: ChunkCoord,
        blocks: ChunkBlocks,
    },
    ChunkUnload {
        coords: ChunkCoord,
    },
    BlockUpdate {
        pos: BlockPos,
        kind: BlockType,
    },
    ShowEntity {
        entity_id: u64,
        kind: EntityKind,
        x: f32,
        y: f32,
        z: f32,
    },
    EntityDisappear {
        entity_id: u64,
    },
    /// Authoritative position of the receiving player themself. Sent on the
    /// unreliable channel whenever it changes (and repeated a few ticks
    /// after it stops so a single dropped packet can't leave the client
    /// parked at a stale spot). Appended last so existing variant indices
    /// don't shift.
    YourPosition {
        x: f32,
        y: f32,
        z: f32,
    },
    /// Mount (or atomically replace) the named UI screen. `xml` is the SDUI
    /// document (see client `ui::parse`). ReliableOrdered. If the client
    /// fails to parse it, the previously mounted screen (if any) is kept.
    UiShow {
        id: String,
        xml: String,
    },
    /// Unmount the named UI screen. Unknown ids are ignored. ReliableOrdered.
    UiHide {
        id: String,
    },
    RegisterKeybind {
        id: String,          // e.g. "mymod:cast_spell"
        default_key: String, // e.g. "KeyF", "Space", "KeyE"
        description: String, // e.g. "Cast Primary Ability"
        category: String,    // e.g. "Combat"
    },
    UnregisterKeybind {
        id: String,
    },
    /// Server-authored catalog for the current session. The server remains the
    /// authority on which assets are required; client hashes only avoid transfers.
    AssetManifest {
        assets: Vec<AssetHash>,
        base_url: String,
    },
    /// Authenticated, short-lived HTTP(S) URL for a fastpak delta archive.
    AssetDownload {
        url: String,
        assets: Vec<AssetHash>,
    },
    /// Ask the client to make these assets resident through their registered handlers.
    AssetLoad {
        paths: Vec<String>,
    },
    /// New or changed assets are available; download and apply them immediately.
    AssetUpdate {
        url: String,
        assets: Vec<AssetHash>,
    },
    /// Files removed from the authoritative catalog during development.
    AssetRemove {
        paths: Vec<String>,
    },
    /// The server could not create the requested pak; clients may join with fallbacks.
    AssetDownloadFailed {
        reason: String,
    },
    /// A server-side mod asks clients to run a package-declared visual action.
    /// `payload_json` is interpreted by the matching client shader manifest;
    /// clients without a handler ignore it. Appended to preserve existing
    /// bincode variant indices.
    ModAction {
        id: String,
        payload_json: String,
    },
    /// Authenticated identity of the receiving client, for correlating local
    /// model transforms with server-authored actions. Appended for wire compatibility.
    YourPlayerId(PlayerId),
    /// Spawn a GLTF model instance on the client. `path` is a server-manifest
    /// asset path (e.g. `models/duck.glb`). TEMPORARY test API.
    /// Appended for wire compatibility.
    SpawnModel {
        entity_id: u64,
        path: String,
        x: f32,
        y: f32,
        z: f32,
        scale: f32,
    },
    /// Despawn a previously spawned model instance. Unknown ids are ignored.
    /// TEMPORARY test API. Appended for wire compatibility.
    DespawnModel {
        entity_id: u64,
    },
    /// Bring an entity into existence on the client. `model` is a catalog
    /// asset path (`models/girl.glb`) or `None` for entities that render
    /// nothing on their own (a block entity renders its block instead).
    /// ReliableOrdered. Appended last so existing variant indices don't shift.
    SpawnEntity {
        entity_id: EntityId,
        kind: EntityKind,
        state: EntityState,
        model: Option<String>,
        scale: f32,
    },
    /// Remove an entity. Unknown ids are ignored. ReliableOrdered.
    /// Appended for wire compatibility.
    DespawnEntity {
        entity_id: EntityId,
    },
    /// Authoritative transform of an existing entity. Sent on the unreliable
    /// channel; the client interpolates between updates, so a dropped packet
    /// costs smoothness but never correctness. Appended for wire compatibility.
    EntityStateUpdate {
        entity_id: EntityId,
        state: EntityState,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeybindState {
    Pressed,
    Released,
}

pub const HOTBAR_SIZE: usize = 9;

// ---- SDUI limits, shared so server-side validation matches the client ----

/// Max size of one `UiShow.xml` payload.
pub const UI_MAX_XML_BYTES: usize = 64 * 1024;
/// Max length of `ui_id` / `element_id`.
pub const UI_MAX_ID_LEN: usize = 64;
/// Max length of `UiAction.action` and `UiAction.data`.
pub const UI_MAX_ACTION_LEN: usize = 256;
/// Max UI screens mounted at the same time on one client.
pub const UI_MAX_OPEN: usize = 32;
/// Max length of `UiDrag.source_id` / `target_id` — reuses the id budget
/// since both are `UiAction.element_id`-shaped.
pub const UI_DRAG_MAX_ID_LEN: usize = UI_MAX_ID_LEN;
/// Maximum namespaced mod action identifier sent to clients.
pub const MOD_ACTION_MAX_ID_BYTES: usize = 128;
/// Maximum UTF-8 JSON payload size for one mod action packet.
pub const MOD_ACTION_MAX_PAYLOAD_BYTES: usize = 4096;
/// Maximum asset path length for a `SpawnModel` request.
pub const MODEL_PATH_MAX_LEN: usize = 256;
/// Maximum decoded bytes for one GLB model. Models dwarf shaders/textures;
/// one oversized asset must not stall joins or exhaust GPU buffers.
pub const GLB_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Mod action identifiers use the same `namespace:name` form as keybinds.
pub fn is_valid_mod_action_id(id: &str) -> bool {
    if id.is_empty() || id.len() > MOD_ACTION_MAX_ID_BYTES {
        return false;
    }
    let Some((namespace, name)) = id.split_once(':') else {
        return false;
    };
    !namespace.is_empty()
        && !name.is_empty()
        && namespace
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b'/'))
}
