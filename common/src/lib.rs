use serde::{Deserialize, Serialize};

mod chunk_blocks;
pub mod inventory;

pub use chunk_blocks::{CHUNK_SIZE, CHUNK_VOLUME, ChunkBlocks};

pub const PROTOCOL_ID: u64 = 1;
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
}

impl BlockType {
    /// Every variant, indexed by its `u8` discriminant (checked by a test in
    /// chunk_blocks.rs).
    pub const ALL: [BlockType; 10] = [
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    Player,
}

/// NOTE: no longer `Copy` — `UiAction` owns `String`s. Anything that copied a
/// `ClientMessage` implicitly now needs `.clone()` (or a move).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    Move {
        x: f32,
        y: f32,
        z: f32,
    },
    PlaceBlock {
        pos: BlockPos,
        kind: BlockType,
    },
    BreakBlock {
        pos: BlockPos,
    },
    SelectHotbar(u8),
    ToggleInventory,
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
    /// The receiving player's own hotbar, sent on ReliableOrdered whenever
    /// it changes (join, pickup, place/break consuming a stack, hotbar
    /// select). `slots[i]` is `None` for an empty slot. There is no
    /// visible UI yet (see inventory.rs on both sides) — this exists so
    /// the client can know what `kind` to put in its next `PlaceBlock`
    /// without guessing. Appended last, same reason as `YourPosition`.
    YourInventory {
        slots: [Option<(BlockType, u8)>; HOTBAR_SIZE],
        selected: u8,
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
