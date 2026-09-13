use serde::{Deserialize, Serialize};

pub const PROTOCOL_ID: u64 = 0;
pub const SERVER_ADDR: &str = "127.0.0.1:5000";
pub const TICK_RATE: f32 = 20.0;

pub type LevelId = u32;
pub type PlayerId = u64;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    Air,
    Stone,
    Dirt,
    Grass,
    Sand,
    Water,
    Snow,
    Glass,
    Log,
    Leaves,
}

impl BlockType {
    pub fn is_air(self) -> bool {
        matches!(self, BlockType::Air)
    }

    pub fn is_solid(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }
}

pub type ChunkBlocks = Vec<BlockType>;

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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ClientMessage {
    Move { x: f32, y: f32, z: f32 },
    PlaceBlock { pos: BlockPos, kind: BlockType },
    BreakBlock { pos: BlockPos },
    SelectHotbar(u8),
    ToggleInventory,
    SetRenderDistance(u8),
}

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
}
