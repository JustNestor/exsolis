use serde::{Deserialize, Serialize};
mod chunk_blocks;
pub use chunk_blocks::{CHUNK_SIZE, CHUNK_VOLUME, ChunkBlocks};

pub const PROTOCOL_ID: u64 = 1;
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

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    pub const ALL: [BlockType; 10] = [
        Self::Air,
        Self::Stone,
        Self::Dirt,
        Self::Grass,
        Self::Sand,
        Self::Water,
        Self::Snow,
        Self::Glass,
        Self::Log,
        Self::Leaves,
    ];
    pub fn from_u8(id: u8) -> Option<Self> {
        Self::ALL.get(id as usize).copied()
    }

    pub fn is_solid(&self) -> bool {
        !matches!(self, Self::Air | Self::Water)
    }

    pub fn is_air(&self) -> bool {
        self == &Self::Air
    }
}

// pub type ChunkBlocks = Vec<BlockType>;

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
