use serde::{Deserialize, Serialize};

pub const PROTOCOL_ID: u64 = 0;

pub const SERVER_ADDR: &str = "127.0.0.1:5000";

pub type PlayerId = u64;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ClientMessage {
    Move { x: f32, y: f32, z: f32 },
    PlaceBlock,
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
}
