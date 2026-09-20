//! Server-authoritative player movement.
//!
//! Clients send their movement *intent* (a direction vector) only when it
//! changes (see client/player.rs), so the server has to remember it and keep
//! integrating every tick: `PlayerInput` holds the last direction received,
//! `apply_player_input` moves `PlayerPosition` along it each tick, and
//! `send_position_updates` tells the owning client where it actually is.

use super::*;
use chunks::PlayerPosition;
use glam::Vec3;

/// Units/second the server moves a player along their input direction.
pub const SERVER_MOVE_SPEED: f32 = 8.0;

/// How many consecutive ticks a resting position keeps being re-sent after
/// the last change. Position snapshots ride the unreliable channel, so
/// without this a single lost "final" packet would leave the client stuck
/// short of where the server thinks it is.
const POSITION_REPEATS: u8 = 3;

/// Last movement direction this player's client reported (already
/// sanitised: finite, length <= 1). Persists between packets.
#[derive(Component, Debug, Default)]
pub struct PlayerInput(pub Vec3);

/// Tracks what we last told the client so we only send on change.
#[derive(Component, Debug)]
pub struct PositionSync {
    last_sent: Vec3,
    repeats_left: u8,
}

impl Default for PositionSync {
    fn default() -> Self {
        Self {
            // NaN never compares equal, so the first tick always sends the
            // spawn position.
            last_sent: Vec3::NAN,
            repeats_left: 0,
        }
    }
}

pub struct MovementPlugin;

impl Plugin for MovementPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                apply_player_input.after(recive_packets),
                send_position_updates.after(apply_player_input),
            ),
        );
    }
}

fn apply_player_input(
    time: Res<bevy_time::Time>,
    mut query: Query<(&PlayerInput, &mut PlayerPosition)>,
) {
    let dt = time.delta_secs();
    for (input, mut pos) in &mut query {
        if input.0 != Vec3::ZERO {
            pos.0 += input.0 * SERVER_MOVE_SPEED * dt;
        }
    }
}

fn send_position_updates(
    mut server: ResMut<RenetServer>,
    mut query: Query<(&Player, &PlayerPosition, &mut PositionSync)>,
) {
    for (player, pos, mut sync) in &mut query {
        if pos.0 != sync.last_sent {
            sync.last_sent = pos.0;
            sync.repeats_left = POSITION_REPEATS;
        }
        if sync.repeats_left == 0 {
            continue;
        }
        sync.repeats_left -= 1;

        let msg = bincode::serialize(&ServerMessage::YourPosition {
            x: pos.0.x,
            y: pos.0.y,
            z: pos.0.z,
        })
        .unwrap();
        server.send_message(player.0, DefaultChannel::Unreliable, msg);
    }
}
