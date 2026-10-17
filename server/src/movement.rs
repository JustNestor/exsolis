use crate::net::{self, OutgoingPacket};

use super::*;
use chunks::PlayerPosition;
use glam::Vec3;

pub const SERVER_MOVE_SPEED: f32 = 8.0;
const POSITION_REPEATS: u8 = 3;

#[derive(Component, Debug, Default)]
pub struct PlayerInput(pub Vec3);

#[derive(Component, Debug, Default, tesjit_schema::TsSchema)]
pub struct PlayerLook {
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Component, Debug)]
pub struct PositionSync {
    last_sent: Vec3,
    last_look: (f32, f32),
    repeats_left: u8,
}

impl Default for PositionSync {
    fn default() -> Self {
        Self {
            last_sent: Vec3::NAN,
            last_look: (f32::NAN, f32::NAN),
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
                apply_player_input
                    .after(events::dispatch_incoming)
                    .in_set(net::GameplaySet),
                send_position_updates
                    .after(apply_player_input)
                    .in_set(net::GameplaySet),
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
    mut query: Query<(&Player, &PlayerPosition, &PlayerLook, &mut PositionSync)>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    for (player, pos, look, mut sync) in &mut query {
        if pos.0 != sync.last_sent || (look.yaw, look.pitch) != sync.last_look {
            sync.last_sent = pos.0;
            sync.last_look = (look.yaw, look.pitch);
            sync.repeats_left = POSITION_REPEATS;
        }
        if sync.repeats_left == 0 {
            continue;
        }
        sync.repeats_left -= 1;

        // Authoritative position update to owner
        outgoing.write(OutgoingPacket::to_unreliable(
            player.0,
            ServerMessage::YourPosition {
                x: pos.0.x,
                y: pos.0.y,
                z: pos.0.z,
            },
        ));

        // Broadcast move update to all other connected peers
        outgoing.write(OutgoingPacket::broadcast_except_unreliable(
            player.0,
            ServerMessage::PlayerMoved {
                player_id: player.0,
                x: pos.0.x,
                y: pos.0.y,
                z: pos.0.z,
                yaw: look.yaw,
                pitch: look.pitch,
            },
        ));
    }
}
