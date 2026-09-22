use super::*;
use chunks::PlayerPosition;
use glam::Vec3;

pub const SERVER_MOVE_SPEED: f32 = 8.0;
const POSITION_REPEATS: u8 = 3;

#[derive(Component, Debug, Default)]
pub struct PlayerInput(pub Vec3);

#[derive(Component, Debug)]
pub struct PositionSync {
    last_sent: Vec3,
    repeats_left: u8,
}

impl Default for PositionSync {
    fn default() -> Self {
        Self {
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
    mut query: Query<(&mut PlayerInput, &mut PlayerPosition)>,
) {
    let dt = time.delta_secs();
    for (mut input, mut pos) in &mut query {
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

        // Authoritative position update to owner
        let msg = bincode::serialize(&ServerMessage::YourPosition {
            x: pos.0.x,
            y: pos.0.y,
            z: pos.0.z,
        })
        .unwrap();
        server.send_message(player.0, DefaultChannel::Unreliable, msg);

        // Broadcast move update to all other connected peers
        let move_msg = bincode::serialize(&ServerMessage::PlayerMoved {
            player_id: player.0,
            x: pos.0.x,
            y: pos.0.y,
            z: pos.0.z,
        })
        .unwrap();
        server.broadcast_message_except(player.0, DefaultChannel::Unreliable, move_msg);
    }
}
