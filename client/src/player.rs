// src/player.rs — client-side camera + input.
//
// The server is authoritative over position. This module:
//   - FlyCamera + look + cursor grab: purely local (rotation is instant).
//   - Movement input: WASD/space/ctrl are sampled into a direction vector
//     and sent to the server as ClientMessage::Move ONLY WHEN IT CHANGES
//     (throttled to the server tick rate). The server keeps applying the
//     last direction every tick, so holding a key costs zero packets.
//   - Movement result: the server sends its authoritative position back
//     (ServerMessage::YourPosition); the camera's Transform eases toward it.
//     There is no client-side prediction yet, so movement has about one
//     round trip plus one server tick of latency. If that starts to feel
//     bad, add prediction with input sequence numbers + reconciliation.

use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_ecs::message::MessageReader;
use bevy_ecs::prelude::*;
use bevy_input::mouse::{MouseButton, MouseMotion};
use bevy_input::{ButtonInput, keyboard::KeyCode};
use bevy_log::info;
use bevy_renet::RenetClient;
use bevy_renet::renet::DefaultChannel;
use bevy_transform::prelude::Transform;
use bevy_window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use exsolis_common::{ClientMessage, SPAWN_POS, ServerMessage, TICK_RATE};
use glam::{EulerRot, Quat, Vec3};

use crate::streaming::RemotePlayer;

#[derive(Component)]
pub struct LocalPlayer;

/// Latest authoritative position from the server (`None` until the first
/// snapshot arrives). The local Transform eases toward it.
#[derive(Component, Default)]
pub struct ServerTarget(pub Option<Vec3>);

/// Exponential-follow rate (1/s) toward the server position. ~15 hides the
/// 20 Hz snapshot cadence without adding noticeable lag.
const FOLLOW_RATE: f32 = 15.0;

/// If we're further than this from the server (teleport, big desync), snap
/// instead of easing.
const SNAP_DISTANCE: f32 = 8.0;

#[derive(Component)]
pub struct FlyCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub sensitivity: f32,
    pub eye_height: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            fov_y: 60_f32.to_radians(),
            sensitivity: 0.0015,
            eye_height: 1.62,
        }
    }
}

impl FlyCamera {
    pub fn eye_position(&self, transform: &Transform) -> Vec3 {
        transform.translation + Vec3::Y * self.eye_height
    }

    /// World -> clip matrix for the renderer. Vulkan's NDC has +Y pointing
    /// DOWN, but `glam::Mat4::perspective_rh` produces OpenGL-style +Y up,
    /// so Y is flipped here; without it the whole world renders upside-down.
    /// (That flip also reverses triangle winding as the rasterizer sees it,
    /// which is why the pipeline in render.rs uses `FrontFace::Clockwise`
    /// for the clockwise-from-outside faces produced by mesh.rs.)
    pub fn view_projection(&self, transform: &Transform, aspect: f32) -> glam::Mat4 {
        let eye = self.eye_position(transform);
        let forward = transform.rotation * Vec3::NEG_Z;
        let mut proj = glam::Mat4::perspective_rh(self.fov_y, aspect, 0.1, 1000.0);
        proj.y_axis.y *= -1.0;
        proj * glam::Mat4::look_at_rh(eye, eye + forward, Vec3::Y)
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_local_player).add_systems(
            Update,
            (
                toggle_cursor_grab,
                fly_camera_look,
                send_movement_input,
                (receive_server_position, follow_server_position).chain(),
            ),
        );
    }
}

fn spawn_local_player(mut commands: Commands) {
    commands.spawn((
        LocalPlayer,
        FlyCamera::default(),
        Transform::from_translation(Vec3::from(SPAWN_POS)),
        ServerTarget::default(),
    ));
}

/// Unlike the single-player version, this has no inventory-open check —
/// there's no inventory UI on the client yet. Add that guard back in
/// once one exists, or clicking to grab the cursor will fight with it.
fn toggle_cursor_grab(
    mouse: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut windows: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = windows.single_mut() else {
        return;
    };

    if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    } else if mouse.just_pressed(MouseButton::Left) && cursor.grab_mode == CursorGrabMode::None {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

fn fly_camera_look(
    mut motion: MessageReader<MouseMotion>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    mut query: Query<(&mut FlyCamera, &mut Transform), With<LocalPlayer>>,
) {
    let grabbed = windows
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);

    let mut delta = Vec3::ZERO;
    for ev in motion.read() {
        delta.x += ev.delta.x;
        delta.y += ev.delta.y;
    }

    if !grabbed || delta.length_squared() == 0.0 {
        return;
    }

    for (mut cam, mut transform) in &mut query {
        cam.yaw -= delta.x * cam.sensitivity;
        cam.pitch -= delta.y * cam.sensitivity;
        cam.pitch = cam.pitch.clamp(-89_f32.to_radians(), 89_f32.to_radians());
        transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, cam.pitch, 0.0);
    }
}

/// What we last told the server, so we only send when something changed.
#[derive(Default)]
struct InputSyncState {
    last_sent: Vec3,
    last_sent_at: Option<Instant>,
}

/// Samples WASD (+ Space up / Ctrl down) into a world-space direction and
/// sends it to the server only when it changes.
///
/// - Sent over ReliableOrdered: since we don't resend, a lost "stop" packet
///   would otherwise leave the server moving the player forever.
/// - Throttled to the server tick rate: turning while walking changes the
///   direction every frame, but the server only integrates at TICK_RATE, so
///   anything faster is wasted. A change blocked by the throttle isn't lost;
///   it's compared against `last_sent` again next frame and goes out as soon
///   as the interval elapses.
/// - Releasing the cursor counts as "stop", otherwise Esc while holding W
///   would keep the player running.
fn send_movement_input(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    query: Query<&Transform, With<LocalPlayer>>,
    mut client: ResMut<RenetClient>,
    mut sync: Local<InputSyncState>,
) {
    if !client.is_connected() {
        // A (re)connection gets a fresh server-side player with zero input,
        // so reset our baseline and resend whatever is held once connected.
        *sync = InputSyncState::default();
        return;
    }

    let Ok(transform) = query.single() else {
        return;
    };

    let grabbed = windows
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);

    let mut direction = Vec3::ZERO;
    if grabbed {
        let forward_3d = transform.rotation * Vec3::NEG_Z;
        let forward = Vec3::new(forward_3d.x, 0.0, forward_3d.z).normalize_or_zero();
        let right_3d = transform.rotation * Vec3::X;
        let right = Vec3::new(right_3d.x, 0.0, right_3d.z).normalize_or_zero();

        if keys.pressed(KeyCode::KeyW) {
            direction += forward;
        }
        if keys.pressed(KeyCode::KeyS) {
            direction -= forward;
        }
        if keys.pressed(KeyCode::KeyD) {
            direction += right;
        }
        if keys.pressed(KeyCode::KeyA) {
            direction -= right;
        }
        if keys.pressed(KeyCode::Space) {
            direction += Vec3::Y;
        }
        if keys.pressed(KeyCode::ControlLeft) {
            direction -= Vec3::Y;
        }
        direction = direction.normalize_or_zero();
    }

    if direction.abs_diff_eq(sync.last_sent, 1e-3) {
        return;
    }

    let now = Instant::now();
    let min_interval = Duration::from_secs_f32(1.0 / TICK_RATE);
    if let Some(at) = sync.last_sent_at {
        if now.duration_since(at) < min_interval {
            return;
        }
    }

    let msg = bincode::serialize(&ClientMessage::Move {
        x: direction.x,
        y: direction.y,
        z: direction.z,
    })
    .unwrap();
    client.send_message(DefaultChannel::ReliableOrdered, msg);
    sync.last_sent = direction;
    sync.last_sent_at = Some(now);
}

/// Drains authoritative position snapshots. Nothing else reads the
/// Unreliable server->client channel (chunks use ReliableOrdered, read in
/// streaming.rs), so this can't steal anyone else's messages.
fn receive_server_position(
    mut client: ResMut<RenetClient>,
    mut query: Query<(&mut Transform, &mut ServerTarget), With<LocalPlayer>>,
    mut remote: Query<(&RemotePlayer, &mut ServerTarget), Without<LocalPlayer>>,
) {
    let Ok((mut transform, mut target)) = query.single_mut() else {
        return;
    };

    while let Some(bytes) = client.receive_message(DefaultChannel::Unreliable) {
        let Ok(msg) = bincode::deserialize::<ServerMessage>(&bytes) else {
            continue;
        };
        match msg {
            ServerMessage::YourPosition { x, y, z } => {
                let pos = Vec3::new(x, y, z);
                if target.0.is_none() {
                    transform.translation = pos;
                }
                target.0 = Some(pos);
            }
            ServerMessage::PlayerMoved { player_id, x, y, z } => {
                let pos = Vec3::new(x, y, z);
                for (p, mut t) in &mut remote {
                    if p.0 == player_id {
                        t.0 = Some(pos);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Eases the camera toward the last server position (frame-rate independent).
fn follow_server_position(
    mut query: Query<(&mut Transform, &ServerTarget)>,
    mut last_frame: Local<Option<Instant>>,
) {
    let now = Instant::now();
    let dt = last_frame
        .replace(now)
        .map(|prev| now.duration_since(prev).as_secs_f32())
        .unwrap_or(0.0)
        .min(0.1);

    for (mut transform, target) in &mut query {
        let Some(target_pos) = target.0 else {
            continue;
        };

        let delta = target_pos - transform.translation;
        if delta.length_squared() > SNAP_DISTANCE * SNAP_DISTANCE {
            transform.translation = target_pos;
            continue;
        }
        let alpha = 1.0 - (-FOLLOW_RATE * dt).exp();
        transform.translation += delta * alpha;
    }
}
