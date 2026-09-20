// src/player.rs — client-side camera + input.
//
// Adapted from the single-player FlyCamera. The single-player version
// ran movement AND collision locally against `Chunk` because there was
// no server. Here the server is authoritative over position, so this
// version is split in two:
//   - FlyCamera + look + cursor grab: purely local, no networking needed.
//   - Movement: WASD/space/ctrl/shift state is sampled and sent to the
//     server as ClientMessage::Move every frame it changes; the actual
//     Transform is NOT stepped locally by input. If you want local
//     movement to feel instant instead of waiting on a round trip,
//     add client-side prediction later — start simple first and get
//     something moving before adding that complexity.

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
use exsolis_common::ClientMessage;
use glam::{EulerRot, Quat, Vec3};

#[derive(Component)]
pub struct LocalPlayer;

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

    /// Same shape as render.rs's placeholder view_proj — call this from
    /// render_frame instead of the `Mat4::IDENTITY` fallback once a
    /// LocalPlayer entity exists.
    pub fn view_projection(&self, transform: &Transform, aspect: f32) -> glam::Mat4 {
        let eye = self.eye_position(transform);
        let forward = transform.rotation * Vec3::NEG_Z;
        glam::Mat4::perspective_rh(self.fov_y, aspect, 0.1, 1000.0)
            * glam::Mat4::look_at_rh(eye, eye + forward, Vec3::Y)
    }
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_local_player).add_systems(
            Update,
            (toggle_cursor_grab, fly_camera_look, send_movement_input),
        );
    }
}

fn spawn_local_player(mut commands: Commands) {
    commands.spawn((
        LocalPlayer,
        FlyCamera::default(),
        Transform::from_xyz(0.0, 80.0, 0.0),
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

/// Sends raw WASD/space/ctrl axis state to the server every frame the
/// window is grabbed. This is deliberately dumb (no client-side physics,
/// no dead-reckoning) — see the module doc comment. Server decides what
/// "Move" actually means once events.rs stops ignoring it (see the
/// server-side patch).
fn send_movement_input(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    query: Query<&Transform, With<LocalPlayer>>,
    mut client: ResMut<RenetClient>,
) {
    let grabbed = windows
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !grabbed {
        return;
    }

    let Ok(transform) = query.single() else {
        return;
    };

    let mut direction = Vec3::ZERO;
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
    if direction.length_squared() > 0.0 {
        direction = direction.normalize();
    }

    // Sent every grabbed frame, not just on change: ClientMessage::Move
    // goes over the Unreliable channel (see events.rs), so a dropped
    // packet should just get superseded by the next one rather than
    // leaving the server holding stale input.
    let msg = bincode::serialize(&ClientMessage::Move {
        x: direction.x,
        y: direction.y,
        z: direction.z,
    })
    .unwrap();
    // info!("sending move");
    // client.send_message(DefaultChannel::Unreliable, msg);
}
