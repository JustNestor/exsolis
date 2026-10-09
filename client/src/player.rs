// src/player.rs — client-side camera + input.

use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_camera::{Camera, Camera2d, Viewport};
use bevy_ecs::message::{MessageReader, MessageWriter};
use bevy_ecs::prelude::*;
use bevy_input::mouse::MouseMotion;
use bevy_input::{ButtonInput, keyboard::KeyCode};
use bevy_log::info;
use bevy_transform::prelude::Transform;
use bevy_ui::IsDefaultUiCamera;
use bevy_window::{CursorGrabMode, CursorOptions, PrimaryWindow, Window};
use exsolis_common::{ClientMessage, SPAWN_POS, ServerMessage, TICK_RATE};
use glam::{EulerRot, Quat, UVec2, Vec3};

use crate::net::{GameplaySet, IncomingPacket, NetConnected, OutgoingPacket};
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
                fly_camera_look,
                send_movement_input,
                sync_ui_layout_camera,
                (receive_server_position, follow_server_position).chain(),
            )
                .in_set(GameplaySet),
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

    commands.spawn((
        Camera2d,
        IsDefaultUiCamera,
        Camera {
            order: 100,
            viewport: Some(Viewport {
                physical_position: UVec2::ZERO,
                physical_size: UVec2::new(100, 100),
                depth: 0.0..1.0,
            }),
            ..Default::default()
        },
    ));
}

fn sync_ui_layout_camera(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cameras: Query<&mut Camera>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let w = window.resolution.physical_width().max(1);
    let h = window.resolution.physical_height().max(1);

    for mut camera in &mut cameras {
        if let Some(ref mut vp) = camera.viewport {
            vp.physical_size = UVec2::new(w, h);
            vp.physical_position = UVec2::ZERO;
        } else {
            camera.viewport = Some(Viewport {
                physical_position: UVec2::ZERO,
                physical_size: UVec2::new(w, h),
                depth: 0.0..1.0,
            });
        }
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

#[derive(Default)]
struct InputSyncState {
    last_sent: Vec3,
    last_look: (f32, f32),
    last_sent_at: Option<Instant>,
}

fn send_movement_input(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&CursorOptions, With<PrimaryWindow>>,
    query: Query<(&Transform, &FlyCamera), With<LocalPlayer>>,
    connected: Res<NetConnected>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut sync: Local<InputSyncState>,
) {
    if !connected.get() {
        *sync = InputSyncState::default();
        return;
    }

    let Ok((transform, cam)) = query.single() else {
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

    let changed_move = !direction.abs_diff_eq(sync.last_sent, 1e-3);
    let changed_look =
        (cam.yaw - sync.last_look.0).abs() > 1e-3 || (cam.pitch - sync.last_look.1).abs() > 1e-3;
    if !changed_move && !changed_look {
        return;
    }

    let now = Instant::now();
    let min_interval = Duration::from_secs_f32(1.0 / TICK_RATE);
    if let Some(at) = sync.last_sent_at {
        if now.duration_since(at) < min_interval {
            return;
        }
    }

    if direction != Vec3::ZERO || changed_move {
        outgoing.write(OutgoingPacket::reliable(ClientMessage::MoveAndLook {
            x: direction.x,
            y: direction.y,
            z: direction.z,
            yaw: cam.yaw,
            pitch: cam.pitch,
        }));
    } else {
        outgoing.write(OutgoingPacket::reliable(ClientMessage::Look {
            yaw: cam.yaw,
            pitch: cam.pitch,
        }));
    }
    sync.last_sent = direction;
    sync.last_look = (cam.yaw, cam.pitch);
    sync.last_sent_at = Some(now);
}

fn receive_server_position(
    mut incoming: MessageReader<IncomingPacket>,
    mut query: Query<(&mut Transform, &mut ServerTarget), With<LocalPlayer>>,
    mut remote: Query<(&RemotePlayer, &mut ServerTarget, &mut Transform), Without<LocalPlayer>>,
) {
    let Ok((mut transform, mut target)) = query.single_mut() else {
        return;
    };

    for packet in incoming.read() {
        match packet.0 {
            ServerMessage::YourPosition { x, y, z } => {
                let pos = Vec3::new(x, y, z);
                if target.0.is_none() {
                    transform.translation = pos;
                }
                target.0 = Some(pos);
            }
            ServerMessage::PlayerMoved {
                player_id,
                x,
                y,
                z,
                yaw,
                pitch,
            } => {
                let pos = Vec3::new(x, y, z);
                for (p, mut t, mut remote_transform) in &mut remote {
                    if p.0 == player_id {
                        t.0 = Some(pos);
                        remote_transform.rotation = Quat::from_euler(
                            EulerRot::YXZ,
                            yaw + std::f32::consts::PI,
                            -pitch,
                            0.0,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

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
