// src/player.rs — client-side camera + input.

use std::time::{Duration, Instant};

use bevy_app::prelude::*;
use bevy_camera::{Camera, Camera2d, Viewport};
use bevy_ecs::message::{MessageReader, MessageWriter};
use bevy_ecs::prelude::*;
use bevy_input::mouse::MouseMotion;
use bevy_input::{ButtonInput, keyboard::KeyCode};
use bevy_log::info;
use bevy_time::Time;
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

    /// The direction the camera looks, as a unit vector.
    ///
    /// Built from the camera's **own** yaw/pitch, not from the entity's
    /// `Transform` rotation. Those are deliberately different things now:
    /// the transform carries the *body's* yaw (which is what the model is
    /// drawn with), while only the camera knows the full pitch. Reading the
    /// direction out of the transform is what used to force the body to
    /// pitch with the camera — and, because the model's origin is at its
    /// feet, to swing its head through an enormous arc.
    pub fn forward(&self) -> Vec3 {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0) * Vec3::NEG_Z
    }

    pub fn view_projection(&self, transform: &Transform, aspect: f32) -> glam::Mat4 {
        let eye = self.eye_position(transform);
        let forward = self.forward();
        let mut proj = glam::Mat4::perspective_rh(self.fov_y, aspect, 0.1, 1000.0);
        proj.y_axis.y *= -1.0;
        proj * glam::Mat4::look_at_rh(eye, eye + forward, Vec3::Y)
    }

    /// The same view with an explicit depth range.
    ///
    /// The first-person viewmodel uses a much shorter range than the world
    /// (0.01..0.6 rather than 0.1..1000) so the hands have usable depth
    /// precision right in front of the camera. The world near plane is far
    /// too coarse for an arm held half a metre from the eye.
    pub fn view_projection_range(
        &self,
        transform: &Transform,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> glam::Mat4 {
        let eye = self.eye_position(transform);
        let forward = self.forward();
        let mut proj = glam::Mat4::perspective_rh(self.fov_y, aspect, near, far);
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
                drive_local_animator,
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
        // The local player's avatar is a first-person viewmodel: it goes
        // through the same rig/animation path a remote player's does, but
        // `first_person` draws only the arms, in their own depth range,
        // from in front of the camera.
        crate::anim::Animator::new(),
        crate::gltf::GltfModelInstance::new(exsolis_server::PLAYER_MODEL)
            .with_scale(exsolis_server::PLAYER_MODEL_SCALE)
            .first_person(VIEWMODEL_OFFSET)
            .with_world_offset(MODEL_VISUAL_LIFT),
        bevy_transform::prelude::GlobalTransform::default(),
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
        // Only the yaw goes into the entity transform. The transform is the
        // *body's* facing, and the body is drawn from it, so putting pitch in
        // here tilted the entire avatar whenever the camera moved — pivoting
        // around the feet, since that is where the model's origin sits, which
        // threw the head a long way off screen. The camera builds its own
        // view from `FlyCamera::forward()`, which still has the full pitch.
        transform.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, 0.0, 0.0);
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

/// Drive the local player's avatar animation.
///
/// Remote entities take their speed from the server's authoritative
/// `EntityState`. The local player is different: it moves on its own input
/// and the server's copy arrives a round trip later, so waiting for it would
/// make the first-person hands lag behind the camera. Differencing the local
/// transform is both available immediately and exactly what the player is
/// seeing.
///
/// The head look is applied additively on top of whatever body clip is
/// playing, because in this rig the head is an independent branch — that is
/// what lets the body keep walking while the head tracks the crosshair.
fn drive_local_animator(
    time: Res<Time>,
    mut last_pos: Local<Option<Vec3>>,
    mut query: Query<(&Transform, &FlyCamera, &mut crate::anim::Animator), With<LocalPlayer>>,
) {
    let dt = time.delta_secs();
    for (transform, camera, mut animator) in &mut query {
        let position = transform.translation;
        let instant = match (last_pos.replace(position), dt) {
            (Some(previous), dt) if dt > f32::EPSILON => {
                let delta = position - previous;
                Vec3::new(delta.x, 0.0, delta.z).length() / dt
            }
            // First frame (or a paused one): no basis for a speed.
            _ => animator.speed,
        };
        // Frame-rate independent smoothing, matched to the server's so both
        // sides cross the walk/run threshold at the same moment.
        let alpha = 1.0 - (-SPEED_SMOOTHING * dt).exp();
        animator.speed += (instant - animator.speed) * alpha.clamp(0.0, 1.0);

        // The avatar's yaw comes from the body's facing, but the head
        // tracks where the player is actually looking, so pass the
        // difference.
        let body_yaw = transform.rotation.to_euler(EulerRot::YXZ).0;
        animator.look(wrap_angle(camera.yaw - body_yaw), -camera.pitch);
    }
}

/// Where the first-person viewmodel's origin — the shoulder line, after
/// [`crate::anim::rig`] recentres the arms — sits relative to the eye.
///
/// Straight down and a little forward, so the arms hang into the lower part
/// of the frame rather than around the camera. This is the number to tune by
/// eye; everything else about the viewmodel derives from it.
const VIEWMODEL_OFFSET: Vec3 = Vec3::new(0.0, -0.42, -0.40);

/// How far the local avatar's body is raised above its authoritative
/// position. Purely visual — the player's collision box, movement and ray
/// casts all still use the server's position.
///
/// Needed because the eye sits at `FlyCamera::eye_height` (1.62) while
/// `girl.glb`, halo included, is about 3.19 * 0.565 = 1.80 units tall and its
/// actual skull is below the halo. Drawn at the raw position the figure
/// therefore reads as standing too low relative to the camera. Tune by eye;
/// anything sane is 0.0..0.4.
const MODEL_VISUAL_LIFT: Vec3 = Vec3::new(0.0, 0.15, 0.0);

/// How fast the smoothed local speed chases the measured one. Must match
/// `entities::sync_player_entities` on the server.
const SPEED_SMOOTHING: f32 = 8.0;

/// Shortest signed difference between two angles, in radians.
fn wrap_angle(a: f32) -> f32 {
    a.rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}
