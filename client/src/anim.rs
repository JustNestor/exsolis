//! glTF node-hierarchy animation: skeleton baking, clip sampling, clip-role
//! translation and transitions.
//!
//! # Why node hierarchy and not skinning
//!
//! `assets/models/girl2.glb` is a **node-hierarchy rig**: it has 1908 nodes,
//! 11 animation clips and zero `skins`. Every mesh primitive is rigidly
//! parented to a bone node (`LeftArm_964`, `RightForeArm_1009`, ...). So a
//! vertex has exactly one influence and linear blend skinning degenerates to
//! a single matrix per vertex.
//!
//! We still emit `JOINTS_0`/`WEIGHTS_0` and run a skinning vertex shader,
//! because that is the only way to get the whole body — 795 primitives today
//! — into **one draw call**, and because a genuinely skinned asset then works
//! with no extra code.
//!
//! # Static baking
//!
//! Only 62 of the 1908 nodes are animated by any clip. Everything else is
//! folded into vertex positions at load, so per-frame work is proportional
//! to the joint count (62) and not the node count (1908).

use bevy_ecs::prelude::*;
use bevy_log::{debug, warn};
use glam::{EulerRot, Mat4, Quat, Vec3};

pub mod rig;

/// Hard cap on joints per model. The palette buffer is sized from this, so a
/// pathological asset is rejected at load rather than at buffer creation.
pub const MAX_JOINTS: usize = 256;

/// What a clip is *for*. Clip names in shipped assets are frequently
/// destroyed (mojibake, or absent), so gameplay never keys off them — it
/// keys off the role this clip was classified as.
///
/// See [`classify_clip`] for how a role is inferred and [`parse_manifest`]
/// for how it is overridden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClipRole {
    /// A held pose (a clip with a single key). The state before anything
    /// else is known, and the target when a model has no locomotion clip.
    Rest,
    /// Standing still, looping.
    Idle,
    /// Walking.
    Walk,
    /// Same topology as `Walk`, played slower.
    WalkSlow,
    /// Running. Distinguished from `Walk` by root-motion amplitude.
    Run,
    /// Drives only a tail chain. Played as an additive layer over whatever
    /// body clip is active — never as a state of its own.
    TailSway,
    /// A hand/body emote with a long duration.
    EmoteHands,
    /// Seated. Distinguished from standing clips by driving legs and upper
    /// arms but *not* forearms.
    Sit,
    SitAlt,
    RestAlt,
    /// No manifest entry and no confident classification. Playable, but
    /// never selected automatically.
    Unknown(u16),
}

impl ClipRole {
    /// Locomotion clips, ordered by the speed they represent. A clip's
    /// natural speed comes from its root motion, not from its name.
    pub fn is_locomotion(self) -> bool {
        matches!(self, ClipRole::Walk | ClipRole::WalkSlow | ClipRole::Run)
    }

    /// Clips that must never be entered through the transition graph
    /// because they layer on top of another clip instead of replacing it.
    pub fn is_additive(self) -> bool {
        matches!(self, ClipRole::TailSway)
    }

    pub fn as_str(self) -> String {
        match self {
            ClipRole::Rest => "rest".into(),
            ClipRole::Idle => "idle".into(),
            ClipRole::Walk => "walk".into(),
            ClipRole::WalkSlow => "walk_slow".into(),
            ClipRole::Run => "run".into(),
            ClipRole::TailSway => "tail_sway".into(),
            ClipRole::EmoteHands => "emote_hands".into(),
            ClipRole::Sit => "sit".into(),
            ClipRole::SitAlt => "sit_alt".into(),
            ClipRole::RestAlt => "rest_alt".into(),
            ClipRole::Unknown(i) => format!("unknown{i}"),
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "rest" => ClipRole::Rest,
            "idle" => ClipRole::Idle,
            "walk" => ClipRole::Walk,
            "walk_slow" => ClipRole::WalkSlow,
            "run" => ClipRole::Run,
            "tail_sway" => ClipRole::TailSway,
            "emote_hands" => ClipRole::EmoteHands,
            "sit" => ClipRole::Sit,
            "sit_alt" => ClipRole::SitAlt,
            "rest_alt" => ClipRole::RestAlt,
            _ => return None,
        })
    }
}

/// Which local property a channel drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimPath {
    Translation,
    Rotation,
    Scale,
}

/// How a channel's keys are interpolated between samples.
///
/// glTF defines four interpolation modes. Every one of them is now decoded;
/// before, anything that was not `Linear` was dropped with a warning, which
/// silently froze the joints of any Blender or Blockbench "smooth" export
/// (CUBICSPLINE) or any `STEP` holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    #[default]
    Linear,
    Step,
    CubicSpline,
}

impl Interpolation {
    /// Map a glTF sampler's interpolation onto ours.
    pub fn from_gltf(m: gltf::animation::Interpolation) -> Self {
        match m {
            gltf::animation::Interpolation::Linear => Interpolation::Linear,
            gltf::animation::Interpolation::Step => Interpolation::Step,
            gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
        }
    }
}

/// One animated property of one joint.
#[derive(Debug, Clone)]
pub struct AnimChannel {
    /// Index into [`Skeleton::joints`].
    pub joint: usize,
    pub path: AnimPath,
    /// How [`Self::values`] are interpolated.
    pub interpolation: Interpolation,
    /// Key times in seconds, ascending.
    pub times: Vec<f32>,
    /// Packed key values. For `Linear`/`Step`: 3 floats per key for
    /// translation/scale, 4 for rotation — kept flat so the hot sampling path
    /// is a single slice read.
    ///
    /// For `CubicSpline`: **three** entries per key, in the order
    /// `in_tangent, value, out_tangent`, so the buffer is 3x longer than the
    /// key count. That is the glTF `CUBICSPLINE` layout exactly, which is why
    /// it can be consumed in place with no re-pack.
    pub values: Vec<f32>,
}

impl AnimChannel {
    fn components(&self) -> usize {
        match self.path {
            AnimPath::Rotation => 4,
            _ => 3,
        }
    }

    /// How many value entries a single key occupies in [`Self::values`].
    fn stride(&self) -> usize {
        match self.interpolation {
            Interpolation::CubicSpline => 3,
            _ => 1,
        }
    }

    /// Byte offset of key `i`'s *value* (the middle third of a CUBICSPLINE
    /// triple; the only entry for Linear/Step).
    #[inline]
    fn value_at(&self, i: usize) -> &[f32] {
        let comps = self.components();
        let base = i * self.stride() * comps;
        let offset = if self.interpolation == Interpolation::CubicSpline {
            comps // skip the in-tangent
        } else {
            0
        };
        &self.values[base + offset..base + offset + comps]
    }

    /// Hermite spline interpolation for one component.
    ///
    /// This is the formula from the glTF spec (§3.7.3.2) and is identical to
    /// bevy's `cubic_spline_interpolation` in
    /// `bevy_animation/src/gltf_curves.rs`, so values match bit-for-bit.
    #[inline]
    fn cubic_component(
        v0: f32,
        out_tan0: f32,
        in_tan1: f32,
        v1: f32,
        s: f32,
        dt: f32,
    ) -> f32 {
        let c0 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
        let c1 = dt * (s * s * s - 2.0 * s * s + s);
        let c2 = -2.0 * s * s * s + 3.0 * s * s;
        let c3 = dt * (s * s * s - s * s);
        v0 * c0 + out_tan0 * c1 + v1 * c2 + in_tan1 * c3
    }

    /// Sample at `time`, writing the joint's local TRS into `out`.
    ///
    /// Times outside the clip clamp to the end keys. Handles all three
    /// glTF interpolation modes (see [`Interpolation`]).
    fn sample_into(&self, time: f32, out: &mut Transform) {
        let comps = self.components();
        let last = self.times.len().saturating_sub(1);
        if last == 0 {
            return;
        }
        // `partition_point` gives the first key at or after `time`.
        let hi = self.times.partition_point(|t| *t < time).min(last);
        let lo = hi.saturating_sub(1);
        let (t0, t1) = (self.times[lo], self.times[hi]);
        let span = t1 - t0;
        // A repeated key time is legal (a hold); treat it as a step to the
        // later value rather than dividing by zero.
        let f = if span > f32::EPSILON {
            ((time - t0) / span).clamp(0.0, 1.0)
        } else if time >= t1 {
            1.0
        } else {
            0.0
        };

        match self.interpolation {
            Interpolation::Step => {
                // Step holds the earlier key until the next one is reached.
                let key = if time >= t1 { hi } else { lo };
                self.write_value(self.value_at(key), out);
            }
            Interpolation::Linear => {
                let a = self.value_at(lo);
                let b = self.value_at(hi);
                self.write_lerp(a, b, f, out);
            }
            Interpolation::CubicSpline => {
                // Each key holds [in_tangent, value, out_tangent]
                let u_base = lo * 3 * comps;
                let v_base = hi * 3 * comps;
                let in_tan_u = &self.values[u_base..u_base + comps];
                let val_u = &self.values[u_base + comps..u_base + 2 * comps];
                let out_tan_u = &self.values[u_base + 2 * comps..u_base + 3 * comps];
                let in_tan_v = &self.values[v_base..v_base + comps];
                let val_v = &self.values[v_base + comps..v_base + 2 * comps];

                match self.path {
                    AnimPath::Translation => {
                        let mut t = [0.0f32; 3];
                        for i in 0..3 {
                            t[i] = Self::cubic_component(
                                val_u[i], out_tan_u[i], in_tan_v[i], val_v[i], f, span,
                            );
                        }
                        out.translation = Vec3::from_array(t);
                    }
                    AnimPath::Scale => {
                        let mut s = [0.0f32; 3];
                        for i in 0..3 {
                            s[i] = Self::cubic_component(
                                val_u[i], out_tan_u[i], in_tan_v[i], val_v[i], f, span,
                            );
                        }
                        out.scale = Vec3::from_array(s);
                    }
                    AnimPath::Rotation => {
                        let mut q = [0.0f32; 4];
                        for i in 0..4 {
                            q[i] = Self::cubic_component(
                                val_u[i], out_tan_u[i], in_tan_v[i], val_v[i], f, span,
                            );
                        }
                        // Cubic interpolation does not preserve unit length,
                        // so the quaternion must be renormalised exactly like
                        // bevy's `CubicRotationCurve` does.
                        out.rotation = Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize();
                    }
                }
            }
        }
    }

    /// Write a raw value slice into `out` (STEP path).
    #[inline]
    fn write_value(&self, a: &[f32], out: &mut Transform) {
        match self.path {
            AnimPath::Translation => out.translation = Vec3::from_array([a[0], a[1], a[2]]),
            AnimPath::Scale => out.scale = Vec3::from_array([a[0], a[1], a[2]]),
            AnimPath::Rotation => {
                out.rotation = Quat::from_xyzw(a[0], a[1], a[2], a[3]).normalize()
            }
        }
    }

    /// Lerp two raw value slices into `out` (LINEAR path).
    #[inline]
    fn write_lerp(&self, a: &[f32], b: &[f32], f: f32, out: &mut Transform) {
        match self.path {
            AnimPath::Translation => {
                out.translation = Vec3::from_array([a[0], a[1], a[2]])
                    .lerp(Vec3::from_array([b[0], b[1], b[2]]), f);
            }
            AnimPath::Scale => {
                out.scale = Vec3::from_array([a[0], a[1], a[2]])
                    .lerp(Vec3::from_array([b[0], b[1], b[2]]), f);
            }
            AnimPath::Rotation => {
                let qa = Quat::from_xyzw(a[0], a[1], a[2], a[3]);
                let qb = Quat::from_xyzw(b[0], b[1], b[2], b[3]);
                out.rotation = qa.slerp(qb, f).normalize();
            }
        }
    }
}

/// Interpolate two local poses. Rotations go through slerp so a joint does
/// not shrink through the midpoint; translations and scales lerp.
fn blend_transform(a: Transform, b: Transform, t: f32) -> Transform {
    Transform {
        translation: a.translation.lerp(b.translation, t),
        rotation: a.rotation.slerp(b.rotation, t).normalize(),
        scale: a.scale.lerp(b.scale, t),
    }
}

/// Local TRS of a joint in some pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// One bone: the animated nodes plus every ancestor they need composed
/// through.
#[derive(Debug, Clone)]
pub struct Joint {
    pub name: String,
    /// Parent joint index, or `None` for a joint under the model root.
    pub parent: Option<usize>,
    pub rest_local: Transform,
    /// Accumulated world matrix of the non-joint ancestors between
    /// [`Joint::parent`] and this joint. Constant across every pose, so it
    /// is folded in per frame instead of being recomputed.
    pub static_prefix: Mat4,
    /// `inverse(rest_world)`. Multiplying a pose's world matrix by this
    /// maps the joint back to its rest orientation.
    pub inv_bind: Mat4,
}

/// The animated part of a model: joints, their rest pose, and the clips
/// that drive them.
#[derive(Debug, Clone, Default)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    pub clips: Vec<AnimationClip>,
    /// Joint index -> the clip roles that have a channel for it. Used by
    /// masked blending to decide whether a joint is upper or lower body.
    upper_body: Vec<bool>,
}

/// Structural facts about one clip, used to infer its [`ClipRole`].
#[derive(Debug, Clone, Default)]
pub struct ClipStats {
    pub duration: f32,
    pub max_keys: usize,
    /// Joint names this clip drives.
    pub driven: Vec<String>,
    /// Peak-to-peak vertical motion of the root/hip translation channels.
    pub root_motion: f32,
}

impl ClipStats {
    fn drives(&self, prefix: &str) -> bool {
        self.driven.iter().any(|n| n.starts_with(prefix))
    }

    /// True when the clip moves legs or arms but leaves the forearms and
    /// feet alone — the seated signature in this rig.
    fn looks_seated(&self) -> bool {
        self.drives("LeftLeg") && self.drives("RightLeg") && !self.drives("LeftForeArm")
    }

    fn only_tail(&self) -> bool {
        self.driven.iter().all(|n| n.starts_with("Tail")) && !self.driven.is_empty()
    }
}

/// Infer a clip's role from what it drives.
///
/// This *proposes*. [`parse_manifest`] overrides it, because the inference
/// reads joint-name prefixes and a re-export that renames bones would
/// otherwise silently reclassify every clip. Any disagreement between the
/// manifest and this function is logged at load.
pub fn classify_clip(stats: &ClipStats) -> ClipRole {
    // A clip with a single key is a held pose, not a loop.
    if stats.max_keys <= 1 || stats.duration <= f32::EPSILON {
        return ClipRole::Rest;
    }
    if stats.only_tail() {
        return ClipRole::TailSway;
    }
    let drives_legs = stats.drives("LeftLeg") && stats.drives("RightLeg");
    let drives_arms = stats.drives("LeftArm") && stats.drives("RightArm");
    if !drives_legs {
        // Upper-body-only motion over a long clip reads as an emote.
        if drives_arms && stats.duration > 2.0 {
            return ClipRole::EmoteHands;
        }
        return ClipRole::Unknown(0);
    }
    if stats.looks_seated() {
        return ClipRole::Sit;
    }
    // Among the standing locomotion clips, root motion separates them:
    // `walk` carries ~0.05 units of vertical bob and `run` ~0.18.
    if stats.root_motion > 0.12 {
        ClipRole::Run
    } else if stats.duration > 1.6 {
        ClipRole::WalkSlow
    } else {
        ClipRole::Walk
    }
}

/// A decoded, role-assigned clip.
#[derive(Debug, Clone)]
pub struct AnimationClip {
    /// Raw name from the file. Kept for diagnostics only — see [`ClipRole`].
    pub name: String,
    pub role: ClipRole,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
    pub stats: ClipStats,
    /// Horizontal stride length implied by the clip's own root motion,
    /// used to scale playback rate so footfalls match real speed.
    pub natural_speed: f32,
}

/// Which parts of the body a transition blends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mask {
    Full,
    /// Legs and hips only. Valid because in this rig the head/hair branch
    /// is separate from the leg branch, so a lower-body blend leaves the
    /// head exactly where it was.
    LowerBody,
    UpperBody,
}

/// How a transition eases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Easing {
    Linear,
    CubicInOut,
}

impl Easing {
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
        }
    }
}

/// One `(from, to)` edge of the transition graph.
#[derive(Debug, Clone, Copy)]
pub struct Transition {
    pub duration: f32,
    pub easing: Easing,
    pub mask: Mask,
}

impl Transition {
    const DEFAULT: Transition = Transition {
        duration: 0.25,
        easing: Easing::CubicInOut,
        mask: Mask::Full,
    };

    /// The transition to use between two clips.
    ///
    /// The table encodes what actually reads well on this rig rather than
    /// one uniform cross-fade: walk and run share a body posture, so only
    /// the legs need to cross-fade; sitting is far from every standing pose
    /// and needs a longer settle; and an emote snaps in fast but eases out
    /// slowly so it does not look like it was cut off.
    pub fn between(from: ClipRole, to: ClipRole) -> Transition {
        use ClipRole::*;
        // Additive clips layer over whatever is playing and never transition.
        if to.is_additive() || from.is_additive() {
            return Transition {
                duration: 0.0,
                easing: Easing::Linear,
                mask: Mask::Full,
            };
        }
        match (from, to) {
            // Same locomotion family: only the legs differ, and the upper
            // body holds its pose across the change.
            (Walk, Run)
            | (Run, Walk)
            | (WalkSlow, Walk)
            | (Walk, WalkSlow)
            | (WalkSlow, Run)
            | (Run, WalkSlow) => Transition {
                duration: 0.20,
                easing: Easing::CubicInOut,
                mask: Mask::LowerBody,
            },
            // Held poses are far from everything; give them room.
            (Rest, _) | (RestAlt, _) => Transition {
                duration: 0.35,
                easing: Easing::CubicInOut,
                mask: Mask::Full,
            },
            (_, Rest) | (_, RestAlt) => Transition {
                duration: 0.30,
                easing: Easing::CubicInOut,
                mask: Mask::Full,
            },
            // Seated is its own cluster.
            (Sit, _) | (SitAlt, _) | (_, Sit) | (_, SitAlt) => Transition {
                duration: 0.40,
                easing: Easing::CubicInOut,
                mask: Mask::Full,
            },
            // Snap into an emote, ease out of it.
            (_, EmoteHands) => Transition {
                duration: 0.15,
                easing: Easing::CubicInOut,
                mask: Mask::Full,
            },
            (EmoteHands, _) => Transition {
                duration: 0.45,
                easing: Easing::CubicInOut,
                mask: Mask::Full,
            },
            _ => Transition::DEFAULT,
        }
    }
}

/// Playback state of one clip.
#[derive(Debug, Clone, Copy)]
pub struct PlayState {
    pub clip: u32,
    pub time: f32,
}

/// How long one first-person arm swing takes, in seconds. Matches the
/// cadence of a mining swing rather than a click, so repeated clicks read as
/// one continuous motion instead of a stutter.
pub const SWING_DURATION: f32 = 0.25;

/// Additive arm swing for the first-person viewmodel.
///
/// This lives on the *viewmodel instance matrix*, not on the bone palette:
/// the palette is shared with the body, and swinging the body from a mouse
/// click would make other players' arms flail.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct UseSwing {
    /// 1 at the start of a swing, easing back to 0.
    pub phase: f32,
}

impl UseSwing {
    /// Start a swing. Re-triggering mid-swing restarts it, which is what
    /// makes holding the button read as repeated swings.
    pub fn trigger(&mut self) {
        self.phase = 1.0;
    }

    /// Advance the swing. Returns true while it is still in progress.
    pub fn advance(&mut self, dt: f32) -> bool {
        if self.phase <= 0.0 {
            return false;
        }
        self.phase -= dt / SWING_DURATION;
        // Snap to finished rather than creeping toward zero: a residue of
        // 1e-8 would leave the hand permanently a hair off its rest pose.
        if self.phase <= 1e-4 {
            self.phase = 0.0;
        }
        self.phase > 0.0
    }

    /// Swing angle at the current phase.
    ///
    /// A half sine over the swing: zero at both ends, peak in the middle.
    /// `phase` counts down from 1 to 0 across the swing, so the progress
    /// through the swing is `1 - phase`, mapped onto 0..PI — feeding the
    /// raw 0..1 progress to `sin` would only ever climb the first quarter
    /// of the curve and never reach the peak.
    pub fn angle(&self) -> f32 {
        if self.phase <= 0.0 {
            return 0.0;
        }
        let progress = 1.0 - self.phase;
        (progress * std::f32::consts::PI).sin() * SWING_ANGLE
    }
}

/// Peak swing angle in radians: enough to read clearly without the hand
/// leaving the frame.
const SWING_ANGLE: f32 = 55f32.to_radians();

/// Drives one animated instance: which clip is playing, how far through it,
/// and how far into a cross-fade we are.
#[derive(Debug, Clone, Copy, bevy_ecs::component::Component)]
pub struct Animator {
    pub current: PlayState,
    /// The outgoing clip during a transition; `None` when settled.
    pub previous: Option<PlayState>,
    /// Seconds since the transition started.
    pub blend_elapsed: f32,
    /// Authoritative horizontal speed in blocks/second, pushed by the
    /// server with each entity state update. Drives both clip choice and
    /// playback rate.
    pub speed: f32,
    pub airborne: bool,
    /// Additive head look, radians. Applied on top of whatever clip is
    /// playing, because the head is an independent branch of this rig.
    pub head_yaw: f32,
    pub head_pitch: f32,
}

/// Head-look limits, in radians.
pub const HEAD_YAW_LIMIT: f32 = 60f32.to_radians();
pub const HEAD_PITCH_LIMIT: f32 = 40f32.to_radians();

impl Animator {
    /// A new animator sitting on clip 0.
    pub fn new() -> Self {
        Self {
            current: PlayState { clip: 0, time: 0.0 },
            previous: None,
            blend_elapsed: 0.0,
            speed: 0.0,
            airborne: false,
            head_yaw: 0.0,
            head_pitch: 0.0,
        }
    }

    /// Ask for `clip`. Does nothing if it is already playing, so callers can
    /// request the locomotion clip every frame without churn.
    pub fn play(&mut self, clip: u32, _transition: Transition) {
        if clip == self.current.clip {
            return;
        }
        self.previous = Some(self.current);
        self.current.clip = clip;
        self.current.time = 0.0;
        self.blend_elapsed = 0.0;
    }

    /// Point the head additively, in radians relative to the body's facing.
    /// Clamped so the avatar never folds over backwards.
    pub fn look(&mut self, yaw: f32, pitch: f32) {
        self.head_yaw = yaw.clamp(-HEAD_YAW_LIMIT, HEAD_YAW_LIMIT);
        self.head_pitch = pitch.clamp(-HEAD_PITCH_LIMIT, HEAD_PITCH_LIMIT);
    }

    /// Blend weight of the outgoing clip, 1 at the start of a transition and
    /// 0 once it has finished.
    pub fn previous_weight(&self, duration: f32) -> Option<f32> {
        let prev = self.previous?;
        if duration <= f32::EPSILON {
            return None;
        }
        let t = (self.blend_elapsed / duration).clamp(0.0, 1.0);
        (t < 1.0).then_some(1.0 - t)
    }

    /// Advance time. `rate` scales playback so footfall cadence matches the
    /// authoritative movement speed.
    pub fn advance(&mut self, dt: f32, rate: f32) {
        self.current.time += dt * rate.max(0.0);
        if let Some(prev) = &mut self.previous {
            prev.time += dt * rate.max(0.0);
        }
        self.blend_elapsed += dt;
    }
}

impl Default for Animator {
    fn default() -> Self {
        Self::new()
    }
}

impl Skeleton {
    /// Whether a joint belongs to the upper body, by the bone-name
    /// convention of this rig. Used to apply [`Mask::LowerBody`] blends.
    pub fn is_upper_body(&self, joint: usize) -> bool {
        self.upper_body.get(joint).copied().unwrap_or(true)
    }

    /// The first clip with the given role, if the model has one.
    pub fn clip_with_role(&self, role: ClipRole) -> Option<u32> {
        self.clips
            .iter()
            .position(|c| c.role == role)
            .map(|i| i as u32)
    }

    /// Pick the locomotion clip that best matches an authoritative speed.
    ///
    /// The server's smoothed speed drives this rather than client-side
    /// velocity differencing, because a dropped unreliable update would
    /// otherwise make the walk/run blend flicker.
    pub fn locomotion_for_speed(&self, speed: f32) -> Option<u32> {
        let run = self.clip_with_role(ClipRole::Run)?;
        let walk = self.clip_with_role(ClipRole::Walk).unwrap_or(run);
        // Threshold sits above a brisk walk and below a jog; the clips
        // themselves decide the actual playback rate via `natural_speed`.
        Some(if speed > 3.2 { run } else { walk })
    }

    /// The role a clip was assigned, so a transition can look up the right
    /// `(from, to)` edge in the graph.
    pub fn role_of(&self, clip: u32) -> ClipRole {
        self.clips
            .get(clip as usize)
            .map(|c| c.role)
            .unwrap_or(ClipRole::Unknown(u16::MAX))
    }

    /// The transition to use between two clips.
    pub fn transition(&self, from_clip: u32, to_clip: u32) -> Transition {
        Transition::between(self.role_of(from_clip), self.role_of(to_clip))
    }

    /// Compose the bone palette for a pose that may be mid-transition.
    ///
    /// `blend` is `None` once a transition has settled, in which case this
    /// is just [`Skeleton::compose_palette`]. Otherwise the two poses are
    /// blended **per joint** according to the transition's [`Mask`] and then
    /// composed once.
    ///
    /// Blending the finished bone matrices instead would be wrong for a
    /// masked transition: a lower-body blend has to leave the head's
    /// matrices exactly as the incoming clip posed them, and averaging two
    /// world-space matrices is not the same as posing the joint once.
    pub fn compose_blend(
        &self,
        locals: &mut [Transform],
        previous: Option<(u32, f32, Transition, f32)>,
        worlds: &mut [Mat4],
        bones: &mut [Mat4],
        scratch_locals: &mut Vec<Transform>,
    ) {
        let Some((from_clip, from_time, transition, weight)) = previous else {
            self.compose_palette(locals, worlds, bones);
            return;
        };
        if transition.duration <= f32::EPSILON || weight <= 0.0 {
            self.compose_palette(locals, worlds, bones);
            return;
        }
        let eased = transition.easing.apply(weight);

        scratch_locals.clear();
        scratch_locals.resize(self.joints.len(), Transform::default());
        self.sample(from_clip, from_time, scratch_locals);

        for (index, slot) in locals.iter_mut().enumerate() {
            let Some(from) = scratch_locals.get(index).copied() else {
                continue;
            };
            let blended = blend_transform(from, *slot, eased);
            *slot = match transition.mask {
                Mask::Full => blended,
                // A masked joint keeps the pose it already had — for
                // `LowerBody` that is the incoming clip's upper body, which
                // is what makes walk -> run readable instead of a full-body
                // mush.
                Mask::LowerBody if self.is_upper_body(index) => *slot,
                Mask::UpperBody if !self.is_upper_body(index) => *slot,
                Mask::LowerBody | Mask::UpperBody => blended,
            };
        }
        self.compose_palette(locals, worlds, bones);
    }

    /// Sample `clip` at `time` into `out`, one entry per joint. Joints the
    /// clip does not drive fall back to their rest pose, which is what makes
    /// a partial clip (a tail sway) safe to play on its own.
    pub fn sample(&self, clip: u32, time: f32, out: &mut [Transform]) {
        for (slot, joint) in out.iter_mut().zip(self.joints.iter()) {
            *slot = joint.rest_local;
        }
        let Some(clip) = self.clips.get(clip as usize) else {
            return;
        };
        let t = if clip.duration > 0.0 {
            time.rem_euclid(clip.duration)
        } else {
            0.0
        };
        for channel in &clip.channels {
            if let Some(slot) = out.get_mut(channel.joint) {
                channel.sample_into(t, slot);
            }
        }
    }

    /// Compose the bone palette for a pose: `palette[j] = world[j] *
    /// inv_bind[j]`, which maps joint-local vertex space to model space.
    ///
    /// Only joints are visited — the static ancestor chain is a precomputed
    /// constant per joint — so this is O(joints), not O(nodes).
    pub fn compose_palette(&self, locals: &[Transform], worlds: &mut [Mat4], palette: &mut [Mat4]) {
        for (j, joint) in self.joints.iter().enumerate() {
            let local = locals.get(j).copied().unwrap_or_default();
            let parent_world = joint.parent.map_or(Mat4::IDENTITY, |p| worlds[p]);
            let world = parent_world * joint.static_prefix * local.to_mat4();
            worlds[j] = world;
            palette[j] = world * joint.inv_bind;
        }
    }
}

// --------------------------------------------------------------- manifest --

/// Parse a clip manifest.
///
/// The format is deliberately trivial so no new dependency is needed:
///
/// ```text
/// # comment
/// models/girl2.glb
/// 0 rest
/// 1 walk
/// ```
///
/// The manifest is **authoritative**. Without an entry, roles come from
/// [`classify_clip`], which reads joint-name prefixes and would misfire on a
/// re-export that renames bones.
pub fn parse_manifest(text: &str) -> std::collections::HashMap<String, Vec<ClipRole>> {
    let mut out = std::collections::HashMap::new();
    let mut current: Option<String> = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if !line.contains(char::is_whitespace) {
            current = Some(line.to_string());
            out.entry(current.clone().unwrap()).or_default();
            continue;
        }
        let Some((index, role)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let (Ok(index), Some(role)) = (index.trim().parse::<u16>(), ClipRole::parse(role.trim()))
        else {
            warn!("clip manifest: skipping malformed line {line:?}");
            continue;
        };
        if let Some(path) = &current {
            let roles: &mut Vec<ClipRole> = out.entry(path.clone()).or_default();
            if roles.len() <= index as usize {
                roles.resize(index as usize + 1, ClipRole::Unknown(u16::MAX));
            }
            roles[index as usize] = role;
        }
    }
    out
}

/// Apply a manifest over classifier output, logging disagreements.
pub fn apply_manifest(clips: &mut [AnimationClip], manifest: &[(String, ClipRole)]) {
    for (index, clip) in clips.iter_mut().enumerate() {
        let Some((_, role)) = manifest.get(index) else {
            if clip.role != ClipRole::Unknown(u16::MAX) {
                debug!(
                    "clip {index} ({:?}) has no manifest entry; classifier said {}",
                    clip.name,
                    clip.role.as_str()
                );
            }
            continue;
        };
        if clip.role != *role && !matches!(clip.role, ClipRole::Unknown(_)) {
            warn!(
                "clip {index} ({:?}): manifest says {} but the classifier inferred {}",
                clip.name,
                role.as_str(),
                clip.role.as_str()
            );
        }
        clip.role = *role;
    }
}

/// The authoritative role table for the two Blockbench exports of this avatar.
///
/// Measured from the files (see documentation/plans/ENTITIES_ANIMATION_PLAN.md §2.2). Every
/// clip name except `walk` and `extra6` was destroyed by the exporter, so
/// these indices are the only stable identifier.
///
/// Both paths are listed because they are the *same* rig in two containers:
/// `girl.glb` is the node-hierarchy original that ships as `PLAYER_MODEL`, and
/// `girl2.glb` is the skinned re-export with identical clips. Keying on only
/// one of them leaves whichever is running with no manifest at all, which is
/// how the shipped avatar once fell back to the structural classifier.
pub const GIRL_CLIP_ROLES: &[(&str, ClipRole)] = &[
    ("models/girl.glb#0", ClipRole::Rest),
    ("models/girl.glb#1", ClipRole::Walk),
    ("models/girl.glb#2", ClipRole::Walk),
    ("models/girl.glb#3", ClipRole::TailSway),
    ("models/girl.glb#4", ClipRole::EmoteHands),
    ("models/girl.glb#5", ClipRole::WalkSlow),
    ("models/girl.glb#6", ClipRole::Sit),
    ("models/girl.glb#7", ClipRole::Run),
    ("models/girl.glb#8", ClipRole::Idle),
    ("models/girl.glb#9", ClipRole::SitAlt),
    ("models/girl.glb#10", ClipRole::RestAlt),
    ("models/girl2.glb#0", ClipRole::Rest),
    ("models/girl2.glb#1", ClipRole::Walk),
    ("models/girl2.glb#2", ClipRole::Walk),
    ("models/girl2.glb#3", ClipRole::TailSway),
    ("models/girl2.glb#4", ClipRole::EmoteHands),
    ("models/girl2.glb#5", ClipRole::WalkSlow),
    ("models/girl2.glb#6", ClipRole::Sit),
    ("models/girl2.glb#7", ClipRole::Run),
    ("models/girl2.glb#8", ClipRole::Idle),
    ("models/girl2.glb#9", ClipRole::SitAlt),
    ("models/girl2.glb#10", ClipRole::RestAlt),
];

/// Clip roles bundled in the binary for `path`, if any.
pub fn builtin_roles(path: &str) -> &'static [(&'static str, ClipRole)] {
    // The table is stored as one slice with a fixed entry per model, so
    // slicing keeps the path -> role mapping in one place.
    if path == "models/girl.glb" {
        &GIRL_CLIP_ROLES[0..11]
    } else if path == "models/girl2.glb" {
        &GIRL_CLIP_ROLES[11..]
    } else {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(driven: &[&str], duration: f32, keys: usize, root: f32) -> ClipStats {
        ClipStats {
            duration,
            max_keys: keys,
            driven: driven.iter().map(|s| s.to_string()).collect(),
            root_motion: root,
        }
    }

    /// Build a translation channel with an explicit interpolation mode.
    fn translation_channel(
        times: Vec<f32>,
        values: Vec<f32>,
        interpolation: Interpolation,
    ) -> AnimChannel {
        AnimChannel {
            joint: 0,
            path: AnimPath::Translation,
            interpolation,
            times,
            values,
        }
    }

    #[test]
    fn linear_translation_interpolates_the_midpoint() {
        let channel = translation_channel(
            vec![0.0, 1.0],
            vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            Interpolation::Linear,
        );
        let mut out = Transform::default();
        channel.sample_into(0.5, &mut out);
        assert_eq!(out.translation, Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn step_translation_holds_until_the_next_key() {
        let channel = translation_channel(
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0],
            Interpolation::Step,
        );
        let mut out = Transform::default();

        channel.sample_into(0.0, &mut out);
        assert_eq!(out.translation, Vec3::ZERO, "t=0 takes key 0");
        channel.sample_into(0.5, &mut out);
        assert_eq!(out.translation, Vec3::ZERO, "t=0.5 still holds key 0");
        channel.sample_into(1.0, &mut out);
        assert_eq!(out.translation, Vec3::new(10.0, 0.0, 0.0), "t=1 switches");
        channel.sample_into(1.5, &mut out);
        assert_eq!(out.translation, Vec3::new(10.0, 0.0, 0.0));
        channel.sample_into(2.0, &mut out);
        assert_eq!(out.translation, Vec3::new(10.0, 10.0, 0.0));
        // Past the last key it clamps rather than dropping to zero.
        channel.sample_into(3.0, &mut out);
        assert_eq!(out.translation, Vec3::new(10.0, 10.0, 0.0));
    }

    #[test]
    fn cubic_spline_hits_its_key_values_exactly() {
        // Two keys, each [in_tangent, value, out_tangent].
        // Values 0 -> 10 with zero in/out tangents at both ends and a mild
        // out-tangent on key 0 so the curve is not a straight line.
        let values = vec![
            // key 0
            0.0, 0.0, 0.0, // in_tangent (unused by the first segment)
            0.0, 0.0, 0.0, // value
            10.0, 0.0, 0.0, // out_tangent
            // key 1
            10.0, 0.0, 0.0, // in_tangent
            10.0, 0.0, 0.0, // value
            0.0, 0.0, 0.0, // out_tangent (unused)
        ];
        let channel = translation_channel(vec![0.0, 1.0], values, Interpolation::CubicSpline);
        let mut out = Transform::default();

        channel.sample_into(0.0, &mut out);
        assert_eq!(out.translation, Vec3::ZERO, "curve starts at key 0's value");
        channel.sample_into(1.0, &mut out);
        assert_eq!(
            out.translation,
            Vec3::new(10.0, 0.0, 0.0),
            "curve ends at key 1's value"
        );
    }

    #[test]
    fn cubic_spline_stays_inside_the_tangent_bounds() {
        // A monotone ramp whose out/in tangents are flat must not overshoot.
        let values = vec![
            0.0, 0.0, 0.0, // key0 in_tangent
            0.0, 0.0, 0.0, // key0 value
            0.0, 0.0, 0.0, // key0 out_tangent (flat: no overshoot)
            0.0, 0.0, 0.0, // key1 in_tangent
            10.0, 0.0, 0.0, // key1 value
            0.0, 0.0, 0.0, // key1 out_tangent
        ];
        let channel = translation_channel(vec![0.0, 1.0], values, Interpolation::CubicSpline);
        let mut out = Transform::default();
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            channel.sample_into(t, &mut out);
            assert!(
                out.translation.x >= -1e-5 && out.translation.x <= 10.0 + 1e-5,
                "t={t}: x={} escaped [0,10]",
                out.translation.x
            );
        }
    }

    #[test]
    fn cubic_spline_rotation_is_renormalised() {
        // Cubic interpolation destroys unit length; the sampler must fix it.
        let values = vec![
            0.0, 0.0, 0.0, 0.0, // key0 in_tangent
            0.0, 0.0, 0.0, 1.0, // key0 value: identity
            0.0, 0.0, 0.0, 0.0, // key0 out_tangent
            0.0, 0.0, 0.0, 0.0, // key1 in_tangent
            0.0, 0.7, 0.0, 0.7, // key1 value: ~180deg around Y
            0.0, 0.0, 0.0, 0.0, // key1 out_tangent
        ];
        let channel = AnimChannel {
            joint: 0,
            path: AnimPath::Rotation,
            interpolation: Interpolation::CubicSpline,
            times: vec![0.0, 1.0],
            values,
        };
        let mut out = Transform::default();
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            channel.sample_into(t, &mut out);
            assert!(
                (out.rotation.length() - 1.0).abs() < 1e-4,
                "t={t}: |q|={} is not unit length",
                out.rotation.length()
            );
        }
    }

    #[test]
    fn step_rotation_snaps_between_keys() {
        let channel = AnimChannel {
            joint: 0,
            path: AnimPath::Rotation,
            interpolation: Interpolation::Step,
            times: vec![0.0, 1.0],
            values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.7, 0.0, 0.7],
        };
        let mut out = Transform::default();

        channel.sample_into(0.5, &mut out);
        assert_eq!(out.rotation, Quat::IDENTITY, "first half holds identity");
        channel.sample_into(1.0, &mut out);
        let expected = Quat::from_xyzw(0.0, 0.7, 0.0, 0.7).normalize();
        assert!(
            out.rotation.dot(expected).abs() > 0.999,
            "second half should be the second key"
        );
    }

    #[test]
    fn a_single_key_clip_is_a_held_pose() {
        assert_eq!(
            classify_clip(&stats(&["LeftArm_1", "LeftLeg_2"], 0.0, 1, 0.0)),
            ClipRole::Rest
        );
    }

    #[test]
    fn a_tail_only_clip_is_additive() {
        let role = classify_clip(&stats(&["Tail_182", "Tail2_180"], 3.0, 53, 0.0));
        assert_eq!(role, ClipRole::TailSway);
        assert!(role.is_additive());
    }

    #[test]
    fn root_motion_separates_walk_from_run() {
        // The real clip numbers: walk bobs 0.05, run 0.181.
        assert_eq!(
            classify_clip(&stats(
                &[
                    "LeftLeg_52",
                    "RightLeg_97",
                    "LeftArm_964",
                    "RightArm_1021",
                    "LeftForeArm_960",
                    "RightForeArm_1009",
                ],
                1.0,
                19,
                0.050
            )),
            ClipRole::Walk
        );
        assert_eq!(
            classify_clip(&stats(
                &[
                    "LeftLeg_52",
                    "RightLeg_97",
                    "LeftArm_964",
                    "RightArm_1021",
                    "LeftForeArm_960",
                    "RightForeArm_1009",
                ],
                6.0,
                109,
                0.181
            )),
            ClipRole::Run
        );
    }

    #[test]
    fn legs_without_forearms_is_a_seated_pose() {
        assert_eq!(
            classify_clip(&stats(
                &["LeftLeg_52", "RightLeg_97", "LeftArm_964", "LeftFoot_50"],
                4.958,
                86,
                0.0
            )),
            ClipRole::Sit
        );
    }

    #[test]
    fn the_shipped_manifest_reproduces_the_measured_classification() {
        // The classifier is only a fallback; for the girl asset the manifest
        // is what ships. Guard the two obvious disagreements so a change to
        // either side is noticed.
        let walk = stats(
            &[
                "LeftLeg_52",
                "RightLeg_97",
                "LeftArm_964",
                "RightArm_1021",
                "LeftForeArm_960",
                "RightForeArm_1009",
            ],
            1.0,
            19,
            0.050,
        );
        assert_eq!(classify_clip(&walk), ClipRole::Walk);
        assert_eq!(GIRL_CLIP_ROLES[1].1, ClipRole::Walk);
        assert_eq!(GIRL_CLIP_ROLES[7].1, ClipRole::Run);
    }

    #[test]
    fn manifest_overrides_are_applied_by_index() {
        let text = "\
# a comment
models/test.glb
0 rest
2 run
";
        let parsed = parse_manifest(text);
        let roles = parsed.get("models/test.glb").unwrap();
        assert_eq!(roles[0], ClipRole::Rest);
        assert_eq!(roles[2], ClipRole::Run);
        // Index 1 was never listed.
        assert!(!roles[1].is_locomotion());
    }

    #[test]
    fn walk_run_transition_only_moves_the_lower_body() {
        let t = Transition::between(ClipRole::Walk, ClipRole::Run);
        assert_eq!(t.mask, Mask::LowerBody);
        assert_eq!(t.duration, 0.20);
        // Anything else gets a full-body cross-fade.
        let t = Transition::between(ClipRole::Idle, ClipRole::Walk);
        assert_eq!(t.mask, Mask::Full);
    }

    #[test]
    fn additive_clips_never_transition() {
        let t = Transition::between(ClipRole::Walk, ClipRole::TailSway);
        assert_eq!(t.duration, 0.0);
    }

    #[test]
    fn easing_is_monotonic_and_pinned_at_both_ends() {
        for e in [Easing::Linear, Easing::CubicInOut] {
            assert_eq!(e.apply(0.0), 0.0);
            assert_eq!(e.apply(1.0), 1.0);
            assert_eq!(e.apply(-5.0), 0.0);
            assert_eq!(e.apply(5.0), 1.0);
            let mut last = 0.0;
            for i in 0..=20 {
                let v = e.apply(i as f32 / 20.0);
                assert!(v >= last, "{e:?} went backwards at {i}");
                last = v;
            }
        }
    }

    fn channel(path: AnimPath, times: &[f32], values: &[f32]) -> AnimChannel {
        AnimChannel {
            joint: 0,
            path,
            interpolation: Interpolation::Linear,
            times: times.to_vec(),
            values: values.to_vec(),
        }
    }

    #[test]
    fn sampling_interpolates_between_keys() {
        let c = channel(
            AnimPath::Translation,
            &[0.0, 1.0],
            &[0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
        );
        let mut t = Transform::default();
        c.sample_into(0.5, &mut t);
        assert!(
            (t.translation.x - 5.0).abs() < 1e-5,
            "got {:?}",
            t.translation
        );
        // Clamps outside the clip instead of extrapolating.
        c.sample_into(-1.0, &mut t);
        assert_eq!(t.translation.x, 0.0);
        c.sample_into(9.0, &mut t);
        assert_eq!(t.translation.x, 10.0);
    }

    #[test]
    fn a_repeated_key_time_holds_then_steps() {
        // Two keys at the same time is a hold, not a divide by zero.
        let c = channel(AnimPath::Translation, &[0.0, 0.5, 0.5, 1.0], &[0.0; 12]);
        let mut t = Transform::default();
        c.sample_into(0.5, &mut t);
        assert!(t.translation.x.is_finite());
    }

    #[test]
    fn rotations_are_normalized_after_slerp() {
        let c = channel(
            AnimPath::Rotation,
            &[0.0, 1.0],
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7071, 0.7071],
        );
        let mut t = Transform::default();
        c.sample_into(0.5, &mut t);
        assert!(
            (t.rotation.length() - 1.0).abs() < 1e-3,
            "got {:?}",
            t.rotation
        );
    }

    #[test]
    fn driving_the_same_clip_is_a_no_op() {
        let mut a = Animator::new();
        a.play(3, Transition::DEFAULT);
        // Already on 3, so a second request must not restart the clip or
        // reset the blend — callers request the locomotion clip every frame.
        a.current.time = 1.5;
        a.play(3, Transition::DEFAULT);
        assert_eq!(a.current.clip, 3);
        assert_eq!(a.current.time, 1.5, "play must not rewind the clip");
    }

    #[test]
    fn a_transition_fades_the_previous_clip_out_to_zero() {
        let mut a = Animator::new();
        a.play(1, Transition::DEFAULT);
        assert!(a.previous_weight(0.25).is_some());
        a.blend_elapsed = 0.125;
        assert!((a.previous_weight(0.25).unwrap() - 0.5).abs() < 1e-5);
        a.blend_elapsed = 0.25;
        assert!(a.previous_weight(0.25).is_none(), "must settle");
    }

    #[test]
    fn time_wraps_instead_of_running_away() {
        let mut a = Animator::new();
        a.current.time = 0.0;
        a.advance(10.0, 1.0);
        assert_eq!(a.current.time, 10.0);
        // The clip itself wraps during sampling.
    }

    #[test]
    fn the_swing_rests_at_zero_and_returns_to_it() {
        let mut swing = UseSwing::default();
        assert_eq!(swing.angle(), 0.0);
        assert!(!swing.advance(0.1), "an idle swing must not run");

        swing.trigger();
        // The swing starts *at* the rest pose and leaves it over the next
        // few frames, so the first frame is legitimately zero.
        assert_eq!(swing.angle(), 0.0, "a swing starts from the rest pose");
        let mut moved = false;
        for _ in 0..(SWING_DURATION * 60.0) as usize {
            swing.advance(1.0 / 60.0);
            moved |= swing.angle() > 0.0;
        }
        assert!(moved, "a triggered swing must actually move");
        assert_eq!(swing.phase, 0.0);
        assert_eq!(swing.angle(), 0.0, "must settle back to the rest pose");
    }

    #[test]
    fn the_swing_peaks_partway_through() {
        let mut swing = UseSwing::default();
        swing.trigger();
        let mut peak = 0.0f32;
        // Step finely enough to land near the middle of the swing.
        let steps = (SWING_DURATION * 200.0) as usize;
        for _ in 0..steps {
            swing.advance(SWING_DURATION / steps as f32);
            peak = peak.max(swing.angle());
        }
        assert!(
            peak > SWING_ANGLE * 0.9,
            "swing peaked at {peak}, expected close to {SWING_ANGLE}"
        );
    }

    #[test]
    fn retriggering_restarts_the_swing() {
        let mut swing = UseSwing::default();
        swing.trigger();
        for _ in 0..10 {
            swing.advance(0.01);
        }
        let before = swing.phase;
        assert!(before < 1.0);
        swing.trigger();
        assert_eq!(swing.phase, 1.0, "holding the button repeats the swing");
    }

    #[test]
    fn head_look_is_clamped_so_the_avatar_never_folds_over() {
        let mut a = Animator::new();
        a.look(10.0, -10.0);
        assert_eq!(a.head_yaw, HEAD_YAW_LIMIT);
        assert_eq!(a.head_pitch, -HEAD_PITCH_LIMIT);
        a.look(0.1, 0.2);
        assert!((a.head_yaw - 0.1).abs() < 1e-6);
        assert!((a.head_pitch - 0.2).abs() < 1e-6);
    }

    /// The head and legs are separate branches of this rig, which is what
    /// makes a lower-body mask possible at all.
    #[test]
    fn upper_body_classification_separates_legs_from_the_head() {
        let skeleton = Skeleton {
            joints: vec![
                Joint {
                    name: "Head_822".into(),
                    parent: None,
                    rest_local: Transform::default(),
                    static_prefix: Mat4::IDENTITY,
                    inv_bind: Mat4::IDENTITY,
                },
                Joint {
                    name: "LeftLeg_52".into(),
                    parent: None,
                    rest_local: Transform::default(),
                    static_prefix: Mat4::IDENTITY,
                    inv_bind: Mat4::IDENTITY,
                },
                Joint {
                    name: "RightFoot_95".into(),
                    parent: None,
                    rest_local: Transform::default(),
                    static_prefix: Mat4::IDENTITY,
                    inv_bind: Mat4::IDENTITY,
                },
            ],
            clips: Vec::new(),
            upper_body: vec![true, false, false],
        };
        assert!(skeleton.is_upper_body(0), "head is upper body");
        assert!(!skeleton.is_upper_body(1), "thigh is lower body");
        assert!(!skeleton.is_upper_body(2), "foot is lower body");
    }

    /// A lower-body transition must leave the head's pose alone. If the
    /// mask were applied to the composed matrices instead of the local
    /// poses, the head would drift halfway toward the other clip.
    #[test]
    fn a_lower_body_transition_leaves_the_head_at_the_incoming_pose() {
        let skeleton = Skeleton {
            joints: vec![
                Joint {
                    name: "Head_822".into(),
                    parent: None,
                    rest_local: Transform::default(),
                    static_prefix: Mat4::IDENTITY,
                    inv_bind: Mat4::IDENTITY,
                },
                Joint {
                    name: "LeftLeg_52".into(),
                    parent: None,
                    rest_local: Transform::default(),
                    static_prefix: Mat4::IDENTITY,
                    inv_bind: Mat4::IDENTITY,
                },
            ],
            clips: Vec::new(),
            upper_body: vec![true, false],
        };
        // Incoming pose: head rotated, leg at rest.
        let mut locals = vec![
            Transform {
                rotation: Quat::from_rotation_y(1.0),
                ..Default::default()
            },
            Transform::default(),
        ];
        let mut worlds = vec![Mat4::IDENTITY; 2];
        let mut bones = vec![Mat4::IDENTITY; 2];
        let mut scratch = Vec::new();
        skeleton.compose_blend(
            &mut locals,
            // Outgoing pose would have put the head somewhere else; with no
            // clips the sampler falls back to rest, so the head must end up
            // exactly where the incoming pose put it.
            Some((
                0,
                0.0,
                Transition::between(ClipRole::Walk, ClipRole::Run),
                0.5,
            )),
            &mut worlds,
            &mut bones,
            &mut scratch,
        );
        assert!(
            (locals[0].rotation.to_euler(EulerRot::YXZ).0 - 1.0).abs() < 1e-3,
            "head moved during a lower-body blend: {:?}",
            locals[0].rotation
        );
    }

    #[test]
    fn a_full_body_blend_does_move_everything() {
        let skeleton = Skeleton {
            joints: vec![Joint {
                name: "Head_822".into(),
                parent: None,
                rest_local: Transform::default(),
                static_prefix: Mat4::IDENTITY,
                inv_bind: Mat4::IDENTITY,
            }],
            clips: Vec::new(),
            upper_body: vec![true],
        };
        let mut locals = vec![Transform {
            rotation: Quat::from_rotation_y(1.0),
            ..Default::default()
        }];
        let mut worlds = vec![Mat4::IDENTITY];
        let mut bones = vec![Mat4::IDENTITY];
        let mut scratch = Vec::new();
        skeleton.compose_blend(
            &mut locals,
            Some((0, 0.0, Transition::DEFAULT, 0.5)),
            &mut worlds,
            &mut bones,
            &mut scratch,
        );
        // Halfway from rest to the incoming pose.
        let yaw = locals[0].rotation.to_euler(EulerRot::YXZ).0;
        assert!(
            (yaw - 0.5).abs() < 1e-3,
            "expected a half-way blend, got {yaw}"
        );
    }
}
