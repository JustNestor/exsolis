//! Animation curve implementations for all glTF interpolation modes.
//!
//! Supports:
//! - LINEAR (UnevenSampleAutoCurve equivalent)
//! - STEP (SteppedKeyframeCurve)
//! - CUBICSPLINE (CubicKeyframeCurve for Vec3/Vec4, CubicRotationCurve for Quat)
//! - Constant (single keyframe)

use std::fmt::Debug;

use glam::{Quat, Vec3, Vec4};
use thiserror::Error;

/// Animation interpolation mode from glTF
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMode {
    Linear,
    Step,
    CubicSpline,
}

/// A time-value pair for keyframes
#[derive(Debug, Clone, Copy)]
pub struct Keyframe<T> {
    pub time: f32,
    pub value: T,
}

/// Trait for sampling animation curves
pub trait AnimationCurve<T>: Debug + Send + Sync
where
    T: Send + Sync,
{
    /// Returns the time domain of the curve
    fn domain(&self) -> (f32, f32);
    
    /// Samples the curve at time t (clamped to domain)
    fn sample_clamped(&self, t: f32) -> T;
    
    /// Samples the curve at time t (unchecked, for performance)
    fn sample_unchecked(&self, t: f32) -> T {
        self.sample_clamped(t)
    }
}

/// Errors for curve construction
#[derive(Error, Debug)]
pub enum CurveError {
    #[error("Insufficient keyframes: need at least 2, got {0}")]
    InsufficientKeyframes(usize),
    #[error("Keyframe times must be strictly increasing")]
    NonIncreasingTimes,
    #[error("Invalid cubic spline data: values length must be 3x times length")]
    InvalidCubicSplineData,
    /// The value buffer is neither one-per-key (Linear/Step) nor
    /// three-per-key (CubicSpline). Silently truncating would drop the tail
    /// of a clip and leave a joint frozen on a stale value.
    #[error("{values_given} values for {divisor} expected slots")]
    LengthMismatch {
        /// Number of values given.
        values_given: usize,
        /// Number of slots the data should have filled.
        divisor: usize,
    },
}

/// ---------- LINEAR INTERPOLATION ----------

/// Linear interpolation curve (equivalent to bevy's UnevenSampleAutoCurve)
#[derive(Debug, Clone)]
pub struct LinearCurve<T> {
    times: Vec<f32>,
    values: Vec<T>,
}

impl<T: Clone + Debug + PartialEq> LinearCurve<T> {
    pub fn new(keyframes: Vec<Keyframe<T>>) -> Result<Self, CurveError> {
        if keyframes.len() < 2 {
            return Err(CurveError::InsufficientKeyframes(keyframes.len()));
        }
        
        let mut times = Vec::with_capacity(keyframes.len());
        let mut values = Vec::with_capacity(keyframes.len());
        
        for (i, kf) in keyframes.iter().enumerate() {
            if i > 0 && kf.time <= times[i - 1] {
                return Err(CurveError::NonIncreasingTimes);
            }
            times.push(kf.time);
            values.push(kf.value.clone());
        }
        
        Ok(Self { times, values })
    }
}

impl<T> AnimationCurve<T> for LinearCurve<T>
where
    T: Clone + Debug + PartialEq + lerp::Lerp<f32> + Send + Sync,
{
    fn domain(&self) -> (f32, f32) {
        (self.times[0], *self.times.last().unwrap())
    }
    
    fn sample_clamped(&self, t: f32) -> T {
        let t = t.clamp(self.times[0], *self.times.last().unwrap());
        
        // Find the segment
        let idx = self.times.partition_point(|&time| time < t);
        if idx == 0 {
            return self.values[0].clone();
        }
        if idx >= self.times.len() {
            return self.values.last().unwrap().clone();
        }
        
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let v0 = &self.values[idx - 1];
        let v1 = &self.values[idx];
        
        if t1 - t0 < f32::EPSILON {
            return v1.clone();
        }
        
        let factor = (t - t0) / (t1 - t0);
        v0.lerp(v1, factor)
    }
}

/// Linear interpolation for Vec3
impl lerp::Lerp<f32> for Vec3 {
    fn lerp(&self, other: &Self, factor: f32) -> Self {
        Vec3::lerp(*self, *other, factor)
    }
}

/// Linear interpolation for Quat (uses slerp)
impl lerp::Lerp<f32> for Quat {
    fn lerp(&self, other: &Self, factor: f32) -> Self {
        self.slerp(*other, factor)
    }
}

/// Linear interpolation for f32
impl lerp::Lerp<f32> for f32 {
    fn lerp(&self, other: &Self, factor: f32) -> Self {
        self + (other - self) * factor
    }
}

/// ---------- STEP INTERPOLATION ----------

/// Step interpolation curve (holds value until next keyframe)
#[derive(Debug, Clone)]
pub struct StepCurve<T> {
    times: Vec<f32>,
    values: Vec<T>,
}

impl<T: Clone + Debug> StepCurve<T> {
    pub fn new(keyframes: Vec<Keyframe<T>>) -> Result<Self, CurveError> {
        if keyframes.len() < 2 {
            return Err(CurveError::InsufficientKeyframes(keyframes.len()));
        }
        
        let mut times = Vec::with_capacity(keyframes.len());
        let mut values = Vec::with_capacity(keyframes.len());
        
        for (i, kf) in keyframes.iter().enumerate() {
            if i > 0 && kf.time <= times[i - 1] {
                return Err(CurveError::NonIncreasingTimes);
            }
            times.push(kf.time);
            values.push(kf.value.clone());
        }
        
        Ok(Self { times, values })
    }
}

impl<T: Clone + Debug + Send + Sync> AnimationCurve<T> for StepCurve<T> {
    fn domain(&self) -> (f32, f32) {
        (self.times[0], *self.times.last().unwrap())
    }
    
    fn sample_clamped(&self, t: f32) -> T {
        let t = t.clamp(self.times[0], *self.times.last().unwrap());
        
        // Find the last keyframe at or before t
        let idx = self.times.partition_point(|&time| time <= t);
        if idx == 0 {
            self.values[0].clone()
        } else {
            self.values[idx - 1].clone()
        }
    }
}

/// ---------- CUBIC SPLINE INTERPOLATION ----------

/// Cubic spline interpolation for Vec3/Vec4 (in-tangent, value, out-tangent per keyframe)
#[derive(Debug, Clone)]
pub struct CubicSplineCurve<T> {
    times: Vec<f32>,
    // Values stored as [in_tangent, value, out_tangent] per keyframe
    values: Vec<T>,
}

impl<T: Clone + Debug + Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>> CubicSplineCurve<T> {
    pub fn new(times: Vec<f32>, values: Vec<T>) -> Result<Self, CurveError> {
        if times.len() < 2 {
            return Err(CurveError::InsufficientKeyframes(times.len()));
        }
        if values.len() != times.len() * 3 {
            return Err(CurveError::InvalidCubicSplineData);
        }
        
        for i in 1..times.len() {
            if times[i] <= times[i - 1] {
                return Err(CurveError::NonIncreasingTimes);
            }
        }
        
        Ok(Self { times, values })
    }
}

impl<T> AnimationCurve<T> for CubicSplineCurve<T>
where
    T: Clone + Debug + Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + Send + Sync,
{
    fn domain(&self) -> (f32, f32) {
        (self.times[0], *self.times.last().unwrap())
    }
    
    fn sample_clamped(&self, t: f32) -> T {
        let t = t.clamp(self.times[0], *self.times.last().unwrap());
        
        // Find the segment
        let idx = self.times.partition_point(|&time| time < t);
        if idx == 0 {
            return self.values[1]; // First keyframe's value
        }
        if idx >= self.times.len() {
            let last_idx = (self.times.len() - 1) * 3 + 1;
            return self.values[last_idx]; // Last keyframe's value
        }
        
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let dt = t1 - t0;
        
        if dt < f32::EPSILON {
            return self.values[idx * 3 + 1];
        }
        
        let s = (t - t0) / dt;
        
        // Each keyframe has 3 values: [in_tangent, value, out_tangent]
        let u = &self.values[(idx - 1) * 3..idx * 3];
        let v = &self.values[idx * 3..(idx + 1) * 3];
        
        // u[1] = value at t0, u[2] = out_tangent at t0
        // v[0] = in_tangent at t1, v[1] = value at t1
        cubic_spline_interpolation(u[1], u[2], v[0], v[1], s, dt)
    }
}

/// Cubic spline interpolation helper (same formula as bevy)
fn cubic_spline_interpolation<T>(
    value_start: T,
    tangent_out_start: T,
    tangent_in_end: T,
    value_end: T,
    lerp: f32,
    step_duration: f32,
) -> T
where
    T: Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
{
    // Coefficients from glTF spec
    let c0 = 2.0 * lerp * lerp * lerp - 3.0 * lerp * lerp + 1.0;
    let c1 = step_duration * (lerp * lerp * lerp - 2.0 * lerp * lerp + lerp);
    let c2 = -2.0 * lerp * lerp * lerp + 3.0 * lerp * lerp;
    let c3 = step_duration * (lerp * lerp * lerp - lerp * lerp);
    
    value_start * c0 + tangent_out_start * c1 + value_end * c2 + tangent_in_end * c3
}

/// Cubic spline interpolation for Quaternions (specialized for rotation)
#[derive(Debug, Clone)]
pub struct CubicRotationCurve {
    times: Vec<f32>,
    // Values stored as [in_tangent, value, out_tangent] per keyframe (Vec4)
    values: Vec<Vec4>,
}

impl CubicRotationCurve {
    pub fn new(times: Vec<f32>, values: Vec<Vec4>) -> Result<Self, CurveError> {
        if times.len() < 2 {
            return Err(CurveError::InsufficientKeyframes(times.len()));
        }
        if values.len() != times.len() * 3 {
            return Err(CurveError::InvalidCubicSplineData);
        }
        
        for i in 1..times.len() {
            if times[i] <= times[i - 1] {
                return Err(CurveError::NonIncreasingTimes);
            }
        }
        
        Ok(Self { times, values })
    }
}

impl AnimationCurve<Quat> for CubicRotationCurve {
    fn domain(&self) -> (f32, f32) {
        (self.times[0], *self.times.last().unwrap())
    }
    
    fn sample_clamped(&self, t: f32) -> Quat {
        let t = t.clamp(self.times[0], *self.times.last().unwrap());
        
        let idx = self.times.partition_point(|&time| time < t);
        if idx == 0 {
            return Quat::from_vec4(self.values[1].normalize());
        }
        if idx >= self.times.len() {
            let last_idx = (self.times.len() - 1) * 3 + 1;
            return Quat::from_vec4(self.values[last_idx].normalize());
        }
        
        let t0 = self.times[idx - 1];
        let t1 = self.times[idx];
        let dt = t1 - t0;
        
        if dt < f32::EPSILON {
            return Quat::from_vec4(self.values[idx * 3 + 1].normalize());
        }
        
        let s = (t - t0) / dt;
        
        let u = &self.values[(idx - 1) * 3..idx * 3];
        let v = &self.values[idx * 3..(idx + 1) * 3];
        
        let interpolated = cubic_spline_interpolation(u[1], u[2], v[0], v[1], s, dt);
        Quat::from_vec4(interpolated.normalize())
    }
}

/// ---------- CONSTANT CURVE (single keyframe) ----------

/// Constant curve (single keyframe, holds value forever)
#[derive(Debug, Clone)]
pub struct ConstantCurve<T> {
    value: T,
}

impl<T: Clone + Debug> ConstantCurve<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone + Debug + Send + Sync> AnimationCurve<T> for ConstantCurve<T> {
    fn domain(&self) -> (f32, f32) {
        (0.0, f32::MAX)
    }
    
    fn sample_clamped(&self, _t: f32) -> T {
        self.value.clone()
    }
}

/// ---------- CURVE FACTORY ----------

/// Creates a Vec3 curve from glTF data
///
/// `values` must be exactly as long as `times` for Linear/Step, and three
/// times as long for CubicSpline (`in_tangent, value, out_tangent` per key).
pub fn create_vec3_curve(
    mode: InterpolationMode,
    times: Vec<f32>,
    values: Vec<[f32; 3]>,
) -> Result<Box<dyn AnimationCurve<Vec3>>, CurveError> {
    let expected = match mode {
        InterpolationMode::CubicSpline => times.len() * 3,
        _ => times.len(),
    };
    if values.len() != expected {
        return Err(CurveError::LengthMismatch {
            values_given: values.len(),
            divisor: expected,
        });
    }

    let vec3_values: Vec<Vec3> = values.into_iter().map(Vec3::from).collect();
    match mode {
        InterpolationMode::Linear => {
            let keyframes: Vec<Keyframe<Vec3>> = times
                .into_iter()
                .zip(vec3_values)
                .map(|(t, v)| Keyframe { time: t, value: v })
                .collect();
            Ok(Box::new(LinearCurve::new(keyframes)?))
        }
        InterpolationMode::Step => {
            let keyframes: Vec<Keyframe<Vec3>> = times
                .into_iter()
                .zip(vec3_values)
                .map(|(t, v)| Keyframe { time: t, value: v })
                .collect();
            Ok(Box::new(StepCurve::new(keyframes)?))
        }
        InterpolationMode::CubicSpline => {
            Ok(Box::new(CubicSplineCurve::new(times, vec3_values)?))
        }
    }
}

/// Creates a Quat curve from glTF data (handles cubic spline specially)
///
/// Same length contract as [`create_vec3_curve`].
pub fn create_quat_curve(
    mode: InterpolationMode,
    times: Vec<f32>,
    values: Vec<[f32; 4]>,
) -> Result<Box<dyn AnimationCurve<Quat>>, CurveError> {
    let expected = match mode {
        InterpolationMode::CubicSpline => times.len() * 3,
        _ => times.len(),
    };
    if values.len() != expected {
        return Err(CurveError::LengthMismatch {
            values_given: values.len(),
            divisor: expected,
        });
    }

    match mode {
        InterpolationMode::CubicSpline => {
            let vec4_values: Vec<Vec4> = values.into_iter().map(Vec4::from).collect();
            Ok(Box::new(CubicRotationCurve::new(times, vec4_values)?))
        }
        InterpolationMode::Linear => {
            let keyframes: Vec<Keyframe<Quat>> = times
                .into_iter()
                .zip(values)
                .map(|(t, v)| Keyframe {
                    time: t,
                    value: Quat::from_xyzw(v[0], v[1], v[2], v[3]),
                })
                .collect();
            Ok(Box::new(LinearCurve::new(keyframes)?))
        }
        InterpolationMode::Step => {
            let keyframes: Vec<Keyframe<Quat>> = times
                .into_iter()
                .zip(values)
                .map(|(t, v)| Keyframe {
                    time: t,
                    value: Quat::from_xyzw(v[0], v[1], v[2], v[3]),
                })
                .collect();
            Ok(Box::new(StepCurve::new(keyframes)?))
        }
    }
}

/// Creates a Vec3 scale curve from glTF data
pub fn create_scale_curve(
    mode: InterpolationMode,
    times: Vec<f32>,
    values: Vec<[f32; 3]>,
) -> Result<Box<dyn AnimationCurve<Vec3>>, CurveError> {
    create_vec3_curve(mode, times, values)
}

/// lerp trait for types that can be linearly interpolated
mod lerp {
    pub trait Lerp<T> {
        fn lerp(&self, other: &Self, factor: T) -> Self;
    }
}