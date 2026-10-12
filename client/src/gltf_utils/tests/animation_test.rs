//! Tests for animation curves (all interpolation modes)

use crate::gltf_utils::animation::curves::*;
use glam::{Quat, Vec3, Vec4};

#[test]
fn test_linear_curve_vec3() {
    let keyframes = vec![
        Keyframe { time: 0.0, value: Vec3::new(0.0, 0.0, 0.0) },
        Keyframe { time: 1.0, value: Vec3::new(10.0, 10.0, 10.0) },
    ];
    
    let curve = LinearCurve::new(keyframes).unwrap();
    
    // At t=0, should be at start
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(0.0, 0.0, 0.0));
    
    // At t=1, should be at end
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(10.0, 10.0, 10.0));
    
    // At t=0.5, should be halfway
    assert_eq!(curve.sample_clamped(0.5), Vec3::new(5.0, 5.0, 5.0));
    
    // Clamping: before start
    assert_eq!(curve.sample_clamped(-1.0), Vec3::new(0.0, 0.0, 0.0));
    
    // Clamping: after end
    assert_eq!(curve.sample_clamped(2.0), Vec3::new(10.0, 10.0, 10.0));
}

#[test]
#[ignore = "Quaternion slerp at 180deg needs investigation"]
fn test_linear_curve_quat() {
    let keyframes = vec![
        Keyframe { time: 0.0, value: Quat::IDENTITY },
        Keyframe { time: 1.0, value: Quat::from_rotation_y(std::f32::consts::PI) },
    ];
    
    let curve = LinearCurve::new(keyframes).unwrap();
    
    // At t=0, identity
    assert_eq!(curve.sample_clamped(0.0), Quat::IDENTITY);
    
    // At t=1, 180 deg around Y - use angle_between since q and -q represent same rotation
    let end = curve.sample_clamped(1.0);
    assert!(end.angle_between(Quat::from_rotation_y(std::f32::consts::PI)) < 0.1);
    
    // At t=0.5, 90 deg around Y (slerp) - slerp at 180deg can be numerically unstable
    let mid = curve.sample_clamped(0.5);
    assert!((mid.angle_between(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))).abs() < 0.1);
}

#[test]
fn test_step_curve() {
    let keyframes = vec![
        Keyframe { time: 0.0, value: Vec3::new(0.0, 0.0, 0.0) },
        Keyframe { time: 1.0, value: Vec3::new(10.0, 0.0, 0.0) },
        Keyframe { time: 2.0, value: Vec3::new(10.0, 10.0, 0.0) },
    ];
    
    let curve = StepCurve::new(keyframes).unwrap();
    
    // Step holds value until next keyframe
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(0.5), Vec3::new(0.0, 0.0, 0.0)); // Still first value
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(10.0, 0.0, 0.0)); // Switches at t=1
    assert_eq!(curve.sample_clamped(1.5), Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(2.0), Vec3::new(10.0, 10.0, 0.0));
    assert_eq!(curve.sample_clamped(3.0), Vec3::new(10.0, 10.0, 0.0)); // Clamped to last
}

#[test]
fn test_cubic_spline_curve_vec3() {
    // Two keyframes with cubic spline: each has [in_tangent, value, out_tangent]
    // For a simple linear-like motion, tangents should be (value_end - value_start) / dt
    let times = vec![0.0, 1.0];
    let values = vec![
        // Keyframe 0: in_tangent, value, out_tangent
        Vec3::new(0.0, 0.0, 0.0),  // in_tangent (unused for first)
        Vec3::new(0.0, 0.0, 0.0),  // value at t=0
        Vec3::new(10.0, 0.0, 0.0), // out_tangent
        // Keyframe 1: in_tangent, value, out_tangent
        Vec3::new(10.0, 0.0, 0.0), // in_tangent
        Vec3::new(10.0, 0.0, 0.0), // value at t=1
        Vec3::new(0.0, 0.0, 0.0),  // out_tangent (unused for last)
    ];
    
    let curve = CubicSplineCurve::new(times, values).unwrap();
    
    // At t=0, should be at start value
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(0.0, 0.0, 0.0));
    
    // At t=1, should be at end value
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(10.0, 0.0, 0.0));
    
    // At t=0.5, should be smoothly interpolated
    let mid = curve.sample_clamped(0.5);
    // With zero tangents at endpoints and matching tangents, should be smooth
    assert!(mid.x > 0.0 && mid.x < 10.0);
}

#[test]
#[ignore = "Quaternion cubic spline interpolation needs investigation"]
fn test_cubic_rotation_curve() {
    let times = vec![0.0, 1.0];
    let values = vec![
        // Keyframe 0
        Vec4::new(0.0, 0.0, 0.0, 0.0),  // in_tangent
        Vec4::new(0.0, 0.0, 0.0, 1.0),  // value (identity quat)
        Vec4::new(0.0, 1.0, 0.0, 0.0),  // out_tangent
        // Keyframe 1
        Vec4::new(0.0, 1.0, 0.0, 0.0),  // in_tangent
        Vec4::new(0.0, 1.0, 0.0, 1.0),  // value (180 deg around Y)
        Vec4::new(0.0, 0.0, 0.0, 0.0),  // out_tangent
    ];
    
    let curve = CubicRotationCurve::new(times, values).unwrap();
    
    // At t=0, identity
    assert_eq!(curve.sample_clamped(0.0), Quat::IDENTITY);
    
    // At t=1, 180 deg around Y
    let end = curve.sample_clamped(1.0);
    assert!((end.angle_between(Quat::from_rotation_y(std::f32::consts::PI))).abs() < 0.1);
    
    // At t=0.5, should be smoothly interpolated
    let mid = curve.sample_clamped(0.5);
    assert!(mid.angle_between(Quat::IDENTITY) > 0.0);
    assert!(mid.angle_between(Quat::from_rotation_y(std::f32::consts::PI)) > 0.0);
}

#[test]
fn test_constant_curve() {
    let curve = ConstantCurve::new(Vec3::new(5.0, 5.0, 5.0));
    
    // Always returns the same value
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(5.0, 5.0, 5.0));
    assert_eq!(curve.sample_clamped(100.0), Vec3::new(5.0, 5.0, 5.0));
    assert_eq!(curve.sample_clamped(-100.0), Vec3::new(5.0, 5.0, 5.0));
    
    // Domain is infinite
    assert_eq!(curve.domain(), (0.0, f32::MAX));
}

#[test]
fn test_create_vec3_curve_linear() {
    let times = vec![0.0, 1.0, 2.0];
    let values = vec![
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
    ];
    
    let curve = create_vec3_curve(InterpolationMode::Linear, times, values).unwrap();
    
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(10.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(2.0), Vec3::new(10.0, 10.0, 0.0));
    assert_eq!(curve.sample_clamped(0.5), Vec3::new(5.0, 0.0, 0.0));
}

#[test]
fn test_create_vec3_curve_step() {
    let times = vec![0.0, 1.0];
    let values = vec![
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ];
    
    let curve = create_vec3_curve(InterpolationMode::Step, times, values).unwrap();
    
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(0.5), Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(10.0, 0.0, 0.0));
}

#[test]
#[ignore = "Quaternion cubic spline interpolation needs investigation"]
fn test_create_quat_curve_cubic_spline() {
    let times = vec![0.0, 1.0];
    // For cubic spline, need 3 values per keyframe: [in_tangent, value, out_tangent]
    let values = vec![
        // Keyframe 0
        [0.0, 0.0, 0.0, 0.0],  // in_tangent
        [0.0, 0.0, 0.0, 1.0],  // value (identity)
        [0.0, 0.0, 0.0, 0.0],  // out_tangent
        // Keyframe 1
        [0.0, 0.0, 0.0, 0.0],  // in_tangent
        [0.0, 1.0, 0.0, 1.0],  // value (180 deg Y)
        [0.0, 0.0, 0.0, 0.0],  // out_tangent
    ];
    
    let curve = create_quat_curve(InterpolationMode::CubicSpline, times, values).unwrap();
    
    assert_eq!(curve.sample_clamped(0.0), Quat::IDENTITY);
    let end = curve.sample_clamped(1.0);
    assert!((end.angle_between(Quat::from_rotation_y(std::f32::consts::PI))).abs() < 0.1);
}

#[test]
fn test_create_scale_curve() {
    let times = vec![0.0, 1.0];
    let values = vec![
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ];
    
    let curve = create_scale_curve(InterpolationMode::Linear, times, values).unwrap();
    
    assert_eq!(curve.sample_clamped(0.0), Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(curve.sample_clamped(1.0), Vec3::new(2.0, 2.0, 2.0));
    assert_eq!(curve.sample_clamped(0.5), Vec3::new(1.5, 1.5, 1.5));
}

#[test]
fn test_curve_error_insufficient_keyframes() {
    let keyframes = vec![Keyframe { time: 0.0, value: Vec3::ZERO }];
    let result = LinearCurve::new(keyframes);
    assert!(matches!(result, Err(CurveError::InsufficientKeyframes(1))));
}

#[test]
fn test_curve_error_non_increasing_times() {
    let keyframes = vec![
        Keyframe { time: 1.0, value: Vec3::ZERO },
        Keyframe { time: 0.5, value: Vec3::ONE },
    ];
    let result = LinearCurve::new(keyframes);
    assert!(matches!(result, Err(CurveError::NonIncreasingTimes)));
}

#[test]
fn test_cubic_spline_error_length_mismatch() {
    let times = vec![0.0, 1.0];
    let values = vec![Vec3::ZERO; 5]; // Wrong: should be 6 (2 * 3)
    let result = CubicSplineCurve::new(times, values);
    assert!(matches!(result, Err(CurveError::InvalidCubicSplineData)));
}
#[test]
fn a_short_value_buffer_is_rejected_rather_than_truncated() {
    // Two keys but only one value. The factory used to zip and silently
    // produce a one-key curve, freezing the joint on a stale value.
    let times = vec![0.0, 1.0];
    let values = vec![[0.0, 0.0, 0.0]];
    let result = create_vec3_curve(InterpolationMode::Linear, times, values);
    assert!(
        matches!(
            result,
            Err(CurveError::LengthMismatch {
                values_given: 1,
                divisor: 2
            })
        ),
        "a short buffer must be an error, got {result:?}"
    );
}

#[test]
fn a_long_value_buffer_is_also_rejected() {
    let times = vec![0.0, 1.0];
    let values = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
    let result = create_vec3_curve(InterpolationMode::Linear, times, values);
    assert!(matches!(
        result,
        Err(CurveError::LengthMismatch {
            values_given: 3,
            divisor: 2
        })
    ));
}

#[test]
fn cubic_spline_needs_three_values_per_key() {
    let times = vec![0.0, 1.0];
    // Two entries: fine for Linear, not for CubicSpline.
    let values = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let result = create_vec3_curve(InterpolationMode::CubicSpline, times, values);
    assert!(matches!(
        result,
        Err(CurveError::LengthMismatch {
            values_given: 2,
            divisor: 6
        })
    ));
}

#[test]
fn the_quat_factory_validates_too() {
    let times = vec![0.0, 1.0];
    let values = vec![[0.0, 0.0, 0.0, 1.0]];
    let result = create_quat_curve(InterpolationMode::Linear, times, values);
    assert!(matches!(result, Err(CurveError::LengthMismatch { .. })));
}

#[test]
fn a_consistent_value_buffer_is_accepted() {
    let times = vec![0.0, 1.0, 2.0];
    let values = vec![[0.0; 3], [1.0; 3], [2.0; 3]];
    assert!(create_vec3_curve(InterpolationMode::Linear, times, values).is_ok());
}
