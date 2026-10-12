//! Tests for coordinate system conversion

use crate::gltf_utils::convert_coordinates::*;
use glam::{Mat4, Quat, Vec3, Vec4};

#[test]
fn test_vec3_convert_coordinates() {
    // glTF +Z forward -> Engine -Z forward
    let gltf_vec = Vec3::new(1.0, 2.0, 3.0);
    let engine_vec = gltf_vec.convert_coordinates();
    assert_eq!(engine_vec, Vec3::new(-1.0, 2.0, -3.0));
}

#[test]
fn test_array3_convert_coordinates() {
    let gltf_arr: [f32; 3] = [1.0, 2.0, 3.0];
    let engine_arr = gltf_arr.convert_coordinates();
    assert_eq!(engine_arr, [-1.0, 2.0, -3.0]);
}

#[test]
fn test_array4_convert_coordinates_tangent() {
    // Tangent: xyz = tangent vector, w = handedness
    let gltf_tangent: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let engine_tangent = gltf_tangent.convert_coordinates();
    // x and z flipped, w (handedness) preserved
    assert_eq!(engine_tangent, [-1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_gltf_to_engine_mat4() {
    let mat = gltf_to_engine_mat4();
    // Should be scale(-1, 1, -1) - axes are Vec4
    assert_eq!(mat.x_axis, Vec4::new(-1.0, 0.0, 0.0, 0.0));
    assert_eq!(mat.y_axis, Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(mat.z_axis, Vec4::new(0.0, 0.0, -1.0, 0.0));
    assert_eq!(mat.w_axis, Vec4::new(0.0, 0.0, 0.0, 1.0));
}

#[test]
fn test_convert_gltf_matrix() {
    // Identity matrix in glTF
    let gltf_matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    
    let engine_mat = convert_gltf_matrix(&gltf_matrix);
    // Identity matrix in glTF converts to identity in engine (C * I * C = I since C = C^-1)
    assert_eq!(engine_mat, Mat4::IDENTITY);
}

#[test]
fn the_basis_change_is_a_rotation_not_a_reflection() {
    // diag(-1,1,-1) has det = +1, so it is R_y(pi) and conjugating a
    // quaternion by it is legitimate. If this ever became det = -1 the
    // quaternion conjugation would silently mirror.
    assert_eq!(gltf_to_engine_mat4().determinant(), 1.0);
}

#[test]
fn an_identity_rotation_survives_conjugation() {
    let converted =
        convert_gltf_transform(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
    assert_eq!(converted.w_axis, Vec4::new(-1.0, 2.0, -3.0, 1.0));
    // No rotation introduced by the conversion itself.
    assert!(converted
        .to_scale_rotation_translation()
        .1
        .abs_diff_eq(Quat::IDENTITY, 1e-6));
}

/// The matrix and decomposed paths must agree, otherwise the two encodings of
/// the same node render in two different places. The decomposed path has to
/// conjugate the rotation; a plain flip of x/z mirrors it.
#[test]
fn matrices_and_decomposed_transforms_agree() {
    // A deliberately awkward transform: 60deg about Y then 36deg about X,
    // non-uniform scale, non-zero translation.
    let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_3)
        * Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let translation = Vec3::new(1.5, -2.25, 0.75);
    let scale = Vec3::new(2.0, 1.0, 3.0);

    let from_trs = convert_gltf_transform(
        &translation.to_array(),
        &[rotation.x, rotation.y, rotation.z, rotation.w],
        &scale.to_array(),
    );

    // Build the equivalent glTF matrix (T * R * S) and convert it.
    let gltf_matrix =
        Mat4::from_scale_rotation_translation(scale, rotation, translation).to_cols_array_2d();
    let from_matrix = convert_gltf_matrix(&gltf_matrix);

    for i in 0..4 {
        let a = from_trs.col(i);
        let b = from_matrix.col(i);
        assert!(
            (a - b).length() < 1e-4,
            "column {i} differs: TRS path gave {a:?}, matrix path gave {b:?}"
        );
    }
}

#[test]
fn test_convert_gltf_transform() {
    // Translation: (1, 2, 3) in glTF -> (-1, 2, -3) in engine
    // Rotation: identity
    // Scale: (1, 1, 1)
    let translation = [1.0, 2.0, 3.0];
    let rotation = [0.0, 0.0, 0.0, 1.0]; // Identity quaternion
    let scale = [1.0, 1.0, 1.0];
    
    let mat = convert_gltf_transform(&translation, &rotation, &scale);
    
    // Translation should be converted
    assert_eq!(mat.w_axis.x, -1.0);
    assert_eq!(mat.w_axis.y, 2.0);
    assert_eq!(mat.w_axis.z, -3.0);
}

#[test]
fn test_convert_gltf_transform_with_rotation() {
    // 90 degree rotation around Y in glTF
    let translation = [0.0, 0.0, 0.0];
    let rotation = [0.0, 0.70710677, 0.0, 0.70710677]; // 90 deg around Y
    let scale = [1.0, 1.0, 1.0];
    
    let mat = convert_gltf_transform(&translation, &rotation, &scale);
    
    // The rotation should be properly converted
    // In glTF: +Z forward, in engine: -Z forward
    // A 90deg Y rotation in glTF points +X to +Z
    // In engine: same rotation points +X to -Z
    // So the matrix should reflect this
    assert!(mat.x_axis.z < 0.0); // Forward is now -Z
}