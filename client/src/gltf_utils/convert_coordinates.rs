//! Coordinate system conversion from glTF (+Z forward) to engine coordinates.
//!
//! glTF uses a right-handed coordinate system with:
//! - Forward: +Z
//! - Up: +Y
//! - Right: -X
//!
//! This module provides conversions for positions, normals, and tangents.

use glam::{Mat4, Quat, Vec3};

/// Trait for converting from glTF coordinates to engine coordinates.
pub trait ConvertCoordinates {
    /// Converts from glTF coordinates to engine coordinate system.
    fn convert_coordinates(self) -> Self;
}

impl ConvertCoordinates for Vec3 {
    fn convert_coordinates(self) -> Self {
        Vec3::new(-self.x, self.y, -self.z)
    }
}

impl ConvertCoordinates for [f32; 3] {
    fn convert_coordinates(self) -> Self {
        [-self[0], self[1], -self[2]]
    }
}

impl ConvertCoordinates for [f32; 4] {
    fn convert_coordinates(self) -> Self {
        // Tangent conversion: q' = r q r* where r is the rotation from glTF to engine
        // For tangents (xyz = tangent, w = handedness), we flip x and z
        [-self[0], self[1], -self[2], self[3]]
    }
}

/// 4x4 matrix that converts from glTF coordinate system to engine coordinate system.
///
/// This is a **proper rotation** of π about Y (`det = 1`), not a reflection: so
/// it is its own inverse and conjugating by it transforms any rotation
/// correctly. (`diag(-1,1,-1)` is exactly `R_y(π)`.)
pub fn gltf_to_engine_mat4() -> Mat4 {
    Mat4::from_scale(Vec3::new(-1.0, 1.0, -1.0))
}

/// The same basis change expressed as a quaternion, for conjugating the
/// decomposed path below.
pub fn gltf_to_engine_quat() -> Quat {
    // π about Y in xyzw.
    Quat::from_xyzw(0.0, 1.0, 0.0, 0.0)
}

/// Convert a glTF matrix (column-major `f32[4][4]`) to engine Mat4 with
/// coordinate conversion.
///
/// `B · M · B` is the correct similarity transform, and since `B = B⁻¹`
/// this also happens to be `B · M · B⁻¹`.
pub fn convert_gltf_matrix(matrix: &[[f32; 4]; 4]) -> Mat4 {
    let m = Mat4::from_cols_array_2d(matrix);
    let b = gltf_to_engine_mat4();
    b * m * b
}

/// Convert a glTF decomposed transform to engine Mat4 with coordinate conversion.
///
/// Translation and scale just get `x`/`z` flipped. The **rotation has to be
/// conjugated** — without that, every non-identity orientation comes out
/// mirrored, which is invisible on axis-aligned frames and wrong everywhere
/// else. Matrix and decomposed forms therefore agree, which
/// `matrices_and_decomposed_transforms_agree` pins down.
pub fn convert_gltf_transform(
    translation: &[f32; 3],
    rotation: &[f32; 4],
    scale: &[f32; 3],
) -> Mat4 {
    let translation = Vec3::new(-translation[0], translation[1], -translation[2]);
    let q = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let b = gltf_to_engine_quat();
    let rotation = b * q * b.conjugate();
    let scale = Vec3::new(scale[0], scale[1], scale[2]);

    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}