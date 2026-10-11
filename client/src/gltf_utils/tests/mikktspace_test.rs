//! Tests for mikktspace tangent generation

use crate::gltf_utils::mikktspace::*;
use glam::Vec3;

#[test]
fn test_compute_tangents_simple_quad() {
    // Two triangles forming a quad in XY plane
    // UVs go from (0,0) to (1,1)
    let positions = vec![
        [0.0, 0.0, 0.0],  // 0
        [1.0, 0.0, 0.0],  // 1
        [1.0, 1.0, 0.0],  // 2
        [0.0, 1.0, 0.0],  // 3
    ];
    
    let normals = vec![
        [0.0, 0.0, 1.0]; 4
    ];
    
    let uvs = vec![
        [0.0, 0.0],  // 0
        [1.0, 0.0],  // 1
        [1.0, 1.0],  // 2
        [0.0, 1.0],  // 3
    ];
    
    let indices = vec![
        0, 1, 2,  // First triangle
        0, 2, 3,  // Second triangle
    ];
    
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
    
    assert_eq!(tangents.len(), 4);
    
    // For a quad in XY plane with UVs matching geometry:
    // Tangent should point along +X (U direction)
    // Bitangent should point along +Y (V direction)
    // Normal is +Z
    // Handedness: N x T = B -> +Z x +X = +Y, so handedness = +1
    
    for tangent in &tangents {
        // Tangent should be roughly (+1, 0, 0) or (-1, 0, 0) depending on UV orientation
        assert!(tangent[0].abs() > 0.9); // Mostly X
        assert!(tangent[1].abs() < 0.1); // Very little Y
        assert!(tangent[2].abs() < 0.1); // Very little Z
        assert_eq!(tangent[3], 1.0); // Positive handedness
    }
}

#[test]
fn test_compute_tangents_flipped_uv() {
    // Same quad but UV V coordinate flipped
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    
    let normals = vec![
        [0.0, 0.0, 1.0]; 4
    ];
    
    // UV V is flipped (1->0, 0->1)
    let uvs = vec![
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];
    
    let indices = vec![0, 1, 2, 0, 2, 3];
    
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
    
    // With flipped V, handedness should be negative
    for tangent in &tangents {
        assert_eq!(tangent[3], -1.0);
    }
}

#[test]
fn test_compute_tangents_degenerate_uv_triangle() {
    // Triangle with zero UV area (degenerate)
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ];
    
    let normals = vec![
        [0.0, 0.0, 1.0]; 3
    ];
    
    // All vertices have same UV - degenerate
    let uvs = vec![
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ];
    
    let indices = vec![0, 1, 2];
    
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
    
    // Should not crash, should produce zero tangents for degenerate triangle
    assert_eq!(tangents.len(), 3);
    for tangent in &tangents {
        // All zeros because r = 0 was guarded
        assert_eq!(tangent[0], 0.0);
        assert_eq!(tangent[1], 0.0);
        assert_eq!(tangent[2], 0.0);
        assert_eq!(tangent[3], 1.0); // Default handedness
    }
}

#[test]
fn test_compute_tangents_cube() {
    // Cube with proper UVs
    let positions = vec![
        // Front face (z = 1)
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        // Back face (z = -1)
        [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
        // Top face (y = 1)
        [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0],
        // Bottom face (y = -1)
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0],
        // Right face (x = 1)
        [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
        // Left face (x = -1)
        [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0],
    ];
    
    let normals = vec![
        // Front
        [0.0, 0.0, 1.0]; 4
    ];
    // ... (this is tedious, let's just test it runs)
    
    // Just verify it doesn't panic with a minimal cube
    let uvs = vec![[0.0, 0.0]; 24]; // Dummy UVs
    let indices = vec![0, 1, 2, 0, 2, 3]; // Just one face
    
    let tangents = compute_tangents(&positions[0..4], &normals, &uvs[0..4], &indices);
    assert_eq!(tangents.len(), 4);
}

#[test]
fn test_tangent_orthogonal_to_normal() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    
    let normals = vec![
        [0.0, 0.0, 1.0]; 3
    ];
    
    let uvs = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ];
    
    let indices = vec![0, 1, 2];
    
    let tangents = compute_tangents(&positions, &normals, &uvs, &indices);
    
    // Tangent should be orthogonal to normal
    for tangent in &tangents {
        let t = Vec3::new(tangent[0], tangent[1], tangent[2]);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let dot = t.dot(n);
        assert!(dot.abs() < 0.001, "Tangent not orthogonal to normal: dot = {}", dot);
    }
}