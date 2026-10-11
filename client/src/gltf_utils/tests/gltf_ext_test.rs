//! Tests for glTF extension utilities

use crate::gltf_utils::gltf_ext::*;

#[test]
fn test_primitive_topology() {
    assert_eq!(mesh::primitive_topology(gltf::mesh::Mode::Triangles).unwrap(), PrimitiveTopology::TriangleList);
    assert_eq!(mesh::primitive_topology(gltf::mesh::Mode::TriangleStrip).unwrap(), PrimitiveTopology::TriangleStrip);
    assert_eq!(mesh::primitive_topology(gltf::mesh::Mode::Lines).unwrap(), PrimitiveTopology::LineList);
    assert_eq!(mesh::primitive_topology(gltf::mesh::Mode::LineStrip).unwrap(), PrimitiveTopology::LineStrip);
    assert_eq!(mesh::primitive_topology(gltf::mesh::Mode::Points).unwrap(), PrimitiveTopology::PointList);
    
    // Unsupported mode
    assert!(mesh::primitive_topology(gltf::mesh::Mode::TriangleFan).is_err());
}

#[test]
fn test_material_alpha_mode() {
    // We can't easily create a gltf::Material in tests without the full glTF document
    // This test is more of a compile-check
    let _opaque = AlphaMode::Opaque;
    let _mask = AlphaMode::Mask(0.5);
    let _blend = AlphaMode::Blend;
}

#[test]
fn test_uv_channel() {
    // Can't easily test without gltf::Material
}

#[test]
fn test_needs_tangents() {
    // Can't easily test without gltf::Material
}

#[test]
fn test_texture_sampler_default() {
    let settings = texture::SamplerSettings::default();
    assert_eq!(settings.mag_filter, gltf::texture::MagFilter::Linear);
    assert_eq!(settings.min_filter, gltf::texture::MinFilter::LinearMipmapLinear);
    assert_eq!(settings.wrap_s, gltf::texture::WrappingMode::Repeat);
    assert_eq!(settings.wrap_t, gltf::texture::WrappingMode::Repeat);
}

#[test]
fn test_texture_transform_to_affine2() {
    // Can't easily create gltf::texture::Info without full document
}

#[test]
fn test_is_mag_nearest() {
    // Can't easily test without gltf::texture::Info
}

#[test]
fn test_scene_check_for_cycles() {
    // Would need a gltf::Document with cycles
    // This is more of an integration test
}

#[test]
fn test_node_transform_matrix() {
    // Identity matrix
    let matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    
    // We can't easily create a gltf::Node in isolation
    // This test would need a proper glTF document
}