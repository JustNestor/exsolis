//! Mesh loading helpers from glTF.

use gltf::mesh::Mode;

/// Primitive topology for rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

/// Maps the primitive topology from glTF to engine.
pub fn primitive_topology(mode: Mode) -> Result<PrimitiveTopology, String> {
    match mode {
        Mode::Points => Ok(PrimitiveTopology::PointList),
        Mode::Lines => Ok(PrimitiveTopology::LineList),
        Mode::LineStrip => Ok(PrimitiveTopology::LineStrip),
        Mode::Triangles => Ok(PrimitiveTopology::TriangleList),
        Mode::TriangleStrip => Ok(PrimitiveTopology::TriangleStrip),
        mode => Err(format!("Unsupported primitive mode: {:?}", mode)),
    }
}

/// Generates a name for a primitive from its mesh and material.
pub fn primitive_name(mesh: &gltf::Mesh, material: &gltf::Material) -> String {
    let mesh_name = mesh.name().unwrap_or("Mesh");
    if let Some(material_name) = material.name() {
        format!("{mesh_name}.{material_name}")
    } else {
        mesh_name.to_string()
    }
}