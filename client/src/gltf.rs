//! GLTF model loading support.
//!
//! Loads GLTF/GLB files and converts them to engine-compatible mesh data.
//! Supports meshes, materials, and textures referenced in the GLTF.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use gltf::{Document, Gltf, image::Format as GltfImageFormat};
use image::DynamicImage;

use crate::{
    assets::{AssetLoader, LoadedAsset},
    mesh::ChunkVertex,
    texture::{TextureManager, sampled_image_index},
};

/// A loaded GLTF model containing meshes and their associated data.
#[derive(Debug, Clone)]
pub struct GltfModel {
    pub meshes: Vec<GltfMesh>,
    pub materials: Vec<GltfMaterial>,
    pub textures: Vec<GltfTexture>,
}

/// A single mesh from a GLTF file.
#[derive(Debug, Clone)]
pub struct GltfMesh {
    pub name: Option<String>,
    pub primitives: Vec<GltfPrimitive>,
}

/// A mesh primitive (draw call) with vertex data and material reference.
#[derive(Debug, Clone)]
pub struct GltfPrimitive {
    pub vertices: Vec<GltfVertex>,
    pub indices: Vec<u32>,
    pub material_index: Option<usize>,
}

/// Vertex data from GLTF, converted to engine format.
#[derive(Debug, Clone, Copy)]
pub struct GltfVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// Material data from GLTF.
#[derive(Debug, Clone)]
pub struct GltfMaterial {
    pub name: Option<String>,
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<usize>,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub normal_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub double_sided: bool,
}

/// Texture data from GLTF.
#[derive(Debug, Clone)]
pub struct GltfTexture {
    pub name: Option<String>,
    pub image_data: Vec<u8>,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
}

/// Alpha mode for materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask,
    Blend,
}

/// Loaded GLTF asset that can be stored in the asset registry.
#[derive(Debug)]
pub struct GltfAsset {
    pub model: GltfModel,
    pub source_path: String,
}

/// Decode a GLTF/GLB file from bytes.
pub fn decode_gltf(bytes: &[u8]) -> Result<LoadedAsset, String> {
    let gltf = Gltf::from_slice(bytes)
        .map_err(|e| format!("Failed to parse GLTF: {}", e))?;
    
    let model = load_gltf_model(&gltf, bytes)
        .map_err(|e| format!("Failed to load GLTF model: {}", e))?;
    
    Ok(Arc::new(GltfAsset {
        model,
        source_path: String::new(), // Will be set by caller if needed
    }))
}

/// Load a GLTF model from a parsed document.
fn load_gltf_model(gltf: &Gltf, bytes: &[u8]) -> Result<GltfModel, String> {
    let document = gltf.document();
    let buffers = gltf.buffers().collect::<Vec<_>>();
    let images = gltf.images().collect::<Vec<_>>();
    
    // Load textures first
    let mut textures = Vec::new();
    for image in &images {
        let texture = load_gltf_texture(image, &buffers)?;
        textures.push(texture);
    }
    
    // Load materials
    let mut materials = Vec::new();
    for material in document.materials() {
        let mat = load_gltf_material(&material, &textures)?;
        materials.push(mat);
    }
    
    // Load meshes
    let mut meshes = Vec::new();
    for mesh in document.meshes() {
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            let prim = load_gltf_primitive(&primitive, &buffers, &materials)?;
            primitives.push(prim);
        }
        meshes.push(GltfMesh {
            name: mesh.name().map(String::from),
            primitives,
        });
    }
    
    Ok(GltfModel {
        meshes,
        materials,
        textures,
    })
}

/// Load a texture from GLTF image data.
fn load_gltf_texture(
    image: &gltf::Image,
    buffers: &[gltf::Buffer],
) -> Result<GltfTexture, String> {
    let (image_data, mime_type) = match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            let buffer = &buffers[view.buffer().index()];
            let start = view.offset() as usize;
            let end = start + view.length() as usize;
            let data = &buffer.0[start..end];
            (data.to_vec(), mime_type.to_string())
        }
        gltf::image::Source::Uri { uri, .. } => {
            // For external URIs, we'd need to load them separately
            // For now, return an error - external textures not yet supported
            return Err(format!("External texture URIs not yet supported: {}", uri));
        }
    };
    
    // Try to decode the image to get dimensions
    let (width, height) = if let Ok(img) = image::load_from_memory(&image_data) {
        img.dimensions()
    } else {
        (0, 0)
    };
    
    Ok(GltfTexture {
        name: image.name().map(String::from),
        image_data,
        mime_type,
        width,
        height,
    })
}

/// Load a material from GLTF.
fn load_gltf_material(
    material: &gltf::Material,
    _textures: &[GltfTexture],
) -> Result<GltfMaterial, String> {
    let pbr = material.pbr_metallic_roughness();
    
    let base_color_factor = pbr.base_color_factor();
    let base_color_texture = pbr.base_color_texture().map(|t| t.texture().index());
    
    let normal_texture = material.normal_texture().map(|t| t.texture().index());
    let occlusion_texture = material.occlusion_texture().map(|t| t.texture().index());
    let emissive_texture = material.emissive_texture().map(|t| t.texture().index());
    
    let alpha_mode = match material.alpha_mode() {
        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
        gltf::material::AlphaMode::Mask => AlphaMode::Mask,
        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
    };
    
    Ok(GltfMaterial {
        name: material.name().map(String::from),
        base_color_factor,
        base_color_texture,
        metallic_factor: pbr.metallic_factor(),
        roughness_factor: pbr.roughness_factor(),
        normal_texture,
        occlusion_texture,
        emissive_factor: material.emissive_factor(),
        emissive_texture,
        alpha_mode,
        alpha_cutoff: material.alpha_cutoff(),
        double_sided: material.double_sided(),
    })
}

/// Load a mesh primitive from GLTF.
fn load_gltf_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::Buffer],
    _materials: &[GltfMaterial],
) -> Result<GltfPrimitive, String> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
    
    // Read positions (required)
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("Primitive missing positions")?
        .collect();
    
    // Read normals (optional, default to up)
    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|n| n.collect())
        .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    
    // Read UVs (optional, default to 0,0)
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|u| u.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    
    // Read colors (optional, default to white)
    let colors: Vec<[f32; 4]> = reader
        .read_colors(0)
        .map(|c| c.into_rgba_f32().collect())
        .unwrap_or_else(|| vec![[1.0, 1.0, 1.0, 1.0]; positions.len()]);
    
    // Read indices (optional)
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|i| i.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    
    // Combine into vertices
    let mut vertices = Vec::with_capacity(positions.len());
    for i in 0..positions.len() {
        vertices.push(GltfVertex {
            position: positions[i],
            normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0, 1.0]),
        });
    }
    
    Ok(GltfPrimitive {
        vertices,
        indices,
        material_index: primitive.material().map(|m| m.index()),
    })
}

/// Convert GLTF vertices to engine ChunkVertex format.
/// This is a simplified conversion - in practice you'd want more sophisticated handling.
pub fn gltf_vertices_to_chunk_vertices(
    vertices: &[GltfVertex],
    texture_id: u32,
) -> Vec<ChunkVertex> {
    vertices
        .iter()
        .map(|v| ChunkVertex {
            position: v.position,
            uv: v.uv,
            sampled_text: texture_id,
            shade: 1.0, // GLTF models handle their own shading
        })
        .collect()
}

/// Register GLTF loaders in the asset registry.
pub fn register_gltf_loaders(registry: &mut crate::assets::AssetHandlerRegistry) {
    registry.register("gltf", decode_gltf);
    registry.register("glb", decode_gltf);
}
