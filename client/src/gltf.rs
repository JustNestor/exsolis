//! GLTF model loading support.
//!
//! Loads GLTF/GLB files and converts them to engine-compatible mesh data.
//! Supports meshes, materials, and textures referenced in the GLTF.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_log::{info, warn};
use bevy_transform::prelude::{GlobalTransform, Transform};
use exsolis_common::ServerMessage;
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano_taskgraph::descriptor_set::SampledImageId;

use crate::{
    assets::{AssetCacheChanged, AssetHandlerRegistry, AssetLoader, LoadedAsset},
    mesh::ChunkVertex,
    net::IncomingPacket,
    render::{
        RenderState,
        allocator::{GpuMeshAllocation, MeshBufferAllocator},
        gpu::{GltfDrawCall, PendingWrite, upload_gltf_writes},
    },
    texture::{TextureManager, sampled_image_index},
    vulkan_prelude::VulkanContext,
};

/// Max materials across all uploaded models (matches the storage buffer).
const MAX_GLTF_MATERIALS: usize = 256;
/// Max model instances drawn per frame (matches the instance buffer).
const MAX_GLTF_INSTANCES: usize = 1024;

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

/// GPU-compatible vertex format for GLTF models (with tangents for normal mapping).
#[derive(BufferContents, Vertex, Clone, Copy, Debug)]
#[repr(C)]
pub struct GltfGpuVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
    #[format(R32G32B32A32_SFLOAT)]
    pub tangent: [f32; 4],  // xyz = tangent, w = bitangent sign
}

/// GPU-side mesh representation uploaded to vertex/index buffers.
#[derive(Clone)]
pub struct GltfGpuMesh {
    pub allocation: GpuMeshAllocation,
    pub primitives: Vec<GltfGpuPrimitive>,
    pub bounding_sphere: BoundingSphere,
}

/// A single draw call within a GLTF mesh.
#[derive(Debug, Clone)]
pub struct GltfGpuPrimitive {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub material_index: usize,
    pub bounding_box: Aabb,
}

/// Bounding sphere for frustum culling.
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// GPU-side material data (uploaded to storage buffer).
#[derive(BufferContents, Clone, Copy, Debug)]
#[repr(C)]
pub struct GltfGpuMaterial {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: u32,      // bindless index, 0xFFFFFFFF = none
    pub metallic_roughness: [f32; 2], // x = metallic, y = roughness
    pub normal_texture: u32,
    pub occlusion_texture: u32,
    pub emissive_factor: [f32; 3],
    pub emissive_texture: u32,
    pub alpha_mode: u32,              // 0 = Opaque, 1 = Mask, 2 = Blend
    pub alpha_cutoff: f32,
    pub double_sided: u32,
    pub _padding: [u32; 2],
}

impl Default for GltfGpuMaterial {
    fn default() -> Self {
        Self {
            base_color_factor: [1.0, 1.0, 1.0, 1.0],
            base_color_texture: 0xFFFFFFFF,
            metallic_roughness: [0.0, 1.0],
            normal_texture: 0xFFFFFFFF,
            occlusion_texture: 0xFFFFFFFF,
            emissive_factor: [0.0, 0.0, 0.0],
            emissive_texture: 0xFFFFFFFF,
            alpha_mode: 0,
            alpha_cutoff: 0.5,
            double_sided: 0,
            _padding: [0, 0],
        }
    }
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

/// Texture data from GLTF (decoded to RGBA8 pixels at load time).
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
    if bytes.len() > exsolis_common::GLB_MAX_BYTES {
        return Err(format!(
            "GLB too large ({} > {} bytes)",
            bytes.len(),
            exsolis_common::GLB_MAX_BYTES
        ));
    }
    let model = load_gltf_model(bytes)
        .map_err(|e| format!("Failed to load GLTF model: {}", e))?;

    Ok(Arc::new(GltfAsset {
        model,
        source_path: String::new(), // Will be set by caller if needed
    }))
}

/// Load a GLTF model from raw bytes using `import_slice`, which resolves
/// embedded (GLB BIN, base64 data URIs) buffers and decodes images.
fn load_gltf_model(bytes: &[u8]) -> Result<GltfModel, String> {
    let (document, buffers, images) = gltf::import_slice(bytes)
        .map_err(|e| format!("Failed to parse GLTF: {}", e))?;

    // Load textures from already-decoded image data
    let mut textures = Vec::with_capacity(images.len());
    for (idx, image) in images.iter().enumerate() {
        textures.push(load_gltf_texture(idx, &document, image)?);
    }

    // Load materials
    let mut materials = Vec::new();
    for material in document.materials() {
        let mat = load_gltf_material(&material)?;
        materials.push(mat);
    }

    // Load meshes
    let mut meshes = Vec::new();
    for mesh in document.meshes() {
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            let prim = load_gltf_primitive(&primitive, &buffers)?;
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

/// Convert decoded `image::Data` to RGBA8 pixels.
fn load_gltf_texture(
    idx: usize,
    document: &gltf::Document,
    image: &gltf::image::Data,
) -> Result<GltfTexture, String> {
    let pixel_count = (image.width * image.height) as usize;
    let rgba: Vec<u8> = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for px in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        gltf::image::Format::R8G8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for px in image.pixels.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[1], 0, 255]);
            }
            out
        }
        gltf::image::Format::R8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for &v in &image.pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        gltf::image::Format::B8G8R8A8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for px in image.pixels.chunks_exact(4) {
                out.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
            }
            out
        }
        gltf::image::Format::B8G8R8 => {
            let mut out = Vec::with_capacity(pixel_count * 4);
            for px in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[2], px[1], px[0], 255]);
            }
            out
        }
        _ => {
            return Err(format!(
                "Unsupported GLTF image format {:?} for image {}",
                image.format, idx
            ));
        }
    };

    let name = document
        .images()
        .nth(idx)
        .and_then(|img| img.name().map(String::from));

    Ok(GltfTexture {
        name,
        image_data: rgba,
        mime_type: format!("{:?}", image.format),
        width: image.width,
        height: image.height,
    })
}

/// Load a material from GLTF.
fn load_gltf_material(
    material: &gltf::Material,
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
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        double_sided: material.double_sided(),
    })
}

/// Load a mesh primitive from GLTF.
fn load_gltf_primitive(
    primitive: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
) -> Result<GltfPrimitive, String> {
    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    
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
        material_index: primitive.material().index(),
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

/// Compute tangents for normal mapping (Mikktspace algorithm).
fn compute_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let vertex_count = positions.len();
    let mut tan1 = vec![[0.0f32; 3]; vertex_count];
    let tan2 = vec![[0.0f32; 3]; vertex_count];

    for chunk in indices.chunks(3) {
        let i0 = chunk[0] as usize;
        let i1 = chunk[1] as usize;
        let i2 = chunk[2] as usize;

        let v0 = glam::Vec3::from(positions[i0]);
        let v1 = glam::Vec3::from(positions[i1]);
        let v2 = glam::Vec3::from(positions[i2]);

        let uv0 = glam::Vec2::from(uvs[i0]);
        let uv1 = glam::Vec2::from(uvs[i1]);
        let uv2 = glam::Vec2::from(uvs[i2]);

        let delta_pos1 = v1 - v0;
        let delta_pos2 = v2 - v0;
        let delta_uv1 = uv1 - uv0;
        let delta_uv2 = uv2 - uv0;

        let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
        let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;

        tan1[i0][0] += tangent.x; tan1[i0][1] += tangent.y; tan1[i0][2] += tangent.z;
        tan1[i1][0] += tangent.x; tan1[i1][1] += tangent.y; tan1[i1][2] += tangent.z;
        tan1[i2][0] += tangent.x; tan1[i2][1] += tangent.y; tan1[i2][2] += tangent.z;
    }

    let mut tangents = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let n = glam::Vec3::from(normals[i]);
        let mut t = glam::Vec3::from(tan1[i]);

        // Gram-Schmidt orthogonalize
        t = (t - n * n.dot(t)).normalize_or_zero();

        // Calculate handedness
        let tan2_vec = glam::Vec3::from(tan2[i]);
        let handedness = if n.cross(t).dot(tan2_vec) < 0.0 { -1.0 } else { 1.0 };

        tangents.push([t.x, t.y, t.z, handedness]);
    }

    tangents
}

/// Compute bounding sphere for a set of positions.
fn compute_bounding_sphere(positions: &[[f32; 3]]) -> BoundingSphere {
    if positions.is_empty() {
        return BoundingSphere { center: [0.0, 0.0, 0.0], radius: 0.0 };
    }

    // Ritter's algorithm for approximate bounding sphere
    let mut min = glam::Vec3::from(positions[0]);
    let mut max = min;

    for pos in positions {
        let p = glam::Vec3::from(*pos);
        min = min.min(p);
        max = max.max(p);
    }

    let center = (min + max) * 0.5;
    let mut radius_sq = 0.0;

    for pos in positions {
        let p = glam::Vec3::from(*pos);
        let d = (p - center).length_squared();
        if d > radius_sq {
            radius_sq = d;
        }
    }

    BoundingSphere {
        center: center.into(),
        radius: radius_sq.sqrt(),
    }
}

/// Compute AABB for a set of positions.
fn compute_aabb(positions: &[[f32; 3]]) -> Aabb {
    if positions.is_empty() {
        return Aabb { min: [0.0, 0.0, 0.0], max: [0.0, 0.0, 0.0] };
    }

    let mut min = glam::Vec3::from(positions[0]);
    let mut max = min;

    for pos in positions {
        let p = glam::Vec3::from(*pos);
        min = min.min(p);
        max = max.max(p);
    }

    Aabb { min: min.into(), max: max.into() }
}

/// Convert CPU GltfVertex to GPU GltfGpuVertex with computed tangents.
pub fn gltf_vertices_to_gpu_vertices(
    vertices: &[GltfVertex],
    indices: &[u32],
) -> Vec<GltfGpuVertex> {
    let positions: Vec<_> = vertices.iter().map(|v| v.position).collect();
    let normals: Vec<_> = vertices.iter().map(|v| v.normal).collect();
    let uvs: Vec<_> = vertices.iter().map(|v| v.uv).collect();
    let colors: Vec<_> = vertices.iter().map(|v| v.color).collect();

    let tangents = compute_tangents(&positions, &normals, &uvs, indices);

    vertices
        .iter()
        .enumerate()
        .map(|(i, v)| GltfGpuVertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv,
            color: v.color,
            tangent: tangents[i],
        })
        .collect()
}

/// Convert CPU GltfMaterial to GPU GltfGpuMaterial.
pub fn gltf_material_to_gpu_material(
    mat: &GltfMaterial,
    texture_indices: &[Option<SampledImageId>],
) -> GltfGpuMaterial {
    let base_color_texture = mat
        .base_color_texture
        .and_then(|idx| texture_indices.get(idx).and_then(|t| *t))
        .map(sampled_image_index)
        .unwrap_or(0xFFFFFFFF);

    let normal_texture = mat
        .normal_texture
        .and_then(|idx| texture_indices.get(idx).and_then(|t| *t))
        .map(sampled_image_index)
        .unwrap_or(0xFFFFFFFF);

    let occlusion_texture = mat
        .occlusion_texture
        .and_then(|idx| texture_indices.get(idx).and_then(|t| *t))
        .map(sampled_image_index)
        .unwrap_or(0xFFFFFFFF);

    let emissive_texture = mat
        .emissive_texture
        .and_then(|idx| texture_indices.get(idx).and_then(|t| *t))
        .map(sampled_image_index)
        .unwrap_or(0xFFFFFFFF);

    let alpha_mode = match mat.alpha_mode {
        AlphaMode::Opaque => 0,
        AlphaMode::Mask => 1,
        AlphaMode::Blend => 2,
    };

    GltfGpuMaterial {
        base_color_factor: mat.base_color_factor,
        base_color_texture,
        metallic_roughness: [mat.metallic_factor, mat.roughness_factor],
        normal_texture,
        occlusion_texture,
        emissive_factor: mat.emissive_factor,
        emissive_texture,
        alpha_mode,
        alpha_cutoff: mat.alpha_cutoff,
        double_sided: if mat.double_sided { 1 } else { 0 },
        _padding: [0, 0],
    }
}

/// Data prepared for GPU upload (vertex/index buffers + materials).
pub struct GltfUploadData {
    pub mesh: GltfGpuMesh,
    pub vertices: Vec<GltfGpuVertex>,
    pub indices: Vec<u32>,
    pub materials: Vec<GltfGpuMaterial>,
}

/// Upload a GLTF model to GPU buffers (prepares data for upload).
/// `texture_indices` must parallel `model.textures` (see
/// `TextureManager::load_gltf_textures`); texture uploading itself is owned
/// by the caller so it can be redone independently of mesh uploads.
pub fn upload_gltf_model(
    allocator: &mut MeshBufferAllocator,
    model: &GltfModel,
    texture_indices: &[Option<SampledImageId>],
) -> Result<GltfUploadData, String> {

    // Convert materials to GPU format
    let mut gpu_materials = Vec::with_capacity(model.materials.len());
    for mat in &model.materials {
        gpu_materials.push(gltf_material_to_gpu_material(mat, &texture_indices));
    }

    // Collect all vertex/index data
    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();
    let mut gpu_primitives = Vec::new();

    for (mesh_idx, mesh) in model.meshes.iter().enumerate() {
        for prim in &mesh.primitives {
            let vertex_start = all_vertices.len() as u32;
            let index_start = all_indices.len() as u32;

            // Convert vertices with tangents
            let gpu_vertices = gltf_vertices_to_gpu_vertices(&prim.vertices, &prim.indices);
            all_vertices.extend(gpu_vertices);
            all_indices.extend(prim.indices.iter().map(|&i| i + vertex_start));

            // Compute bounds
            let positions: Vec<_> = prim.vertices.iter().map(|v| v.position).collect();
            let bounding_box = compute_aabb(&positions);

            gpu_primitives.push(GltfGpuPrimitive {
                index_count: prim.indices.len() as u32,
                first_index: index_start,
                vertex_offset: vertex_start as i32,
                material_index: prim.material_index.unwrap_or(0),
                bounding_box,
            });
        }
    }

    if all_vertices.is_empty() {
        return Err("GLTF model has no geometry".into());
    }

    // Allocate GPU buffers
    let allocation = allocator
        .allocate(all_vertices.len() as u32, all_indices.len() as u32)
        .ok_or("Failed to allocate GPU mesh buffers")?;

    // Compute overall bounding sphere
    let all_positions: Vec<_> = all_vertices.iter().map(|v| v.position).collect();
    let bounding_sphere = compute_bounding_sphere(&all_positions);

    Ok(GltfUploadData {
        mesh: GltfGpuMesh {
            allocation,
            primitives: gpu_primitives,
            bounding_sphere,
        },
        vertices: all_vertices,
        indices: all_indices,
        materials: gpu_materials,
    })
}

/// ECS component: place a loaded GLTF model in the world.
#[derive(Component, Debug, Clone)]
pub struct GltfModelInstance {
    pub model_path: String,
}

impl GltfModelInstance {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

/// One uploaded model: GPU allocation + per-primitive draw info + materials.
pub struct UploadedGltf {
    pub mesh: GltfGpuMesh,
    pub materials: Vec<GltfGpuMaterial>,
    /// Base slot of this model's materials in the global material buffer.
    pub material_base: u32,
}

/// Server-assigned id for a spawned model (see `SpawnModel`/`DespawnModel`).
/// TEMPORARY test API.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelEntityId(pub u64);

/// All GPU-uploaded GLTF models, keyed by asset path.
#[derive(Resource, Default)]
pub struct GltfModels {
    pub uploaded: std::collections::HashMap<String, UploadedGltf>,
    pub pending_paths: std::collections::HashSet<String>,
    /// `TextureManager::generation` textures were last uploaded into.
    pub texture_generation: u64,
    /// Instance paths warned about once (model not uploaded yet).
    pub warned_missing: std::collections::HashSet<String>,
}

/// Upload newly arrived (or hot-reloaded) GLTF assets to the dedicated GPU
/// buffers. Models whose materials would overflow the global material buffer
/// are skipped with a warning and left out of `uploaded` (their instances
/// are skipped by sync until they fit).
pub fn upload_gltf_assets(
    vk: Option<Res<VulkanContext>>,
    render_state: Option<ResMut<RenderState>>,
    textures: Option<ResMut<TextureManager>>,
    assets: Option<Res<AssetHandlerRegistry>>,
    mut models: ResMut<GltfModels>,
    mut changed: MessageReader<AssetCacheChanged>,
) {
    let (Some(vk), Some(mut render_state), Some(mut textures), Some(assets)) =
        (vk, render_state, textures, assets)
    else {
        return;
    };

    let changed_glb: std::collections::HashSet<&str> = changed
        .read()
        .flat_map(|event| event.assets.iter())
        .filter(|asset| asset.path.ends_with(".glb"))
        .map(|asset| asset.path.as_str())
        .collect();

    // Newly resident + hot-reloaded models need mesh uploads.
    let mut to_upload: Vec<String> = assets
        .iter_loaded::<GltfAsset>()
        .map(|(path, _)| path.to_owned())
        .filter(|path| !models.uploaded.contains_key(path) || changed_glb.contains(path.as_str()))
        .collect();

    // A manager rebuild wiped all bindless ids. Resident GltfAssets retain
    // everything, so redo those models fully (textures + meshes + materials).
    let manager_regenerated = models.texture_generation != textures.generation;
    if manager_regenerated {
        for path in models.uploaded.keys() {
            if !to_upload.contains(path) {
                to_upload.push(path.clone());
            }
        }
        if !to_upload.is_empty() {
            warn!(
                "gltf textures wiped by texture rebuild (gen {} -> {}); re-uploading {} model(s), old bindless ids retained",
                models.texture_generation,
                textures.generation,
                to_upload.len()
            );
        }
    }
    to_upload.sort();

    if to_upload.is_empty() {
        return;
    }

    // Free previous GPU allocations for re-uploaded models (swap-under-
    // instances: entities keyed by path keep working, see B1).
    for path in &to_upload {
        if let Some(old) = models.uploaded.remove(path) {
            render_state.gltf_allocator.free(old.mesh.allocation);
        }
    }
    if !changed_glb.is_empty() {
        // Textures are re-uploaded below; the old bindless ids are retained
        // (no GPU unload policy — same as PNGs). Long dev sessions with many
        // model hot-reloads will accumulate VRAM; tracked as tech debt.
        warn!(
            "gltf hot-reload: re-uploading {} model(s); old bindless textures retained",
            changed_glb.len()
        );
    }

    let mut vertex_writes = Vec::new();
    let mut index_writes = Vec::new();
    for path in &to_upload {
        let Some(asset) = assets.get::<GltfAsset>(path) else {
            continue;
        };
        let indices = textures.load_gltf_textures(&vk, &asset.model.textures);
        let ok = indices.iter().filter(|i| i.is_some()).count();
        if ok < indices.len() {
            warn!("gltf textures: {ok}/{} uploaded for {path}", indices.len());
        }
        match upload_gltf_model(&mut render_state.gltf_allocator, &asset.model, &indices) {
            Ok(data) => {
                info!(
                    "gltf uploaded {path}: {} verts, {} prims, {} materials, textures {:?}",
                    data.vertices.len(),
                    data.mesh.primitives.len(),
                    data.materials.len(),
                    indices
                        .iter()
                        .map(|id| id.map(sampled_image_index))
                        .collect::<Vec<_>>(),
                );
                // NOTE: GLTF stride (64 B), not ChunkVertex — hence slots.
                let offset_v = data.mesh.allocation.vertex_slot() as u64
                    * std::mem::size_of::<GltfGpuVertex>() as u64;
                let offset_i = data.mesh.allocation.index_slot() as u64
                    * std::mem::size_of::<u32>() as u64;
                vertex_writes.push(PendingWrite {
                    offset_bytes: offset_v,
                    data: data.vertices,
                });
                index_writes.push(PendingWrite {
                    offset_bytes: offset_i,
                    data: data.indices,
                });
                models.uploaded.insert(
                    path.clone(),
                    UploadedGltf {
                        mesh: data.mesh,
                        materials: data.materials,
                        material_base: 0, // assigned below
                    },
                );
            }
            Err(e) => warn!("gltf upload failed for {path}: {e}"),
        }
    }
    if vertex_writes.is_empty() {
        return;
    }

    // Pack materials for all uploaded models from slot 0; skip models that
    // would overflow the buffer (their draws are skipped by sync).
    let mut paths: Vec<String> = models.uploaded.keys().cloned().collect();
    paths.sort();
    let mut material_data = Vec::new();
    let mut base = 0u32;
    let mut skipped = Vec::new();
    for path in &paths {
        let count = models.uploaded[path].materials.len() as u32;
        if base + count > MAX_GLTF_MATERIALS as u32 {
            skipped.push(path.clone());
            continue;
        }
        models.uploaded.get_mut(path).unwrap().material_base = base;
        material_data.extend_from_slice(&models.uploaded[path].materials);
        base += count;
    }
    for path in skipped {
        if let Some(old) = models.uploaded.remove(&path) {
            render_state.gltf_allocator.free(old.mesh.allocation);
        }
        warn!("gltf materials overflow ({MAX_GLTF_MATERIALS} slots); skipped {path}");
    }

    let material_writes = [PendingWrite {
        offset_bytes: 0,
        data: material_data,
    }];
    let (material_buf, instance_buf) = {
        let shared = render_state.shared.read().unwrap();
        (
            shared.gltf_material_buffer_id.unwrap(),
            shared.gltf_instance_buffer_id.unwrap(),
        )
    };
    upload_gltf_writes(
        &vk,
        render_state.gltf_vertex_buffer_id,
        render_state.gltf_index_buffer_id,
        material_buf,
        instance_buf,
        &vertex_writes,
        &index_writes,
        &material_writes,
        &[],
    );
    models.texture_generation = textures.generation;
}

/// Collect `GltfModelInstance` transforms into the instance buffer and stage
/// per-primitive draws for the frame. Instances whose model is not uploaded
/// (still downloading, failed, or skipped) are skipped with a one-time warn.
pub fn sync_gltf_instances(
    vk: Option<Res<VulkanContext>>,
    render_state: Option<ResMut<RenderState>>,
    mut models: ResMut<GltfModels>,
    instances: Query<(&GltfModelInstance, &GlobalTransform)>,
) {
    let (Some(vk), Some(mut render_state)) = (vk, render_state) else {
        return;
    };

    for (instance, _) in instances.iter() {
        if !models.uploaded.contains_key(&instance.model_path)
            && models.warned_missing.insert(instance.model_path.clone())
        {
            warn!(
                "gltf instance references model with no GPU upload: {}",
                instance.model_path
            );
        }
    }

    // Deterministic model order so draws and instance ranges line up.
    let mut paths: Vec<&String> = models.uploaded.keys().collect();
    paths.sort();

    let mut matrices: Vec<glam::Mat4> = Vec::new();
    let mut draws: Vec<GltfDrawCall> = Vec::new();
    for path in paths {
        let uploaded = &models.uploaded[path];
        let first_instance = matrices.len() as u32;
        for (instance, transform) in instances.iter() {
            if instance.model_path != *path {
                continue;
            }
            if matrices.len() >= MAX_GLTF_INSTANCES {
                warn!("gltf instance overflow ({MAX_GLTF_INSTANCES}); dropping extras");
                break;
            }
            matrices.push(transform.to_matrix());
        }
        let instance_count = matrices.len() as u32 - first_instance;
        if instance_count == 0 {
            continue;
        }
        for prim in &uploaded.mesh.primitives {
            draws.push(GltfDrawCall {
                index_count: prim.index_count,
                first_index: uploaded.mesh.allocation.first_index() + prim.first_index,
                vertex_offset: uploaded.mesh.allocation.vertex_offset()
                    + prim.vertex_offset,
                material_index: uploaded.material_base + prim.material_index as u32,
                first_instance,
                instance_count,
            });
        }
    }

    let (instance_buf, material_buf) = {
        let mut shared = render_state.shared.write().unwrap();
        shared.gltf_instance_count = matrices.len() as u32;
        shared.gltf_draws = draws;
        (
            shared.gltf_instance_buffer_id.unwrap(),
            shared.gltf_material_buffer_id.unwrap(),
        )
    };
    if matrices.is_empty() {
        return;
    }
    upload_gltf_writes(
        &vk,
        render_state.gltf_vertex_buffer_id,
        render_state.gltf_index_buffer_id,
        material_buf,
        instance_buf,
        &[],
        &[],
        &[],
        &[PendingWrite {
            offset_bytes: 0,
            data: matrices,
        }],
    );
}

/// TEMPORARY test API: spawn/despawn model entities from server messages.
pub fn handle_spawn_model_messages(
    mut incoming: MessageReader<IncomingPacket>,
    mut commands: Commands,
    existing: Query<(Entity, &ModelEntityId)>,
) {
    for packet in incoming.read() {
        match &packet.0 {
            ServerMessage::SpawnModel {
                entity_id,
                path,
                x,
                y,
                z,
                scale,
            } => {
                if !path.ends_with(".glb") || path.len() > exsolis_common::MODEL_PATH_MAX_LEN
                {
                    warn!("spawn_model: rejected path {path:?}");
                    continue;
                }
                // Replace semantics: one live entity per server id.
                for (entity, id) in existing.iter() {
                    if id.0 == *entity_id {
                        commands.entity(entity).despawn();
                    }
                }
                commands.spawn((
                    ModelEntityId(*entity_id),
                    GltfModelInstance::new(path.clone()),
                    Transform::from_xyz(*x, *y, *z).with_scale(glam::Vec3::splat(*scale)),
                    GlobalTransform::default(),
                ));
            }
            ServerMessage::DespawnModel { entity_id } => {
                for (entity, id) in existing.iter() {
                    if id.0 == *entity_id {
                        commands.entity(entity).despawn();
                    }
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_duck_glb() {
        decode_test_model("duck.glb");
    }

    fn decode_test_model(model: &str) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../assets/models")
            .join(model);
        let bytes = std::fs::read(&path).expect("duck.glb test model missing");
        let loaded = decode_gltf(&bytes).expect("decode_gltf failed");
        let asset = loaded
            .as_ref()
            .downcast_ref::<GltfAsset>()
            .expect("wrong asset type");
        assert!(
            !asset.model.meshes.is_empty(),
            "duck has no meshes"
        );
        let vert_count: usize = asset.model.meshes.iter()
            .flat_map(|mesh| mesh.primitives.iter())
            .map(|prim| prim.vertices.len())
            .sum();
        assert!(vert_count > 0, "duck has no vertices");
        // Tangent generation + material conversion (GPU prep path).
        let prim = &asset.model.meshes[0].primitives[0];
        let gpu = gltf_vertices_to_gpu_vertices(&prim.vertices, &prim.indices);
        assert_eq!(gpu.len(), prim.vertices.len());
        assert!(gpu.iter().all(|v| v.tangent.iter().all(|c| c.is_finite())));
    }
}
