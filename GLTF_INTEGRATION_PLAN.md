# GLTF Model Integration Plan

## Status: Phase 1 (CPU→GPU foundation) — implemented, compiles

`cargo check -p exsolis_client` and `cargo check --workspace` pass.
Test target builds (`cargo check --tests`). Shaders are inline in
`render/gpu.rs` (matching the `mesh.rs` pattern) rather than separate
`.glsl` files.

## Current State Analysis

**What exists:**
- ✅ GLTF/GLB parsing via `gltf` crate (v0.16) in `client/src/gltf.rs`
- ✅ Asset loaders registered for `.gltf` and `.glb` extensions
- ✅ CPU-side data structures: `GltfModel`, `GltfMesh`, `GltfPrimitive`, `GltfMaterial`, `GltfTexture`
- ✅ Asset delivery pipeline (server → client via fastpak/HTTP)
- ✅ Bindless texture system for PNG textures
- ✅ Vulkano task-graph renderer with chunk, avatar, effects, and UI passes
- ✅ Mesh buffer allocator (`MeshBufferAllocator`) for GPU vertex/index buffers

**What's missing:**
- ❌ GPU upload of GLTF mesh data (vertex/index buffers)
- ❌ GPU upload of GLTF textures (embedded or external)
- ❌ PBR material pipeline (metallic/roughness, normal maps, emissive)
- ❌ Render pass for GLTF models
- ❌ Instancing system for placing GLTF models in the world
- ❌ Animation/skeletal support
- ❌ External texture reference resolution (URI-based textures)
- ❌ LOD support
- ❌ Morph targets

---

## Integration Architecture

### 1. Data Flow

```
Server assets/ (GLTF/GLB files)
    │
    ▼
Client asset worker (download + extract)
    │
    ▼
AssetHandlerRegistry → decode_gltf() → GltfAsset (CPU)
    │
    ├──▶ GltfTexture → TextureManager → GPU bindless textures
    │
    └──▶ GltfMesh → GltfGpuMesh → GPU vertex/index buffers (MeshBufferAllocator)
         │
         ▼
    GltfModelInstance (ECS component) → Render pass → Draw
```

### 2. New Components

```rust
// ECS component for placing a GLTF model in the world
#[derive(Component)]
pub struct GltfModelInstance {
    pub model_path: String,        // e.g., "models/tree.glb"
    pub material_overrides: Vec<MaterialOverride>, // optional per-instance material changes
    pub lod_bias: f32,             // for LOD selection
}

// GPU-side mesh representation (uploaded to vertex/index buffers)
pub struct GltfGpuMesh {
    pub allocation: GpuMeshAllocation,
    pub primitives: Vec<GltfGpuPrimitive>,
    pub bounding_sphere: BoundingSphere, // for frustum culling
}

pub struct GltfGpuPrimitive {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub material_index: usize,
    pub bounding_box: Aabb,
}

// Material data uploaded to GPU (uniform/storage buffer)
pub struct GltfGpuMaterial {
    pub base_color_factor: [f32; 4],
    pub base_color_texture: Option<SampledImageId>, // bindless
    pub metallic_roughness: [f32; 2],               // metallic, roughness
    pub normal_texture: Option<SampledImageId>,
    pub occlusion_texture: Option<SampledImageId>,
    pub emissive_factor: [f32; 3],
    pub emissive_texture: Option<SampledImageId>,
    pub alpha_mode: u32,        // 0=Opaque, 1=Mask, 2=Blend
    pub alpha_cutoff: f32,
    pub double_sided: u32,
}
```

---

## Implementation Phases

### Phase 1: Core GPU Upload (Week 1-2)

**Goal:** Upload GLTF meshes and textures to GPU, render a single model.

#### 1.1 Extend `TextureManager` for GLTF Textures
- Add method to load textures from `GltfTexture` (already in memory)
- Support different formats (JPEG, PNG, KTX2, Basis Universal)
- Handle sRGB vs linear color space correctly
- Generate mipmaps for all textures

```rust
// In texture.rs
impl TextureManager {
    pub fn load_gltf_textures(
        &mut self,
        vk: &VulkanContext,
        gltf_textures: &[GltfTexture],
    ) -> Result<Vec<Option<SampledImageId>>, String> { ... }
}
```

#### 1.2 Create `GltfGpuMesh` Upload System
- New system: `upload_gltf_meshes` in `render/meshing.rs` or new `gltf.rs` module
- Uses `MeshBufferAllocator` to sub-allocate vertex/index buffers
- Converts `GltfVertex` → engine vertex format (position, normal, uv, color, tangent?)
- Creates `GltfGpuMesh` resource stored in `RenderStateShared`

```rust
// New vertex format for GLTF models (extends ChunkVertex)
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub struct GltfVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
    #[format(R32G32B32A32_SFLOAT)]  // For normal mapping
    pub tangent: [f32; 4],
}
```

#### 1.3 Basic GLTF Render Pass
- New shader: `gltf.vert` / `gltf.frag` (PBR metallic/roughness)
- New pipeline in `RenderStateShared`
- Draw call in `ChunkRenderTask::execute()` or new render task
- Push constants: view_proj, model matrix, material index, samplers

---

### Phase 2: Material System & Instancing (Week 2-3)

**Goal:** Full PBR materials, multiple instances, frustum culling.

#### 2.1 Material Buffer
- Upload `GltfGpuMaterial` array to a storage buffer (bindless)
- One buffer per model, or global material buffer
- Support material overrides per instance

#### 2.2 Instancing
- Instance buffer with model matrices (like avatars but more flexible)
- Per-instance data: model matrix, material override index, custom data
- Frustum culling on CPU (using `bounding_sphere`)

#### 2.3 Alpha Mode Handling
- **Opaque**: Standard depth prepass + forward
- **Mask** (alpha cutoff): Same as opaque but with `discard` in fragment shader
- **Blend**: Separate transparent pass (after opaque, sorted back-to-front)

---

### Phase 3: Advanced Features (Week 3-4+)

#### 3.1 Animation/Skeletal
- Load skin data from GLTF (joints, weights, inverse bind matrices)
- Vertex shader skinning (compute or vertex shader)
- Animation sampling system

#### 3.2 LOD Support
- Load multiple meshes per GLTF (using `extensions` or naming convention)
- Distance-based LOD selection

#### 3.3 Morph Targets
- Blend shapes for facial animation, etc.
- GPU buffer for morph weights

#### 3.4 KTX2/Basis Universal Textures
- Transcode to GPU-compressed formats (BC7, ASTC, ETC2)
- Faster loading, lower VRAM

---

## Detailed Task Breakdown

### Phase 1 Tasks

| Task | File | Description |
|------|------|-------------|
| 1.1 | `texture.rs` | Add `load_gltf_textures()` method |
| 1.2 | `gltf.rs` | Add `GltfGpuMesh`, `GltfGpuPrimitive`, `GltfGpuMaterial` |
| 1.3 | `gltf.rs` | Add `upload_gltf_model()` function |
| 1.4 | `render/gpu.rs` | Add `GltfVertex` format, PBR pipeline |
| 1.5 | `render/gpu.rs` | Add GLTF pipeline to `RenderStateShared` |
| 1.6 | `render/gpu.rs` | Add GLTF draw call in `ChunkRenderTask::execute()` |
| 1.7 | `render/mod.rs` | Register new systems |
| 1.8 | `assets.rs` | Trigger GLTF upload on `AssetCacheChanged` |

### Phase 2 Tasks

| Task | File | Description |
|------|------|-------------|
| 2.1 | `gltf.rs` | Material buffer upload system |
| 2.2 | `gltf.rs` | Instance buffer management |
| 2.3 | `gltf.rs` | `GltfModelInstance` component + spawn system |
| 2.4 | `render/gpu.rs` | Instanced draw call |
| 2.5 | `render/gpu.rs` | Alpha mode handling (separate passes) |
| 2.6 | `render/gpu.rs` | Frustum culling (CPU-side) |

### Phase 3 Tasks

| Task | File | Description |
|------|------|-------------|
| 3.1 | `gltf.rs` | Skin/joint loading |
| 3.2 | `render/gpu.rs` | Skinning vertex shader |
| 3.3 | `gltf.rs` | Animation system |
| 3.4 | `gltf.rs` | LOD loading/selection |
| 3.5 | `texture.rs` | KTX2/Basis transcoder integration |

---

## Shader Design (PBR Metallic/Roughness)

### Vertex Shader (`gltf.vert`)
```glsl
#version 450
#include <vulkano.glsl>

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    SamplerId sampler_id;
    SamplerId linear_sampler_id;
} push_constants;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 color;
layout(location = 4) in vec4 tangent;

layout(location = 0) out vec3 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec4 v_color;
layout(location = 4) out vec4 v_tangent;

// Instance data
layout(set = 0, binding = 0, std430) readonly buffer InstanceData {
    mat4 model_matrices[];
} instances;

layout(set = 0, binding = 1, std430) readonly buffer MaterialData {
    GltfMaterial materials[];
} materials;

struct GltfMaterial {
    vec4 base_color_factor;
    uint base_color_texture;   // bindless index or 0xFFFFFFFF
    vec2 metallic_roughness;
    uint normal_texture;
    uint occlusion_texture;
    vec3 emissive_factor;
    uint emissive_texture;
    uint alpha_mode;           // 0=Opaque, 1=Mask, 2=Blend
    float alpha_cutoff;
    uint double_sided;
    // padding
};

void main() {
    uint instance_id = gl_InstanceIndex;
    mat4 model = instances.model_matrices[instance_id];
    
    vec4 world_pos = model * vec4(position, 1.0);
    v_position = world_pos.xyz;
    v_normal = mat3(transpose(inverse(model))) * normal;
    v_uv = uv;
    v_color = color;
    v_tangent = tangent;
    
    gl_Position = push_constants.view_proj * world_pos;
}
```

### Fragment Shader (`gltf.frag`)
```glsl
#version 450
#include <vulkano.glsl>

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
    SamplerId sampler_id;
    SamplerId linear_sampler_id;
} push_constants;

layout(location = 0) in vec3 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec4 v_color;
layout(location = 4) in vec4 v_tangent;

layout(set = 0, binding = 1, std430) readonly buffer MaterialData {
    GltfMaterial materials[];
} materials;

layout(location = 0) out vec4 f_color;

// Standard PBR functions: BRDF, GGX, Schlick, etc.
// ...

void main() {
    uint material_index = /* from push constant or instance data */;
    GltfMaterial mat = materials[material_index];
    
    // Sample base color texture
    vec4 base_color = mat.base_color_factor;
    if (mat.base_color_texture != 0xFFFFFFFF) {
        base_color *= texture(sampler2D(mat.base_color_texture, push_constants.sampler_id), v_uv);
    }
    base_color *= v_color;
    
    // Alpha test for MASK mode
    if (mat.alpha_mode == 1u && base_color.a < mat.alpha_cutoff) {
        discard;
    }
    
    // PBR lighting calculation...
    // Normal mapping, metallic/roughness, IBL, etc.
    
    f_color = vec4(final_color.rgb, base_color.a);
}
```

---

## Integration Points

### 1. Asset System (`assets.rs`)
```rust
// In poll_workers, after loading assets:
fn load_gltf_assets(
    assets: &[AssetHash],
    paths: &AssetCachePaths,
    handlers: &mut AssetHandlerRegistry,
    vk: &VulkanContext,  // Need access to VK context
) -> Vec<String> {
    // Load GLTF assets and upload to GPU
}
```

### 2. Render System (`render/mod.rs`)
```rust
impl Plugin for RendererPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(TexturePlugin)
            // ...
            .add_systems(Update, (
                // ...
                gltf::upload_gltf_meshes.run_if(resource_exists::<RenderState>),
                gltf::prepare_gltf_materials.run_if(resource_exists::<RenderState>),
                gltf::cull_and_queue_instances.run_if(resource_exists::<RenderState>),
                // ...
            ).chain().in_set(crate::net::GameplaySet));
    }
}
```

### 3. ECS Spawning (User-facing)
```rust
// In a mod or game system:
commands.spawn((
    GltfModelInstance {
        model_path: "models/character.glb".to_string(),
        material_overrides: vec![],
        lod_bias: 1.0,
    },
    Transform::from_xyz(0.0, 1.0, 0.0),
    GlobalTransform::default(),
    Visibility::default(),
));
```

---

## Performance Considerations

1. **Batch uploads**: Upload all new GLTF meshes/textures in single command buffer
2. **Bindless everything**: Textures, materials, instance data all via bindless
3. **Frustum culling**: CPU-side culling using bounding spheres before draw
4. **LOD**: Reduce triangle count at distance
5. **Occlusion culling**: Future - GPU-driven occlusion queries
6. **Meshlets**: Future - mesh shader pipeline for complex models

---

## Dependencies to Add

```toml
# client/Cargo.toml
gltf = { version = "0.16", features = ["image"] }
image = { version = "0.25", features = ["png", "jpeg", "ktx2"] }
# For KTX2/Basis transcoding:
# basis_universal = "0.13"
# ktx2 = "0.10"
```

---

## Testing Strategy

1. **Unit tests**: `gltf.rs` functions (parsing, vertex conversion)
2. **Integration test**: Load a test GLTF, verify GPU buffers created
3. **Visual test**: Render sample models (cube, sphere, textured quad, PBR sphere)
4. **Stress test**: 1000+ instanced models, measure frame time
5. **Memory test**: Verify no leaks on hot-reload/unload

---

## Rollout Plan

1. **Feature flag**: `gltf_models` feature in client Cargo.toml
2. **Dev branch**: Implement Phase 1-2 behind flag
3. **Dogfood**: Test with artist-provided models
4. **Optimize**: Profile, fix bottlenecks
5. **Merge**: Enable by default

---

## Open Questions

1. **Coordinate system**: GLTF is right-handed Y-up; engine is right-handed Y-up? Verify.
2. **Vertex format**: Do we need tangents for normal mapping? (Yes - add to `GltfVertex`)
3. **Index format**: GLTF supports u16/u32; we use u32. Handle conversion.
4. **External textures**: How to resolve URI references? (Download via asset system)
5. **GLTF extensions**: `KHR_materials_unlit`, `KHR_materials_specular`, `KHR_texture_basisu`?
6. **Scene graph**: GLTF has nodes/scenes. Support full hierarchy or just mesh list?

---

## Appendix: File Structure After Integration

```
client/src/
├── gltf.rs                    # Extended: GPU upload, materials, instancing
├── assets.rs                  # Modified: trigger GLTF GPU upload
├── texture.rs                 # Extended: load_gltf_textures()
├── render/
│   ├── mod.rs                 # Modified: register GLTF systems
│   ├── gpu.rs                 # Extended: GLTF pipeline, draw calls
│   ├── meshing.rs             # Extended: GLTF mesh upload system
│   └── shaders/
│       ├── gltf.vert          # NEW: PBR vertex shader
│       ├── gltf.frag          # NEW: PBR fragment shader
│       └── gltf_shadow.vert   # NEW: Shadow pass (future)
└── main.rs                    # No changes needed
```