# GLTF Model Integration Plan

## Status: Phase 1 (CPU→GPU foundation) — implemented, compiles, decode-tested

`cargo check --workspace` passes. `gltf::tests::decode_duck_glb` passes
(real GLB 2.0 asset: meshes, vertices, finite tangents). Shaders are inline in
`render/gpu.rs` (matching the `mesh.rs` pattern) rather than separate
`.glsl` files.

### Agreed decisions (2026-10-09)
- **GLB-only**: server manifests and sends only `.glb`. `.gltf` sources are
  auto-converted server-side (A6); raw sources never ship.
- **Hot-reload swaps under instances**: entities keyed by asset path keep
  working; the upload system frees + reallocates GPU buffers (B1).
- **Eager decode at sync**: all manifest `.glb` decoded like effect shaders.
- **Suggestions accepted**: 64 MiB per-model cap (`GLB_MAX_BYTES`, enforced
  server-side at manifest time and client-side at decode), bindless texture
  leak on hot-reload logged as tech debt.
- **Color fix (2026-10-09)**: `TextureManager` rebuilds replaced the whole
  resource, invalidating GLTF bindless ids (models rendered untextured after
  any PNG-bearing asset sync). Fixed with a `generation` counter: uploads
  stamp the generation they built against and redo fully on mismatch.

### Test assets (`assets/models/`)
- `duck.glb` (120 KiB, GLB 2.0) — primary test model, covered by unit test.
- `avocado.glb` (8.3 MiB, GLB 2.0) — larger/stress test model.
- `dragon/`, `girl/` source dirs are **glTF 1.0** (assimp export). The server
  auto-converts them to `models/dragon/result.glb`, `models/girl/result.glb`
  at catalog time (`server/src/gltf_pack.rs`, tested); the raw dirs are NOT
  usable by the client loader and never ship (filtered at pak-add).
  Do not add new 1.0 assets; re-export as 2.0/GLB instead.

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
7. **Model distribution**: GLB-only for now, or `.gltf`+`.bin` pairs too? (see Hot-reload item 4)
8. **Load trigger**: eager-decode all manifest `.glb` at sync, or server-driven
   `request_load` per spawn? (see Server-download item 3)

---

## TODO: Server-download integration & hot-reload

### A. Server-download integration (NOT yet done)

Finding: `is_client_asset_path` (`server/src/assets.rs:617`) allowlists only
`fxm | wgsl | png | xml | json`. `.glb` / `.gltf` / `.bin` never enter the
manifest, so models are currently never downloaded at all. Client-side loader
registration alone is insufficient.

| # | Task | File | Description | Status |
|---|------|------|-------------|--------|
| A1 | Manifest allowlist | `server/src/assets.rs` | `glb` ships; `.gltf` sources auto-convert (see A6), raw sources never ship | ✅ done |
| A6 | Server auto-convert | `server/src/gltf_pack.rs` (new) | Rust: v1.0→GLB (geometry 1:1, diffuse→PBR base color) + v2.0 external-embed; `.glb` written next to source when stale, served from disk via `make_delta`; committed `.glb` wins on collision | ✅ done + unit-tested |
| A2 | Size guard | `server/src/assets.rs` | Per-model warn/skip above a cap (e.g. 64 MiB); models are orders of magnitude larger than shaders | ✅ done (`GLB_MAX_BYTES`, skip+warn in `walk_assets`) |
| A3 | Eager decode at sync | `client/src/assets.rs` | Extend `load_effect_assets` (or add `load_model_assets`) to force-decode manifest `.glb`/`.gltf` at `SyncReady`, like `fxm`/`wgsl` today. Otherwise a model only decodes if the server sends an explicit `AssetLoad` for it | ✅ done (`glb` added to eager match) + decode-side `GLB_MAX_BYTES` guard |
| A4 | GPU upload hook | `client/src/gltf.rs` | Fill in `upload_gltf_assets` body: for each resident `GltfAsset`, call `upload_gltf_model` → `upload_gltf_writes`, record in `GltfModels.uploaded` | ✅ done (incl. material packing ≤256 slots, B1 free+re-upload) |
| A5 | Test | manual | Place a `.glb` under `assets/`, join, verify download + resident `GltfAsset` + GPU upload | ⏳ user verification (duck+avocado via `mods/models_test.lua`) |

### B. Hot-reload

What already works: `WorkerEvent::Updated` re-runs the registered `decode_gltf`
loader, so a changed `.glb` becomes a fresh resident `GltfAsset`, and
`AssetCacheChanged` fires. What is missing is everything GPU-side.

| # | Task | File | Description |
|---|------|------|-------------|
| B1 | Re-upload on change | `client/src/gltf.rs` | `upload_gltf_assets` reads `AssetCacheChanged`; for changed `.glb` paths: free old `gltf_allocator` allocation, re-run `upload_gltf_model`, swap `GltfModels.uploaded` entry. Instances keyed by path keep working | ✅ done |
| B2 | Texture refresh | `client/src/texture.rs` | Re-run `load_gltf_textures` for the new pixels; update material bindless indices. Note: old bindless ids leak (no unload policy — same as PNGs, acceptable for dev, track as tech debt) | ✅ done (re-upload via `upload_gltf_model`; leak logged with warn on hot-reload) |
| B3 | Material buffer rewrite | `client/src/gltf.rs` | Rewrite that model's material range via `upload_gltf_writes` material writes | ✅ done (whole-buffer repack ≤256 slots on any upload) |
| B4 | Removal | `client/src/gltf.rs` | On `AssetRemove` / manifest prune: free GPU allocation, drop `GltfModels` entry, hide dependent `GltfModelInstance`s (or fall back to placeholder). Textures retained per existing no-unload policy | ⏳ todo |
| B5 | External-URI constraint | `client/src/gltf.rs` | Document: hot-reload of `.gltf`+`.bin`/external-URI pairs is unsupported until custom buffer resolution lands (`import_slice` reads externals from CWD and will fail); GLB + embedded base64 only | ✅ documented (GLB-only decision) |

### C. TEMPORARY scripting spawn API (test only — remove before merge)

| # | Task | File | Description | Status |
|---|------|------|-------------|--------|
| C1 | Protocol | `common/src/lib.rs` | `SpawnModel { entity_id, path, x, y, z, scale }` + `DespawnModel { entity_id }` appended last (bincode index stability); `MODEL_PATH_MAX_LEN` | ✅ done |
| C2 | Lua kinds | `server/src/scripting.rs` | `spawn_model` / `despawn_model` in `build_server_message` with path/scale validation | ✅ done |
| C3 | Client handler | `client/src/gltf.rs` | `handle_spawn_model_messages`: spawn `GltfModelInstance`+`Transform` entities (replace semantics per server id), `ModelEntityId` mapping for despawn | ✅ done, registered in `RendererPlugin` chain |
| C4 | Test script | `mods/models_test.lua` | On `player_join`: duck (id 1) + avocado (id 2) + dragon (id 3) + girl (id 4) near spawn | ✅ done |
| C5 | In-game verification | manual | Join, confirm all four models render near spawn | ⏳ user verification |

**Verification checklist for the session**: (1) server manifest includes
`models/*.glb`; (2) client downloads + eager-decodes
them (no `asset handler failed` warnings); (3) all four models render near
spawn; (4) `cargo test -p exsolis_client --bin exsolis gltf` passes (already ✅).

Fixed this session: `setup_renderer` panic — shader inputs (`in_position`,
…) did not match `Vertex`-derived attribute names (`position`, …); renamed
shader-side. `cargo check` passing does NOT catch this class of error (it is
a runtime `definition()` validation), only a real renderer startup does.

Fixed next: `create_info.stages[0]` validation — `pipeline_layout_from_stages`
carries only the bindless global set, dropping our set-1 storage bindings.
GLTF pipeline now builds its layout explicitly as
`[global_set_layout, local]` (same shape as `create_effect_pipeline_layout`),
and the descriptor set allocates against that local layout.

Fixed next: completely black models — CPU `GltfGpuMaterial` (72 B packed)
did not match the GLSL std430 layout (vec2/vec3 padding + 96 B stride), so
every material field read garbage. Both sides reordered to vec4s-first with
zero implicit padding (80 B / stride 80), locked by
`gltf_gpu_material_layout` unit test. Lesson: any GPU struct with vec2/vec3
members needs an explicit layout test; vec4s + scalars ordered
largest-first need none.

Fixed next: sky giant — measured POSITION bounds show the test models are
wildly different scales (duck 165u, avocado 0.04u, dragon 51u, girl 2.7u).
`mods/models_test.lua` scales each to ~2-3 world units. Always measure
bounds before choosing spawn scales.

Fixed next: exploded multi-primitive models — `vertex_offset` was applied
TWICE (indices pre-offset to model-global at upload AND `prim.vertex_offset`
added again in the draw; Vulkan adds `vertexOffset` on top). Primitive 0
(vertex_start 0) rendered fine, which is exactly why single-primitive duck
looked perfect while dragon/girl shattered. Draws now use only the
allocation base. Lesson: pick ONE offset scheme and assert it — pre-baked
indices pair with allocation-only `vertexOffset`.

Fixed next: texture/image index confusion — materials stored
`texture().index()` (textures-array space) but our texture list parallels
`images()`. Now resolves via `texture().source().index()`; covered by the
faithfulness test asserting every material texture ref < image count.

Deferred with reason: global alpha blending (all test models are Opaque;
blending needs Phase 2.5 sorting + depth-write policy, not a blanket flag),
`SampledImageId(_, 256)` (mirrors mesh.rs exactly; shared constant if ever
changed), image layout transitions (identical code shape to the working PNG
path — taskgraph handles it).

Fixed next: stretched textures + black wing gaps on converted models.
UV/mip investigation showed the pipeline samples faithfully (duck perfect,
girl's face details correct) — converted UVs are tiny authored islands
(~0.02 span magnified over whole quads), i.e. asset quality, not a bug.
The black gaps are backface-culled thin membranes: v1 carries no sidedness,
so the converter now emits `doubleSided: true` and the client draws those
primitives with a no-cull pipeline variant (grouped binds). Converter output
is version-stamped (`CONVERTER_VERSION`, via `asset.generator`) so future
converter changes force regeneration instead of freshness-skipping.

Fixed next: dark transparent regions (wing membranes, tails). Two parts:
(a) universal `alpha < 0.004` discard — fully-transparent texels (black RGB
in cutout atlases) no longer render as opaque black under any mode;
(b) full Blend support — `alpha_blend` flag per draw, blend + blend_nocull
pipelines (`AttachmentBlend::alpha()`, depth-write off), transparent groups
drawn last in a 4-pass loop. Converter emits `MASK` + 0.5 cutoff for textured
v1 materials (opaque textures never trip it), so converted cutouts resolve
in the opaque pass.

Fixed next: **anisotropic filtering** — the primary cause of "blurred"
textures at oblique angles. The `linear_sampler` and `block_sampler` were
created without anisotropy (`anisotropy: None`). Enabled by:
1. Adding `anisotropy: Some(16.0)` to both samplers in `vulkan_prelude.rs`
2. The vulkano git commit used already supports the `anisotropy` field in
   `SamplerCreateInfo`; the Cargo feature flag was not needed (field exists
   in base API). Vulkan clamps to device max.
Result: textures at oblique angles now remain sharp; mipmap selection uses
the correct LOD based on the anisotropic footprint.

Fixed next (bevy_gltf parity investigation): three code-side improvements
from comparing with `temp/bevy/crates/bevy_gltf` —
(a) tangents are now computed only when a normal-mapped material exists
(bevy runs mikktspace only on `needs_tangents`; ours ran a hand-rolled pass
always — pure perf, no visual change since tangents are dead data without
normal maps);
(b) the authored sampler `magFilter` is honored for base color: Nearest
selects the block (nearest-mag) sampler instead of linear, giving crisp
texels on pixel-art-ish assets (all current models author Linear, so no
change for them — stored as `mag_nearest` in the material buffer);
(c) UV-span diagnostic: upload logs `max_uv_span` per model and warns below
0.1 — tiny islands magnified over whole parts are the proven cause of
"blurry" model textures, now visible in logs instead of requiring a
debugger.
Investigation verdict stands: attribute reading (float/normalized ints),
TEXCOORD handling, image upload, mipmaps, and sampling are equivalent to
bevy_gltf for our data; the remaining blur is asset UV quality.

---

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