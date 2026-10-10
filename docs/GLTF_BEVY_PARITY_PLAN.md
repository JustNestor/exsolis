# glTF load / render / animation — bevy parity gap analysis

Status: **investigation + plan (no code changed yet)**

Sources of truth for this document:

- Bevy 0.20.0-dev checkouts under `temp/bevy/crates/` (the workspace builds bevy
  0.19, close enough for these paths).
- Working reference app: `../bevyanimtest` (loads `girl.glb`, plays `walk`).
- This repo: `client/src/gltf.rs`, `client/src/anim.rs`,
  `client/src/anim/rig.rs`, `client/src/render/gpu.rs`.
- Asset measurements read straight out of the GLB chunks (not assumed):
  `assets/models/girl.glb` and `assets/models/girl2.glb`.

---

## 0. Asset audit — the two "girl" files are structurally different

| | `girl.glb` (shipped) | `girl2.glb` (added, not shipped) |
|---|---|---|
| Nodes | 1908, single root | 1114, `blockbench_export` root |
| Meshes / primitives | 795 / 795 (one 36-vertex box each) | 1 / 1 |
| Vertices | 19 080 | 76 320 |
| **Skins** | **0** | **1** (`JOINTS_0` u16 × `WEIGHTS_0` f32, **rigid: every weight is (1,0,0,0)**) |
| Animated nodes | 62 (81 incl. ancestors) | 62 (82 incl. ancestors) |
| Clips | 11, all `LINEAR`, 227 rot / 22 trans / 56 scale channels | 11, identical counts |
| Material | 1, `MASK`, `doubleSided`, WebP texture | 1, `MASK`+cutoff 0.05, `doubleSided`, PNG, sampler `magFilter: 9728` |
| TANGENT attribute | **present on every primitive** (byteStride 48) | absent |
| Mesh-node transforms | all identity | identity |
| World extent | 1.370 × 3.193 × 3.139, y −0.005…3.188 | −0.685…0.685 x, ~same height |

`server/src/lib.rs:12` still ships `PLAYER_MODEL = "models/girl.glb"` (scale
`0.565`), while `client/src/anim.rs:797-817` `GIRL_CLIP_ROLES` /
`builtin_roles` now key on **`models/girl2.glb`**.

**Consequence (already observable):** the avatar that actually runs has *no*
clip manifest, so all 11 clips get their role from the structural classifier,
and `cargo test -p exsolis_client` fails:

```
anim::rig::tests::girl_clips_decode_with_the_measured_shapes
  assert_eq!(manifest.len(), clips.len(), "manifest covers every clip")
  left: 0   right: 11
```

(second, unrelated failing test: `ui::parse::tests::unknown_element_is_error` —
`<img>` is now a real element, so the test's "unknown element" fixture is stale.)

---

## 1. How bevy loads and runs a glb

`temp/bevy/crates/bevy_gltf/src/loader/mod.rs`

1. **Import** — `gltf::import` with resolved buffers + decoded images
   (`load_gltf`, L249).
2. **Mesh assets, per primitive** (L780-900). Every accessor *data type* is
   converted, not just f32: `vertex_attributes.rs` maps u8/u16/u32/float,
   normalized or not, into `VertexAttributeValues` (`F32`, `U32`, `S16x2`,
   `Unorm8x4`, …). Topology comes from `Mode` (`primitive_topology`,
   `loader/gltf_ext/mesh.rs`).
   - **Missing normals** → `mesh.duplicate_vertices(); mesh.compute_flat_normals()`
     (L826-841).
   - **Missing tangents, only when the material needs them** (normal map present)
     → `mesh.generate_tangents()` (mikktspace) (L843-848).
   - **Morph targets** → `mesh.set_morph_targets(...)` (L798-816).
   - Attributes are then **compressed** for the GPU (`compressed_mesh`).
3. **Scene, one entity per node** (L1044-1150). Node transforms become real
   `Transform`s in a hierarchy — nothing is baked.
   - Skinning is *not* baked either: each skinned mesh entity gets
     `SkinnedMesh { joints: Vec<Entity>, inverse_bindposes }` (L1125-1137).
   - Every animation-**root** node gets an `AnimationPlayer` (L1112-1123).
4. **Textures / materials** (`loader/gltf_ext/material.rs`). Per glTF spec,
   base-colour and emissive images are uploaded **sRGB**, normal/occlusion/
   metallic-roughness **linear**; sampler wraps/filters are honoured; the
   material's `tex_coord` index picks the TEXCOORD set and a missing one warns
   (`uv_channel`, L57-72). `KHR_materials_unlit` → unlit material;
   `KHR_materials_anisotropy|clearcoat|specular` have loader hooks.

**Animation** (`bevy_animation/src/gltf_curves.rs` + loader L318-590):
per channel a `VariableCurve` keyed by `AnimationTargetId` (the node's *name
path*) and the property path. Interpolation is mapped 1:1:

| glTF sampler | bevy curve |
|---|---|
| `LINEAR` | `UnevenSampleAutoCurve` |
| `STEP` | `SteppedKeyframeCurve` |
| `CUBICSPLINE` | `CubicKeyframeCurve` |
| single keyframe | `ConstantCurve` |

`AnimationClip` is an asset (`Animation#0`, plus `named_animations`),
`AnimationGraph` composes several clips into nodes, `AnimationPlayer` plays them
with speed/repeat/pause, `AnimationTransitions` cross-fades — exactly the API
`../bevyanimtest/src/main.rs` uses.

**Skinning on the GPU** (`bevy_pbr/src/render/skin.rs`): `SkinUniforms` gathers
each joint entity's **current `GlobalTransform` × inverse bindpose** every frame,
packs all skins into one growing buffer with an offset allocator
(`MAX_TOTAL_JOINTS` = 2³⁰), and hands the shader a per-mesh byte offset. Nothing
about skinning is capped at 256 joints, and no animation knowledge lives in the
loader or the renderer — `bevy_animation` drives plain `Transform`s and the
renderer reads the result.

**Rendering** (`bevy_pbr`): GGX + Smith, IBL/env, the scene's actual
`DirectionalLight`/`PointLight`/ambient, shadows, ACES tonemap with proper sRGB
EOTF, normal mapping through a TBN (tangent when available, otherwise
screen-space derivatives), `double_sided` → cull mode, `alpha_mode` → opaque /
mask-discard / blend-sorted, and Aabb-based frustum culling.

---

## 2. Where we diverge — ranked

### P0 — the shipped avatar has no clip manifest (one-line fix, breaks tests)

`client/src/anim.rs:797-817` keys on `models/girl2.glb`; the shipped model is
`models/girl.glb`. `builtin_roles` must match by model, and it must stay in sync
with whatever `PLAYER_MODEL` ships.

Fix: key the role table on *both* paths (or on the basename+shape, e.g. match
`girl.glb` / `girl.glb`/`girl2.glb`, both measure the same), and tighten the rig
test so it can't drift again. Long term, ship the manifest as an asset
(`docs/ENTITIES_ANIMATION_PLAN.md:93` promises `assets/anim/avatars.toml`, which
`parse_manifest` (`anim.rs:735`) already parses) instead of a `const`.

### P0 — every non-LINEAR animation channel is silently dropped

`client/src/anim/rig.rs:705-794` (`decode_clips`) has
`if sampler.interpolation() != Linear { warn; continue }`. Both current girls are
all-LINEAR, so this is invisible today — but **any Blockbench "smooth"/Blender
re-export uses STEP or CUBICSPLINE, and those joints freeze while the rest of the
clip plays**. Bevy supports all three plus single-key constant curves.

Fix: add `Step` and `CubicSpline` to `AnimPath` sampling
(`anim.rs:144-183`). CUBICSPLINE needs a 3× wider value array (in-tangent,
value, out-tangent) and tangent-based evaluation, exactly like bevy's
`CubicKeyframeCurve`.

### P0 — tangents from the file are ignored, and the generator is broken

`client/src/gltf.rs:462-516` (`load_gltf_primitive`) never calls
`reader.read_tangents()`, even though `girl.glb` ships TANGENT on all 795
primitives. Instead `compute_tangents` (`gltf.rs:542-604`) runs, and it has three
real defects:

1. `tan2` is allocated and **never accumulated**, so `handedness` is always `+1`
   (`gltf.rs:593-598`). A UV-mirrored atlas (Blockbench / the dragon) yields a
   flipped bitangent → normal maps are lit from the wrong side.
2. No area/angle weighting (mikktspace weights each vertex's contribution by the
   opposite triangle angle) → tangents visibly rotate across large triangles.
3. `r = 1.0 / (delta_uv1.x*delta_uv2.y - delta_uv1.y*delta_uv2.x)` has no
   degenerate-triangle guard → `inf`/`NaN` tangents poison the TBN for any
   zero-area-UV triangle.

Fix: read TANGENT when present (glTF's w is the handedness), and only fall back
to a guarded, angle-weighted mikktspace-style generator when a normal map is
actually present (bevy's `needs_tangents(&primitive.material())` rule,
loader L843).

### P1 — skinning is rigid-only, and triangles are split by their first vertex

`client/src/anim/rig.rs:144-181` picks each vertex's **dominant** joint, then
`split_by_joint` (rig.rs:432-483) assigns a whole triangle to the first vertex's
joint. girl2 happens to be rigid (`WEIGHTS_0` is `(1,0,0,0)` for all 76 320
verts, verified), so it survives — but any rig with real weights tears, and even
a rigid mesh can lose a triangle whose vertices straddle two bones.

Bevy stores 4 joint indices + 4 normalized weights per vertex and blends them in
the shader. Our `GltfGpuVertex` (`gltf.rs:84-102`) has one `joint: u32` slot.

Fix: carry `joints: [u16;4] + weights: [f32;4]` (or at least 2 influences),
skinning in `gltf_vs` (gpu.rs:146-147) as a weighted sum, and drop
`split_by_joint` entirely once weights are per-vertex.

### P1 — the skinned path double-applies the mesh node transform

In `Rig::build` the skinned branch still bakes vertices with
`to_joint = world[index]` (rig.rs:123, 193). Per the spec a skinned primitive's
bind matrix already lives in the joint chain, so the mesh node's own transform is
applied twice the moment it isn't identity. Harmless on both current girls (all
mesh nodes are identity) — a landmine for the next asset.

### P1 — a model needing >255 joints fails to load entirely

`assign_joints` (`rig.rs:584-589`) returns `Err("model needs N joints, limit is
256")` → `decode_gltf` errors → nothing renders. girl2 has 1113 skin joints (82
animated) and would be fine, but a Blender export with more animated nodes is a
hard failure with no fallback. Bevy's `MAX_JOINTS` is a shader constant over a
shared, allocator-managed buffer, and the pack-offset scheme means skins don't
compete for a per-model budget.

Fix: raise the cap, and on overflow keep the N most-influential joints and fall
the rest to `STATIC_JOINT` (warn) instead of dropping the model.

### P1 — lighting, gamma and colour space in `gltf_fs`

`client/src/render/gpu.rs:276-361`:

- one **hardcoded** directional light (`normalize(vec3(0.5,1.0,0.3))`,
  `light_color = vec3(1.0,0.95,0.9)*3.0`), `ambient = vec3(0.03)`, no shadows, no
  IBL. `../bevyanimtest` instead uses `GlobalAmbientLight{brightness: 300}` and a
  real `DirectionalLight` with `shadow_maps_enabled`. This is the single biggest
  reason the same asset looks different between the two apps.
- colour space: `base_color_factor` is multiplied in and then the whole product
  goes through `srgb_to_linear` (`gpu.rs:289-298`) with `pow(c, 2.2)`. glTF's
  factor, `COLOR_0` and `emissive_factor` are **already linear** — only the
  base-colour/emissive *textures* are sRGB. So every non-white factor/vertex tint
  is gamma-shifted twice, and ACES is applied to the wrong curve.
- magnitude filter defaults to **Nearest** (`gltf.rs:424-426`) unless the sampler
  says otherwise. Correct for our pixel-art dragon/girl; wrong for any normal
  textured asset (bevy honours the sampler).

Fix: pass light + ambient through push constants/uniforms from the ECS, keep sRGB
conversion to textures only, and use the real sRGB EOTF.

### P2 — props and non-player models can't be told what to play

`Animator::new()` (`anim.rs:528`) is clip 0, and `locomotion_for_speed`
(anim.rs:607) always resolves to walk/run when both roles exist. A spawned prop
with a lid animation or a spinning part therefore plays an arbitrary clip (or
rest) forever. Bevy gives every `AnimationPlayer` an explicit
`play(index, Duration)` + repeat, and the app in `../bevyanimtest/src/main.rs`
iterates `named_animations` to pick by name.

Fix: a `play_clip(path_substring_or_index, repeat, speed)` API plus a
`GltfModelInstance::playing` field, and keep the locomotion auto-select for
player entities only.

### P2 — other gaps (small but all bevy-covered)

- `play()` ignores its `Transition` argument (`anim.rs:542-550`) — no weights, no
  speed, no additive layering, and no `previous` clip survival across the
  `Mask::LowerBody` blend edge cases the tests already document.
- `animator.look()` / `airborne` are written (`player.rs:393`, `entities.rs:263`)
  but **never read by the pose compositor** — the head-look feature is inert.
- No morph-target support (`decode_clips` drops `MorphTargetWeights`;
  `GltfVertex` has no slot for deltas).
- No `KHR_materials_unlit`, no `KHR_texture_transform`, no
  `KHR_materials_specular/clearcoat/anisotropy`.
- Only `TEXCOORD_0` is read; a material with `tex_coord: 1` silently uses the
  wrong UVs (bevy maps and warns).
- `MAX_PALETTE_INSTANCES = 64` (`gltf.rs:43`) silently drops animated instances
  past 64 per model (`sync_gltf_instances`, gltf.rs:1289).
- `compute_aabb` / `compute_bounding_sphere` are computed and then never used —
  there is **no frustum culling** for models at all.
- `build_arms` (rig.rs:289-347) runs on *every* model, warns on all non-rigged
  ones, and silently disables first-person arms when the name-prefix heuristic
  matches >35% of the geometry.
- No shadows for models.

---

## 3. Plan

### Stage 1 — make the shipped avatar correct (½ day, no renderer work)

1. Fix the manifest keying (`anim.rs:797-817`) so `models/girl.glb` is covered;
   move the table into a shipped manifest file that `parse_manifest` reads
   (`assets/anim/avatars.toml`, already promised by the docs), falling back to
   the classifier. Add a test that asserts every shipped `PLAYER_MODEL` resolves
   a non-empty manifest.
2. Fix the stale `ui::parse` test (unrelated, but `cargo test` should be green).
3. Read `TANGENT` when the file has it; fix `compute_tangents`
   (angle weighting, real `tan2` accumulation, degenerate guard) and gate the
   fallback on `needs_tangents`.
4. Don't bake skinned primitives by the mesh node's world matrix
   (rig.rs:144-181).

### Stage 2 — animation fidelity (2–3 days)

1. `AnimChannel::sample_into`: `Step`, `CubicSpline` (in/value/out tangents),
   single-key `Constant`. Keep the linear path hot.
2. Playback control per instance: `play / repeat / speed / pause` on `Animator`
   (honour the `Transition` argument), and an explicit clip selection for props.
3. Wire `head_yaw`/`head_pitch` into `compose_palette` as an additive head
   rotation (it's already computed).
4. Add a Rust-side evaluator test that re-samples a known clip against the
   values bevy would produce, for each interpolation mode (a tiny synthetic glb
   is enough — no GPU needed).

### Stage 3 — real skinning (2–3 days)

1. `GltfVertex`/`GltfGpuVertex`: `joints: [u16;4]`, `weights: [f32;4]` (normalise
   and drop zero weights at upload so the common rigid case still costs one
   matrix).
2. `gltf_vs`: weighted skin matrix + weighted normal matrix; delete
   `split_by_joint`.
3. `gltf_fs`: skip the `mat3(transpose(inverse(...)))` per-vertex when all four
   weights are the same joint (the common case) — keep it for the blended one.

### Stage 4 — lighting/colour (2 days)

1. Move light direction/colour/intensity and ambient out of the shader into push
   constants fed from ECS light entities; add shadow-map support for models
   (this repo already has a depth image to borrow).
2. sRGB only on base-colour/emissive textures; proper sRGB EOTF/`OETF`; keep
   factors linear.
3. Honour the sampler's mag/min filter per material (already plumbed via
   `mag_nearest`) — decide it from the asset, not globally.

### Stage 5 — robustness + throughput (1–2 days)

1. Joint-limit fallback instead of a hard error; raise `MAX_JOINTS`.
2. Second texture set (`tex_coord > 0`), `KHR_materials_unlit`,
   `KHR_texture_transform`.
3. Frustum culling from the already-computed `bounding_sphere`.
4. Morph targets (at minimum `MASK`-adjacent weight animation on a dedicated
   buffer).
5. `MAX_PALETTE_INSTANCES` → grow or spill into a second draw group.

### Tests that must exist before any of this is called done

| Test | Guards |
|---|---|
| `manifest covers every shipped model` | Stage 1 |
| `reader tangents are used; generator matches a reference pose` | Stage 1 |
| `step/cubic/constant sampling matches expected values` | Stage 2 |
| `a weighted skin bends without tearing` (synthetic 2-bone mesh, >1 weight) | Stage 3 |
| `bounding sphere frustum cull keeps the model on screen` | Stage 5 |
