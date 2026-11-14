# Rendering

The client does **not** use Bevy Render. The renderer is hand-written on [Vulkano](https://github.com/vulkano-rs/vulkano) because the engine needs bindless textures, a custom mesh pipeline, a separate first-person pass and a UI pass with its own camera — none of which Bevy Render exposes at that level.

Bevy is still doing the work it is good at: the ECS, the schedule, the input, the UI widget tree.

## Layout

```
client/src/render/
├── allocator.rs   sub-allocated vertex/index buffers
├── gpu.rs         pipelines, resources, the task graph, the frame submit
├── meshing.rs     the 3-stage chunk mesher
├── effects.rs     routing server-authored mod actions
├── effects_gpu.rs the effect instances and their compute passes
├── world_data.rs  the block-data window effect shaders can read
└── ui_pass.rs     the UI pass
```

The split exists so the GPU allocator, the mesher, the frame submit and the avatar rendering can change independently.

## The frame

`gpu::render_frame` runs once per tick, after the gameplay systems, and does roughly this:

1. **World pass** — the chunk meshes and the glTF instances, into the colour target with depth testing.
2. **First-person pass** — the arms, with their **own depth buffer and their own camera near/far range**, so the hands are never clipped by the world near plane.
3. **UI pass** — the Bevy UI entity tree, alpha-blended with **depth test and depth write disabled**, in the same subpass as the world after the world geometry.

The UI pass shares the world subpass, so the whole frame is one render pass with a dependency graph derived from the attachments — no explicit barriers between the world and the UI.

## The chunk mesher

`render/meshing.rs` implements an async, staged pipeline:

```
invalidate → queue → poll → budgeted upload
```

| Stage | What it does |
|---|---|
| **invalidate** | A changed chunk is marked dirty and its old GPU mesh is released. |
| **queue** | Dirty chunks are handed to `AsyncComputeTaskPool` as background tasks. |
| **poll** | Finished meshes are collected when their task completes. |
| **upload** | Ready meshes are written to the GPU buffer. |

Two budgets keep a boundary-crossing burst from becoming a frame spike:

```rust
const MAX_MESH_TASKS_STARTED_PER_FRAME: usize = 8;
const MAX_UPLOADS_PER_FRAME: usize = 4;
```

The mesher is deliberately symmetric in all three axes. A 32³ cubic chunk means a player moving in *any* direction — including straight up — can dirty a burst of neighbours at once, not just the horizontal ring a column-chunk world would produce.

### Texture invalidation

Mesh snapshots capture the **asset texture ids** they reference. When a texture is replaced (an asset hot reload), the affected meshes are rebuilt so they point at the newly registered sampled-image ids instead of the stale ones.

## The buffer allocator

`render/allocator.rs` sub-allocates the vertex and index buffers out of one large GPU allocation using `offset-allocator`. Chunks come and go constantly as a player moves, so the buffers are never reallocated — only the sub-ranges inside them are handed out and returned.

This is also what makes the *bindless* texture table viable: a texture id is just an index, and the table lives in one place for the lifetime of the process.

## Bindless textures

`client/src/texture.rs` is a port of the previous project's texture loader, with one deliberate difference:

> The old version panicked (`.unwrap_or_else(|_| panic!(...))`) on a missing/bad file. This one skips it and falls back to whichever texture loaded first for any block that wanted it — a missing asset should look wrong, not crash the game.

The loader discovers PNGs, batches them up, uploads them, and registers them as `SampledImageId`s. A mesh vertex references a texture by **id**, not by descriptor set, so adding a texture never rebinds anything.

`TextureMapping` decides which texture key a block face uses (top, side, bottom, …).

## Models

### Voxel block models

The mesher only ever sees a `BakedModel` — quads and triangles in 0–1 block space, with 0–1 UVs and texture keys like `"minecraft:block/stone"`. Each source game implements the `BlockPack` trait:

| Loader | Source | Notes |
|---|---|---|
| `loader/mcmodel.rs` | Latest-Minecraft JSON | `blockstates/*.json` + `models/block/*.json` + `.mcmeta`. Walks `parent` chains, resolves `#texture` variables. |
| `loader/obj.rs` | OBJ + MTL | Minimal loader for Forge-style static models. Triangulates n-gons as fans. |
| `loader/bbmodel.rs` | Blockbench `.bbmodel` | Cubes, meshes, groups, textures. Bakes the rest pose. |

Known Minecraft simplifications, all counted in the `LoadReport`: weighted variant arrays take the first entry, `uvlock` is ignored, element-level `rotation` falls back to a textured cube, and non-default states bake the default variant (chunk storage is id-only).

Every baked model carries a `RenderType`:

| Render type | Drawn as |
|---|---|
| `Opaque` | The default. |
| `Cutout` | Alpha-tested — leaves, glass, flowers, crops. No blending. |
| `Translucent` | Blended, drawn after the opaque pass — water, ice. |

### glTF

`client/src/gltf.rs` and `client/src/gltf_utils/` are a hand-written glTF loader: node hierarchy, materials, meshes, textures, and MikkTSpace tangents. The loader is covered by its own test suite in `client/src/gltf_utils/tests/` (animation, coordinate conversion, extensions, tangents, vertex attributes).

The rigging and animation side is covered in [entities-and-animation.md](entities-and-animation.md).

## The UI pass

`render/ui_pass.rs` takes the Bevy UI entity tree and draws it. The UI is a normal Bevy UI — nodes, `Style`, `Interaction` — but the **drawing** is the engine's own, so it can run in the same subpass as the world and after it.

The UI pass keeps its own draw list (`UiDrawList`) and syncs it to the GPU once per frame (`ui_pass::sync`).

## Mod effects

`render/effects.rs` routes server-authored `ModAction` packets to the matching client shader package; `render/effects_gpu.rs` owns the effect instances and their compute passes; `render/world_data.rs` maintains the block-data window that effect shaders read.

See [mod-effects.md](mod-effects.md) for the protocol.

## Diagnostics

The client has a `tracy` cargo feature that wires `tracing-tracy` in, for profiling the frame.
