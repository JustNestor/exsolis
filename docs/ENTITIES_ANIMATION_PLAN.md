# Entities, Block Entities & Animation — Plan

Branch: `entities-anim` (from `main`).

## Status: implemented

Everything in the task list below is done except the v1 clip synthesis noted
in A1, which turned out to be unnecessary — see §5.

## 1. Scope

Three coupled pieces of work, in dependency order:

1. **Entity system** — players become entities, block entities, a generic
   kind + mod-facing spawn API.
2. **Animation system** — glTF node-hierarchy animation (skeletal-equivalent),
   clip translation, and a transition layer with per-model strategies.
3. **First-person hands** — an arms-only viewmodel from the same GLB, drawn
   in a separate depth range.

## 2. Asset audit (measured, not assumed)

All numbers below were read out of the actual files.

### 2.1 `assets/models/girl.glb` — the avatar

Self-contained GLB 2.0, 1 968 188 bytes (under `GLB_MAX_BYTES` = 64 MiB),
so it ships through the asset pak with **no `.gltf` conversion**.

| Property | Value |
|---|---|
| Nodes | 1908, single root, max depth 18 |
| Node transforms | 927 nodes carry TRS; **no node carries a `matrix`** |
| Meshes / primitives | 795 / 795 — each primitive is one 36-vertex box |
| Vertices / indices / tris | 19 080 / 27 030 / 9 010 |
| Materials | 1 (`MASK`, `doubleSided`), 1 texture (**WebP**), 1 sampler |
| Vertex layout | interleaved, `byteStride` 48 (POSITION/NORMAL/TANGENT) |
| Skins | **0** |
| Animations | **11** |
| World extent | 1.370 × 3.193 × 3.139 units, y from −0.005 to 3.188 |

It is a **node-hierarchy rig, not a skin**: every mesh is rigidly parented to a
bone node (`LeftArm_964`, `LeftForeArm_960`, `RightArm_1021`,
`RightForeArm_1009`, `RightHand_1006`, `LeftLeg_52`, `LeftLowerLeg_51`,
`LeftFoot_50`, …). All 795 mesh nodes have an animated ancestor, so the whole
body is driven by the clips.

This is *better* than skinning for us: rigid binding means every vertex has
exactly one influence, so we get skinned-style batching (one draw call) with
none of the weight-blending complexity — but we still carry a joint index per
vertex and skin in the vertex shader so a genuinely skinned asset also works.

Two asset quirks the loader has to cope with, both found by reading the file:

- The texture is **WebP**. The `gltf` 0.16 importer only recognises PNG and
  JPEG — even with its `guess_mime_type` feature, every other format maps to
  "unsupported image encoding" and the whole file is rejected. The server now
  normalizes embedded textures to PNG while building the catalog, and the
  client does the same for models loaded from disk.
- Accessor `min`/`max` do **not** match the actual vertex data for some
  primitives (node 901 declares y ∈ [0, 0.125]; the buffer holds ±0.89).
  Anything that measures bounds from accessor metadata will disagree with the
  real geometry, so the bounds test composes node matrices over accessor
  bounds and never reads `byteStride`-interleaved data naively.

Animation cost is tiny: **62 animated nodes** across all 11 clips. The other
1846 nodes are static and get baked into vertex positions at load.

### 2.2 Clip translation — the names are destroyed

Every clip name except `walk` and `extra6` is mojibake: the exporter wrote
GBK bytes that were already replaced by `U+FFFD` before packing, so the
original text is **unrecoverable** (`'锟斤拷锟斤拷'` and friends are all
`ef bf bd` repeats). Names cannot be the source of truth.

We translate structurally instead — a clip is classified by *which nodes it
drives*, its duration, key count and root-motion amplitude:

| # | Raw name | dur | keys | drives | Inferred role |
|---|---|---|---|---|---|
| 0 | mojibake | 0.000 s | 1 | upper body + hair/tail | `rest` (static pose) |
| 1 | `walk` | 1.000 s | 19 | full body + root motion 0.05 | `walk` |
| 2 | mojibake | 1.000 s | 19 | walk set + `Tail` | `walk` (tail variant) |
| 3 | mojibake | 3.000 s | 53 | `Tail`, `Tail2..6` only | `tail_sway` (additive) |
| 4 | mojibake | 6.000 s | 109 | full body + `RightHand_1006` | `emote_hands` |
| 5 | mojibake | 3.000 s | 57 | walk set ×3 length | `walk_slow` |
| 6 | mojibake | 4.958 s | 86 | legs + upper arms, **no forearms** | `sit` |
| 7 | `extra6` | 6.000 s | 109 | full body, **root motion 0.181** | `run` |
| 8 | mojibake | 6.042 s | 109 | full body, no root motion | `idle` |
| 9 | mojibake | 4.958 s | 86 | legs + upper arms, no forearms | `sit_alt` |
| 10 | mojibake | 0.000 s | 1 | arms + legs | `rest_alt` (static pose) |

The classifier **proposes**; it does not decide. `assets/anim/avatars.toml`
holds the authoritative index→role mapping with a `verified` flag, and the
loader logs any disagreement between manifest and classifier. Shipping a new
rigged avatar means editing that file, not the classifier.

### 2.3 Today

`client/src/render/avatars.rs` draws a stone-textured cube head
(`build_player_head_mesh`), one instanced draw. 795 primitives per girl would
be 795 draw calls, since `sync_gltf_instances` emits one `GltfDrawCall` per
primitive and the vertex shader has a single model matrix.

### 2.4 Other models

`duck.glb` (3 nodes), `dragon/result.gltf`, `labs.glb`, `avocado.glb` have no
animation support requirement in this pass, but must keep working unchanged —
so skinning/animation is strictly opt-in per model.

## 3. Architecture

### Stage A — packer: preserve what is there

`server/src/gltf_pack.rs`

- `convert_v1`: glTF 1.0 has no animation support in this converter. It gains
  a **clip synthesis** fallback (see §3.5) so v1 sources (dragon, old girl)
  still get something to play.
- `convert_v2`: currently clones the document, so `animations`/`skins` already
  survive. Add a validation pass that rejects assets whose animation data we
  cannot honour (e.g. `CUBICSPLINE` on an unsupported path, weights > 4) with
  a clear message instead of silently mis-rendering.
- Bump `CONVERTER_VERSION` so stale `.glb`s regenerate.

### Stage B — protocol: entities

`common/src/lib.rs` (append-only, bincode indices preserved)

```rust
pub type EntityId = u64;

pub enum EntityKind {
    Player,
    Block,
    Item,
    Prop,
}

pub struct EntityState {
    pub x: f32, pub y: f32, pub z: f32,
    pub yaw: f32, pub pitch: f32,
    pub anim: Option<AnimState>,     // locomotion, for the client
}
```

New `ServerMessage` variants (all appended last):

- `SpawnEntity { entity_id, kind, pos, yaw, pitch, model: Option<String>, scale }`
- `DespawnEntity { entity_id }`
- `EntityStateUpdate { entity_id, state }` — unreliable, coalesced
- `EntityAnim { entity_id, clip: u16, time: f32, speed: f32 }`

`PlayerJoined`/`PlayerLeft`/`PlayerMoved` are kept for one release as thin
adapters over the entity messages so the client can migrate independently.

`EntityKind::Player` reuses the existing stub; `ShowEntity`/`EntityDisappear`
are removed once `SpawnEntity`/`DespawnEntity` land.

### Stage C — server: entity registry + block entities

- `server/src/entities.rs`: `EntityRegistry` resource
  (`id → (kind, Entity, spawn state)`), monotonic id allocation, spawn/despawn
  with broadcast, and a despawn pass for entities that leave a player's
  render distance.
- `spawn_joined_player` spawns a `Player` **entity** and registers it; the
  existing `PlayerMap` becomes a lookup into the registry.
- Block entities: a block type flagged in the type DB creates an entity bound
  to its `BlockPos`. Components:
  - `BlockEntity { pos: BlockPos, kind: BlockType }`
  - `Container` (reuses `common::inventory::Inventory`, the same zero-based
    slot container `mods/container.lua` already models)
  - `TickScheduled { at: u64 }` — mods can request a tick; the server runs it
    on the tick thread with the same bounds as `api.on_tick`
  - Breaking the block despawns the entity and drops its container contents.
- Mod API (`server/src/scripting.rs`), replacing the TEMPORARY stubs:
  `api.spawn_entity`, `api.despawn_entity`, `api.set_entity_state`,
  `api.get_block_entity`, `api.list_entities`. The server allocates ids, so
  a mod cannot pick a colliding one. All ids are strings at the Lua boundary
  (`u64` does not fit a Lua number), matching the existing
  `player_id_to_lua` convention.

### Stage D — client: glTF skeleton + skinning + animation

`client/src/gltf.rs`

New decode stage (after `load_gltf_model`):

1. Build the node graph: `parent[]`, local TRS (decompose `matrix` when
   present — the girl uses `matrix`, not TRS), rest-pose world matrices.
2. Compute the **joint set**: nodes animated by any clip, plus all their
   ancestors. Everything else is static and gets folded into vertex positions.
3. **Bake** static transforms: each primitive's vertices are transformed by
   `inverse(joint_world) * node_world` and re-emitted in joint-local space.
4. Assign `JOINTS_0` (rigid: `j,0,0,0`) and `WEIGHTS_0` (`1,0,0,0`) per vertex.
5. **Merge** all primitives sharing a material into one primitive. The girl is
   single-material → **1 draw call instead of 795**.
6. Bake the arms viewmodel: a second merged primitive containing only
   primitives whose joint is in the arm set (`LeftArm`, `LeftForeArm`,
   `RightArm`, `RightForeArm`, `RightHand`, plus the hair/garment arm chains
   `H_LeftArm*`, `H_RightArm*`, `H_above_Left*`, `H_above_Right*`,
   `Hoodie_Left`, `Hoodie_Right`, `H_hm_*`), matched by node-name prefix.
   Reported as `warn!` when the match set is empty or covers implausibly much
   of the body, so a re-export that renames bones fails loudly.

`client/src/anim.rs` (new) and `client/src/anim/rig.rs`

`rig.rs` does the graph work: walk the scene, pick the joint set (nodes
driven by a clip plus the ancestors needed to compose through — 62 nodes for
the girl, out of 1908), rewrite each primitive's vertices into model space,
merge primitives by material, and cut the arms viewmodel by bone-name prefix.
`anim.rs` holds clip decoding, sampling, role translation and the transition
graph.

```rust
pub enum ClipRole { Rest, Idle, Walk, WalkSlow, Run, TailSway, EmoteHands, Sit, SitAlt, RestAlt, Unknown(u16) }

pub struct AnimationClip {
    pub role: ClipRole,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,   // target joint, path, key times, values
    pub loop_: bool,
}

pub struct Animator {                 // one per animated instance
    pub current: PlayState,
    pub previous: Option<PlayState>,
    pub blend: f32,                    // 0..1 cross-fade progress
    pub speed: f32,
    pub time: f32,
}
```

Sampling: linear interpolation between keys, per-channel time (clips have
differing key counts per channel), quaternion channels normalized after lerp,
looping by `duration`. Rest-pose channels (nodes absent from a clip) fall back
to the clip's own rest value, so cross-fades do not snap.

`client/src/render/gpu.rs`

- `GltfGpuVertex` gains `joints: [u16; 4]` and `weights: [f32; 4]`.
- New storage buffer `gltf_palette_buffer_id`: bone matrices, laid out
  `[instance][joint]`, `MAX_SKINNED_INSTANCES × MAX_JOINTS × mat4`.
  Sizing: 62 joints × 64 B = 3 968 B per instance; a 32-instance cap is
  127 KiB. Unskinned models consume zero palette.
- `gltf_vs` applies `Σ w_i · palette[instance][joint_i] · pos` and rotates
  normals by the same weighted matrices. A `SKINNED` push-constant flag keeps
  the unskinned path byte-identical (one branch, uniform).
- `upload_gltf_writes` gains a palette write.

### Stage E — transitions

A transition is a `(from, to, strategy)` edge. `client/src/anim.rs` holds a
transition graph per clip role; unknown edges fall back to `CrossFade`.

```rust
pub struct Transition {
    pub duration: f32,
    pub easing: Easing,               // Linear, CubicInOut, ...
    pub mask: Mask,                   // LowerBody, UpperBody, Full
    pub scale_root_motion: bool,
}
```

Base graph:

| from → to | duration | mask | notes |
|---|---|---|---|
| `*` → `*` (default) | 0.25 s | Full | cubic in-out |
| `Rest` → `*` | 0.35 s | Full | rest is a held pose; longer reads better |
| `*` → `Rest` | 0.30 s | Full | |
| `Walk` ⇄ `Run` | 0.20 s | LowerBody | upper body keeps its current pose |
| `Idle` ⇄ `Walk` | 0.25 s | Full | |
| `Walk` ⇄ `WalkSlow` | 0.20 s | Full | same topology, only rate differs |
| `Sit` ⇄ `*` | 0.40 s | Full | sit is far from the standing set |
| any → `EmoteHands` | 0.15 s | Full | snap in fast, snap out slow (0.45 s) |

A masked transition blends the two poses **per joint** and then composes
once. Blending the finished bone matrices instead would drag the head halfway
toward the other clip even on a lower-body mask — which is exactly the thing
the mask exists to prevent.

**Strategies specific to this model**, all enabled because its rig has the
node layout to support them:

1. **Masked lower-body blending.** `Head_822`/`AllHead_823` and the hair chain
   are separate branches from `LeftLeg_52`/`RightLeg_97`. A lower-body mask
   therefore blends legs and leaves the head untouched — this is what makes
   walk→run readable instead of a full-body mush.
2. **Additive head look.** Because the head is an independent branch, the
   player's yaw/pitch is applied *on top of* whatever clip is playing, as an
   additive offset clamped to ±60° yaw / ±40° pitch, blended in over 0.15 s.
   The head then tracks where the player looks while the body keeps walking.
3. **Root-motion scaling.** `walk` carries 0.05 units of vertical root motion
   and `extra6` (= `run`) carries 0.181, which is how the classifier tells
   them apart without a readable name. Playback rate scales with the
   authoritative speed, so the footfall cadence matches how fast the entity
   is actually moving.
4. **Speed is authoritative.** The server measures horizontal speed per tick
   and smooths it; the client never differences its own interpolated
   positions. Position updates ride the unreliable channel, and a dropped
   one would make the walk/run blend flicker. The client and server use the
   same smoothing constant so the local player's hands and everyone else's
   avatars cross the threshold at the same moment.
5. **Additive tail.** `tail_sway` (clip 3) drives only `Tail`/`Tail2..6`. It is
   playable on its own (joints it does not drive fall back to rest) and is
   excluded from the transition graph — a zero-length transition, not a body
   state.

Not implemented: footfall phase lock. It wants the outgoing pose's foot
height sampled per phase, which is worth doing only once walk/run blending
looks wrong on screen.

### Stage F — players as entities, and hands

- `client/src/streaming.rs`: `RemotePlayer` becomes an entity mirroring
  `SpawnEntity`/`EntityStateUpdate`; `LocalPlayer` gets its own entity too, so
  first- and third-person share one code path.
- `client/src/render/avatars.rs`: the stone head is deleted. Remote players
  draw the girl model via the entity's `model` path, yaw-corrected (the avatar
  faces local +Z, the camera looks along −Z, so yaw is turned by π — the same
  correction already in `render_frame`).
- **Hands viewmodel**: a new pass after the world pass that
  - binds the arms-only merged primitive,
  - uses its **own depth buffer**, cleared at the start of the pass, so the
    hands can neither be clipped by world geometry nor occlude it,
  - projects with a **0.01 .. 0.6 range** instead of the world's
    0.1 .. 1000 — the world near plane is far too coarse for an arm held
    half a metre from the eye,
  - sits at its own contiguous instance range so the body draw does not put
    the local player in front of the camera,
  - is placed from the eye plus a fixed offset, and swings on use. The swing
    is additive on the **viewmodel instance matrix**, not the bone palette:
    the palette is shared with the body, so driving it from a mouse click
    would make other players' arms flail.
- `FrameData`/`GltfRenderTask` gain a second draw list; `sync_gltf_instances`
  fills it from `Animator` state.

## 4. Risks

| Risk | Mitigation |
|---|---|
| `girl.glb` is 1.9 MB and the pak ships whole models | already under `GLB_MAX_BYTES`; fine |
| Quaternion `matrix` decomposition drift | glTF matrices are affine; decompose, and unit-test round-trip on the real asset |
| Palette buffer growth if a model has many joints | hard `MAX_JOINTS` cap (256) with a loud `warn!`; girl needs 62 |
| 795 primitives merged → one 19 080-vertex draw | single-material merge is safe; multi-material keeps one primitive per material |
| Clip-role inference wrong for a future re-export | manifest is authoritative, classifier only proposes, mismatch is logged |
| Arms prefix match silently matches nothing | match coverage asserted at load, logged |

## 5. Task list

All done except where noted.

- [x] A1 packer: embedded-texture normalization to PNG, `CONVERTER_VERSION`
      bump. **Not** done: v1 clip synthesis — the only v1 sources are
      `dragon` and the old `girl`, neither of which is an avatar, and
      synthesizing clips for a model with no rig would produce a bone
      hierarchy nobody authored. Clips come from rigged assets instead.
- [x] B1 `EntityId`/`EntityKind`/`EntityState`, `SpawnEntity`/`DespawnEntity`/
      `EntityStateUpdate`, `PROTOCOL_ID` → 3
- [x] C1 `server/src/entities.rs` registry + spawn/despawn/broadcast
- [x] C2 players registered as entities (`PlayerMap` is now a view over it)
- [x] C3 block entities + `Container` + spill-on-break. **Partially**:
      `TickScheduled` is not implemented — nothing needs it yet, and
      `api.on_tick` already gives mods a tick hook.
- [x] C4 mod API (`spawn_entity`, `despawn_entity`, `set_entity_state`,
      `get_block_entity`, `list_entities`)
- [x] D1 node graph, TRS decomposition, joint set, static bake
- [x] D2 joint index in `GltfGpuVertex` + skinning in `gltf_vs`. Rigid
      binding means one influence, so the shader applies one matrix rather
      than a weighted sum.
- [x] D3 per-instance palette buffer + upload
- [x] D4 primitive merge (795 → 1 draw call)
- [x] E1 `anim.rs`: clip decode, sampler, `ClipRole` translation
- [x] E2 clip manifest. Landed as a Rust table (`builtin_roles`) rather than
      a `.toml` file, to avoid a new dependency for eleven entries; the
      parser for the documented file format exists and is tested.

## 7. Test coverage

The rig tests run against the real `girl.glb` rather than fixtures, because
every bug found while building this was a disagreement between the file and
an assumption about it:

- 795 primitives merge into one draw call with no vertices lost
- the rest-pose palette is the identity (a wrong `inv_bind` shows up here)
- the baked geometry reproduces the model's world extent
- joint parents always precede their children and the chain terminates
- every clip decodes with the measured duration and channel count
- the arms viewmodel is a small fraction of the body, not all of it
- transition masks leave the head alone, and full-body blends do not
- [x] E3 transition graph + masked/additive strategies
- [x] F1 client entity components replacing `RemotePlayer`
- [x] F2 stone-cube head → girl model
- [x] F3 arms viewmodel mesh + separate-depth pass
- [x] F4 first-person swing animation

## 6. Things that bit

Recorded because they are all easy to reintroduce:

- **The loader ignored the scene graph.** Meshes were read per-`document.
  meshes()` with no node transforms at all, so a model whose geometry lives
  across 795 nodes rendered collapsed at the origin. `anim::rig` walks the
  graph; there is a bounds test that fails loudly if a node transform goes
  missing.
- **Baking space.** Vertices are baked into *model* space (where the rest
  pose lives), which is what `palette[j] = posed_world · inverse(bind_world)`
  consumes. Baking into joint-local space silently squashes the model.
- **Joint ordering.** Parents must precede children in the joint array.
  Sorting by node depth guarantees it; a node's parent can have a *higher*
  index than the node (the girl's scene root is node 1907 of 1908), so a
  single ascending pass computing depth reads not-yet-computed values.
- **`inv_bind` must be composed exactly the way `compose_palette` composes
  at runtime.** Any other order desynchronises the two and the model comes
  apart in the rest pose.
- **WebP.** See §2.1.
- **Merging by material is unconditional.** The joint lives on the vertex,
  so one draw call can span any number of bones.
- **Draw ranges are exclusive.** A draw covers `first_instance ..
  first_instance + instance_count` wholesale, so first-person instances
  have to sit in their own contiguous range or the local player's body
  appears in front of the camera.
- **Palette stride is hardcoded as 256 in the shader** and must match
  `MAX_GLTF_JOINTS`; the two stages' push constants must agree byte for
  byte, so a per-model stride would need threading through both.