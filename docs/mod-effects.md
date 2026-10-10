# Mod visual effects

This document records the effect API direction and the first protocol slice.
The client runs no mod scripts. Server mods decide when an action happens;
clients route that action to a shader package and run it locally.

## Server API

The server Lua API exposes a broadcast helper. Keybind event tables contain
`player_id`, `id`, `pressed`, and the current server position as
`position = { x, y, z }`:

```lua
api.on("keybind", function(ev)
    if ev.id == "example:fly" and ev.pressed then
        if not ev.position then return end
        api.broadcast_action("example:player_fly", {
            player_id = ev.player_id,
            position = ev.position,
            strength = 1.0,
        })
    end
end)
```

`broadcast_action` sends one reliable, ordered `ModAction` packet to all
players. Use `broadcast_action_unreliable` for frequent updates that can be
superseded by the next update. The action ID is namespaced (`namespace:name`). Payloads are JSON
objects with finite numbers, strings, booleans, nulls, arrays, and objects;
nesting, value count, action ID length, and encoded payload size are bounded.
Player IDs should be passed as strings because Lua numbers cannot exactly
represent every `u64`.

Key registration remains independent: a mod registers the keybind, and its
server-side `keybind` or `on_tick` handler decides whether to broadcast an
action. Actions need not be tied to a built-in game event.
`api.get_player_pos` remains a scoped pull helper for `api.on_tick`; event
handlers should use their event data instead of retaining a world-read
callback beyond that tick scope.

## Client package direction

A server asset package declares action routes, payload field types, WGSL entry
points, render phases, and resource access. The client matches a received
action ID against those routes, validates and packs its payload, then dispatches
the declared compute pass. Unknown action IDs are ignored. Packages contain
data and WGSL, never client-side scripts.

WGSL will receive only declared engine resources. The proposed stable frame
binding includes frame index, elapsed time, delta time, viewport, camera
position, and view-projection matrix. Actions that identify an entity can
also request its model transform; the engine supplies model and model-view-
projection matrices rather than asking each mod to reconstruct them.

Compute passes can run for incoming actions and/or each frame. Each incoming
action creates an independent effect instance with its own copy of every
declared buffer, its payload, and a snapshot of its requested player transform.
Compute and graphics passes use that instance's buffers, so repeated actions
and actions from different players can overlap without overwriting each other.
The manifest's `max_instances` bounds the pool (default 8, maximum 64), and
`instance_lifetime_seconds` controls when a slot becomes reusable (default 3
seconds). If the pool is full, the oldest instance is reused.

An action can also carry an `instance_key` (see [Keyed instances](#keyed-instances-and-action-modes)),
which turns the pool from "fire and forget" into addressable slots: `spawn`
with a key restarts the live instance with that key, `update` heartbeats it,
and `stop` fades it out.

Manifest buffers are package-owned storage allocations. The renderer places
them in a bounded arena and binds them to shader group 1 in declaration
order. Server-synced PNG files are uploaded by the existing texture loader
into group 0's bindless set; a manifest declares them in `textures`
(`{ "id": "spark", "path": "mods/fly/effects/spark.png" }`), and shaders index
them through the frame buffer's texture table
(`textures[frame.textures[i]]`, in declaration order, capacity 8). `images` is
reserved for GPU-owned images and is currently rejected when non-empty;
sampled, storage, and render-target image declarations are not executable
yet.

Pass `reads`, `writes`, and `after` are validated at load time, but do not
create one Vulkano task-graph node per pass or narrow access to individual
resources yet. Compute passes execute in the shared compute node before the
renderer node; graphics passes execute in the renderer node at their declared
phase. The task graph declares read/write access to the shared effect arena
for both nodes, and explicit shader-storage barriers separate dispatches and
draws. This is enough for the buffer-only fly example, but packages must not
depend on arbitrary cross-pass ordering beyond that schedule.

The first manifest format is JSON with the `.fxm` extension. An effect package
can include this `example:fly` manifest and the referenced WGSL files:

```json
{
  "format_version": 1,
  "id": "example:fly",
  "max_instances": 16,
  "instance_lifetime_seconds": 1.25,
  "frame_inputs": ["frame_index", "time", "delta_time", "viewport", "view_projection", "model"],
  "actions": [
    {
      "id": "example:player_fly",
      "compute_pass": "spawn",
      "fields": [
        { "name": "strength", "source": "strength", "type": "f32" },
        { "name": "position", "source": "position", "type": "vec3" }
      ],
      "model_player_id": "player_id"
    }
  ],
  "passes": [
    {
      "id": "spawn",
      "kind": "compute",
      "shader": "mods/fly/effects/fly.wgsl",
      "entry_point": "spawn",
      "workgroups": [128, 1, 1],
      "writes": ["particles"]
    },
    {
      "id": "simulate",
      "kind": "compute",
      "shader": "mods/fly/effects/fly.wgsl",
      "entry_point": "simulate",
      "run_each_frame": true,
      "workgroups": [128, 1, 1],
      "reads": ["particles"],
      "writes": ["particles", "alive", "args"],
      "after": ["spawn"]
    },
    {
      "id": "draw",
      "kind": "graphics",
      "vertex": { "shader": "mods/fly/effects/fly.wgsl", "entry_point": "vs_main" },
      "fragment": { "shader": "mods/fly/effects/fly.wgsl", "entry_point": "fs_main" },
      "phase": "world_transparent",
      "blend": "additive",
      "depth": "test",
      "vertex_count": 6,
      "instance_count": 8192,
      "indirect": { "buffer": "args", "offset": 0 },
      "reads": ["particles", "alive", "args"],
      "after": ["simulate"]
    }
  ],
  "buffers": [
    { "id": "particles", "bytes": 262144, "persistent": true },
    { "id": "alive", "bytes": 32768, "persistent": false },
    { "id": "args", "bytes": 16, "persistent": false }
  ],
  "images": [],
  "textures": [
    { "id": "spark", "path": "mods/fly/effects/spark.png" }
  ]
}
```

`model_player_id` is a decimal string in the payload because JSON numbers
cannot safely represent every player ID. For event positions that must match
across clients, include the server's world position in the payload and bind it
as a `vec3`; client-side model transforms are interpolated snapshots and can
differ slightly between observers. Payload fields use an explicit type set
(`i32`, `u32`, `f32`, vectors, and `mat4`); the client packs them with WGSL
storage-layout alignment and rejects missing or out-of-range values.
Manifest and resource sizes are bounded by the client. Version 1 can declare
frame inputs for frame index, time, delta time, viewport, camera position,
view-projection, model, and model-view-projection matrices. The renderer now
collects frame timing, camera position, player model transforms, and queued
actions into its per-frame data. The join handshake includes the receiving
player’s ID so model transforms can include both remote and local players.

Each instance receives the full declared buffer set. Its buffers are
initialized when the instance starts; persistent buffers then keep their
simulation state until that instance expires. Non-persistent buffers are
cleared before each active frame's compute passes.

The GPU ABI is now available at descriptor set `@group(1)`: binding 0 is the
packed action struct, binding 1 is the engine `FrameData` struct, binding 24 is
the engine world block buffer, and bindings 2 onward are the manifest's buffers
in declaration order. `FrameData` starts with `frame_index`, `time_seconds`,
`delta_time`, `viewport: vec2<f32>`, `camera_position: vec3<f32>`, and the
`view_projection`, `model`, and `model_view_projection` matrices; after that
classic 240-byte block come the bindless texture table
(`sampler_nearest`, `sampler_linear`, `texture_count`, `textures: array<u32, 8>`),
the per-instance `instance_age` / `stop_age` / `instance_slot`, and the world
window description (`world_dims`, `world_anchor`). WGSL struct layout supplies
the padding between those fields. A manifest must request a frame input before
the engine fills that field. The action binding's WGSL struct fields must match
its manifest fields in order and type.

## Engine WGSL imports

Mod shaders no longer copy engine ABI declarations. A line consisting of
`//!import <name>` is replaced at load time by the engine-owned declarations
below, so a struct that grows with the engine (the frame layout has grown
twice) needs no mod edits. Imports expand once per file and pull in what they
depend on; `enable ...;` lines must still come first.

| Import | What it declares |
|---|---|
| `engine/frame` | `FrameData` + `@group(1) @binding(1) var<storage, read> frame` |
| `engine/bindless` | `samplers` / `textures` binding arrays (group 0; needs `enable wgpu_binding_array;`) |
| `engine/world` | block data in a window of chunks around the camera (group 1, binding 24; needs `engine/frame`) |

The fly mod's shader opens with:

```wgsl
enable wgpu_binding_array;
//!import engine/frame
//!import engine/bindless
//!import engine/world
```

## Block data for effects (`engine/world`)

Effects run entirely on the GPU, so anything they should react to — ground to
bounce on, water to splash in — has to be there already. The client keeps a
window of chunks (`client/src/render/world_data.rs`) mirrored into one storage
buffer that every effect pass can bind: chunk tags followed by CHUNK_SIZE³
packed voxels (u16 per block: id in bits 0..8, solid/transparent/air flags in
bits 8..11). A chunk always lives in cell `c mod window`, so sliding the
window never moves data; only chunks that newly enter it are written, nearest
first, a bounded number per frame. The import exposes `world_block`,
`world_solid_at`, `world_ground_below`, and friends; anything not loaded reads
as `WORLD_UNLOADED`, so a spark flying into unstreamed terrain slides on
regardless.

## Keyed instances and action modes

`spawn` (the default) remains fire-and-forget. An action that instead names a
payload path in `instance_key` addresses a live instance: `spawn` with a key
restarts that instance instead of stacking a second one, `update` runs an
optional one-shot pass and extends its lifetime (a heartbeat), and `stop` runs
an optional one-shot pass, then keeps the instance alive for
`linger_seconds` of fade-out. `follow_model` (with `model_player_id`) keeps
refreshing the instance's model matrix from the player's transform every
frame, so a continuous effect rides along with its owner. Shaders read the
instance's age, the seconds since its stop (`stop_age`, -1 while running), and
its slot from the frame buffer.

The fly mod's `server.lua` demonstrates this with a keyed aura (`KeyG` toggles
it, `KeyH` boosts the running instance) alongside the original Space burst;
its `assets/aura.fxm` is the manifest for that keyed, follow-model effect.

## Indirect draws and dispatches

A pass may declare `indirect: { "buffer": <id>, "offset": <n> }` instead of
static draw counts or workgroups: graphics read a `DrawIndirectCommand`
(4 × u32) from that buffer, compute reads a `DispatchIndirectCommand` (3 × u32).
A non-persistent buffer is zeroed every frame, so a compute pass can compact
alive particles into a list and write the draw's `instance_count` with atomics,
with no counting pass and no reset pass. The engine declares
`INDIRECT_COMMAND_READ` access and `BufferUsage::INDIRECT_BUFFER` on the effect
arena for manifests that use it.

Graphics passes use procedural vertices (`vertex_index`) and manifest draw
counts. The executor runs compute nodes before world graphics in the Vulkano
graph, then runs graphics passes in their phase. Per-instance buffers share a
16 MiB client arena with descriptor offsets aligned to the device's storage
buffer alignment. Updated WGSL or manifests rebuild the package set after the
active frame completes.

The fly mod is installed and runnable in this checkout at
[`mods/fly`](/home/nestor/prj/exsolis/mods/fly). Its source package is in
[`examples/mod-effects/fly`](/home/nestor/prj/exsolis/examples/mod-effects/fly).
To install it on another server, copy the example directory's contents into
`mods/fly`. The server discovers Lua files recursively and scans each mod's
`assets/` directory recursively. It publishes `.fxm`, `.wgsl`, `.png`, `.xml`,
and `.json` files automatically; `.lua` files stay server-side. Mod asset paths
are namespaced as `mods/<mod-name>/...`, so the manifest references the shader
as `mods/fly/effects/fly.wgsl`. The client loads manifests and WGSL on asset
sync, then runs the fly burst when players press Space, and the keyed aura
when they press G (H boosts the running one). A mod with several manifests
(`effect.fxm`, `aura.fxm`) just lists each `.fxm` under its `assets/`.

From the workspace root, start the server and client in separate terminals:

```sh
cargo run -p exsolis_server --bin exsolis_server_bin
cargo run -p exsolis_client --bin exsolis
```

The default asset folder remains the source for shared game assets. Set
`EXSOLIS_MODS_DIR` to use a different server mod directory; the client asset
catalog remains controlled by `EXSOLIS_ASSET_DIR`.

Naga validates WGSL and generates SPIR-V per entry point and stage; the
manifest selects which compiled entry points become pipelines. Engine
`//!import` lines expand before parsing, so mod shaders keep no copy of the
engine ABI. A replacement package becomes active only after manifest
validation, shader validation, SPIR-V generation, and pipeline creation all
succeed.

## Implemented and remaining

The code now includes the server action packet/API, client action event
boundary, Naga validation and per-entry-point SPIR-V generation, automatic
loading of `.fxm` manifests and `.wgsl` assets, manifest validation, and typed
payload packing into queued compute action events. The renderer builds
compute and procedural graphics pipelines, runs dispatches through the
Vulkano task graph, and supplies the declared frame matrices and timing.

Keyed instance modes (spawn/update/stop with `instance_key`, `follow_model`,
`linger_seconds`), GPU-written indirect draw/dispatch arguments, the bindless
texture table, the engine WGSL imports, and the block-data world window are
all implemented and exercised by the fly mod (burst + keyed aura).

The executor still runs all compute passes in one shared compute node and all
graphics passes in the renderer node at their declared phase, rather than one
task-graph node per pass with per-resource access narrowing. Manifests that
declare images are rejected at pipeline-load time. Those remain follow-up
work, as does scheduling passes against individual buffers.
