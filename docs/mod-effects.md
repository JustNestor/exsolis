# Mod visual effects

This document records the effect API direction and the first protocol slice.
The client runs no mod scripts. Server mods decide when an action happens;
clients route that action to a shader package and run it locally.

## Server API

The server Lua API exposes a broadcast helper:

```lua
api.on("keybind", function(ev)
    if ev.id == "example:fly" and ev.pressed then
        local position = api.get_player_pos(ev.player_id)
        if not position then return end
        api.broadcast_action("example:player_fly", {
            player_id = ev.player_id,
            position = position,
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

Compute passes can run for incoming actions and/or each frame. Each action is
packed into a temporary GPU storage range and dispatched immediately; the
shader can update package-owned persistent buffers. Graphics passes consume
those buffers. The temporary payload range and persistent simulation state
have separate lifetimes, so discarding the action input cannot erase live
particles.

Manifest buffers are package-owned storage allocations. The renderer places
them in a bounded arena and binds them to shader group 1 in declaration
order. Server-synced PNG files are uploaded by the existing texture loader
into group 0's bindless set, but version 1 does not yet provide effect
manifests with a stable way to name those texture handles. `images` is
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
  "frame_inputs": ["time", "delta_time", "view_projection", "model"],
  "actions": [{
    "id": "example:player_fly",
    "compute_pass": "spawn",
    "fields": [
      { "name": "strength", "source": "strength", "type": "f32" }
    ],
    "model_player_id": "player_id"
  }],
  "passes": [
    {
      "id": "spawn",
      "kind": "compute",
      "shader": "effects/fly.wgsl",
      "entry_point": "spawn",
      "workgroups": [128, 1, 1],
      "writes": ["particles"]
    },
    {
      "id": "simulate",
      "kind": "compute",
      "shader": "effects/fly.wgsl",
      "entry_point": "simulate",
      "run_each_frame": true,
      "workgroups": [128, 1, 1],
      "reads": ["particles"],
      "writes": ["particles"],
      "after": ["spawn"]
    },
    {
      "id": "draw",
      "kind": "graphics",
      "vertex": { "shader": "effects/fly.wgsl", "entry_point": "vs_main" },
      "fragment": { "shader": "effects/fly.wgsl", "entry_point": "fs_main" },
      "phase": "world_transparent",
      "blend": "additive",
      "depth": "test",
      "vertex_count": 6,
      "instance_count": 8192,
      "reads": ["particles"],
      "after": ["simulate"]
    }
  ],
  "buffers": [{ "id": "particles", "bytes": 262144, "persistent": true }],
  "images": []
}
```

`model_player_id` is a decimal string in the payload because JSON numbers
cannot safely represent every player ID. Payload fields use an explicit type
set (`i32`, `u32`, `f32`, vectors, and `mat4`); the client packs them with
WGSL storage-layout alignment and rejects missing or out-of-range values.
Manifest and resource sizes are bounded by the client. Version 1 can declare
frame inputs for frame index, time, delta time, viewport, camera position,
view-projection, model, and model-view-projection matrices. The renderer now
collects frame timing, camera position, player model transforms, and queued
actions into its per-frame data. The join handshake includes the receiving
player’s ID so model transforms can include both remote and local players.

The GPU ABI is now available at descriptor set `@group(1)`: binding 0 is the
packed action struct, binding 1 is the engine `FrameData` struct, and bindings
2 onward are the manifest's buffers in declaration order. `FrameData` has
`frame_index`, `time_seconds`, `delta_time`, `viewport: vec2<f32>`,
`camera_position: vec3<f32>`, and the `view_projection`, `model`, and
`model_view_projection` matrices. WGSL struct layout supplies the padding
between those fields. A manifest must request a frame input before the engine
fills that field. The action binding's WGSL struct fields must match its
manifest fields in order and type.

Graphics passes use procedural vertices (`vertex_index`) and manifest draw
counts. The executor runs compute nodes before world graphics in the Vulkano
graph, then runs graphics passes in their phase. Package buffers share a
16 MiB client arena with descriptor offsets aligned to the device's storage
buffer alignment. Persistent buffers are initialized once; transient buffers
are cleared each frame. Updated WGSL or manifests rebuild the package set
after the active frame completes.

An executable mod package is in
[`examples/mod-effects/fly`](/home/nestor/prj/exsolis/examples/mod-effects/fly).
Copy its `.fxm` and WGSL files into the server's asset catalog and run its Lua
handler as a server mod. The example spawns and simulates particles in compute,
then draws them as additive quads when players press Space.

Naga validates WGSL and generates SPIR-V per entry point and stage; the
manifest selects which compiled entry points become pipelines. naga-oil is a
dependency for a future reusable WGSL import/composition step, but imports
are not part of the version 1 loader yet. A replacement package becomes
active only after manifest validation, shader validation, SPIR-V generation,
and pipeline creation all succeed.

## Implemented and remaining

The code now includes the server action packet/API, client action event
boundary, Naga validation and per-entry-point SPIR-V generation, automatic
loading of `.fxm` manifests and `.wgsl` assets, manifest validation, and typed
payload packing into queued compute action events. The renderer builds
compute and procedural graphics pipelines, runs dispatches through the
Vulkano task graph, and supplies the declared frame matrices and timing.

The executor currently supports storage buffers and procedural graphics
passes. Manifests that declare images are rejected at pipeline-load time.
Effect texture lookup, independently scheduled task-graph passes, and
per-resource graph access declarations remain follow-up work. The fly
example uses only supported buffer resources and exercises the action,
compute, frame matrix, and draw path.
