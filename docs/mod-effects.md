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

A server asset package will declare action routes, payload field types, WGSL
entry points, render phases, and resource access. The client will match a
received action ID against those routes, validate and pack its payload, then
enqueue it for the declared compute pass. Unknown action IDs are ignored.
Packages contain data and WGSL, never client-side scripts.

WGSL will receive only declared engine resources. The proposed stable frame
binding includes frame index, elapsed time, delta time, viewport, camera
position, and view-projection matrix. Actions that identify an entity can
also request its model transform; the engine supplies model and model-view-
projection matrices rather than asking each mod to reconstruct them.

Compute passes can run for incoming actions and/or each frame. They read a
bounded action-event buffer and update package-owned persistent buffers.
Graphics passes consume those buffers. Event input and persistent simulation
state have separate lifetimes, so clearing consumed events cannot erase live
particles.

The manifest may declare buffers and images, but only through supported
resource kinds, formats, access modes, sizes, and lifetime policies. It can
declare pass dependencies; the engine translates those into Vulkano task
graph nodes and resource access declarations. It cannot inject arbitrary
Vulkano callbacks or bypass graph synchronization. Bindless texture handles
are logical asset references resolved by the client.

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
then draws them as additive billboards when players press Space.

Naga validates WGSL and generates SPIR-V per entry point and stage; the
manifest selects which compiled entry points become pipelines. naga-oil
composition can add reusable WGSL imports. A replacement package becomes
active only after manifest validation, shader validation, SPIR-V generation,
and pipeline creation all succeed.

## Implemented and remaining

The code now includes the server action packet/API, client action event
boundary, Naga validation and per-entry-point SPIR-V generation, automatic
loading of `.fxm` manifests and `.wgsl` assets, manifest validation, and typed
payload packing into queued compute action events. It also validates pass
dependencies and resource declarations.

The executor currently supports storage buffers and procedural graphics
passes. Manifest-declared images and per-pass image synchronization are not
implemented yet; manifests that declare images are rejected at pipeline-load
time. The graph currently exposes the shared storage arena to effect compute
and graphics nodes, so `reads` and `writes` are validated metadata and do not
yet narrow access per individual buffer. The fly example uses only supported
buffer resources and exercises the action, compute, frame matrix, and draw
path.
