# Architecture

exsolis is split into a **server** that owns the world and a **client** that draws it, with a shared crate for the wire contract. The server is authoritative for everything: blocks, positions, inventories, entities and UI. The client is a state-driven view.

## The workspace

```
exsolis/
├── common/     exsolis_common   — protocol, block registry, chunk packing, inventory
├── server/     exsolis_server   — the authoritative simulation
├── client/     exsolis_client   — the Vulkano renderer and the input/UI layer
└── crates/
    ├── fastpak/            — the .pak asset archive format
    ├── tesjit/             — the TypeScript → Cranelift JIT
    ├── tesjit-dwarf/       — the DWARF → types.tdb exporter
    └── tesjit-schema/      — the shared schema for the type database
```

### `common` — the contract

`exsolis_common` is deliberately small and deliberately shared. It holds everything both sides must agree on, and nothing either side could own alone:

| Item | Why it lives here |
|---|---|
| `ClientMessage` / `ServerMessage` | The wire protocol. Both sides encode and decode them. |
| `PROTOCOL_ID` | A mismatch aborts the connection instead of desyncing. |
| `BlockRegistry`, `BlockId`, `BlockDef` | Numeric ids go into chunk palettes on the wire; names go into scripts. |
| `ChunkBlocks`, `ChunkCoord`, `CHUNK_SIZE` | The chunk encoding and its geometry. |
| `Inventory`, `Slot`, `MAX_STACK` | The authoritative inventory format and its rules. |
| `Face`, `Facing` | Which side of a block was clicked, and placement orientation. |
| `EntityId`, `EntityKind`, `EntityState` | The entity model both sides mirror. |
| Limits (`UI_MAX_OPEN`, `MOD_ACTION_MAX_PAYLOAD_BYTES`, …) | Validation constants shared so the server and client agree on what "too big" means. |

The rule of thumb: **if a value appears in a message, it is defined here.**

### `server` — the authoritative simulation

The server is a Bevy app with a headless plugin set (`bevy_app`, `bevy_ecs`, `bevy_renet`) — no windowing, no rendering. Its plugins:

| Plugin | Responsibility |
|---|---|
| `net` | Renet server, the channel setup, and the single place packets are drained. |
| `events` | Fans incoming packets into per-feature request messages. |
| `chunks` | The chunk world, generation tasks, and per-player streaming. |
| `blocks` | The `ServerBlocks` registry resource. |
| `entities` | The entity registry, spawn/despawn, and entity state broadcast. |
| `interaction` | Server-authoritative place/break, reach checks, hotbar selection. |
| `movement` | Applies player input and broadcasts position. |
| `creative` | Creative-tab give and inventory sync. |
| `scripting` | The Lua bridge (one long-lived VM, event dispatch, request drain). |
| `ts_backend` / `ts_ecs` | The TypeScript JIT backend and its typed ECS layer. |
| `ui` | Server-side UI screens and the UI template renderer. |
| `assets` | The asset catalog, the Axum HTTP service, and hot reload. |

### `client` — the view

The client is a Bevy app with the **full** plugin set, but it does **not** use Bevy Render. The renderer is hand-written on Vulkano (`client/src/render/`), because the engine needs bindless textures, a custom mesh pipeline, and a separate UI pass with its own camera.

| Module | Responsibility |
|---|---|
| `render/allocator` | Sub-allocates the vertex/index buffers out of one big GPU allocation. |
| `render/meshing` | The 3-stage chunk mesher: invalidate → queue → poll → budgeted upload. |
| `render/gpu` | Pipelines, buffers, the task graph, and the frame submit. |
| `render/effects` / `effects_gpu` | Server-authored mod visual effects. |
| `render/ui_pass` | The UI pass, drawn after the world with its own camera. |
| `chunk` / `chunk_registry` / `streaming` | Client chunk storage and the receive side of streaming. |
| `gltf` / `gltf_utils` / `anim` | The glTF loader, skinning, and the animation clip layer. |
| `ui/` | The XML parser, the Bevy UI entity tree, and the action round trip. |
| `assets` | The persistent client-side cache and the download worker. |
| `player` / `interaction` / `net` | Input, block targeting, and the Renet client. |

### `crates/` — the reusable pieces

`fastpak`, `tesjit`, `tesjit-dwarf` and `tesjit-schema` are standalone crates with their own READMEs and their own test suites. They are not part of the game — they are the pieces of the game that would be useful elsewhere.

## The data flow

```
              ┌──────────────────────────────────────────────┐
              │                  SERVER                      │
  client ───▶ │  RenetServer ──▶ events::dispatch_incoming   │
              │        ▲                    │                │
              │        │                    ▼                │
              │        │        interaction / movement / …    │
              │        │                    │                │
              │        │                    ▼                │
              │        │            ChunkWorld, entities,    │
              │        │            inventories (authority)  │
              │        │                    │                │
              │        │            scripting ──▶ Lua / TS  │
              │        │                    │                │
              └────────┼────────────────────┼────────────────┘
                       │                    │
        Renet (UDP)   │                    │  Renet (UDP)
        reliable +    │                    ▼
        unreliable    │            ServerMessage → client
                       │
                       └── Axum HTTP (`:5001`) ──▶ asset .pak
```

**Every** server→client message goes through `net::OutgoingPacket` and is flushed centrally by `net.rs`. No gameplay system touches `RenetServer` directly. The same shape holds on the client: `net.rs` drains the socket into `IncomingPacket`, and every feature reads from that message stream.

That indirection is what makes the transport swappable and what keeps the feature modules honest — you can read any feature file and see exactly what it sends and receives.

## The tick

The server runs at **20 Hz** (`TICK_RATE` in `common`). One Bevy `Update` pass is one tick, and the systems are ordered explicitly:

1. `net::drain_incoming` — read the socket into `IncomingPacket`.
2. `events::dispatch_incoming` — route packets to feature requests.
3. Feature systems: movement, interaction, chunks, entities, creative, scripting.
4. `net::flush_outgoing` — write queued packets to the socket.

Chunk generation is the exception: it runs on `AsyncComputeTaskPool`, so a slow chunk never stalls the tick. Completed chunks are polled back in and streamed on a later tick.

## Authority rules

These are the invariants the whole codebase leans on:

- **Position.** The client sends an *input direction* and a *look angle*; the server integrates the movement and broadcasts the result. The client never sets its own authoritative position.
- **Blocks.** A place or break is a request. The server re-checks reach against its own record of where the player is, re-checks that the target chunk is loaded, re-checks that the target cell is air (or non-air), and re-checks the player's inventory. Only then does it mutate and broadcast.
- **Inventory.** The server owns every stack. The client renders what `InventorySync` says. Clicking a slot sends an action; the server resolves it and sends a new sync.
- **UI.** The server owns every screen. The client parses the XML it was given and reports clicks; it never decides what a button does.

## Where to start reading the code

| If you want to… | Start at |
|---|---|
| Understand the protocol | `common/src/lib.rs` |
| See how a packet becomes a world change | `server/src/events.rs` → `server/src/interaction.rs` |
| Understand chunk streaming | `server/src/chunks.rs` |
| Understand the renderer | `client/src/render/gpu.rs`, `client/src/render/meshing.rs` |
| Write a mod | [TypeScript modding](scripting/typescript-modding.md) or [Lua modding](lua-modding.md) |
| Understand the JIT | [TypeScript runtime](scripting/typescript-runtime.md) |
