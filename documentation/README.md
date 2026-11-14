# exsolis documentation

Everything about how the engine works, written for people who want to change it.

## Reading order

If you are new to the codebase, read these in order:

1. **[Architecture](architecture.md)** — the crates, the crates' responsibilities, and how data flows between them.
2. **[World, blocks and inventory](world-and-blocks.md)** — chunks, the block registry, world generation, and the authoritative inventory.
3. **[Networking](networking.md)** — the Renet transport, the two channels, and every message on the wire.
4. **[Rendering](rendering.md)** — the Vulkano pipeline, chunk meshing, the glTF loader, and animation.
5. **[Entities](entities-and-animation.md)** — the entity registry, entity kinds, and server-authoritative entity state.

## Modding

| Document | What it covers |
|---|---|
| **[TypeScript modding — how to use it](scripting/typescript-modding.md)** | Writing a TS mod: the language subset, the host API, typed ECS queries, resources, `Commands.spawn`, and worked examples. |
| **[TypeScript runtime — how it works](scripting/typescript-runtime.md)** | The compiler pipeline, the host ABI, the sandbox, hot reload, and the DWARF type-layout database. |
| **[Lua modding](lua-modding.md)** | The `api.*` surface, events, and the shared request translator. |
| **[Mod visual effects](mod-effects.md)** | Server-authored actions, client shader packages, and instance pooling. |

## UI

| Document | What it covers |
|---|---|
| **[XML UI — how to use it](ui/xml-ui.md)** | The XML tag/attribute vocabulary, the template mini-language, actions, and worked examples. |
| **[XML UI — how it works](ui/xml-ui-internals.md)** | The server renderer, the client parser, the entity tree, and the action round trip. |

## Operations

| Document | What it covers |
|---|---|
| **[Assets](assets.md)** | The `.pak` archive, the HTTP delivery protocol, caching, and hot reload. |
| **[Build and tooling](build-and-tooling.md)** | Build profiles, the `just` recipes, the export toolchain, and the benchmarks. |

## Reference material

- **[Blockbench skills](blockbench/)** — the Blockbench MCP skill docs the client loader was built against.
- **[Plans](plans/)** — design documents for work that is planned, spiked, or in flight:

| Plan | Status |
|---|---|
| [`SCRIPTING_ECS_MODS_PLAN.md`](plans/SCRIPTING_ECS_MODS_PLAN.md) | The design behind the TypeScript ECS layer, with the benchmark numbers. |
| [`ENTITIES_ANIMATION_PLAN.md`](plans/ENTITIES_ANIMATION_PLAN.md) | Implemented — entity system, block entities, animation, first-person viewmodel. |
| [`GLTF_BEVY_PARITY_PLAN.md`](plans/GLTF_BEVY_PARITY_PLAN.md) | The glTF loader parity work against Bevy. |
| [`GLTF_GOLDEN_TEST_PLAN.md`](plans/GLTF_GOLDEN_TEST_PLAN.md) | Golden-image tests for the glTF pipeline. |
| [`GLTF_INTEGRATION_PLAN.md`](plans/GLTF_INTEGRATION_PLAN.md) | The glTF import pipeline. |
| [`LEVEL_PARADIGM_PLAN.md`](plans/LEVEL_PARADIGM_PLAN.md) | The planned multi-level server architecture. |
| [`scroll_virtualization_plan.md`](plans/scroll_virtualization_plan.md) | Virtualized scroll lists for the creative grid. |
| [`phase2_brief.md`](plans/phase2_brief.md) | The task brief that drove the typed scripting runtime. |
| [`ASSET_SYSTEM_HANDOFF.md`](plans/ASSET_SYSTEM_HANDOFF.md) | Handoff notes for the asset system work in flight. |

## Conventions

- **The server is the authority.** The client never decides what is true; it renders what it is told and sends intent.
- **Everything crossing the mod boundary is a scalar.** No live `&mut World`, no `Entity`, no Rust type ever reaches a mod.
- **Client claims are requests, not facts.** Reach, inventory, block ids and UI payloads are all re-validated server-side.
- **Comments explain *why*.** The code says what it does; the comments say why it does it that way and what breaks otherwise.
