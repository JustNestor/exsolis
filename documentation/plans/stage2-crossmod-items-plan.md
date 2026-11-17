# Stage 2 — Cross-mod types + Item entities + core hygiene

- **Status:** Part C done (partial C3), Part A done (with limitation), Part B pending
- **Progress log (2026-11-17):**
  - Part C `[x]` C1/C2 done: `HotbarSyncLog`/`send_inventory_sync`/`hotbar_diff`/`sync_new_inventories` moved to `server/src/inventory_sync.rs` (+ `InventorySyncPlugin`); `send_hud` moved to `ui.rs`. **C3 deliberately partial:** `CreativePlugin`, `handle_creative_give`, `give_stack`, `apply_inventory_clicks`, `sorted_block_names` remain in `creative.rs` — removing them now (before Lua removal + first authored bundle) would break inventory/creative flow. Revisit when the game layer exists.
  - Part A `[x]` done: tesjit parses `import type { T } from "source"` (`@ns/name` → `ns:Name`), binds local name to the host-registered layout; TOML `mods/ts/<id>.toml` with `dependencies` only (`server/src/mod_manifest.rs`), missing deps → hard error; `toml = 0.8` + `serde` added. **Limitation:** mod components must currently be registered on the Rust side (via `TsSchemaRegistry`/Bevy world) to be queryable — pure-TS dynamic component registration (`World::register_component` with a dynamic layout) is future work; `QueryPlan::link` looks up `ComponentId`, which only Bevy-registered components have.
- **Date:** 2026-11-17
- **Branch:** `plan/next-stage` (created from `main` after the change-detection merge)
- **Depends on:** change detection in `main` (`Added`/`Changed`, `world.get<T>`, `QuerySystem::finish_tick`)
- **Explicitly out of scope:** Lua removal (moved to `ideas-for-the-future.md`), persistence, HoverCard, HitContext/posttick, dropped-item physics rendering (that is stage 3)
- **Related:** [`ecs-items-and-abilities-plan.md`](ecs-items-and-abilities-plan.md), [`ideas-for-the-future.md`](../ideas-for-the-future.md)

---

## TL;DR

Three independent but connected pieces, in this order:

1. **Cross-mod type imports** — let one TS mod declare a component and another import its type and query it. Today `compile_mod` (`server/src/ts_backend.rs:1697`) pushes every registered engine layout into `JitBuilder` via `builder.component(name, layout)`, and tesjit's checker has **no import handling at all** (`crates/tesjit/src/check.rs`). This is the blocker: without it, mod-defined components cannot be referenced by name from another mod.
2. **Item entities** — replace `Slot { item: BlockId, amount: u8 }` in a fixed array (`common/src/inventory.rs:21,29`) with ECS entities carrying components. Inventory/container slots hold entity handles. Wire messages change from `(String, u8)` tuples to a structured payload.
3. **creative.rs extraction** — split the misnamed module into a neutral inventory-sync contract, shared inventory mechanics, and game-layer scripts (see `ideas-for-the-future.md`). Do this **before** the item refactor so the new `ItemStack` does not inherit the wrong home.

**Checkbox legend:** `[x]` already in `main` · `[ ]` to implement.

---

## 0. Decisions already made (do not re-litigate)

- Types are still **hand-written** (`mods/exsolis.d.ts`), not generated from DWARF. `exsolis.generated.d.ts` is produced by `ComponentRegistry::write_declarations` (`server/src/ts_ecs.rs:297`) and is a *curated engine layout dump*; it stays.
- No manual change labels (`mark_added`/`mark_changed`) — a change means a real write or a real insert/remove.
- `Added<T>`/`Changed<T>` work both inline in the row tuple and in Query's filter position (`crates/tesjit/src/check.rs::desugar_query_row`).
- `world.get<T>(entity)` returns `Option<T>`; the host fn is `__tesjit_world_get` (`server/src/ts_backend.rs`).
- Lua stays for now — removal is queued in `ideas-for-the-future.md`.
- Block-entity persistence is deferred.
- One stage, detailed, with exit tests that another agent can run.

---

## Part C — creative.rs extraction (do this FIRST; it touches the same files as the item refactor)

The reason to go first: `creative.rs` currently owns `HotbarSyncLog`, `send_inventory_sync`, `apply_inventory_clicks` and `give_stack` — all of which the item refactor must change. Extracting them now means the item work edits neutral modules instead of a module named "creative".

- [ ] **C1 — Neutral inventory-sync module.** Create `server/src/inventory_sync.rs`. Move from `creative.rs`:
  - `HotbarSyncLog` (`creative.rs:29`), `HotbarSync` type alias (`:35`), `impl HotbarSyncLog` (`:37`),
  - `send_inventory_sync` (`:86`), `hotbar_diff` (`:141`), `sync_new_inventories` (`:285`).
  - Re-export or `pub(crate)` them so `interaction.rs`, `events.rs`, `scripting.rs`, `ts_backend.rs` switch imports from `crate::creative::…` to `crate::inventory_sync::…`.
  - `common/src/lib.rs` already carries `InventorySyncDiff` — no protocol change needed here.
- [ ] **C2 — Shared inventory mechanics.** Move `give_stack` (`creative.rs:221`) and `apply_inventory_clicks` (`:320`) into `interaction.rs` (or a new `inventory mechanics` module). These are game-agnostic; creative mode is not their owner.
- [ ] **C3 — Remove the game layer from the core.** Delete from `server/src/creative.rs` and `main.rs`:
  - `CreativePlugin` (`creative.rs:22`, registered at `main.rs:150`),
  - `handle_creative_give` (`:228`), `send_hud` (`:159`), `sorted_block_names` (`:211`),
  - `CreativeGiveRequested` message and its bridge in `events.rs`.
  - Keep `ClientMessage::CreativeGive` in the protocol only if stage-3 scripts need it; otherwise delete the variant from `common/src/lib.rs` and bump `PROTOCOL_ID`.
  - The first authored bundle reimplements this as TS systems + XML. That is **not** part of this stage — just make the core stop knowing about it.
- [ ] **C4 — Verify.** `cargo check -p exsolis_server`, `cargo test --bin exsolis_server_bin`. No `creative::` imports remain in `events.rs`, `interaction.rs`, `scripting.rs`, `ts_backend.rs`.

**Exit test for Part C:** `grep -rn "crate::creative" server/src` returns only matches inside `creative.rs` itself (which should now contain only `mod` scaffolding or be deleted). Inventory give/click/sync tests still pass from their new home.

---

## Part A — Cross-mod type imports

### A.0 Current mechanism (verified)

- `compile_mod(path, registry)` (`ts_backend.rs:1697`) builds a `JitBuilder`, calls `builder.host_fn_ref(f)` for each host fn and `builder.component(name, layout)` for each layout in `ComponentRegistry` (`ts_ecs.rs:154`, populated from the Bevy world plus `TsSchemaRegistry` in `from_world_with_schemas`, `ts_ecs.rs:218`).
- `ComponentRegistry::write_declarations` (`ts_ecs.rs:297`) emits `exsolis.generated.d.ts` for engine components.
- `check.rs` resolves type names via `ty_of_x` → `host.component(...)` or a local type alias (`check.rs:2201`). **`import` statements are currently ignored** — there are no `ImportDeclaration` handling branches in `check.rs`.
- `JitBuilder::component` registers one named layout into the checker's host registry. There is one flat namespace; no ownership, no versioning.

### A.1 Mod manifest and type packages

- [ ] **A1.1** The manifest is **TOML and contains only dependencies** — no id, version, type path or ownership fields (those belong to the type package / conventions). One file per mod: `mods/ts/<id>.toml`, where `<id>` is the file stem of the matching `*.ts`:
  ```toml
  # mods/ts/mymod.toml
  dependencies = ["mymod-core", "exsolis-blocks"]
  ```
  A missing manifest means "no dependencies" — legal. Add `toml` as a workspace dependency (not currently present — checked `Cargo.toml`).
- [ ] **A1.2** `mods/types/<id>.d.ts` is hand-written by the mod author (consistent with "types by hand"). Its path is a **convention** keyed by `<id>`, not a manifest field. It declares the exact `interface`s for the components this mod exports. The host validates declared layouts against runtime registrations (A.4).
- [ ] **A1.3** Component ownership is derived from `<id>` + the declarations in the type package, not from a manifest `owns` list. Today `lower_type_alias` registers `Component<T>` aliases (`check.rs:122`) but only in `TypeScope`, not in the Bevy world — see A.4.

### A.2 Compile-time import resolution

- [ ] **A2.1** In `check.rs`, handle `Statement::ImportDeclaration` in pass 0. Recognise:
  ```ts
  import type { Lightning } from "@mymod/lightning";
  ```
  The specifier `@<id>/<component>` maps to a component name `<id>:<Component>`. Resolve by looking up the imported component layout in a new import scope passed to `Lowerer`/`TypeScope`.
- [ ] **A2.2** Merge imported layouts into `ty_of_x` resolution: an imported `Lightning` becomes `Ty::Struct(Arc<StructLayout>)` exactly like `host.component("mymod:Lightning")` would. No new `Ty` variant needed — imports are a *resolution source*, not a new type.
- [ ] **A2.3** Emit a clear diagnostic for unknown imports, non-`import type` imports (value imports are not supported — JIT modules are isolated), and duplicate local names. Add tests in `crates/tesjit/src/tests.rs` following the existing query-filter test at `tests.rs:122`.

### A.3 Host wiring

- [ ] **A3.1** Before compiling a mod, load all `*.toml` manifests (TOML, dependencies only) plus the conventional `mods/types/<id>.d.ts` files, build `name → layout` for every exported component, and pass them to `JitBuilder` via `builder.component(...)`. This is the same call path as engine components (`ts_backend.rs:1713`) — just a second source.
- [ ] **A3.2** `compile_mod` signature takes an extra `&ImportRegistry` (or extend `ComponentRegistry` with `mod_components`). Keep tests compiling: update every `compile_mod`/`reload_with_registry` call site (`ts_backend.rs:1227,1889,2354,2397,2513,2610,2696,3390,3508`).
- [ ] **A3.3** Hot reload: when a dependency's `types/*.d.ts` or manifest changes, **recompile dependants** — the current stamp check (`ts_backend.rs:1239-1267`) only watches each mod's own `*.ts` file. Track a manifest/type stamp and invalidate dependants transitively.

### A.4 Layout safety, ownership, dependencies (no typedb)

- [ ] **A4.1** Validate each declared layout: `size`, `align`, and every field offset from the `.d.ts` against the runtime registration. Reuse the existing check pattern in `TsSchemaRegistry::register` (`ts_ecs.rs:97`) and the "layout changed since compilation" checks in `QueryPlan::link` (`ts_ecs.rs:562,603,640`).
- [ ] **A4.2** Namespace ownership: only the mod whose `<id>` (file stem) matches the prefix may register `mymod:*`. A second mod trying to register or shadow the same name is a hard load error.
- [ ] **A4.3** Dependency check: the TOML manifest lists only dependency **ids** (no version ranges for now — versioning is future work, see `ideas-for-the-future.md`). Hard-fail with a readable message on a missing dependency. No silent empty queries.
- [ ] **A4.4** Layout mismatch or missing dependency → the mod keeps its last good build, logs diagnostics (same policy as `reload_with_registry`'s failure handling, `ts_backend.rs:1248`).

### A.5 Modification sites summary (for the executing agent)

| File | What changes |
|---|---|
| `crates/tesjit/src/check.rs` | pass-0 import handling, `ty_of_x` resolution from imports, diagnostics |
| `crates/tesjit/src/tests.rs` | import parsing tests |
| `server/src/ts_backend.rs` | manifest loader, `compile_mod` extra param, dependency invalidation, hot reload |
| `server/src/ts_ecs.rs` | extend `ComponentRegistry` or new `ImportRegistry` with validation helpers |
| `mods/exsolis.d.ts` | document `import type` usage |
| `Cargo.toml` | add `toml` workspace dependency (no `semver` — versions are future work) |

**Exit test for Part A:** Write two temporary TS mods in a test: `provider.ts` declares `export type Lightning = Component<{ level: i32 }>` (or equivalent) and a manifest owning `mymod:Lightning`; `consumer.ts` does `import type { Lightning } from "@mymod/lightning"` and has `on_tick(q: Query<[Entity, Lightning]>)`. Both compile and link. Remove the provider → consumer fails with "missing dependency", no silent success. Change the `.d.ts` field type → consumer fails with a layout mismatch before any JIT code runs.

---

## Part B — Item entities

### B.0 Current model (verified)

- `Slot { item: BlockId, amount: u8 }`, `MAX_STACK = 64`, `INVENTORY_SIZE = 36` — `common/src/inventory.rs:13,18,21,29`.
- `Inventory` is `[Option<Slot>; 36]` + `selected` + `held`, with `add_item`, `remove_one`, `click`, `absorb`, `stash_held`, `grab` (`:77-236`).
- Wire: `InventorySync { slots: Vec<Option<(String,u8)>>, selected }` and `InventorySyncDiff { changes: Vec<(u8, Option<(String,u8)>)>, selected }` — `common/src/lib.rs:410,433`.
- Block entities use `Container` (`server/src/main.rs:65`) wrapping the same `Inventory`.
- `EntityKind::Item` exists in the protocol (`common/src/lib.rs:148,155`) but no gameplay path spawns it.

### B.1 Item identity

- [ ] **B1.1** Introduce `ItemId = u16` (or a `String` name type) in `common`, with a data-driven registry mirroring `BlockRegistry` (`common/src/blocks.rs:119`). Names are `namespace:item` (e.g. `exsolis:stone`, `mymod:lightning_charm`). For v1, seed it from the block registry: each placeable block gets a same-named item; non-placeable entries (`BlockDef.placeable = false`, `common/src/blocks.rs:29`) become item-only entries.
- [ ] **B1.2** `ItemDef { name, stack: u8 (max), place_as: Option<BlockId>, model: Option<String> }`. The `place_as` link is what lets placement/breaking reuse existing block logic without an enum match.
- [ ] **B1.3** Both sides build the same default registry (same approach as `BlockRegistry::default_registry()`); ids are deterministic. Bump `PROTOCOL_ID` (`common/src/lib.rs:17`).

### B.2 Stack entity and component bundle

- [ ] **B2.1** New `ItemStack { item: ItemId, amount: u8 }` component on an ECS entity. Add `InInventory { owner: Entity, slot: u8 }` (or Bevy 0.19 relationship `#[Relationship]`) to mark membership. A dropped item is the same entity with `InInventory` removed and a `Transform`/state inserted.
- [ ] **B2.2** `Inventory` holds `[Option<Entity>; 36]` + `held: Option<Entity>` + `selected`. All existing methods get entity-handle semantics:
  - `add_item` creates a new stack entity or grows an existing one (needs `Commands`/world access — so `Inventory` methods that mutate now require `&mut World`; plan the signature change carefully, see risk R2).
  - `remove_one` decrements and despawns at zero.
  - `click` moves handles between slots/held.
  - `absorb` moves handles from a container into the player.
- [ ] **B2.3** Stackability rule: two stacks merge iff same `ItemId` **and** equal component sets excluding derived/identity components. Add `components_equal` helper with an opt-in equality registration for unknown components; unknown → unstackable (safe default).
- [ ] **B2.4** TS visibility: register `ItemStack`/`InInventory` as engine components through `TsSchemaRegistry` (`ts_ecs.rs:121`) so mods can query them with `Query<[Entity, ItemStack]>`. This is the first real *game* use of the cross-mod type system.

### B.3 Wire protocol

- [ ] **B3.1** Replace tuple payloads with a struct so new fields can be `#[serde(default)]`:
  ```rust
  pub struct HotbarSlot { pub item: String, pub amount: u8, pub glint: bool }
  ```
  `InventorySync { slots: Vec<Option<HotbarSlot>>, selected: u8 }`, `InventorySyncDiff { changes: Vec<(u8, Option<HotbarSlot>)>, selected: Option<u8> }`. Append-only: add **new message variants** rather than changing existing tuple arities where possible.
- [ ] **B3.2** `SpawnEntity` gains `item: Option<ItemStackSnapshot>` for `EntityKind::Item` (appended field with `#[serde(default)]`).
- [ ] **B3.3** Client: update inventory HUD/creative rendering (`client/src/ui/mod.rs`, `client/src/entities.rs`) to consume the new payload. Client never resolves attributes — presentation only.

### B.4 Dropped items (minimum for this stage)

- [ ] **B4.1** Extend `sync_player_entities` (`server/src/entities.rs:352`) to tick `EntityKind::Item`: gravity + ground snap against `ChunkWorld`, broadcast `EntityStateUpdate` for moving items.
- [ ] **B4.2** Replace every "dropped with a warning" path with an item-entity spawn: creative give overflow, `stash_held`, break overflow, container break spill. Grep for `did not fit` / `dropped` in `server/src/creative.rs`, `scripting.rs`, `common/src/inventory.rs`.
- [ ] **B4.3** TTL despawn + merge on contact (reuse B2.3). Pickup radius → re-insert `InInventory`.

### B.5 Tests

- [ ] **B5.1** Inventory unit tests in `common/src/inventory.rs` ported to entity handles (the existing `click_pickup_and_drop`, `click_right_splits_and_places_single`, etc. at `:244-292`).
- [ ] **B5.2** Server integration test: apply a prefab to a stack entity, verify `Query<[Entity, Added<Lightning>]>` writes derived component, verify stack merges only when components equal.
- [ ] **B5.3** Wire round-trip test: `InventorySyncDiff` with new struct payload decodes on both sides.

**Exit test for Part B:** Place a block from the hotbar → a stack entity is created/moved with its components intact; break it → a dropped item entity appears with the same components; pick it up → `InInventory` restored; overflow spawns an item entity instead of vanishing; no `Slot` in `common/src/inventory.rs`.

---

## Execution order and dependencies

```
Part C (creative.rs)     ← do first; touches the same files as B
     ↓
Part A (cross-mod types) ← independent of B; can be parallelised after C
     ↓
Part B (item entities)   ← uses A for TS queries, uses C's neutral modules
```

Do **not** start B before C: the item refactor will edit `send_inventory_sync`/`apply_inventory_clicks`, and those must already live in neutral modules.

Do **not** start B.2 before A.3: mods need to be able to name `ItemStack`/`InInventory` from TS, otherwise the first authored bundle cannot react to items.

---

## Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | `Inventory` methods currently pure; entity handles require `&mut World` | Split into pure handle-manipulation (slot moves) and world-owning helpers (spawn/despawn); keep as much pure logic as possible for tests |
| R2 | Cross-mod imports change JIT linking for every mod | Landing A first with two test mods proves the path before real content depends on it |
| R3 | Wire change breaks client | Bump `PROTOCOL_ID`, new message variants, client updated in same commit |
| R4 | ItemId registry duplicates BlockRegistry | Generate item entries from blocks at startup; hand-written additions only for item-only content |
| R5 | Hot-reload stamp check misses type-package changes | A3.3 adds transitive invalidation; test with two dependent mods |
| R6 | `MAX_ENTITIES = 4096` reached faster with stack entities | Log + cap already exists (`server/src/entities.rs:36`); monitor in B4 tests |
| R7 | Hand-written layout mismatch between mods | A4.1 validation before any JIT compile; reuse `layout_hash` pattern (`ts_ecs.rs:160`) |

---

## Ground rules

- No central string-keyed attribute registry, no JSON-defined behaviour — components + systems only (from `ecs-items-and-abilities-plan.md`).
- New wire enums are append-only; new message fields get `#[serde(default)]`; bump `PROTOCOL_ID`.
- Client never resolves attributes or trusts mod component bytes.
- When a decision changes, edit this plan and the linked docs, and note the date.
- Lua removal, persistence, HoverCard, HitContext/posttick stay in `ideas-for-the-future.md` / the main plan — not here.
