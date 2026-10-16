# Claude task brief — exsolis scripting phase 2 (full typed runtime)

## Repository
- Root: `/home/nestor/prj/exsolis`
- Working branch: `phase2/modding-api` (already includes merged phase-1 work)
- Default branch: `main` (remote origin/main)
- Build & test:
  - `cargo test -p tesjit` (currently 36 tests pass)
  - `cargo test --bin exsolis_server_bin` (currently 21 tests pass)
  - `cargo check -p tesjit`
  - `cargo check -p exsolis_server`
  - `just bench-scripting` (combined table)
- Note: may use existing subagents in OpenCode, but the brief assumes single-developer work; do not change `temp/bevy` (submodule) unless absolutely needed.

## Objective
Finish phase 2 of the scripting/modding design (`docs/SCRIPTING_ECS_MODS_PLAN.md`) so a TS mod can express typed ECS read/mutate through `Query<T>`/`Res<T>`/`Mut<T>`/`Commands` without per-type host calls, validate that a typed surface actually runs against live world data, then proceed to phase 3 + 4 to the extent possible.

## What is already shipped (do NOT redo)
- Type system: `i8..i64`, `u8..u64`, `f32`, `f64`, `bool`, `Entity`, `Option<T>`, `Tuple`, `Query<T>`, `Res<T>`, `Mut<T>`, `Commands`; UFCS method resolution in `check.rs`.
- `for-of` lowering over arrays with bind/writeback (tests `for_of_arrays_sum`, `for_of_arrays_mutate`).
- `host_fns()` pattern in `server/src/ts_backend.rs`, PullCtx with `world`, `bevy_world`, `components` mapping (currently typed by Player).
- `layout_hash()` on HostRegistry.
- Benchmarks: `crates/tesjit/benches/scripting.rs` with `bench_spike` (a_native, b_dynamic, c1, c1t, c2, d_player_rows) — 100k entities, ns/entity reported.
- Docs: `docs/SCRIPTING_ECS_MODS_PLAN.md` (Phases structure, §6 ForOf sketch, typed-query requirements).
- Demo mods: `mods/ts/{demo_option.ts, typed_rows.ts, component_access.ts, phase2_smoke.ts}`.

## Deliverables required before declaration of phase 2 done

### P2.1 — Place expression writeback
- HIR: reuse `PlaceRead/PlaceWrite/PlaceExpr` (Slot/Field/Index).
- Checker: lower `x = v` and `x.field = v` and `a[i] = v` as `PlaceWrite`.
- Codegen: store into slot, array element, or (later) component column; `ty` of PlaceRead is element type.
- Tests: unit test for `x.field = …` (where possible) and `arr[i] = v` already exercised.

### P2.2 — Typed component query runtime on Bevy world
- App build:
  - Register a thread-local/`TsRuntime` component map (`String -> u64` from `ComponentId::index()`) populated at least for `Player`, `PlayerPosition`. Store it in `TsRuntime` or `PullCtx.components`.
  - Raise an id as `DeclMix!`? No. Implement `declare_component_access::<T: Component>(name: &str)` generic helper usable in plugin build.
- Host ABI:
  - Add `component_access_id(name: string): i64` (already present) returning the id, with trap on missing.
  - Add `world_query_bytes(id: i64, field_offset: i64, type_size: i64, count: i64)`? Or better: a buffer handle pool owning `Vec<u8>` carved from grouped archetypes. For v1 a compaction copying the value bytes into an arena slab is enough, and the mod reads via existing generic helpers (`row_get_f32(index, offset)`, etc.).
  - Use `Field::offset` semantics already in `field_offset(type_name, field)` (from DWARF db, see `server/src/blocks.rs` uses). Emit TS-side constants for offsets & sizes at compile time — but checker only knows string names, so runtime-style is correct.
- TS mod surface (edit `mods/exsolis.d.ts`):
  - `declare function query_field_offset(ty: string, field: string): i32;`
  - `declare function query_type_size(ty: string): i32;`
  - Keep row-based generic `row_*` helpers untouched (backward compat).
- TS demo: `mods/ts/demo_query.ts` that typed-iterates a registered component's queried field via offset and sums into a log.
- Acceptance: a unit/integration test passes that loads `demo_query.ts`, handles a fake world with ≥1 `Player`, and reads back the right total without trapping.

### P2.3 — Run-time isolation rules
- Ensure `runtime.clear_trap()` and per-mod fuel reset stay as today.
- Do not expose mutable `&mut World` directly in host ABI (use `ScriptRequest`-style model already used in Lua).
- Separation of "queries" from "TS mod type-check": none of the type injection should require compiling against concrete Rust component names from TS; names are strings but are registered on the Rust side at app build time (this is a deliberate, documentable tradeoff consistent with `Entity`'s opaque handling).

### P2.4 — Documentation
- Append section in `docs/SCRIPTING_ECS_MODS_PLAN.md` for Phase 2 Typed surface (§4 extension): how a mod author passes type names for the typed helper, example.
- Update `mods/exsolis.d.ts` with the new declarations (see 2.2), keep existing Lua bridge design untouched.

## Phase 3 (optional this run but bounded scope)
- Add criterion group entries in `benches/scripting.rs` for array `for-of`; target < 3 ns/entity.
- Fuzz-style tests for index/bounds (10k randomized cases).
- Do not chase multi-backend Lua numbers (they live in isolated crates already).

## Phase 4 (if time remains after P2 + P3)
- Implement reject-on-layout-change in `ts_reload_system` via `host_registry.layout_hash()` and a previously-stored hash.
- Add test harness in `server/tests/` (or `ts_backend.rs` tests) that reloads a changed mod and asserts old build kept.

## Files to read first (in order)
1. `docs/SCRIPTING_ECS_MODS_PLAN.md` — phased plan + design decisions.
2. `crates/tesjit/src/hir.rs` — `Stmt`, `ExprKind`, `PlaceExpr`.
3. `crates/tesjit/src/check.rs` — lowering pipelines and scope entry points
4. `crates/tesjit/src/codegen.rs` — `stmt`, `arith`, `adapt`, `zero`, `convert`.
5. `crates/tesjit/src/ty.rs` — `Ty` machinery (`is_storable`, `elem_size`, etc.).
6. `crates/tesjit/src/jit.rs` — `runtime_signatures`, `get`, `register_region`.
7. `crates/tesjit/src/runtime.rs` — bindings, allocation, `arena`.
8. `server/src/ts_backend.rs` — `host_fns`, `PullCtx`, `ts_tick_system`, `compile_mod`, `run_guarded`.
9. `mods/exsolis.d.ts` — current typed surface (table in same file).
10. `crates/tesjit/benches/scripting.rs` — criterion benchmarks & spike numbers.
11. `tests.rs` (in `crates/tesjit/src/`) — test patterns.

## Critical rules
- Preserve Lua bridge (`server/src/scripting.rs`) — no change allowed in this slice (event-handler + ScriptRequest model stays untouched).
- Generic over type T means the host never exposes `player_x`-style wall; only indexed/generic runtime strings/Component ids.
- Every change must leave `cargo test -p tesjit` and `cargo test --bin exsolis_server_bin` green, with a focused integration test for each delivered surface.
- Commit at the end of each independently-verifiable sub-step (per §0 design doc: phase ends in testable, revertible deliverable).
- If a sub-task appears to require reshaping `JitBuilder` or `Compiled` APIs beyond additive changes, stop and explain; do not restructure undocumented.

## Evaluation when done
1. TS mod (`mods/ts/demo_query.ts`) compiles and runs an iteration that returns a logged number.
2. `cargo test --bin exsolis_server_ts_mod_compile` suite passes (and pre-existing tests still pass).
3. `cargo bench -p tesjit --bench scripting` produces the spike table without changes to baseline numbers.
4. The Rust-only `cargo check -p tesjit` + `cargo check -p exsolis_server` remain clean.

## Prior context the implementer (Claude) needs
- Prior conversation covered an Options runtime, UFCS registration, HIR scaffolds, Tx runtime helpers. These are in `crates/tesjit/src/*` and exercised by tests; prefer reusing them.
- Bench numbers in the doc should be interpreted as measured (real, not estimated).
- `Temp/bevy` is a submodule often modified locally; do not include those changes in commits.
EOF
echo "written"