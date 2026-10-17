# Scripting with ECS for flexible mods — design document

Status: **implementation approved and underway**. The tesjit query-loop
foundation, Bevy table queries, and typed `Res<T>`/`Mut<T>` system parameters
are implemented, including `With<T>` and `Without<T>` query filters.
`Commands.spawn` with typed component values is implemented and defers entity
insertion until all mods finish the tick. The broader `Commands` API, planned
builder API, and performance/fuzz acceptance work remain outstanding. The
initial runtime currently imports component layouts from `types.tdb`; replacing
that runtime dependency is a planned migration (see §11).

Sources of truth for this document:

- benchmark data: `crates/tesjit/benches/scripting.rs` (criterion, 100k entities)
- isolated Lua backend numbers: `crates/lua-bench-{luajit,luau,lua54}/`, `crates/bms-bench/`
- tesjit type/check/codegen/jit: `crates/tesjit/src/{ty,check,codegen,hir,jit,runtime,host}.rs`
- existing server scripting bridge: `server/src/scripting.rs` (Lua, kept unchanged in v1)
- hot-reload policy decision: reject-on-layout-change (see §7)

---

## 0. Conclusion of the spike (step 2)

`crates/tesjit/benches/scripting.rs :: bench_spike`, 100k entities,
`x += v * dt` per step, release build. All ten combinations agreed to four
significant figures. **ns/entity** (per step):

| label | approach | ns/entity | × vs raw |
|---|---|---:|---:|
| c1 | unsafe Rust over column slices | 0.84 | 1.00 |
| c2 | hand-written Cranelift kernel over column pointers | 0.84 | 1.00 |
| a | native Bevy query (`(&mut Pos, &Vel)`) | 1.65 | 1.96 |
| c1t | c1 + per-row changed-tick store | 1.20 | 1.43 |
| b | Bevy dynamic (`FilteredEntityMut` + `get_by_id`) | 18.22 | 21.69 |
| d | PlayerRow (snapshot + tesjit region loop + write-back) | 25.38 | 30.21 |

Notes:

- Bevy's native query is within ~2× of the raw column ceiling. The gap is
  archetypes/changed-tick bookkeeping; c1t measures +0.36 ns/row of that.
- Dynamic `FilteredEntityMut` is **21×** the raw ceiling — reflection
  dispatch, not language overhead — and out of the same ballpark as Lua's
  BevyModScripting (~8–15 µs/row ≈ 10000× the ceiling). That is the
  measured cost of the "generic access via name lookup" model this design
  deliberately avoids.
- The Cranelift kernel matches the unsafe-Rust loop byte for byte, so
  codegen will not be the bottleneck of the chosen design: the column-ptr
  kernel is the lowering target for `for-of` over a query (see §6).
- PlayerRow snapshots and rejoins every step, so even with a static layout
  it is the worst option; it is retained only as a bridge for mods that
  need it (see §8) and never for the `for-of` path.

Decision taken from these numbers: **Bevy query column layout is the v1
sweet spot** — do not build a separate scripting-owned index; land matches
this, and reject layout-changing hot reload (§7).

---

## 1. Goals and non-goals

### Goals
- Mods in TS get a typed, generic, programmable view of the ECS world:
  `world.resource(Res<T>)`, `world.query().with<T>().without<U>().build()`,
  destructuring assignments, component writes.
- No per-type / per-field host functions: the host exports `resource()` /
  `query()` etc. with the component identity expressed through generic type
  arguments or `ComponentId` values; the type system carries the T.
- No mod-level mutable state: all state lives in components / resources on
  the world. Mods are pure functions of (events, query result, resource).
- Fuel limits; no host panic or UB from any mod behaviour (range-checked
  literals, bounds-checked array access, registerable bounds). Bad bounds and
  Fuel trap into a structured trap; the VM unit keeps its shape.
- Hot reload rejects layout-incompatible builds and keeps the last-good build
  running, degrading gracefully (no world corruption).
- The Lua bridge (`server/src/scripting.rs`) is untouched: it keeps its
  event-handler + `ScriptRequest` model. TS and Lua coexist.

### Non-goals (v1)
- JIT-compiled arbitrary dispatch to *unknown at compile time* component
  types is left to v2: the type system fixes the concrete `T` per mod unit,
  so `Query<T>` is monomorphized at compile time.
- No embedded Lua runtime inside tesjit; each backend stays in its own
  isolated crate (`links = "lua"`).
- No mod-level storage: no static variables, no module-scope `let`s.
- No custom ECS scheduler hooks from mods; mods run inside a single
  user-provided host hook (one tick callback).
- No garbage-collected `Entity` lifetime management from mod code: `Entity`
  is a plain 64-bit handle; validity is host's problem to assert.

---

## 2. Architecture decisions (locked)

1. **TS-only v1.** Lua keeps its existing event-handler + `ScriptRequest`
   model (see `server/src/scripting.rs`). This design targets tesjit on the
   mod side only. The host surface (`ScriptRequest`, `OutgoingPacket`, event
   dispatch) is unchanged.
2. **Criterion spike extension, not a separate binary.** The spike landed in
   the existing `crates/tesjit/benches/scripting.rs`, exposing the
   `bench_spike` function. Combined output comes from `just bench-scripting`.
3. **Reject-on-layout-change hot reload** (§7).
4. **Isolated Lua backends.** mlua-sys `links = "lua"` forbids two backends
   in one cargo graph; each backend is its own crate excluded from the main
   workspace. Host surface is shared across all three runners via the BMS
   header, but the workloads agree on names.
5. **Host surface is generic.** Host functions take values (`ComponentId`,
   `Entity`, `usize`, heap pointers returned as `u64`); every concrete
   component type is a **compile-time** `T`, never a runtime name. See §4.

---

## 3. Type system spec

### 3.1. Primitive types (already implemented in `crates/tesjit/src/ty.rs`)

Integer widths `i8`–`i64` / `u8`–`u64`, `f32`, `f64`, `boolean`, opaque
`Entity` handles, `string`, `T[]`, `void`. Integer ranges enforced at lowering,
`as` casts carry widening/narrowing/conversion per the checker's matrix
(see `check.rs` lowering and `codegen.rs::convert()` and
`sized_integers_wrap_and_compare`, `float_widths_and_casts`,
`entity_handles_compare_and_cast` tests).

### 3.2. `Option<T>` (to be added)

- `Ty::Option(Box<Ty>)`. At the machine level it is an `enum { tag: i8,
  value: T }` — for reference-ish T reuse T's space with a sentinel; for
  value types a tagged union. For the JIT we lower it as a single i64 slot
  holding T with a tag word, following the existing `ArrNew`/`ArrPush`
  calling convention precedent (compile-time layout, no dynamic tagging).
- Standard constructors: `some(x)`, `none<T>()`. Pattern matching via
  `isSome()` / `unwrapOr(default)`. No null pointer dereferences — `unwrap` on
  `None` traps.
- Acceptance: `Option<f32>` and `Option<Entity>` returnable from host
  functions as a single word + tag.

### 3.3. References and place expressions (to be added)

- A place expression (`Ref<T>`) is a *storage location*, not a value. TS
  surfaces: indexing `arr[i]`, field access `row.x`, dereference.
- Distinction is between *value reads* (copy) and *place writes* (mutation).
  Today's `Assign { local, op, value }` handles slots; we extend the HIR with
  `PlaceWrite { place: PlaceExpr, op, value }` and `PlaceRead { place }`.
- `PlaceExpr` is one of: `Slot(usize)` (today, bare local), `Field { base,
  field: &'static str }`, `Index { base, index }`. Codegen for Field/Index
  lowers to a typed pointer with the correct stride. For `Array<T>`, stride
  is `elem_size`.
- Type of `place_read` is `T`; type of `place_write` is `void`.
- No general pointer aliasing is exposed; placed values are checked at lower
  time. This is what keeps codegen free of runtime type information.
- The `&mut` borrowed from a `PlaceWrite` is transient (an instruction
  sequence), not a value — no dangling references across control-flow.

### 3.4. `Query<Q, F>`, `Res<T>`, `Commands` (signature-only types)

These types exist *only* as parameter/return annotations on host functions
and exports. They never materialise as values in JIT code. The host inspects
the function signature at registration to derive a *static* `FilteredAccess`
(as in bevy's `QueryBuilder`) and asserts it against the `run_on_tick` system
schedule.

Representation:
- `Query<T, F>` where T is a tuple `(Comp...)` or single component and F is an
  optional tuple of `With<Comp>` / `Without<Comp>` filters. Query terms and
  filters currently require table storage; filter-only zero-sized marker
  components are supported. Archetype matching is cached as new archetypes
  appear.
- Static field access (e.g. `q.iter().next().x`) is lowered against tuple
  indices.
- `Res<T>` — read-only resource handle.
- `Mut<T>` is the mutable variant of `Res`.
- `Commands` — append-only command buffer; currently supports
  `spawn(component1, component2, ...)` for registered components with supported
  layouts. Typed object literals initialize components, and spawned entities
  become visible on the next tick after all mods have run. It returns `void`;
  `despawn`, `insert`, and resource operations are not implemented. No direct
  world mutation from mod code.

Because they are signature-only, attempts to construct them (`new Query<T>()`,
bind them to locals, store them in arrays) are rejected by the checker.

### 3.5. UFCS-style method resolution

- Calls of shape `a.foo(b, c)` re-resolve to `foo(a, b, c)` where `foo` is a
  function in the module's scope or the host's; the receiver becomes the
  first argument. This is what lets `world.query()` and `q.iter()` be expressed.
- Methods on `Array<T>`: `a.length`, `a.push(x)`, `a[i]` already use this.
- Methods on `Query<T>`: `.iter()`, `.single()`, `.get(entity)`.
- Methods on `Iter<T>`: `.next()`, `.skip`, exhaustiveness.
- Methods on `Commands`: `.spawn(c1, c2, ...)` (implemented); `.despawn(e)` and
  `.insert(e, c)` remain planned.
- Resolution order: module-local → host namespace (`world`, `q`) → builtins.
  No `Self`/trait confusion.

### 3.6. Generics and monomorphisation

- Each mod unit has a *single* compile-time monomorphisation: `T` is fixed
  by the host that loads it. A mod cannot be loaded twice with different T's
  in one VM.
- Generic function declarations (`declare function foo<T>(x: T): T`) are
  *not* allowed in v1; v1 type parameters only appear on builtins/host-safe
  types (`Query<T>`, `Res<T>`, `Option<T>`, `Array<T>`), whose monomorphisation
  is driven by the host side's numeric schema.

---

## 4. Host ABI (generic only — no per-type/per-field calls)

### 4.1. Design rule
- A host function must accept and return only `Entity`, `ComponentId`,
  `usize`, `u64`, numbers, strings, `Array<T>` (strides are pointer
  metadata the host reads from the Array header), and the signature-only
  `Query`/`Res`/`Commands` types.
- All component *identity* is expressed through `ComponentId`, which the host
  derives from `T` via its FFI adapter and exposes through a single numeric.
  Never split into `player_x`, `row_get_f32(i, field)`.

### 4.2. v1 host call catalogue (gating implementation phase 1)

| host symbol | params | ret | semantics in mod terms |
|---|---|---|---|
| `world` builtin | — | world access | zero-arg magic; the registry inserts one `HostFn` named `world` per unit |
| `world_query` | `... (field access encoded in sig)` | `Query<T>` | signature-only |
| `world_resource` | `usize (res-id)` | `Res<T>` | signature-only |
| `world_resource_mut` | `usize` | `Mut<T>` | signature-only |
| `world_commands` | — | `Commands` | signature-only |
| `entity::id` | `Entity` | `u64` | identity without type loss |
| `component::id` | `ComponentId`-identity encoded as sig | — | only for introspection tests |
| `array::len` / `push` / `get` / `set` | — | — | already exist as intrinsics |
| `load_*` / `store_*` | `u64 region, offset, ...` | typed | for mods that *want* raw region access (escape hatch), unchanged |

`iter()` etc. are lowered against the same `Query<T>` signature without a
host call; the JIT statically emits the column loop using
`get_data_slice_for`-equivalent descriptors (§5).

### 4.3. Single registration entry point
The host side has one helper:
```rust
fn declare_component_access<T: Component>(&mut self, ...) -> ComponentId;
```
and mods call back through `HostFn` by name + signature only. This is the
only place where `T` is materialised; the rest is `ComponentId`/`Entity`.

### 4.4. Rejection of bad behaviour
- All host functions assert preconditions; they *trust* the JIT signature
  (already checked at compile) and return `Trap` on bad numeric ranges.
- No panic can cross the host/JIT boundary. The Cranelift codegen wraps
  host calls in fuel guards; host panics are caught with `catch_unwind`
  at the boundary and re-encoded as `TrapKind::HostPanicked`.

---

## 5. Chunk descriptor layout (verified against bevy 0.19.1)

Confirmed APIs:
- `world.storages().tables` — iterator over `Table`.
- `Table::entity_count()`, `Table::get_data_slice_for::<T>(ComponentId)`,
  `Table::get_changed_ticks_slice_for(ComponentId)`.
- `World::change_tick()`, `World::component_id::<T>()`.
- No structural changes (insert/remove components) while descriptors are
  alive — the `while` loop runs synchronously in `run_on_tick`, so the
  described tables remain valid.

Layout of one descriptor (matches `Chunk` in `scripting.rs`):
```rust
struct Chunk {
    len: usize,
    px: *const f32,            // column for first component, etc.
    vx: *const f32,
    ticks: *mut bevy_ecs::change_detection::Tick,  // optional
    // one column per T in Query<T>, in declaration order.
}
```

Higher-order per-mon per-query-kernel descriptors: enumerate `archetypes
.iter()` instead of tables when the query needs join. Tables are the floor
case; archetypes are used only when the mod's query has `With<T>` / `Without<T>`
filters, so we walk `Archetypes` once at schedule time and bake the match
into per-query chunk lists.

Each typed component gets one `*const` column plus its `ticks: *mut Tick`.
Columns are contiguous within one `Table`; the JIT emits one column at a time
(via `SegLoad`/`SegStore`, widened from `i64` for the i8/i16/i32/u8/… case
per the codegen's `convert()`).

### 5.1. Span of the descriptor's lifetime
- Descriptors are rebuilt at the start of *every* `Update` system that
  invokes mods, before any mod call. They are bulk-collected (no
  per-mod allocation that escapes) and dropped when the dispatch finishes.

---

## 6. Codegen lowering sketch for `for-of` over query

New HIR constructs (add to `exprs`, augment check/codegen):

```rust
pub enum Stmt {
    // existing...
    /// for (let item of iter) { ... }  — only iterables that lower to
    /// column walks; checked for exhaustiveness.
    ForOf { iter: Expr, body: Vec<Stmt>, span: Span },
    // existing...
}
```

`ExprKind::Iter` (new — typed iterator handle is pointer-to-an-iterator-frame,
not an array. Lifetime = containing ForOf, not a value):

```rust
ExprKind::Iter { query: Box<Expr> },
ExprKind::IterNext { iter: usize },
```

Lowering pipeline:
1. Checker validates the receiver type (must be `Query<T>` or any
   `Iterable`. Compute `T` and reject unbound types.
2. Codegen builds a `value:` stack frame holding the descriptor index and
   current row; emits per-column advances; increments fuel; checks bounds.
3. For each component in `T`, the in-loop binding (e.g. `for (const x of
   q.iter()) { ... }`) exposes `x` as a typed mutable local that desugars to
   `place_read(query, current_row, comp_i)` and writes back on end-of-block
   when the place was mutated.
4. Destructuring: `for (const {px, vx} of q) …` lowers to separate place
   reads per component plus the iter frame.
5. Mutation via place: `item.x += 1` in the body compiles to
   `place_write(..., 1)`. Codegen chooses the writeback strategy
   (Cranelift store; `ticks.rc` is set via host call).
6. No allocation anywhere in the loop.

The lowering mirrors `c2_cranelift` (hand-written kernel) directly — the
Cranelift kernel in `benches/scripting.rs` is the executable spec — so
expected perf is in the 0.84 ns/entity vicinity plus component indexing and
fuel ticks.

Acceptance benchmark: a TS inner loop at `for-of` form lands no worse than
`+50%` over `a_native` (raw Bevy query, 3.18 ns/entity). Track via
`criterion::` group `spike_100k/q1_forof` (new group site inside
`benches/scripting.rs`).

---

## 7. Hot-reload semantics (reject-on-layout-change)

### 7.1. Hash
- A build's "layout hash" is the SHA-256 of the sorted list of
  (`component name`, `component size`, `component align`, `fields`) for every
  `Component`/`Resource` the mod's types touch plus the `ScriptingPlugin`
  version. Computed by `JitBuilder::build_hash()` over the host registry.
- The hash is stored with the compiled module and compared on reload.

### 7.2. Reload protocol
1. FS watcher observes a change under one mod's root.
2. Re-parse + re-typecheck. If typecheck fails: log + skip; old build keeps
   running (last-good build continues, per accepted default).
3. On typecheck success: compute new hash. If hash ≠ stored hash: **reject**
   reload, log `layout mismatch (old vs new)`, keep old build running. Do
   *not* swap regions; do not crash the VM.
4. On hash match: swap the Compiled entry point and reset regions by
   region-id (no allocation). Fuel counters reset to `DEFAULT_FUEL`.
5. Edge case: the mod renames a field but size/align stay the same. The
   structural hash changes (different field names); reload rejects. That is
   intended — mods cannot silently change semantic layout.

### 7.3. Why not allow layout-change and patch
- The safe alternative would require migration code for every component change;
  a mod author who breaks a component breaks *every* mod that touches it.
  Reject-and-keep-last-good is strictly less surprising.
- Last-good builds degrade: file change shows a toast in the client log;
  inform, not corrupt.

---

## 8. PlayerRow bridge (escape hatch)

- Provide `tesjit` standard library runtime functions `get_row<T>` /
  `set_row<T>` that map an `Entity` to/from a per-mod region pointer (the
  existing `register_region` mechanism), so mods that need an array of
  per-entity scratch data can keep using the `load_f32`/`store_f32` builtin
  style.
- Performance: d_player_rows = 25.38 ns/entity; marked as escape-hatch only,
  never the recommended pattern. Document in the host-side mod template.
- No global state: the bridge stores its buffer *in a resource* (`ModRegion`)
  so the world owns it; a mod receives it by `Mut<ModRegion>`.

---

## 9. Phased plan with acceptance tests

Each phase ends in a testable, revertible deliverable.

### Phase 0 — done
- ✅ Numeric widths i8/u8/f32/…/Entity in `crates/tesjit/src/ty.rs`.
- ✅ Cast matrix in `check.rs`, conversion in `codegen.rs`.
- ✅ 31 tests (incl. `sized_integers_wrap_and_compare`, etc.).
- ✅ Lua backend isolation, `just bench-scripting`, combined table.

### Phase 1 — type system + host ABI cleanup
- Add `Option<T>`, place-expressions (`Ref`/`Place`), `Query/Res/Mut/ Commands`,
  UFCS method resolution.
- `crates/tesjit/src/ty.rs`: new variants.
- `crates/tesjit/src/hir.rs`: `PlaceExpr`, `PlaceRead`, `PlaceWrite`, `ForOf`,
  `Iter`.
- `crates/tesjit/src/check.rs`: resolution rules (UFCS), place-expr
  well-formedness.
- `crates/tesjit/src/codegen.rs`: lowered stores/loads per component stride,
  `PlaceWrite` -> store + tick-write.
- Host side: generic helpers, single `declare_component_access`.
- Acceptance: new tests pass:
  - `option_is_some_none_pattern_roundtrip`
  - `place_write_roundtrip`
  - `query_forof_compiles_and_matches`
  - `ufcs_method_resolution`
- `cargo test -p tesjit` 31 → +15 tests, no regression.

### Phase 2 — modding API shape
- Type-safe `world.query::<T>()`, `world.resource::<T>()` in host-side
  signatures.
- Build a demo mod (TS) that iterates a query and reads a resource; deploy
  under `mods/` and run through the TS loader (new helper that drives a
  `Compiled` per mod unit via a root `main` export).
- Acceptance: the demo compiles, runs, and reports a sensible ns/entity
  above 0.84 (specifically land at < 3 ns/entity total).
- No production server code changes to Lua.

### Phase 3 — performance and fuel hardening
- Fuel policy: each `for-of` iteration consumes one fuel unit per component
  access; default `PER_CALL_FUEL = 1e7` unchanged. Trap surfaces `TrapKind`,
  logged with mod + event name.
- Bench add: `criterion` group `spike_100k/q1_forof`, target < 3 ns/entity
  per row.
- Add fuzz test: random mod bounds (100k cases) must never panic the host.
- Acceptance: `cargo test -p exsolis_server && cargo check -p tesjit` clean,
  `just bench-scripting` prints the new row and runs clean.

### Phase 4 — hot reload + docs
- Implement §7 hash and reject path in a new `ReloadTracker`.
- Keep the Lua `check_and_reload_mods` bridge untouched (Lua reloads under
  its own mechanism).
- Add `MODS.md` at repo root with the demo mod as a template and the
  performance warning.
- Acceptance: unit test `layout_change_rejects_reload_and_keeps_old` plus a
  CLI smoke test `reload-inspect` prints a diff when hash differs.

---

## 10. Open questions and resolved decisions

1. Should mods have read access to component *metadata* (which Components
   exist) via a host call — constant-cost `component::id::<T>` — or should
   the set be fixed compile-time by the host registry? Proposed: fixed
   compile-time, with a host test listing every registered T.
   The component set is currently fixed at compile time.
2. Are mod-spawned entities visible to *other* mods in the same tick?
   Resolved: no. Commands are applied after all mods run, and the entities are
   queryable on the following tick.
3. Should the spawn path take an explicit `Archetype` or infer from the
   `T` tuple? Resolved: infer from typed component values passed as
   `.spawn(c1, c2, ...)`; component object literals require an explicit
   component type annotation.

---

## 11. Runtime layout metadata without `types.tdb`

It is possible, and preferred, to remove `types.tdb` as a runtime prerequisite
for mod ECS access. `types.tdb` is a DWARF export for broad inspection of Rust
types; using it as the source of a live JIT ABI ties mod loading to a separate
binary-export step and can leave the server without typed queries when the
database is missing or stale.

Recommended replacement: an explicit, compile-time schema for opted-in
components and resources. A `TsComponent`/`TsResource` derive (or equivalent
registration macro) emits each exposed type's field names, field types,
`size_of`, `align_of`, and `offset_of!` values from the same Rust compilation
that builds the server. The server registers those schemas with Bevy's runtime
component/resource IDs; the JIT consumes the schema directly. Nested types
must also provide an exposed schema, and unsupported layouts (unions,
zero-sized table components, opaque fields, or recursive by-value types) are
rejected rather than guessed.

This keeps the hot path zero-copy and avoids a generic reflection lookup per
field. It does mean schemas are an explicit allowlist: only registered types
are available to mods, which is a useful safety boundary and makes generated
editor declarations deterministic. Do not derive offsets from Bevy reflection:
reflection describes logical fields, not necessarily their physical Rust
offsets. Keep `types.tdb` as an optional debugging/inspection tool and
temporary migration fallback, not as the source of production JIT layouts.

Migration steps:

1. Add the schema trait/derive and tests for primitive and nested component
   fields, including `repr(Rust)` structs using compiler-provided
   `offset_of!`.
2. Register opted-in component/resource schemas and generate declarations
   from that registry. Require an exact match between schema and Bevy's
   runtime size/alignment before linking.
3. Move production query tests and supported server components off
   `types.tdb`; keep a fixture proving missing `types.tdb` does not disable
   registered schemas.
4. Remove `ServerTypes` from mod compilation/linking. Retain the exporter
   crate only for developer diagnostics if still useful.
""", "path">
/home/nestor/prj/exsolis/docs/SCRIPTING_ECS_MODS_PLAN.md