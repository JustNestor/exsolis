# TypeScript runtime — how it works

This is the engineering deep-dive into `crates/tesjit`: the pipeline, the type system, the host ABI, the sandbox, and the hot reload policy. For how to write a mod, see [typescript-modding.md](typescript-modding.md).

## Pipeline

```
source → oxc parse → typed HIR → Cranelift IR → machine code
            (check.rs)   (hir.rs)    (codegen.rs)
```

The modules map one-to-one:

| Module | Responsibility |
|---|---|
| `src/check.rs` | Parse + lower the oxc AST into the typed HIR; collect **all** diagnostics. |
| `src/hir.rs` | The owned typed IR: names resolved to local slots, every expression typed, every function signature validated. |
| `src/codegen.rs` | Pure structural lowering of the HIR into Cranelift IR. Infallible — no symbol tables, no type expectations, no fallible decisions. |
| `src/jit.rs` | `JitBuilder → Jit → Compiled`: host registration, entry points, invocation. |
| `src/host.rs` | The host function registry and the per-compilation-unit `FnTable`. |
| `src/abi.rs` | The typed wire between JIT code and host code (queries, rows, commands). |

Because the HIR is fully typed before codegen starts, the codegen layer is a pure structural walk — which is what makes batch diagnostics and a future ahead-of-time backend cheap to add.

## The type universe

`src/ty.rs` — every TS-facing type maps to exactly one Cranelift machine type.

```rust
pub enum Ty {
    I8, I16, I32, I64, U8, U16, U32, U64,   // all widths, signed/unsigned distinct
    F32, F64, Bool,
    Entity,                                   // opaque u64
    Option(Box<Ty>),
    Struct(Arc<StructLayout>),
    OwnedStruct(Arc<StructLayout>),
    Tuple(Vec<Ty>),
    Component(Box<Ty>),
    Resource(Box<Ty>),
    Query(Box<Ty>, Vec<QueryFilter>),
    Res(Box<Ty>),
    Mut(Box<Ty>),
    Commands,
    Str, Array(Box<Ty>), Ptr, Void,
}
```

Key design decisions:

- **Integer widths are all distinct**, even where the machine representation is the same (`u32` and `i32` both occupy an i32). Signedness drives division, remainder, shift and comparison semantics, so the checker keeps them apart and requires an explicit `as` to convert.
- **Strings and arrays are machine words** — the pointer is an `i64`, with the real payload behind it.
- **Query/Res/Mut/Commands are signature-only.** They never materialise as locals; they describe the host frame.
- **A struct is either** a mod-defined `repr(C)` POD (`StructLayout::repr_c`) **or** a host-registered engine component layout (`StructLayout::from_parts`, verified against the real byte size/alignment). The latter are looked up by component id in the host's `ComponentRegistry`.

## The host ABI

`src/host.rs` + `src/abi.rs`. The critical constraint: the JIT invokes host code through `extern "C"` with **no closures** — the host has to be a plain function table.

```
Jit code ──extern "C"──▶ FnTable<HostFn> ──▶ HostSig + HostFn
                                   │
                                   └─▶ the host's Rust state, via the PullCtx / per-module runtime
```

Host functions are registered per compilation unit (never shared across modules):

```rust
pub fn host_fns() -> Vec<HostFn> {
    vec![
        HostFn::new("log",        vec![Ty::F64], Ty::Void, host_log as *const u8),
        HostFn::new("send",       vec![Ty::Str, Ty::Str], Ty::Void, host_send as *const u8),
        HostFn::new("broadcast",  vec![Ty::Str], Ty::Void, host_broadcast as *const u8),
        HostFn::new("set_block",  vec![Ty::I32, Ty::I32, Ty::I32, Ty::I32], Ty::Void, host_set_block as *const u8),
        …
    ]
}
```

A host function records its own signature (`HostSig`: parameter types + return type) and the compiler validates every declared function against it — that is the “host signature mismatch” diagnostic.

## The per-tick pull context

`server/src/ts_backend.rs` installs a `PullCtx` for the duration of a tick:

```rust
struct PullCtx {
    positions: HashMap<String, (f32, f32, f32)>,  // player_id -> position
    world: *const ChunkWorld,                     // live chunk storage
    bevy_world: *mut World,                      // generic host functions run cached Queries by registered name
    components: HashMap<String, u64>,           // component name -> Bevy component id
}
```

Because the JIT ABI is flat `extern "C"` with no closures, the context lives in **thread-local state** for the duration of the call and is cleared afterwards, so JIT code can never outlive the tick.

The per-tick driver is an exclusive system — `fn(&mut World) -> ()` — which is what makes the raw chunk pointer sound and keeps mod reads and engine writes ordered.

## The query ABI

A `Query<T, F>` parameter becomes, on the host side, a `QueryPlan`:

```rust
pub(crate) struct QueryPlan {
    terms: Vec<QueryTerm>,
    filters: Vec<(ComponentId, QueryFilterKind)>,
    archetypes_seen: usize,
    tables: Vec<TableId>,
    table_ids: HashSet<u32>,
}
```

`QueryPlan::fill` walks the **Bevy tables** directly and produces a dense `QueryBuffer` with:

- column pointers (`usize` values into the table storage) — the zero-copy path,
- entity bits, when the row has an `Entity` term.

For a `Mut` term, the plan captures the Bevy **change tick** of that component during the fill, so the write-back semantics (a `Mut` borrow updates the changed tick when acquired) are preserved.

`QuerySystem::args` wires the host frame: a query gets its buffer, a `Res`/`Mut` gets the resource pointer, a `Commands` gets the command buffer.

The result is fed to the JIT code as a pointer into a `QueryInput` — and the mod's `for (const row of query)` iterates over those columns without copying.

## Commands

`Commands` is an append-only buffer on the host side; `apply_commands` writes the staged entities into the world after all mods have run (the “after every mod has run” rule).

```rust
impl QuerySystem {
    pub(crate) fn apply_commands(&mut self, world: &mut World) {
        for param in &mut self.params {
            if let LinkedParam::Commands { buffer } = param {
                buffer.apply(world);
            }
        }
    }
}
```

## Sandbox

`src/runtime.rs`.

| Guard | Mechanism | Limit |
|---|---|---|
| Fuel | One unit per function entry and per loop back-edge, charged from the per-call budget. | `DEFAULT_FUEL = 10_000_000`; server arms `PER_CALL_FUEL = 2_000_000` per handler. |
| Memory | A bump arena per module with a **stable backing buffer** (reserved once via virtual memory, never reallocated — JIT code holds pointers into it for the module's lifetime). | `ARENA_CAP = 64 * 1024 * 1024`. |
| Bounds | Array/string access goes through the runtime helpers, which trap before touching memory. | — |
| Traps | A trap records a code and returns a zero value, so the handler returns normally instead of unwinding into Rust. | `TrapKind`: index out of bounds, array overflow, arena out of memory, fuel exhausted, unknown region. |

`Compiled::trap()` / `clear_trap()` let the host read the trap and log it; the handler still returns normally, so one buggy mod cannot take the server down.

The arena is module-scoped: a module never reaches another module's memory, and the module runtime is restored on scope exit.

## Hot reload

`ts_reload_system` (`server/src/ts_backend.rs`) is what runs on a timer / on source change:

1. Build a fresh `ComponentRegistry` from the world plus the registered schemas and the type database.
2. `registry.write_declarations(&runtime.root)` — writes the generated declarations.
3. `runtime.reload_with_registry(Some(&registry))`.

The type database comes from the server binary's DWARF via `tesjit-dwarf`, so layouts are real, not guessed.

### Reject-on-layout-change

The layout hash is taken over the component name, field names/types/offsets and the size. When a recompiled mod's declared layout hash differs from the host's, the **reload is rejected** and the last good build is kept — a stale layout must never silently be interpreted as a fresh one.

## The module ABI in one paragraph

The JIT's machine code never touches the Bevy world directly. It asks the host for:

- a **query buffer** (column pointers + entity bits),
- **resource pointers** (by component type name),
- a **commands buffer**.

Everything else — the chunk world, the component registry, the type database — lives in the host, reached through the pull context. That is the boundary that makes the whole design safe: no raw pointer ever escapes the exclusive tick system that installed it.

## Tests

```sh
cargo test -p tesjit
```

Covers: recursion/loops, host calls, short-circuit logic, `?:`, casts, compound assignment, prefix/postfix updates, and diagnostics (missing return type, host signature mismatch, unknown host function, `break` outside a loop, batched multi-error reporting).
