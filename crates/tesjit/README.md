# tesjit

A minimal, statically-typed TypeScript → Cranelift JIT, meant to be **embedded**
by the exsolis server as the scripting/mod backend.

## Pipeline

```
source → oxc parse ─→ typed HIR ─→ Cranelift IR ─→ machine code
            (check.rs)      (hir.rs)    (codegen.rs)
```

| Module | Responsibility |
|---|---|
| `src/check.rs` | parse + lower oxc AST into typed HIR; collect **all** diagnostics |
| `src/hir.rs` | owned typed IR: names resolved to local slots, every expression typed |
| `src/codegen.rs` | structural Cranelift lowering (infallible, no lookups) |
| `src/jit.rs` | `JitBuilder` → `Jit` → `Compiled`; host registration + entry points |
| `src/host.rs` | host function registry, per compilation unit (no globals) |
| `src/ty.rs` / `src/diag.rs` | the type universe / miette diagnostics |

## Language rules (v1)

- **Every function boundary is annotated** — parameters *and* return type
  (use `: void` explicitly). No `any`/`unknown`.
- Inference is local: `let x = expr` only.
- No implicit conversions; `as` is explicit (`i32`↔`f64`, `boolean`→`i32`/`f64`).
- Types: `i32`, `f64`, `boolean`, `void`.
- Statements: `let`, `if/else`, `while`, `for`, `break`, `continue`, `return`,
  blocks, expression statements.
- Expressions: arithmetic/comparison/bitwise/shift ops, `&&`/`||`, `?:`,
  assignment (`=` and compounds), `++`/`--` (pre/post), calls, `as`, literals.
- Host calls go through `declare function name(...): T;` + registration.

Anything else is a compile-time error (`unsupported statement`/`unsupported
expression`), never silently miscompiled.

## Embedding

```rust
use tesjit::{JitBuilder, Ty};

extern "C" fn host_log(x: f64) { println!("[host] {x}"); }

let src = r#"
declare function log(x: f64): void;
export function go(dt: f64): void { log(dt * 2); }
"#;

let mut jit = JitBuilder::new("mod.ts", src)
    .host_fn("log", vec![Ty::F64], Ty::Void, host_log as *const u8)
    .build()?;
jit.compile(src)?;                 // all diagnostics at once
let compiled = jit.finish()?;

let go: extern "C" fn(f64) = unsafe { compiled.get("go").unwrap() };
go(1.5);
```

Other API surface: `jit.check(src)` (type-check only), `Compiled::exports()`,
`Compiled::entry()`, `Compiled::call_i64()/call_void()` (integer arguments only),
and `Jit::lower()` for tooling that wants the typed HIR.

## CLI

```sh
tesjit run <file.ts>      # compile, call main() if exported
tesjit check <file.ts>    # type-check only, print every diagnostic
tesjit bench <file.ts>    # fib benchmark driver
```

## Host ABI: query rows and layouts

The host can expose real Rust memory to mod code — the shape a Bevy system will
use for query rows and components:

```rust
// Rust side: a value laid out by the real compiler, plus its layout from the
// exported type database (crates/tesjit-dwarf).
let offsets = db.layout("exsolis_common::ChunkBlocks").unwrap();
let id = compiled.register_region("row", ptr, len);
```

```ts
// Mod side: read/write by the offsets the type database reports.
let count: i32 = load_i32(id, 0);
let pos: f64 = load_f64(region_id("row"), 8);
store_f64(id, 8, pos);
```

Every access is bounds-checked against the region length and traps on a bad
offset or unknown id, so a wrong offset never reads host memory. Offsets come
from `types.tdb` (exported from the server binary's DWARF), which is what makes
layouts match without guessing — see `tests::type_db_offsets_drive_region_access`
for an end-to-end test that exports a struct and reads a real Rust value through
it.

Row helpers on the host side (`row_get_f32`, ...) are the ergonomic layer over
this: they compute `index * stride + field` for the mod and go through the same
bounds checks.

## Sandbox (Phase 4)

Mod code cannot take the process down:

| Guard | Mechanism |
|---|---|
| Bounds checks | Every string byte / array element read and write goes through the runtime (`arr_get`/`arr_set`/`str_byte`), which traps before touching memory |
| Fuel | Each function entry and loop back-edge costs one unit (`Compiled::reset_fuel`, default 10M); when it runs out the handler returns zero and records a trap |
| Memory | One stable (never reallocated) 64 MB arena per module, so JIT-held pointers stay valid; exhaustion traps |
| Traps | Recorded in the runtime, read by the host via `Compiled::trap()` / `clear_trap()`; the handler returns normally, one buggy mod never unwinds into Rust |
| Stack | Cranelift stack probes enabled (`enable_probestack`), so large frames cannot skip the guard page |

## Tests

```sh
cargo test -p tesjit
```

Covered: recursion/loops, host calls, short-circuit logic, `?:`, casts,
compound assignment, prefix/postfix updates, plus diagnostics: missing return
type, host signature mismatch, unknown host function, `break` outside a loop,
and batched reporting of multiple errors.
