# TypeScript modding — how to use it

This is the guide for writing a **TypeScript mod** that runs in the exsolis server through the `tesjit` JIT. For the design background, see [plans/SCRIPTING_ECS_MODS_PLAN.md](../plans/SCRIPTING_ECS_MODS_PLAN.md).

## The shape of a mod

A mod lives in `mods/ts/`. It is a set of exported functions:

| Export | Signature | Fires when |
|---|---|---|
| `on_tick` | `(query: Query<T, F>, res?: Res<R>, commands?: Commands) => void` | Every server tick (20 Hz). |
| `on_<event>` | Varies | The named server event. |

Subscriptions are by **export name** — there are no callbacks in the language yet, so the host calls `on_tick` / `on_<event>` when the mod exports it.

Because every function boundary is fully annotated, a mod that exports the wrong shape **fails to compile** (and keeps its last good build) instead of crashing the server.

The canonical "hello world" from `crates/tesjit/src/lib.rs`:

```ts
declare function log(x: f64): void;
declare function player_count(): i32;

export function on_tick(): void {
    let n: i32 = player_count();
    if (n > 0) { log(n as f64); }
}
```

Two things to notice:

- **Every boundary is annotated.** `declare function log(x: f64): void;` — parameters *and* return type. The annotation on the declaration is what the host's signature is checked against.
- **`as` is explicit.** `log(n as f64)` converts `i32 → f64`. There are no implicit conversions.

## What is in the language (v1)

The supported subset, as implemented in `crates/tesjit/src/check.rs` and lowered to the HIR in `crates/tesjit/src/hir.rs`.

### Types

| Type | Machine type | Notes |
|---|---|---|
| `i8`, `i16`, `i32`, `i64` | signed | All four are **distinct**. `u32` and `i32` share a machine representation but differ in signedness semantics; the checker keeps them apart and requires an explicit `as`. |
| `u8`, `u16`, `u32`, `u64` | unsigned | Same, by width. |
| `f32`, `f64` | floats | `number` means `f64`. |
| `boolean` | `I8` machine type | |
| `Entity` | opaque `u64` | No literals, no arithmetic; only `==`/`!=` and explicit `as` casts. |
| `string` | `string` (arena pointer) | |
| `T[]` | array | `[len][cap][elems…]` in the module arena. |
| `Option<T>` | tagged pointer | Heap-allocated `{tag, value}`. |
| Structs / components / resources | machine words | `Component<T>` / `Resource<T>`, `Query<T, F>`, `Res<T>`, `Mut<T>`, `Commands`, `void`. |
| `Query<T, F>`, `Res<T>`, `Mut<T>`, `Commands` | signature-only | Never materialize as locals; they describe the host frame. |

### Statements

| Statement | Notes |
|---|---|
| `let x: T = expr;` | Inference is local to `let`. A declaration must carry a type annotation. |
| `if` / `else` | |
| `while` | |
| `for` (over an array, with write-back) | |
| `break`, `continue` | |
| `return` | Must match the declared return type; no value unless the return type carries one. |
| Block statements | |

### Expressions

| Expression | Notes |
|---|---|
| Arithmetic / comparison / bitwise / shift | |
| `&&`, `\|\|`, ternary `?:` | |
| Assignment `=`, compound assignment (`+=`, …) | |
| `++` / `--`, prefix and postfix | |
| Calls, `as` casts, literals | |

### What is rejected

The checker produces **all diagnostics in one pass** (`miette`-flavoured diagnostics) and reports them together rather than stopping at the first:

- Missing parameter or return type on a function boundary — *“every function boundary is annotated”*.
- Host signature mismatch — the declared signature does not match the registered host function.
- Unknown host function.
- `break` outside a loop, destructuring, spread, `await`, top-level state, struct copy, unsupported operators, unsupported query filters.

Anything that is not in the supported set is a **compile-time error** — never silently miscompiled.

## How to write a typed ECS mod

### The query system

```ts
export function on_tick(
  players: Query<[Mut<PlayerPosition>, Mut<PlayerLook>]>,
): void {
  for (const [pos, look] of players) {
    pos.position.x += 1.0;
    look.yaw = look.yaw + 0.01;
  }
}
```

`Query<T, F>` takes a tuple of terms. `Mut<>` marks a component as writable; `With<Comp>` / `Without<Comp>` are the filters. One term may be `Entity` (an opaque entity handle) and one may be a `Mut<>` of a scalar.

Rules:

- A query system must **return `void`**.
- Conflicting access to one component or resource within a single system is an error — you cannot have `Mut<A>` twice, or `Mut<A>` and `Res<A>`.
- The row is a dense column of the matching Bevy table; reads are zero-copy into the column pointer, writes go back through the column. There is **no snapshot, no copy-back**, and a `Mut` borrow updates Bevy's changed tick when acquired, even if the field is left unchanged.

The full feature set is described in the `Phase 2` notes of the plan, and demonstrated by:

- `mods/ts/showcase_orbit.ts` — a multi-term row over two engine components with nested field writes through the `Vec3` layout, plus an `atan2` implemented from a polynomial.
- `crates/tesjit/benches/scripting.rs` — the benchmark harness.

### Resources

```ts
export function on_tick(
  config: Res<ModConfig>,
  commands: Commands,
): void {
    // read-only
}
```

`Res<T>` is read-only; `Mut<T>` is mutable; both must name a registered struct. A resource must be present when the system is linked, or the system is skipped; if it is removed later, the system is skipped until it returns.

### Commands

`Commands` is an append-only command buffer. `Commands.spawn` accepts registered, supported component values and stages the entity for insertion after all mods have run:

```ts
export function on_tick(commands: Commands): void {
    // stage entities here
}
```

Spawned entities are added after every mod has run, so they are visible to queries on the **next tick**, not later in the current tick.

### The host-provided row data

Not every system is a pure ECS system. The host also exposes:

- `query_players(): i32` — region id of this tick's player rows (0 when there is none).
- `query_row_count(region): i32` — number of rows in the region.
- `row_stride(): i32` — byte size of one row.
- `row_offset(field: string): i32` — byte offset of one row field, or -1 when unknown.
- `row_get_f32(index, field): f64` / `row_set_f32(index, field, value): f64` — read/write a float field of row `index`.
- `row_get_i64(index, field): i32` / `row_set_i64(index, field, value): i32` — read/write a machine-word field (e.g. `"id"`).
- `field_offset(type: string, field: string): i32` and `type_size(type: string): i32` — layouts from the exported type database.

### Option, arrays, strings

- `Option<T>` is heap-allocated and tagged. `some(value)` infers `Option<T>`; `none()` requires a typed assignment site; `isSome()` / `unwrap()` keep unwrapping sound.
- Arrays are heap-allocated with a length and capacity.
- Strings are `[len: i64][utf8 bytes…]` in the module arena.

---

## Examples

### 1. Logging the player count (the canonical smoke test)

`mods/ts/phase2_smoke.ts`:

```ts
declare function world_token(): i64;
declare function log(x: f64): void;

export function on_tick(): void {
    let w: i64 = world_token();
    log(w as f64);
}
```

`mods/ts/component_access.ts`:

```ts
declare function component_access_id(name: string): i64;
declare function log(x: f64): void;

export function on_tick(): void {
    log(component_access_id("Player") as f64);
}
```

### 2. The zero-copy orbit

`mods/ts/showcase_orbit.ts` — see the header comment for the full analysis of what it exercises.

### 3. The typed row

`mods/ts/typed_rows.ts` — reads a row's float field by name through the row ABI.

### 4. Option exercise

`mods/ts/demo_option.ts` — exercises `Option<f64>` / `Option<i64>`, `some`, `none`, `isSome`, `unwrap`.

---

## The host API surface

These are the host functions the server registers. They take the signature listed; the ambient declarations live in `mods/exsolis.d.ts`.

### Messaging

| Host fn | Signature | Meaning |
|---|---|---|
| `log` | `(x: f64) -> void` | Log a number to the server console. |
| `send` | `(player: string, kind: string) -> void` | Send a mod message to one player. |
| `broadcast` | `(kind: string) -> void` | Send a mod message to every player. |

### Blocks

| Host fn | Signature | Meaning |
|---|---|---|
| `set_block` | `(x: i32, y: i32, z: i32, block: i32) -> void` | Place a block. |
| `commands_set_block` | `(entity: i64, x: i32, y: i32, z: i32, block: i32) -> void` | Stage a block placement through `Commands`. |
| `commands_despawn` | `(entity: i64, other: i64) -> void` | Stage a despawn through `Commands`. |
| `get_block` | `(x: i32, y: i32, z: i32) -> i32` | Block id at a position, or 0 (air) if the chunk is not loaded. |

### Players

| Host fn | Signature | Meaning |
|---|---|---|
| `player_count` | `() -> i32` | Number of connected players. |
| `player_id_at` | `(index: i32) -> string` | Player id at a hotbar index. |
| `player_x` / `player_y` / `player_z` | `(id: string) -> f64` | Player position. |

### ECS

| Host fn | Signature | Meaning |
|---|---|---|
| `query_players` | `() -> i32` | Region id for this tick's rows. |
| `query_row_count` | `(region: i32) -> i32` | Row count. |
| `row_stride` | `() -> i32` | Byte size of one row. |
| `row_offset` | `(field: string) -> i32` | Field byte offset, or -1. |
| `row_get_f32` | `(index: i32, field: string) -> f64` | Read a float. |
| `row_set_f32` | `(index: i32, field: string, value: f64) -> f64` | Write a float; returns the previous value. |
| `row_get_i64` | `(index: i32, field: string) -> i32` | Read a machine word. |
| `row_set_i64` | `(index: i32, field: string, value: i32) -> i32` | Write a machine word. |
| `field_offset` | `(type: string, field: string) -> i32` | Byte offset in a named type. |
| `type_size` | `(type: string) -> i32` | Byte size of a named type. |
| `component_access_id` | `(name: string) -> i64` | Numeric Bevy component id. |
| `world_token` | `() -> i64` | Opaque world handle. |
| `__tesjit_commands_spawn` | `(commands: Commands, input: Ptr) -> void` | Stage a spawn through the command buffer. |

### UI

| Host fn | Signature | Meaning |
|---|---|---|
| `render_xml` | `(file: string) -> string` | Expand a template. Data stays on the Rust side for now; pass the resulting XML via `send`. |
| `register_keybind` | `(id: string, key: string, description: string) -> void` | Register a keybind. |

### Fuel

Each handler invocation runs under a fuel budget (`PER_CALL_FUEL = 2_000_000` per call in the server). Function entries and loop back-edges each cost one unit, so a per-tick hook has room for tens of thousands of statements while a spin loop dies inside one tick.

---

## Diagnostics

Compilation collects **all** diagnostics rather than stopping at the first, and the server keeps the last good build when a recompile fails. The diagnostics surface:

- missing or mismatched function boundaries,
- unknown host functions,
- unsupported statements or expressions,
- unsupported types or query filters.

There is never a silent miscompile: anything the compiler cannot lower is a compile-time error.

---

## Where to go next

- [TypeScript runtime — how it works](../scripting/typescript-runtime.md)
- [Lua modding](../lua-modding.md)
- [XML UI](../ui/xml-ui.md)
- The design background lives in [plans/SCRIPTING_ECS_MODS_PLAN.md](../plans/SCRIPTING_ECS_MODS_PLAN.md).
