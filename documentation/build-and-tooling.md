# Build and tooling

## Requirements

- **Rust** with edition 2024 support (stable 1.85+). The workspace uses edition 2024 in `server/`, `client/`, `common/` and the `crates/`.
- **A Vulkan 1.x driver** for the client. The renderer uses Vulkano directly, not Bevy Render.
- **[`just`](https://github.com/casey/just)** for the recipes below. Every recipe is a plain `cargo` command underneath, so `just` is optional.

## Running

```sh
just                      # server in the background, client in the foreground
cargo run --bin exsolis_server_bin
cargo run --bin exsolis
```

The server is the authority; the client needs a running server to connect to.

## The `justfile`

| Recipe | What it does |
|---|---|
| `just` | Runs the server in the background and the client in the foreground, killing the server when the client exits. |
| `just export-types` | Rebuilds the server with full debug info and exports `types.tdb` from its DWARF. |
| `just bench-scripting` | Compares native Rust, Bevy queries and the TypeScript JIT on one identical workload, then prints a combined table. |

### `just export-types`

```sh
RUSTFLAGS="-C debuginfo=2" cargo build --bin exsolis_server_bin
cargo run -p tesjit-dwarf -- target/debug/exsolis_server_bin -o types.tdb
```

The first step forces **full** debuginfo. The workspace dev profile uses `debug = "line-tables-only"`, which carries line tables but no type information — and DWARF layouts are exactly type information. Without this, the export produces an empty database.

The second step reads the DWARF out of the built binary and writes the versioned type database. The CLI round-trips the file it just wrote and reports the reloaded type count, so a load failure at server start is caught at export time.

See [scripting/typescript-runtime.md](scripting/typescript-runtime.md) for what the database is for.

### `just bench-scripting`

This one is a script rather than a single command, because it has to run **isolated cargo graphs**:

- `crates/bms-bench` is excluded from the workspace and built once per Lua backend (`bms-luajit`, `bms-luau`, `bms-lua54`).
- `crates/lua-bench-{luajit,luau,lua54}` are excluded for the same reason.

The exclusion is not cosmetic. `mlua` unifies features per version and `mlua-sys` declares `links = "lua"`, so **two Lua backends can never coexist in one cargo graph**. Each runner has to be a separate subprocess.

The recipe then parses the criterion output and the isolated runner JSON and prints one table:

```
== ecs_step, N=10000, x += v*dt: ns per row per step ==
backend          ns/row    vs native
```

## Build profiles

The workspace `[profile.dev]` is deliberately unusual:

```toml
[profile.dev]
opt-level = 1
debug = 2                       # full DWARF, not line-tables-only
split-debuginfo = "unpacked"
lto = false

[profile.dev.package."*"]
opt-level = 3
debug = "line-tables-only"

[profile.dev.build-override]
opt-level = 3
```

| Setting | Why |
|---|---|
| `opt-level = 1` | The workspace's own code compiles slowly at `opt-level = 0` and the simulation is not the bottleneck. |
| `debug = 2` | The mod bridge loads struct layouts out of DWARF, and line tables carry none. |
| `split-debuginfo = "unpacked"` | The toolchain default. The exporter is pointed at both the binary and the sibling `.dwo` directory, so this stays at the default. |
| `[profile.dev.package."*"] opt-level = 3` | Dependencies stay optimized so the benchmarks measure the engine, not `serde`. |

`crates/fastpak` has its own release profile: `lto = "thin"`, `codegen-units = 1`.

## Cargo features

| Feature | Crate | Effect |
|---|---|---|
| `advanced_worldgen` | `exsolis_server` | The noise-based terrain generator instead of the built-in heightfield. |
| `standalone` | `exsolis_server` | Uses `mimalloc` as the allocator. |
| `tracy` | `exsolis_client` | Wires `tracing-tracy` in for frame profiling. |

## Tests

```sh
cargo test -p tesjit                      # the JIT: compiler, sandbox, diagnostics
cargo test --bin exsolis_server_bin       # the server
cargo test --workspace                    # everything
```

Notable suites:

| Suite | What it covers |
|---|---|
| `crates/tesjit/src/tests.rs` | Recursion and loops, host calls, short-circuit logic, `?:`, casts, compound assignment, prefix/postfix updates, and diagnostics: missing return type, host signature mismatch, unknown host function, `break` outside a loop, batched multi-error reporting. |
| `common/src/chunk_blocks.rs` | Palette packing round trips, the frozen block ids, and rejecting garbage. |
| `common/src/inventory.rs` | Click semantics: pickup, split, swap, overflow rejection, out-of-range no-ops. |
| `client/src/gltf_utils/tests/` | The glTF loader: animation, coordinate conversion, extensions, MikkTSpace tangents, vertex attributes. |

## The standalone crates

The `crates/` directory holds pieces that are not part of the game and each has its own README:

| Crate | What it is |
|---|---|
| [`fastpak`](../crates/fastpak) | The `.pak` asset archive format. |
| [`tesjit`](../crates/tesjit) | The statically-typed TypeScript → Cranelift JIT. |
| [`tesjit-dwarf`](../crates/tesjit-dwarf) | The DWARF → `types.tdb` exporter. |
| [`tesjit-schema`](../crates/tesjit-schema) | The shared schema for the type database. |

Each of `tesjit` and `tesjit-dwarf` also ships a CLI:

```sh
cargo run -p tesjit -- run <file.ts>      # compile, call main() if exported
cargo run -p tesjit -- check <file.ts>    # type-check only, print every diagnostic
cargo run -p tesjit -- bench <file.ts>    # the fib benchmark driver

cargo run -p tesjit-dwarf -- <binary> -o types.tdb [--list substr] [--show Type]
```

## The `.d.ts` generation

After a mod is loaded, the server writes `mods/exsolis.generated.d.ts` for editor completion. That file is **managed by the server** — do not edit it by hand.

`mods/exsolis.d.ts` is the hand-written ambient API; the generated file complements it with the types the server actually registered.

## Environment variables

| Variable | Purpose |
|---|---|
| `EXSOLIS_ASSET_DIR` | The server's asset source directory. |
| `EXSOLIS_ASSET_BIND_ADDR` | The asset HTTP listener address. |
| `EXSOLIS_ASSET_BASE_URL` | The public origin clients download from. |
| `EXSOLIS_ASSET_CACHE_DIR` | The client's persistent cache directory. |
| `EXSOLIS_MODS_DIR` | Where the server looks for mods. |
| `EXSOLIS_TYPES_DB` | The exported type layout database. |
| `EXSOLIS_MC_ASSETS` | An extracted Minecraft asset tree, for mod blocks. |
| `EXSOLIS_TC_ASSETS` | A directory of Thaumcraft `.bbmodel` props. |

The full list, with defaults, is in [assets.md](assets.md) and [networking.md](networking.md).
