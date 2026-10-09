# tesjit-dwarf

Export Rust type layouts from DWARF into a versioned `types.tdb` that the
server loads at startup.

```sh
# 1. build a binary WITH type debug info. The workspace dev profile is
#    `debug = "line-tables-only"` (line tables but no types), so force full
#    debug info — otherwise DWARF has no layouts to read.
RUSTFLAGS="-C debuginfo=2" cargo build --bin exsolis_server_bin

# 2. export its types
cargo run -p tesjit-dwarf -- target/debug/exsolis_server_bin -o types.tdb

# 3. the server loads ./types.tdb (or $EXSOLIS_TYPES_DB) on start
#    (see server/src/typedb.rs; `just export-types` does 1+2)
```

## Why

The script/mod layer (`tesjit`) will need to touch Rust-side data structures,
and the ABI work planned for it is undecided: one candidate is sharing the
*layouts of real Rust structs* with generated code. This crate is the tool that
turns a binary's DWARF into a queryable database of those layouts, so the
decision can be made on data instead of guesses.

## What it records

| Kind | Contents |
|---|---|
| struct / union | size, alignment, ordered `(name, type, offset)` fields |
| enum | discriminant (type + offset) plus per-variant payload fields |
| array | element type + length |
| pointer / reference | pointee name (layout stops at pointers) |
| scalar (base type) | size + alignment |
| alias (`typedef`, `const T`, ...) | `alias -> canonical` map |

Names are demangled and qualified from the DWARF namespace chain, so keys read
like `exsolis_common::ChunkBlocks` and `Vec<T>`, and lookups accept
`<T as Trait>::Assoc` written as `<T>::Assoc`.

Known limitations (all fine for the current purpose, worth remembering):
- DWARF must be present: `types.tdb` is only as complete as the debug info in
  the binary it was exported from (release builds without `-g` yield nothing).
- Niche-encoded enums come out as a plain struct with an artificial tag member.
- Types that were never codegen'ed (unused in the binary) are absent.
- Only `.debug_info` is walked, not `.debug_types`.

## Format

`bincode`-encoded [`TypeDb`](src/schema.rs) with a magic + version header.
Callers validate `head.magic` / `head.version` on load; the CLI round-trips the
file it just wrote and reports the reloaded type count, so a load failure at
server start is caught at export time.

## Library use

```rust
use tesjit_dwarf::{export, TypeDb};

let data = std::fs::read("target/debug/exsolis_server_bin")?;
let (db, stats) = export(&data, "exsolis_server_bin")?;
let def = db.get("exsolis_common::ChunkBlocks").expect("type missing");
println!("{} is {} bytes", def.name, def.size);

let leaves = db.layout("exsolis_common::ChunkBlocks")?; // flat (path, offset) list
```

## CLI

```text
tesjit-dwarf <binary> [-o types.tdb] [--list substr] [--show Type] [-h]
```
