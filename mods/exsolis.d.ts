// exsolis — TypeScript mod API for the tesjit runtime (mods/ts/*.ts).
//
// Every function boundary must be annotated — parameters *and* return type.
// Handlers subscribe by export *name*: `on_tick` for every tick, and
// `on_<event>` for each event below.

// ───────────────────────── host functions ─────────────────────────

/** Log a number to the server console. */
declare function log(x: f64): void;

/** Send a mod message to one player (by decimal id string). */
declare function send(player_id: string, kind: string): void;

/** Send a mod message to every player. */
declare function broadcast(kind: string): void;

/** Place a block (id 0 = air). */
declare function set_block(x: i32, y: i32, z: i32, block: i32): void;

/** Block id at a world position, or 0 (air) if the chunk is not loaded. */
declare function get_block(x: i32, y: i32, z: i32): i32;

/** Number of connected players. */
declare function player_count(): i32;

/** Player id at `index` (use with `player_count`). */
declare function player_id_at(index: i32): string;

/** Player position, or 0 when the id is unknown. */
declare function player_x(id: string): f64;
declare function player_y(id: string): f64;
declare function player_z(id: string): f64;

/** Expand a UI template to client-ready XML. */
declare function render_xml(file: string): string;

/** Register a keybind on the client. */
declare function register_keybind(id: string, default_key: string, description: string): void;

// ───────────────────────── query rows ─────────────────────────
//
// The hands the mod a *view* of the world for one tick: a packed array of rows
// in host memory. Rows are read and written through host functions that own
// the offset math, so no mod ever hardcodes a byte offset — and every layout
// is cross-checked against the type database exported from the server binary
// (`just export-types`), so a Rust-side layout change is caught, not silently
// wrong.

/** Region id of this tick's player rows (0 when there is none). */
declare function query_players(): i32;

/** Number of rows in a query region. */
declare function query_row_count(region: i32): i32;

/** Byte size of one row. */
declare function row_stride(): i32;

/** Byte offset of one row field ("x", "y", "z", "id"), or -1. */
declare function row_offset(field: string): i32;

// ───────────────────────── type layouts ─────────────────────────
//
// Offsets for *any* Rust type in the exported type database, which is what
// makes non-`repr(C)` types usable from mods: rustc reorders struct fields and
// inserts padding, and only the DWARF export knows the real layout. Pass a
// fully-qualified type name and a field; you get bytes (-1 when unknown).
//
//   let off: i32 = field_offset("exsolis_server::chunks::PlayerPosition", "0");

/** Byte offset of `field` inside `ty` (from the exported type database). */
declare function field_offset(ty: string, field: string): i32;

/** Byte size of `ty` (from the exported type database). */
declare function type_size(ty: string): i32;

/** Read a float field of row `index`. */
declare function row_get_f32(index: i32, field: string): f64;

/** Write a float field of row `index`; applied to the world after the tick. */
declare function row_set_f32(index: i32, field: string, value: f64): f64;

/** Read a machine-word field of row `index` (e.g. "id"). */
declare function row_get_i64(index: i32, field: string): i32;

/** Write a machine-word field of row `index`. */
declare function row_set_i64(index: i32, field: string, value: i32): i32;

// ───────────────────────── builtins ─────────────────────────
//
// Runtime helpers shipped with the engine; they take no `declare`, and a mod
// cannot define functions with these names. They are the raw form of the row
// API above — bounds-checked, and intended for reading structures the host has
// not wrapped yet.

declare function load_f64(region: i32, offset: i32): f64;
declare function load_f32(region: i32, offset: i32): f64;
declare function load_i64(region: i32, offset: i32): i32;
declare function load_i32(region: i32, offset: i32): i32;
declare function store_f64(region: i32, offset: i32, value: f64): f64;
declare function store_f32(region: i32, offset: i32, value: f64): f64;
declare function store_i64(region: i32, offset: i32, value: i32): i32;
declare function store_i32(region: i32, offset: i32, value: i32): i32;
declare function region_id(name: string): i32;

// ───────────────────────── handlers ─────────────────────────
//
// Export any of these; the host calls them when the matching thing happens.

/** Once per server tick (20 Hz). */
export function on_tick(): void {}

/** A player connected. */
export function on_player_join(id: string): void {}

/** A player disconnected. */
export function on_player_leave(id: string): void {}

/** A player placed a block. */
export function on_place_block(
  id: string,
  x: i32,
  y: i32,
  z: i32,
  block: i32,
): void {}

/** A player broke a block. */
export function on_break_block(
  id: string,
  x: i32,
  y: i32,
  z: i32,
  block: i32,
): void {}

/** A player selected a hotbar slot. */
export function on_select_hotbar(id: string, index: i32): void {}


/** Opaque 64-bit world handle (TS-visible raw token). */
declare function world_token(): i64;

/** Registered component kind (numeric bevy id). */
declare function component_access_id(name: string): i64;
