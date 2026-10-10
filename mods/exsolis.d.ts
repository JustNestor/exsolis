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
