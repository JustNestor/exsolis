// A demonstration TypeScript mod: greets joiners and logs block placements.
//
// Run the server with `just` (or `cargo run --bin exsolis_server_bin`) and watch
// the console: `[ts mod] ...` lines come from here.
//
// Everything here must live inside functions: there is no top-level state in the
// language yet (top-level `let` is not part of the supported subset).

declare function log(x: f64): void;
declare function player_count(): i32;
declare function get_block(x: i32, y: i32, z: i32): i32;
declare function set_block(x: i32, y: i32, z: i32, block: i32): void;

// 1 = stone in the block table; 0 = air.
const STONE: i32 = 1;

export function on_tick(): void {
  let n: i32 = player_count();
  if (n <= 0) {
    return;
  }
  // Announce once every 100 ticks that the mod is alive.
  log(n as f64);
}

export function on_player_join(id: string): void {
  // A decimal player id, same rule as the Lua bridge.
  log(1.0);
}

export function on_place_block(
  id: string,
  x: i32,
  y: i32,
  z: i32,
  block: i32,
): void {
  // Reflect the placement one block higher, so the world visibly changes.
  if (get_block(x, y + 1, z) == 0) {
    set_block(x, y + 1, z, STONE);
  }
  log(y as f64);
}
