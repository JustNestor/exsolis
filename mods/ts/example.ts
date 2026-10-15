// A demonstration TypeScript mod: greets joiners and logs block placements.
//
// Run the server with `just` (or `cargo run --bin exsolis_server_bin`) and watch
// the console: `[ts mod] ...` lines come from here.
//
// Note: every bit of state lives inside a function. Top-level `let`/`const` is
// not part of the supported subset yet — the mod API in `mods/exsolis.d.ts`
// documents what is.

declare function log(x: f64): void;
declare function player_count(): i32;
declare function get_block(x: i32, y: i32, z: i32): i32;
declare function set_block(x: i32, y: i32, z: i32, block: i32): void;

/// Block ids, kept local so the mod has one place to change them.
///
/// 1 = stone, 0 = air (see the server's block table).
function stone(): i32 {
  return 1;
}

export function on_tick(): void {
  let n: i32 = player_count();
  if (n <= 0) {
    return;
  }
  log(n as f64);
}

export function on_player_join(id: string): void {
  // `id` is the decimal player id, same rule as the Lua bridge.
  log(1.0);
}

export function on_place_block(
  id: string,
  x: i32,
  y: i32,
  z: i32,
  block: i32,
): void {
  // Demo note: this used to mirror the placement one block higher to show
  // the world changing. That demo now lives in git history — placement must
  // not surprise the player with extra blocks.
  log(y as f64);
}
