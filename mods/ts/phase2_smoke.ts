// Phase-2 smoke: read the opaque world token and do nothing else yet.
declare function world_token(): i64;
declare function log(x: f64): void;

export function on_tick(): void {
  let w: i64 = world_token();
  log(w as f64);
}
