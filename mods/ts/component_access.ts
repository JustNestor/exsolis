declare function component_access_id(name: string): i64;
declare function log(x: f64): void;

export function on_tick(): void {
  log(component_access_id("Player") as f64);
}
