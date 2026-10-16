declare function query_players(): i32;
declare function query_row_count(region: i32): i32;
declare function row_stride(): i32;
declare function row_offset(field: string): i32;
declare function row_get_f32(index: i32, field: string): f64;
declare function row_set_f32(index: i32, field: string, value: f64): f64;
declare function log(x: f64): void;

export function on_tick(): void {
  let region: i32 = query_players();
  let n: i32 = query_row_count(region);
  let stride: i32 = row_stride();
  if (n <= 0 || stride <= 0) {
    return;
  }
  // typed view over whatever layout the exported type database recorded.
  let dx_off: i32 = row_offset("0");
  if (dx_off < 0) { return; }
  let i: i32 = 0;
  let total: f64 = 0;
  while (i < n) {
    let px: f64 = row_get_f32(i, "0");
    total += px;
    i += 1;
  }
  log(total);
}
