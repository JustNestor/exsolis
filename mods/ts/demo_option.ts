// Demonstration TypeScript mod: uses the heap-allocated tagged `Option<T>`
// type to hold an optional f64 per tick. Does not touch the world yet — only
// exercises the typed runtime feature introduced in phase 1.
//
// `Option<T>` is a tagged pointer allocated by `some(value)`; `none()` returns
// a tagged null. `isSome()` / `unwrap()` mirror Rust's `Option`.
function identity(x: f64): f64 { return x; }

export function on_tick(): void {
  // `some` infers `Option<f64>` from its argument. `isSome()` keeps
  // `unwrap()` sound — at no point do we read a `None` value.
  let v: Option<f64> = some(3.5);
  if (v.isSome()) {
    let w: f64 = v.unwrap();
    // Avoid logging the exact value, but definitely exercise the branch path.
    identity(w);
  } else {
    // None still allocates a tag; never unwrapped.
    identity(0.0);
  }

  // Plain `none` only appears in a typed assignment site (Option<i64>).
  let missing: Option<i64> = none();
  if (missing.isSome()) {
    // Never taken for a `none` value.
    log(missing.unwrap() as f64);
  }
}

/// 0 = make no claims about the world; the demo only mutates compile-time.
export function on_player_join(id: string): void { log(1.0); }
