// showcase_embers — a wildfire-front simulation that exercises the full
// (test fixture, not a live mod: its components only exist in the fixture
// world built by `showcase_embers_test` in `server/src/ts_backend.rs`.)
// zero-copy ECS query surface in a single `on_tick` system:
//
//   * `Query<[Entity, Mut<Flame>], Without<Firebreak>>`
//       — multi-term row (entity handle + mutable component), archetype
//         filtered: firebreak ("stone") entities never decay or spread;
//   * `Query<Mut<Fuel>, [With<Ignited>, Without<Firebreak>]>
//       — compound filter: only ignited fuel burns, firebreak fuel is
//         pruned by the host before a single column pointer is handed over;
//   * `Mut<FireState>`
//       — tick-local mutable resource (simulation clock + policy knobs);
//   * `Commands`
//       — deferred `spawn`: new flames spawned this tick appear in the
//         world after all mods finish, and are iterated by this same
//         system from the next tick on.
//
// Every component and the resource (`Flame`, `Fuel`, `Ignited`,
// `Firebreak`, `FireState`) must be registered by the host — on a test
// harness that means a `TsSchema`/`ComponentRegistry` fixture; on the real
// server the engine components (`PlayerPosition`, `PlayerLook`, …) take
// their place. `exsolis.generated.d.ts` (written by
// `ComponentRegistry::write_declarations`) provides the interfaces.
//
// Runs as-is under `server` test `showcase_embers_test` and is validated
// tick-by-tick there.

export function on_tick(
  flames: Query<[Entity, Mut<Flame>], Without<Firebreak>>,
  fuel: Query<Mut<Fuel>, [With<Ignited>, Without<Firebreak>]>,
  state: Mut<FireState>,
  commands: Commands,
): void {
  state.ticks += 1.0;

  // Flame front: every live flame decays, drifts with the wind, and —
  // while still hotter than the spread threshold — ignites one child
  // flame at a wind-jittered position. The jitter derives from the
  // entity handle, so it is deterministic per flame.
  for (const [entity, flame] of flames) {
    flame.intensity -= state.decay as f32;
    flame.x += (state.wind_x * 0.1) as f32;
    if (flame.intensity > (state.spread_intensity as f32) && state.spawned < state.spawn_cap) {
      let h: u64 = entity as u64;
      let roll: f64 = (h % (9973 as u64)) as f64 / 9973.0; // [0, 1)
      const child: Flame = {
        x: flame.x + ((roll * state.spread_range) as f32),
        y: flame.y,
        z: flame.z,
        intensity: 1.0,
      };
      const mark: Ignited = {};
      commands.spawn(child, mark);
      state.spawned += 1.0;
    }
  }

  // Ignited fuel burns down. Firebreak and non-ignited fuel never even
  // appear here — the archetype filters prune whole tables host-side.
  for (const f of fuel) {
    f.amount -= state.burn_rate as f32;
    state.total_burned += state.burn_rate;
  }
}
