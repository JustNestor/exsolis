// showcase_orbit — a *visible* showcase of the zero-copy ECS query surface.
//
// Every connected player is teleported onto a circle around the world spawn
// point and spun to face the centre, so joining the server is enough to see
// the mod working. It exercises exactly what the mod API is for:
//
//   * `Query<[Mut<PlayerPosition>, Mut<PlayerLook>], Without<SpawnedByMod>>`
//       — a multi-term row over two *engine* components, written straight into
//         Bevy's table columns (no snapshot, no copy-back);
//   * nested field writes (`pos.position.x = …`) through the `Vec3` layout
//         that `ComponentRegistry::write_declarations` exports as
//         `mods/exsolis.generated.d.ts`.
//
// No trig builtins exist in the supported subset, so the orbit is built the
// incremental way: the per-player (x, z) offset from the centre *is* the
// state. Each tick it is rotated by a fixed 6° step (two multiply-adds with
// precomputed cos/sin) and pulled back onto the radius, which is a circle by
// construction and needs no `sin`, `cos` or `sqrt`.
//
// Yaw uses the standard `atan2(a, b) == 2 * atan(a / (r + b))` identity with a
// polynomial `atan`, so players actually turn towards the middle of the ring
// instead of sliding sideways.

export function on_tick(
  players: Query<[Mut<PlayerPosition>, Mut<PlayerLook>]>,
): void {
  // Circle around SPAWN_POS (0, 28, 0) — see `common/src/lib.rs`.
  const CX: f32 = 0.0;
  const CY: f32 = 28.0;
  const CZ: f32 = 0.0;
  const R: f32 = 1.0; // ring radius
  const BOB: f32 = 4.0; // vertical wave amplitude

  // Orbit step: 6° per tick (server ticks at 20 Hz → one lap every 3 s).
  const COS: f32 = 0.9945219;
  const SIN: f32 = 0.1045285;

  // Seed step: 24° per player, so a full server does not stack every avatar
  // on the same point of the ring.
  const SEED_COS: f32 = 0.9135455;
  const SEED_SIN: f32 = 0.4067366;

  let index: i32 = 0;
  for (const [pos, look] of players) {
    let dx: f32 = pos.position.x - CX;
    let dz: f32 = pos.position.z - CZ;

    // Rotate the offset one step around the Y axis.
    const rx: f32 = dx * COS - dz * SIN;
    const rz: f32 = dx * SIN + dz * COS;
    dx = rx;
    dz = rz;

    let len2: f32 = dx * dx + dz * dz;
    if (len2 < 0.0001) {
      // Player is standing on the centre (fresh join, or walked back in):
      // drop them on the ring at an angle that depends on their slot.
      dx = R;
      dz = 0.0;
      let spin: i32 = 0;
      while (spin < index) {
        const sx: f32 = dx * SEED_COS - dz * SEED_SIN;
        dz = dx * SEED_SIN + dz * SEED_COS;
        dx = sx;
        spin += 1;
      }
    } else {
      // Rotation preserves the radius exactly, but walking/jumping/teleports
      // do not; nudge it back with one damped step (f = 1 when len2 == R²).
      const f: f32 = ((R * R) / len2 + 1.0) * 0.5;
      dx = dx * f;
      dz = dz * f;
    }

    // `dx / R` walks the unit circle by itself, so it doubles as the phase
    // for the vertical wave — no sine needed.
    const wave: f32 = dx / R;
    pos.position.x = CX + dx;
    pos.position.y = CY + BOB * wave;
    pos.position.z = CZ + dz;

    // Face the centre: yaw = atan2(dx, dz) via 2 * atan(dx / (r + dz)).
    let denom: f32 = R + dz;
    let yaw: f32;
    if (denom < 0.001) {
      yaw = 3.1415927;
    } else {
      const t: f32 = dx / denom;
      const t2: f32 = t * t;
      // atan(t) ≈ t * (0.9998660 - 0.3302995 t² + 0.1801410 t⁴ - …)
      const poly: f32 =
        0.999866 +
        t2 * (-0.3302995 + t2 * (0.180141 + t2 * (-0.085133 + t2 * 0.0208351)));
      yaw = 2.0 * t * poly;
    }
    look.yaw = yaw;
    look.pitch = 0.35 * wave;

    index += 1;
  }
}
