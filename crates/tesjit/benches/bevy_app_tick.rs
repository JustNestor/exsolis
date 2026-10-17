//! Idiomatic-bevy comparison: a real `bevy_app::App` with plain
//! `#[derive(Component)]` structs — no `repr(C)`, no hand-built world —
//! ticking `app.update()`, against the same app driven by the TS zero-copy
//! query kernel.
//!
//! The four scenarios per size:
//!   * `app/update_only`      — `app.update()` with zero systems: the
//!     schedule's fixed cost (subtract/compose by hand).
//!   * `app/update_native`    — schedule + two real Rust systems
//!     (`Query<&mut Particle>` + `Query<&mut Lifetime, With<Burning>>`,
//!     `ResMut<Stats>`).
//!   * `app/jit_mod_only`     — descriptor scan + JIT call against
//!     `app.world()`; no schedule. What a TS mod's `on_tick` costs.
//!   * `app/update_ts_mod`    — schedule + (scan + JIT), i.e. `app.update()`
//!     with the Rust systems *replaced* by the TS mod. Compare directly
//!     against `app/update_native`.
//!
//! Component layouts handed to `JitBuilder` are described from the REAL
//! `offset_of!` offsets of the plain Rust structs — exactly what the
//! server's `TsSchema` impls do for engine components (no repr(C)
//! required: `offset_of!` reports where the fields actually live).
//!
//! Run: `cargo bench -p tesjit --bench bevy_app_tick`

use std::sync::Arc;
use std::time::Duration;

use bevy_app::{App, Update};
use bevy_ecs::component::ComponentId;
use bevy_ecs::prelude::*;
use bevy_ecs::query::With;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const DT: f32 = 0.016;
const EMBER_PERCENT: usize = 10;

// Plain bevy components — derive only, layout deliberately unspecified.
#[derive(Component)]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

#[derive(Component)]
struct Lifetime {
    ttl: f32,
}

/// Idiomatic ZST marker.
#[derive(Component)]
struct Burning;

#[derive(Resource, Default)]
struct Stats {
    sum: f64,
}

// ── native systems (the baseline) ─────────────────────────────────────────

fn native_integrate(mut particles: Query<&mut Particle>, mut stats: ResMut<Stats>) {
    let stats = &mut *stats;
    for mut p in particles.iter_mut() {
        p.vx *= 0.99;
        p.x += p.vx * DT;
        stats.sum += f64::from(p.x);
    }
}

fn native_age(mut embers: Query<&mut Lifetime, With<Burning>>, mut stats: ResMut<Stats>) {
    let stats = &mut *stats;
    for mut life in embers.iter_mut() {
        life.ttl -= DT;
        stats.sum += f64::from(life.ttl);
    }
}

// ── world construction through the App ────────────────────────────────────

fn app_with_entities(n: usize, with_systems: bool) -> App {
    let mut app = App::new();
    app.insert_resource(Stats::default());
    if with_systems {
        app.add_systems(Update, (native_integrate, native_age).chain());
    }
    let world = app.world_mut();
    let ember_every = 100 / EMBER_PERCENT;
    for i in 0..n {
        let particle = Particle {
            x: i as f32,
            y: 0.0,
            z: 0.0,
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
        };
        if i % ember_every == 0 {
            world.spawn((particle, Lifetime { ttl: 1.0 }, Burning));
        } else {
            world.spawn(particle);
        }
    }
    app
}

// ── layouts for the JIT checker, described from the real struct offsets ───

fn layout(
    name: &str,
    size: usize,
    align: usize,
    fields: &[(&str, u64, tesjit::Ty, u64)], // (name, offset, ty, width)
) -> Arc<tesjit::StructLayout> {
    let fields: Vec<_> = fields
        .iter()
        .map(|(n, off, ty, width)| {
            assert!(
                *off + *width <= size as u64,
                "{n}: field runs past struct end"
            );
            tesjit::StructField {
                name: (*n).to_owned(),
                ty: ty.clone(),
                offset: *off,
            }
        })
        .collect();
    Arc::new(tesjit::StructLayout::from_parts(name, fields, size as u64, align as u64).unwrap())
}

fn particle_layout() -> Arc<tesjit::StructLayout> {
    use tesjit::Ty;
    const W: u64 = std::mem::size_of::<f32>() as u64;
    layout(
        "Particle",
        std::mem::size_of::<Particle>(),
        std::mem::align_of::<Particle>(),
        &[
            ("x", std::mem::offset_of!(Particle, x) as u64, Ty::F32, W),
            ("y", std::mem::offset_of!(Particle, y) as u64, Ty::F32, W),
            ("z", std::mem::offset_of!(Particle, z) as u64, Ty::F32, W),
            ("vx", std::mem::offset_of!(Particle, vx) as u64, Ty::F32, W),
            ("vy", std::mem::offset_of!(Particle, vy) as u64, Ty::F32, W),
            ("vz", std::mem::offset_of!(Particle, vz) as u64, Ty::F32, W),
        ],
    )
}

fn lifetime_layout() -> Arc<tesjit::StructLayout> {
    use tesjit::Ty;
    layout(
        "Lifetime",
        std::mem::size_of::<Lifetime>(),
        std::mem::align_of::<Lifetime>(),
        &[(
            "ttl",
            std::mem::offset_of!(Lifetime, ttl) as u64,
            Ty::F32,
            std::mem::size_of::<f32>() as u64,
        )],
    )
}

fn stats_layout() -> Arc<tesjit::StructLayout> {
    use tesjit::Ty;
    layout(
        "Stats",
        std::mem::size_of::<Stats>(),
        std::mem::align_of::<Stats>(),
        &[(
            "sum",
            std::mem::offset_of!(Stats, sum) as u64,
            Ty::F64,
            std::mem::size_of::<f64>() as u64,
        )],
    )
}

fn burning_layout() -> Arc<tesjit::StructLayout> {
    Arc::new(
        tesjit::StructLayout::from_parts("Burning", vec![], 0, 1).unwrap(),
    )
}

// ── host-side descriptor scan (what the server's QueryPlan::fill does) ────

struct QueryScan {
    descriptors: Vec<u64>,
    chunk_count: u64,
    rows: u64,
}

/// # Safety — no structural world changes while the pointers are used.
unsafe fn scan_particles(world: &World, particle: ComponentId) -> QueryScan {
    let mut descriptors = Vec::new();
    let mut chunk_count = 0u64;
    let mut rows = 0u64;
    for table in world.storages().tables.iter() {
        let Some(slice) = (unsafe { table.get_data_slice_for::<Particle>(particle) }) else {
            continue;
        };
        tesjit::abi::write_chunk(
            &mut descriptors,
            slice.len() as u64,
            &[slice.as_ptr() as *mut u8],
        );
        chunk_count += 1;
        rows += slice.len() as u64;
    }
    QueryScan {
        descriptors,
        chunk_count,
        rows,
    }
}

/// Tables carrying Lifetime AND the (ZST) Burning marker.
///
/// # Safety — no structural world changes while the pointers are used.
unsafe fn scan_embers(world: &World, lifetime: ComponentId, burning: ComponentId) -> QueryScan {
    let mut descriptors = Vec::new();
    let mut chunk_count = 0u64;
    let mut rows = 0u64;
    for table in world.storages().tables.iter() {
        if !table.has_column(burning) {
            continue;
        }
        let Some(slice) = (unsafe { table.get_data_slice_for::<Lifetime>(lifetime) }) else {
            continue;
        };
        tesjit::abi::write_chunk(
            &mut descriptors,
            slice.len() as u64,
            &[slice.as_ptr() as *mut u8],
        );
        chunk_count += 1;
        rows += slice.len() as u64;
    }
    QueryScan {
        descriptors,
        chunk_count,
        rows,
    }
}

// ── the TS mod kernel (same shape as the native systems) ──────────────────

const TS_ON_TICK: &str = r#"
export function on_tick(
  all: Query<Mut<Particle>>,
  embers: Query<Mut<Lifetime>, With<Burning>>,
  stats: Mut<Stats>,
): void {
  for (const p of all) {
    p.vx *= 0.99;
    p.x += p.vx * 0.016;
    stats.sum += p.x as f64;
  }
  for (const life of embers) {
    life.ttl -= 0.016;
    stats.sum += life.ttl as f64;
  }
}
"#;

fn compile_ts() -> tesjit::Compiled {
    let builder = tesjit::JitBuilder::new("<bevy-app-tick>", Arc::from(TS_ON_TICK))
        .component("Particle", particle_layout())
        .unwrap()
        .component("Lifetime", lifetime_layout())
        .unwrap()
        .component("Burning", burning_layout())
        .unwrap()
        .component("Stats", stats_layout())
        .unwrap();
    let mut jit = builder.build().unwrap();
    jit.compile(TS_ON_TICK).unwrap();
    let compiled = jit.finish().unwrap();
    let guard = compiled.enter();
    // Outermost enter() reset fuel to DEFAULT_FUEL (10M); re-arm so the
    // measured iterations can't exhaust it (a sticky FuelExhausted trap
    // would silently early-return every later call).
    compiled.runtime().reset_fuel(u64::MAX / 4);
    drop(guard);
    compiled.runtime().reset_fuel(u64::MAX / 4);
    compiled
}

fn app_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("bevy_app_tick");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);

    let ts = compile_ts();
    type OnTick = extern "C" fn(i64, i64, i64);
    let run_ts: OnTick = unsafe { ts.get("on_tick").unwrap() };

    for n in [10_000usize, 100_000] {
        // ── empty schedule baseline ─────────────────────────────────────
        {
            let mut app = app_with_entities(n, false);
            let id = BenchmarkId::new("app/update_only", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter(|| {
                    app.update();
                })
            });
        }

        // ── the idiomatic native baseline: schedule + two systems ───────
        {
            let mut app = app_with_entities(n, true);
            let id = BenchmarkId::new("app/update_native", n);
            // Warm-up: the systems must actually run.
            {
                app.update();
                let sum = app.world().resource::<Stats>().sum;
                assert!(sum > 0.0, "native systems did no work");
            }
            group.bench_with_input(id, &n, |b, _| {
                b.iter(|| {
                    app.update();
                    black_box(app.world().resource::<Stats>().sum)
                })
            });
        }

        // ── the TS mod path: scan + JIT against the world ───────────────
        let mut ts_app = app_with_entities(n, false);
        let particle = ts_app.world().component_id::<Particle>().unwrap();
        let lifetime = ts_app.world().component_id::<Lifetime>().unwrap();
        let burning = ts_app.world().component_id::<Burning>().unwrap();
        let _guard = ts.enter();
        ts.runtime().reset_fuel(u64::MAX / 4);
        {
            let world = ts_app.world();
            let all = unsafe { scan_particles(world, particle) };
            let embers = unsafe { scan_embers(world, lifetime, burning) };
            assert_eq!(all.rows as usize, n, "particle scan must see every row");
            assert_eq!(
                embers.rows as usize,
                n * EMBER_PERCENT / 100,
                "Burning filter must see 10%"
            );
            let all_input = tesjit::abi::QueryInput {
                chunks_addr: all.descriptors.as_ptr() as usize as u64,
                chunk_count: all.chunk_count,
            };
            let ember_input = tesjit::abi::QueryInput {
                chunks_addr: embers.descriptors.as_ptr() as usize as u64,
                chunk_count: embers.chunk_count,
            };
            let mut stats = Stats::default();
            run_ts(
                (&all_input as *const tesjit::abi::QueryInput as usize) as i64,
                (&ember_input as *const tesjit::abi::QueryInput as usize) as i64,
                (&mut stats as *mut Stats as usize) as i64,
            );
            assert!(ts.runtime().trap().is_none(), "ts kernel trapped");
            assert!(stats.sum > 0.0, "ts kernel did no work");
        }
        {
            let id = BenchmarkId::new("app/jit_mod_only", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter(|| {
                    let world = ts_app.world();
                    let all = unsafe { scan_particles(world, particle) };
                    let embers = unsafe { scan_embers(world, lifetime, burning) };
                    let all_input = tesjit::abi::QueryInput {
                        chunks_addr: all.descriptors.as_ptr() as usize as u64,
                        chunk_count: all.chunk_count,
                    };
                    let ember_input = tesjit::abi::QueryInput {
                        chunks_addr: embers.descriptors.as_ptr() as usize as u64,
                        chunk_count: embers.chunk_count,
                    };
                    run_ts(
                        (&all_input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&ember_input as *const tesjit::abi::QueryInput as usize) as i64,
                        // Reuse the app's own Stats resource, like the
                        // server hands systems their resource by pointer.
                        (ts_app
                            .world_mut()
                            .resource_mut::<Stats>()
                            .as_mut() as *mut Stats
                            as usize)
                            as i64,
                    );
                    black_box(ts_app.world().resource::<Stats>().sum)
                })
            });
        }

        // ── schedule + TS mod: apples-to-apples vs app/update_native ────
        {
            let mut app = app_with_entities(n, false);
            let particle = app.world().component_id::<Particle>().unwrap();
            let lifetime = app.world().component_id::<Lifetime>().unwrap();
            let burning = app.world().component_id::<Burning>().unwrap();
            let id = BenchmarkId::new("app/update_ts_mod", n);
            group.bench_with_input(id, &n, |b, _| {
                b.iter(|| {
                    app.update();
                    let world = app.world();
                    let all = unsafe { scan_particles(world, particle) };
                    let embers = unsafe { scan_embers(world, lifetime, burning) };
                    let all_input = tesjit::abi::QueryInput {
                        chunks_addr: all.descriptors.as_ptr() as usize as u64,
                        chunk_count: all.chunk_count,
                    };
                    let ember_input = tesjit::abi::QueryInput {
                        chunks_addr: embers.descriptors.as_ptr() as usize as u64,
                        chunk_count: embers.chunk_count,
                    };
                    run_ts(
                        (&all_input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&ember_input as *const tesjit::abi::QueryInput as usize) as i64,
                        (app.world_mut().resource_mut::<Stats>().as_mut() as *mut Stats
                            as usize)
                            as i64,
                    );
                    black_box(app.world().resource::<Stats>().sum)
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, app_benches);
criterion_main!(benches);
