//! Zero-copy TS-mod query kernel benchmarks
//! (`server/tests/fixtures/showcase_embers.ts`'s
//! workload, isolated from the server): the JIT query kernels run over REAL
//! Bevy table columns — the exact `QueryInput`/chunk-descriptor ABI the
//! server's `tick_ecs` uses — and the host scan replicates
//! `ts_ecs::QueryPlan::fill` (per-tick descriptor rebuild included in the
//! timed loop, because the server pays it every tick too).
//!
//! World: N entities in two archetypes — 90% bare `Particle`, 10%
//! `Particle + Lifetime + Burning` — matching the showcase's drifters/embers
//! split. Scenarios, in decreasing generosity toward the JIT:
//!   * `native/readonly_sum`   — bevy `&Particle` loop (memory ceiling)
//!   * `jit/readonly`          — `Query<Particle>` + `Mut<Stats>`
//!   * `native/full_tick`      — bevy `&mut` loops, the full two-query tick
//!   * `jit/integrate`         — `Query<Mut<Particle>>` (read-modify-write)
//!   * `jit/integrate_plus_aged` — showcase-shaped: `Mut<Particle>` query +
//!     `Mut<Lifetime>` query filtered `With<Burning>` (host prunes the 90%
//!     of tables that never reach the kernel)
//!   * `jit/write_one_field`, `jit/write_x_plus_sum` (100k only) —
//!     write-cost isolation: one mutable field store per row vs a store +
//!     sum. Against the native rows this shows write-vs-read asymmetry
//!     (~30ns per written row) is shared by native bevy `&mut` loops —
//!     streaming-write cost of the working set, not JIT overhead.
//!   * `host_scan`             — descriptor rebuild only (subtract from any
//!     jit/* number to isolate the kernel)
//!
//! Run: `cargo bench -p tesjit --bench query_showcase`

use std::sync::Arc;
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::query::With;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const DT: f32 = 0.016;
/// Ember (Particle + Lifetime + Burning) fraction, percent.
const EMBER_PERCENT: usize = 10;

#[repr(C)]
#[derive(Component, Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

#[repr(C)]
#[derive(Component, Clone, Copy)]
struct Lifetime {
    ttl: f32,
}

/// Non-ZST marker so the component owns a real table column the scan can
/// test for archetype membership (bevy gives ZST markers no storage).
#[repr(C)]
#[derive(Component, Clone, Copy)]
struct Burning {
    _since: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Stats {
    sum: f64,
}

fn f32_layout(
    name: &str,
    size: u64,
    fields: &[(&str, u64)],
) -> Arc<tesjit::StructLayout> {
    let fields = fields
        .iter()
        .map(|(n, off)| tesjit::StructField {
            name: (*n).to_owned(),
            ty: tesjit::Ty::F32,
            offset: *off,
        })
        .collect();
    Arc::new(tesjit::StructLayout::from_parts(name, fields, size, 4).unwrap())
}

fn particle_layout() -> Arc<tesjit::StructLayout> {
    f32_layout(
        "Particle",
        std::mem::size_of::<Particle>() as u64,
        &[
            ("x", std::mem::offset_of!(Particle, x) as u64),
            ("y", std::mem::offset_of!(Particle, y) as u64),
            ("z", std::mem::offset_of!(Particle, z) as u64),
            ("vx", std::mem::offset_of!(Particle, vx) as u64),
            ("vy", std::mem::offset_of!(Particle, vy) as u64),
            ("vz", std::mem::offset_of!(Particle, vz) as u64),
        ],
    )
}

fn lifetime_layout() -> Arc<tesjit::StructLayout> {
    f32_layout(
        "Lifetime",
        std::mem::size_of::<Lifetime>() as u64,
        &[("ttl", std::mem::offset_of!(Lifetime, ttl) as u64)],
    )
}

fn burning_layout() -> Arc<tesjit::StructLayout> {
    f32_layout(
        "Burning",
        std::mem::size_of::<Burning>() as u64,
        &[("_since", 0)],
    )
}

fn stats_layout() -> Arc<tesjit::StructLayout> {
    Arc::new(
        tesjit::StructLayout::from_parts(
            "Stats",
            vec![tesjit::StructField {
                name: "sum".to_owned(),
                ty: tesjit::Ty::F64,
                offset: std::mem::offset_of!(Stats, sum) as u64,
            }],
            std::mem::size_of::<Stats>() as u64,
            std::mem::align_of::<Stats>() as u64,
        )
        .unwrap(),
    )
}

/// N entities: `EMBER_PERCENT`% carry Lifetime + Burning, the rest are bare.
fn build_world(n: usize) -> World {
    let mut world = World::new();
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
            world.spawn((particle, Lifetime { ttl: 1.0 }, Burning { _since: 0 }));
        } else {
            world.spawn(particle);
        }
    }
    world
}

/// One query's chunk list, as the server rebuilds it every tick.
struct QueryScan {
    descriptors: Vec<u64>,
    chunk_count: u64,
    rows: u64,
}

/// All tables carrying `Particle` (the unfiltered query).
///
/// # Safety
/// No structural world changes while the returned pointers are used.
unsafe fn scan_particles(world: &World, particle: bevy_ecs::component::ComponentId) -> QueryScan {
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

/// Only tables carrying Lifetime AND Burning — the host-side `With<Burning>`
/// archetype filter; the 90% of tables without Burning never reach the
/// kernel.
///
/// # Safety
/// No structural world changes while the returned pointers are used.
unsafe fn scan_embers(
    world: &World,
    lifetime: bevy_ecs::component::ComponentId,
    burning: bevy_ecs::component::ComponentId,
) -> QueryScan {
    let mut descriptors = Vec::new();
    let mut chunk_count = 0u64;
    let mut rows = 0u64;
    for table in world.storages().tables.iter() {
        let (Some(slice), Some(_)) = (
            unsafe { table.get_data_slice_for::<Lifetime>(lifetime) },
            unsafe { table.get_data_slice_for::<Burning>(burning) },
        ) else {
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

const READONLY_SRC: &str = r#"
export function on_tick(q: Query<Particle>, stats: Mut<Stats>): void {
  for (const p of q) {
    stats.sum += p.x as f64;
  }
}
"#;

const INTEGRATE_SRC: &str = r#"
export function on_tick(q: Query<Mut<Particle>>, stats: Mut<Stats>): void {
  for (const p of q) {
    p.vx *= 0.99;
    p.x += p.vx * 0.016;
    stats.sum += p.x as f64;
  }
}
"#;

const INTEGRATE_PLUS_AGED_SRC: &str = r#"
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

const WRITE_ONE_FIELD_SRC: &str = r#"
export function on_tick(q: Query<Mut<Particle>>): void {
  for (const p of q) {
    p.vx *= 0.99;
  }
}
"#;

const WRITE_X_SUM_SRC: &str = r#"
export function on_tick(q: Query<Mut<Particle>>, stats: Mut<Stats>): void {
  for (const p of q) {
    p.x += p.vx * 0.016;
    stats.sum += p.x as f64;
  }
}
"#;

fn compile(src: &str) -> tesjit::Compiled {
    let name = "<query-showcase>";
    let builder = tesjit::JitBuilder::new(name, Arc::from(src))
        .component("Particle", particle_layout())
        .unwrap()
        .component("Lifetime", lifetime_layout())
        .unwrap()
        .component("Burning", burning_layout())
        .unwrap()
        .component("Stats", stats_layout())
        .unwrap();
    let mut jit = builder.build().unwrap();
    jit.compile(src).unwrap();
    let compiled = jit.finish().unwrap();
    // The hot loop over 100k rows would exhaust the per-call budget the
    // server hands handlers; benches get effectively unlimited fuel.
    compiled.runtime().reset_fuel(u64::MAX / 4);
    compiled
}

fn query_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("ts_query");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);

    // JIT kernels are compiled once; only the tick is timed.
    let readonly = compile(READONLY_SRC);
    let integrate = compile(INTEGRATE_SRC);
    let integrate_plus_aged = compile(INTEGRATE_PLUS_AGED_SRC);
    let write_one_field = compile(WRITE_ONE_FIELD_SRC);
    let write_x_sum = compile(WRITE_X_SUM_SRC);
    let _guard = readonly.enter();
    // The outermost `enter()` just reset this runtime's fuel to
    // DEFAULT_FUEL (10M) — far too little for ~1M hot-loop iterations;
    // a mid-run FuelExhausted trap is sticky and would turn every later
    // call into an early return. Re-arm an effectively unlimited budget.
    readonly.runtime().reset_fuel(u64::MAX / 4);
    integrate.runtime().reset_fuel(u64::MAX / 4);
    integrate_plus_aged.runtime().reset_fuel(u64::MAX / 4);
    write_one_field.runtime().reset_fuel(u64::MAX / 4);
    write_x_sum.runtime().reset_fuel(u64::MAX / 4);

    for n in [10_000usize, 100_000] {
        let world = build_world(n);
        let particle = world.component_id::<Particle>().unwrap();
        let lifetime = world.component_id::<Lifetime>().unwrap();
        let burning = world.component_id::<Burning>().unwrap();
        let mut stats = Stats::default();

        // ── baselines ──────────────────────────────────────────────────
        {
            let mut world = build_world(n);
            let mut particles =
                bevy_ecs::query::QueryState::<&Particle>::new(&mut world);
            let id = BenchmarkId::new("native/readonly_sum", n);
            group.bench_with_input(id.clone(), &n, |b, _| {
                b.iter(|| {
                    let mut sum = 0.0f64;
                    for p in particles.iter(&world) {
                        sum += f64::from(p.x);
                    }
                    black_box(sum)
                })
            });
        }
        {
            let mut world = build_world(n);
            let mut particles =
                bevy_ecs::query::QueryState::<&mut Particle>::new(&mut world);
            let mut embers = bevy_ecs::query::QueryState::<&mut Lifetime, With<Burning>>::new(
                &mut world,
            );
            let id = BenchmarkId::new("native/full_tick", n);
            group.bench_with_input(id.clone(), &n, |b, _| {
                b.iter(|| {
                    let mut sum = 0.0f64;
                    for mut p in particles.iter_mut(&mut world) {
                        p.vx *= 0.99;
                        p.x += p.vx * DT;
                        sum += f64::from(p.x);
                    }
                    for mut life in embers.iter_mut(&mut world) {
                        life.ttl -= DT;
                        sum += f64::from(life.ttl);
                    }
                    black_box(sum)
                })
            });
        }

        // ── host-side archetype scan alone ─────────────────────────────
        let id = BenchmarkId::new("host_scan", n);
        group.bench_with_input(id.clone(), &n, |b, _| {
            b.iter(|| {
                let scan = unsafe { scan_particles(&world, particle) };
                black_box((scan.chunk_count, scan.descriptors.len()))
            })
        });

        // ── JIT kernels (scan + kernel, both in the timed loop) ────────
        unsafe {
            type OnTick1 = extern "C" fn(i64, i64);
            let run_readonly: OnTick1 = readonly.get("on_tick").unwrap();
            let id = BenchmarkId::new("jit/readonly", n);
            {
                let scan = scan_particles(&world, particle);
                assert_eq!(scan.rows as usize, n, "particle scan must see every row");
                let input = tesjit::abi::QueryInput {
                    chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                    chunk_count: scan.chunk_count,
                };
                stats = Stats::default();
                run_readonly(
                    (&input as *const tesjit::abi::QueryInput as usize) as i64,
                    (&mut stats as *mut Stats as usize) as i64,
                );
                assert!(stats.sum > 0.0, "readonly kernel did no work");
                assert!(readonly.runtime().trap().is_none(), "readonly trapped");
            }
            group.bench_with_input(id.clone(), &n, |b, _| {
                b.iter(|| {
                    let scan = scan_particles(&world, particle);
                    let input = tesjit::abi::QueryInput {
                        chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                        chunk_count: scan.chunk_count,
                    };
                    run_readonly(
                        (&input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&mut stats as *mut Stats as usize) as i64,
                    );
                    black_box(stats.sum)
                })
            });

            let run_integrate: OnTick1 = integrate.get("on_tick").unwrap();
            let id = BenchmarkId::new("jit/integrate", n);
            {
                let scan = scan_particles(&world, particle);
                assert_eq!(scan.rows as usize, n, "particle scan must see every row");
                let input = tesjit::abi::QueryInput {
                    chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                    chunk_count: scan.chunk_count,
                };
                stats = Stats::default();
                run_integrate(
                    (&input as *const tesjit::abi::QueryInput as usize) as i64,
                    (&mut stats as *mut Stats as usize) as i64,
                );
                assert!(stats.sum > 0.0, "integrate kernel did no work");
                assert!(integrate.runtime().trap().is_none(), "integrate trapped");
            }
            group.bench_with_input(id.clone(), &n, |b, _| {
                b.iter(|| {
                    let scan = scan_particles(&world, particle);
                    let input = tesjit::abi::QueryInput {
                        chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                        chunk_count: scan.chunk_count,
                    };
                    run_integrate(
                        (&input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&mut stats as *mut Stats as usize) as i64,
                    );
                    black_box(stats.sum)
                })
            });

            type OnTick2 = extern "C" fn(i64, i64, i64);
            let run_full: OnTick2 = integrate_plus_aged.get("on_tick").unwrap();
            let id = BenchmarkId::new("jit/integrate_plus_aged", n);
            {
                let all = scan_particles(&world, particle);
                let embers = scan_embers(&world, lifetime, burning);
                assert_eq!(embers.rows as usize, n / 10, "ember filter must see 10%");
                let all_input = tesjit::abi::QueryInput {
                    chunks_addr: all.descriptors.as_ptr() as usize as u64,
                    chunk_count: all.chunk_count,
                };
                let ember_input = tesjit::abi::QueryInput {
                    chunks_addr: embers.descriptors.as_ptr() as usize as u64,
                    chunk_count: embers.chunk_count,
                };
                stats = Stats::default();
                run_full(
                    (&all_input as *const tesjit::abi::QueryInput as usize) as i64,
                    (&ember_input as *const tesjit::abi::QueryInput as usize) as i64,
                    (&mut stats as *mut Stats as usize) as i64,
                );
                assert!(stats.sum > 0.0, "full kernel did no work");
                assert!(integrate_plus_aged.runtime().trap().is_none(), "full trapped");
            }
            group.bench_with_input(id.clone(), &n, |b, _| {
                b.iter(|| {
                    let all = scan_particles(&world, particle);
                    let embers = scan_embers(&world, lifetime, burning);
                    let all_input = tesjit::abi::QueryInput {
                        chunks_addr: all.descriptors.as_ptr() as usize as u64,
                        chunk_count: all.chunk_count,
                    };
                    let ember_input = tesjit::abi::QueryInput {
                        chunks_addr: embers.descriptors.as_ptr() as usize as u64,
                        chunk_count: embers.chunk_count,
                    };
                    run_full(
                        (&all_input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&ember_input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&mut stats as *mut Stats as usize) as i64,
                    );
                    black_box(stats.sum)
                })
            });
        }

        // ── write-cost isolation (100k only) ──────────────────────────
        if n == 100_000 {
            unsafe {
                type OnTick0 = extern "C" fn(i64);
                let run_write: OnTick0 = write_one_field.get("on_tick").unwrap();
                let id = BenchmarkId::new("jit/write_one_field", n);
                {
                    let scan = scan_particles(&world, particle);
                    let input = tesjit::abi::QueryInput {
                        chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                        chunk_count: scan.chunk_count,
                    };
                    // Prior write benchmarks decays vx toward 0 (0.99^N);
                    // reseed so `* 0.99` is observably smaller.
                    let vx0 = world
                        .storages()
                        .tables
                        .iter()
                        .find_map(|t| unsafe { t.get_data_slice_for::<Particle>(particle) }
                            .map(|s| unsafe { &mut *s[0].get() }))
                        .unwrap();
                    vx0.vx = 1.0;
                    let before_x = vx0.vx;
                    run_write((&input as *const tesjit::abi::QueryInput as usize) as i64);
                    let after_x = vx0.vx;
                    assert!(after_x < before_x, "write kernel did no work");
                    assert!(write_one_field.runtime().trap().is_none(), "write trapped");
                }
                group.bench_with_input(id.clone(), &n, |b, _| {
                    b.iter(|| {
                        let scan = scan_particles(&world, particle);
                        let input = tesjit::abi::QueryInput {
                            chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                            chunk_count: scan.chunk_count,
                        };
                        run_write((&input as *const tesjit::abi::QueryInput as usize) as i64);
                    })
                });

                type OnTick1b = extern "C" fn(i64, i64);
                let run_write_sum: OnTick1b = write_x_sum.get("on_tick").unwrap();
                let id = BenchmarkId::new("jit/write_x_plus_sum", n);
                {
                    let scan = scan_particles(&world, particle);
                    let input = tesjit::abi::QueryInput {
                        chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                        chunk_count: scan.chunk_count,
                    };
                    stats = Stats::default();
                    run_write_sum(
                        (&input as *const tesjit::abi::QueryInput as usize) as i64,
                        (&mut stats as *mut Stats as usize) as i64,
                    );
                    assert!(stats.sum > 0.0, "write+sum kernel did no work");
                    assert!(write_x_sum.runtime().trap().is_none(), "write+sum trapped");
                }
                group.bench_with_input(id.clone(), &n, |b, _| {
                    b.iter(|| {
                        let scan = scan_particles(&world, particle);
                        let input = tesjit::abi::QueryInput {
                            chunks_addr: scan.descriptors.as_ptr() as usize as u64,
                            chunk_count: scan.chunk_count,
                        };
                        run_write_sum(
                            (&input as *const tesjit::abi::QueryInput as usize) as i64,
                            (&mut stats as *mut Stats as usize) as i64,
                        );
                        black_box(stats.sum)
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, query_benches);
criterion_main!(benches);
