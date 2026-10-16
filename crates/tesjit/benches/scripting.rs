//! In-workspace scripting benchmarks (criterion): tesjit vs native Rust vs
//! native Bevy queries.
//!
//! LuaJIT / Luau / PUC-Lua numbers come from the isolated runner crates
//! (`crates/lua-bench-*`, one cargo graph each — mlua-sys declares
//! `links = "lua"`, so two backends can never coexist in one build) and the
//! bevy_mod_scripting dispatch number from `crates/bms-bench`. Run everything
//! with `just bench-scripting`, which prints the combined table.
//!
//! Workloads (identical across all runners):
//! - `fib`: recursive `fibonacci(22)` — call overhead per runtime.
//! - `ecs_step`: 10k rows, `x += v * dt` — loop/iteration cost.
//!   Lua backends keep their tables in Lua; tesjit keeps its arrays in the
//!   module arena; Rust/Bevy iterate native memory. The BMS runner measures
//!   `on_update` dispatch through a headless App (event + full update tick).
//!
//! Run: `cargo bench -p tesjit --bench scripting`

use std::sync::Arc;
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use criterion::black_box;
use tesjit::runtime::DEFAULT_FUEL;

use tesjit::JitBuilder;
use bevy_ecs::component::ComponentId;

// ── Rust baselines ────────────────────────────────────────────────────

fn rust_fib(n: i64) -> i64 {
    if n < 2 { n } else { rust_fib(n - 1) + rust_fib(n - 2) }
}

fn rust_ecs_step(rows: &mut [f64], dt: f64) {
    for pair in rows.chunks_exact_mut(2) {
        pair[0] += pair[1] * dt;
    }
}

// ── tesjit ────────────────────────────────────────────────────────────

const TESJIT_FIB: &str = r#"
function fibonacci(n: i32): i32 {
    if (n < 2) { return n; }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
"#;

const TESJIT_ECS: &str = r#"
function make(n: i32): f64[] {
    let a: f64[] = [0.0];
    for (let i: i32 = 1; i < 2 * n; i++) { a.push(i as f64 * 0.1); }
    return a;
}
function step(a: f64[], dt: f64): void {
    let n: i32 = a.length;
    for (let i: i32 = 0; i < n; i += 2) { a[i] = a[i] + a[i + 1] * dt; }
}
function total(a: f64[]): f64 {
    let s: f64 = 0;
    for (let i: i32 = 0; i < a.length; i += 2) { s += a[i]; }
    return s;
}
"#;

struct TesjitFib {
    compiled: tesjit::Compiled,
    _guard: tesjit::runtime::Guard,
}

impl TesjitFib {
    fn new() -> Self {
        let src: Arc<str> = TESJIT_FIB.into();
        let mut jit = JitBuilder::new("fib", src.clone()).build().unwrap();
        jit.compile(&src).unwrap();
        let compiled = jit.finish().unwrap();
        let guard = compiled.enter();
        Self {
            compiled,
            _guard: guard,
        }
    }

    fn call(&self, n: i32) -> i32 {
        // Production hands each handler a fresh budget (`ts_backend` resets
        // per call); without this the fuel traps after ~100 iterations and
        // the bench times the instant-trap path instead of the work.
        self.compiled.runtime().reset_fuel(DEFAULT_FUEL);
        unsafe {
            let f: extern "C" fn(i32) -> i32 = self.compiled.get("fibonacci").unwrap();
            f(n)
        }
    }
}

struct TesjitEcs {
    compiled: tesjit::Compiled,
    _guard: tesjit::runtime::Guard,
    arr: i64,
}

impl TesjitEcs {
    fn new(n: i32) -> Self {
        let src: Arc<str> = TESJIT_ECS.into();
        let mut jit = JitBuilder::new("ecs", src.clone()).build().unwrap();
        jit.compile(&src).unwrap();
        let compiled = jit.finish().unwrap();
        let guard = compiled.enter();
        let arr = unsafe {
            let make: extern "C" fn(i32) -> i64 = compiled.get("make").unwrap();
            make(n)
        };
        Self {
            compiled,
            _guard: guard,
            arr,
        }
    }

    fn step(&self, dt: f64) {
        self.compiled.runtime().reset_fuel(DEFAULT_FUEL);
        unsafe {
            let step: extern "C" fn(i64, f64) = self.compiled.get("step").unwrap();
            step(self.arr, dt);
        }
    }

    fn total(&self) -> f64 {
        unsafe {
            let total: extern "C" fn(i64) -> f64 = self.compiled.get("total").unwrap();
            total(self.arr)
        }
    }
}

// ── chunk descriptors for column‑ptr bench ──────────────────────────────

struct Chunk {
    len: usize,
    px: *const f32,
    vx: *const f32,
    ticks: *mut bevy_ecs::change_detection::Tick,
}

unsafe fn describe_chunks(world: &bevy_ecs::world::World, pos: ComponentId, vel: ComponentId) -> Vec<Chunk> {
    let mut out = Vec::new();
    for table in world.storages().tables.iter() {
        let (Some(px), Some(vx)) = (unsafe {
            (
                table.get_data_slice_for::<f32>(pos),
                table.get_data_slice_for::<f32>(vel),
            )
        }) else {
            continue;
        };
        let ticks = unsafe {
            table
                .get_changed_ticks_slice_for(pos)
                .map(|s| s.as_ptr() as *mut bevy_ecs::change_detection::Tick)
        }
        .unwrap_or(std::ptr::null_mut());
        out.push(Chunk {
            len: table.entity_count() as usize,
            px: px.as_ptr() as *const f32,
            vx: vx.as_ptr() as *const f32,
            ticks,
        });
    }
    out
}

// ── (c2) hand-written Cranelift kernel over column pointers ──────────────

type KernelFn = unsafe extern "C" fn(i64, i64, i64, f32);

struct Kernel {
    _module: cranelift_jit::JITModule,
    f: KernelFn,
}

fn build_kernel() -> Kernel {
    use cranelift_codegen::ir::{AbiParam, types};
    use cranelift_codegen::isa::TargetFrontendConfig;
    use cranelift_codegen::settings::Configurable;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();
    let flags = cranelift_codegen::settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(flags)
        .unwrap();
    let target: TargetFrontendConfig = isa.frontend_config();
    let mut module = JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::F32));
    let func_id = module
        .declare_function("kernel", Linkage::Local, &sig)
        .unwrap();

    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        use cranelift_codegen::ir::InstBuilder;
        let flags = cranelift_codegen::ir::MachMemFlags::new();
        let entry = b.create_block();
        let head = b.create_block();
        let body = b.create_block();
        let exit = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        let zero = b.ins().iconst(types::I64, 0);
        b.ins().jump(head, &[zero.into()]);
        b.switch_to_block(head);
        let i = b.append_block_param(head, types::I64);
        let n_v = b.block_params(entry)[0];
        let px_v = b.block_params(entry)[1];
        let vx_v = b.block_params(entry)[2];
        let dt_v = b.block_params(entry)[3];
        let done = b.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual,
            i,
            n_v,
        );
        b.ins().brif(done, exit, &[], body, &[]);
        b.switch_to_block(body);
        let off = b.ins().imul_imm(i, 4);
        let pa = b.ins().iadd(px_v, off);
        let va = b.ins().iadd(vx_v, off);
        let x = b.ins().load(types::F32, flags, pa, 0);
        let v = b.ins().load(types::F32, flags, va, 0);
        let step = b.ins().fmul(v, dt_v);
        let nx = b.ins().fadd(x, step);
        b.ins().store(flags, nx, pa, 0);
        let ni = b.ins().iadd_imm(i, 1);
        b.ins().jump(head, &[ni.into()]);
        b.switch_to_block(exit);
        b.seal_block(exit);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(target);
    }
    let _ = target;
    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();
    let ptr = module.get_finalized_function(func_id);
    // SAFETY: signature matches KernelFn exactly.
    let f: KernelFn = unsafe { std::mem::transmute(ptr) };
    Kernel {
        _module: module,
        f,
    }
}

// ── native Bevy ───────────────────────────────────────────────────────

#[derive(bevy_ecs::prelude::Component)]
struct Pos(f32);

#[derive(bevy_ecs::prelude::Component)]
struct Vel(f32);

struct BevyWorld {
    world: bevy_ecs::world::World,
}

impl BevyWorld {
    fn new(n: usize) -> Self {
        let mut world = bevy_ecs::world::World::new();
        for i in 0..n {
            world.spawn((Pos(i as f32 * 0.1), Vel(1.0)));
        }
        Self { world }
    }

    fn step(&mut self, dt: f32) {
        let mut query = self
            .world
            .query::<(&mut Pos, &Vel)>();
        for (mut p, v) in query.iter_mut(&mut self.world) {
            p.0 += v.0 * dt;
        }
    }
}

// ── bevy_mod_scripting lives in crates/bms-bench ────────────────────────
// (isolated graph: its mlua 0.11 would collide with this workspace's
// mlua 0.12 at resolve time — mlua-sys `links = "lua"`).

// ── criterion groups ──────────────────────────────────────────────────

fn bench_spike(c: &mut Criterion) {
    const N: usize = 100_000;
    const DT_F32: f32 = 0.016;

    use bevy_ecs::prelude::*;
    use bevy_ecs::component::ComponentId;
use bevy_ecs::world::World;
    use bevy_ecs::query::QueryBuilder;
    use bevy_ecs::world::FilteredEntityMut;
    use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

    // (a) native Bevy query
    {
        let mut world = bevy_ecs::world::World::new();
        for i in 0..N {
            world.spawn((
                Pos(i as f32 * 0.1),
                Vel(1.0),
            ));
        }
        let mut qs = world.query::<(&mut Pos, &Vel)>();
        let mut group = c.benchmark_group("spike_100k");
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(20);
        group.bench_function("a_native", |b| {
            b.iter(|| {
                for (mut p, v) in qs.iter_mut(&mut world) {
                    p.0 += v.0 * DT_F32;
                }
            })
        });
        group.finish();
    }

    // (b) dynamic FilteredEntityMut + get_by_id
    {
        use bevy_ecs::world::FilteredEntityMut;
        let mut world = bevy_ecs::world::World::new();
        for i in 0..N {
            world.spawn((
                Pos(i as f32 * 0.1),
                Vel(1.0),
            ));
        }
        let (pos, vel) = (
            world.component_id::<Pos>().unwrap(),
            world.component_id::<Vel>().unwrap(),
        );
        let mut qb = QueryBuilder::<FilteredEntityMut>::new(&mut world);
        qb.mut_id(pos);
        qb.ref_id(vel);
        let mut qs = qb.build();
        let mut group = c.benchmark_group("spike_100k");
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(20);
        group.bench_function("b_dynamic", |b| {
            b.iter(|| {
                for mut fem in qs.iter_mut(&mut world) {
                    let vv: f32 =
                        unsafe { *fem.get_by_id(vel).unwrap().deref::<f32>() };
                    let mut p = fem.get_mut_by_id(pos).unwrap();
                    unsafe {
                        let px = p.as_mut().deref_mut::<f32>();
                        *px += vv * DT_F32;
                    }
                }
            })
        });
        group.finish();
    }

    // (c1) unsafe Rust over column slices
    {
        let mut world = bevy_ecs::world::World::new();
        for i in 0..N {
            world.spawn((
                Pos(i as f32 * 0.1),
                Vel(1.0),
            ));
        }
        let (pos, vel) = (
            world.component_id::<Pos>().unwrap(),
            world.component_id::<Vel>().unwrap(),
        );
        let chunks = unsafe { describe_chunks(&world, pos, vel) };
        let mut group = c.benchmark_group("spike_100k");
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(20);
        group.bench_function("c1_columns", |b| {
            b.iter(|| unsafe {
                for c in &chunks {
                    for i in 0..c.len {
                        let x = c.px.add(i).read();
                        let v = c.vx.add(i).read();
                        c.px.add(i).cast_mut().write(x + v * DT_F32);
                    }
                }
            })
        });
        group.bench_function("c1t_columns_ticks", |b| {
            b.iter(|| unsafe {
                for c in &chunks {
                    for i in 0..c.len {
                        let x = c.px.add(i).read();
                        let v = c.vx.add(i).read();
                        c.px.add(i).cast_mut().write(x + v * DT_F32);
                        c.ticks.add(i).write(world.change_tick());
                    }
                }
            })
        });
        group.finish();
    }

    // (c2) Cranelift kernel over column pointers
    {
        let mut world = bevy_ecs::world::World::new();
        for i in 0..N {
            world.spawn((
                Pos(i as f32 * 0.1),
                Vel(1.0),
            ));
        }
        let (pos, vel) = (
            world.component_id::<Pos>().unwrap(),
            world.component_id::<Vel>().unwrap(),
        );
        let chunks = unsafe { describe_chunks(&world, pos, vel) };
        let kernel = build_kernel();
        let mut group = c.benchmark_group("spike_100k");
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(20);
        group.bench_function("c2_cranelift", |b| {
            b.iter(|| unsafe {
                for c in &chunks {
                    (kernel.f)(
                        c.len as i64,
                        c.px as i64,
                        c.vx as i64,
                        DT_F32,
                    );
                }
            })
        });
        group.finish();
    }

    // (d) PlayerRow: snapshot + region loop + write-back
    {
        const TESJIT_STEP: &str = r#"
function step(r: i32, n: i32): void {
    for (let i: i32 = 0; i < n; i++) {
        let base: i32 = i * 16;
        let x: f64 = load_f32(r, base + 8);
        let vv: f64 = load_f32(r, base + 12);
        store_f32(r, base + 8, x + vv * 0.016);
    }
}
"#;

        let mut world = bevy_ecs::world::World::new();
        for i in 0..N {
            world.spawn((
                Pos(i as f32 * 0.1),
                Vel(1.0),
            ));
        }
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        struct Row {
            id: u64,
            x: f32,
            v: f32,
        }
        let mut rows: Vec<Row> = (0..N).map(|i| Row { id: i as u64, x: 0.0, v: 0.0 }).collect();
        let mut jit = tesjit::JitBuilder::new("rows", std::borrow::ToOwned::to_owned(&TESJIT_STEP)).build().unwrap();
        jit.compile(&TESJIT_STEP).unwrap();
        let compiled = jit.finish().unwrap();
        let guard = compiled.enter();
        let region = compiled.register_region(
            "rows",
            rows.as_mut_ptr() as *mut u8,
            N * std::mem::size_of::<Row>(),
        );
        let region_i = region as i32;
        let mut group = c.benchmark_group("spike_100k");
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(20);
        group.bench_function("d_player_rows", |b| {
            b.iter(|| {
                // snapshot
                let mut q = world.query::<(&Pos, &Vel)>();
                for (i, (p, v)) in q.iter(&world).enumerate() {
                    rows[i] = Row {
                        id: i as u64,
                        x: p.0,
                        v: v.0,
                    };
                }
                // jit loop
                compiled.runtime().reset_fuel(tesjit::runtime::DEFAULT_FUEL);
                unsafe {
                    let step: extern "C" fn(i32, i32) = compiled.get("step").unwrap();
                    step(region as i32, N as i32);
                }
                // write-back
                let mut q = world.query::<(&mut Pos, &Vel)>();
                for (i, (mut p, _)) in q.iter_mut(&mut world).enumerate() {
                    p.0 = rows[i].x;
                }
            })
        });
        group.finish();
    }
}

fn bench_fib(c: &mut Criterion) {
    let mut group = c.benchmark_group("fib_22");
    group.sample_size(20);

    group.bench_function("rust", |b| b.iter(|| rust_fib(black_box(22))));

    let jit = TesjitFib::new();
    assert_eq!(jit.call(22), rust_fib(22) as i32);
    group.bench_function("tesjit", |b| b.iter(|| jit.call(black_box(22))));

    group.finish();
}

fn bench_ecs(c: &mut Criterion) {
    const N: usize = 10_000;
    const DT_F64: f64 = 0.016;
    let mut group = c.benchmark_group("ecs_step_10k");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    // Native Rust over one interleaved vec (same layout as the tesjit bench).
    let mut rows: Vec<f64> = (0..2 * N).map(|i| i as f64 * 0.1).collect();
    group.bench_function("rust_vec", |b| {
        b.iter(|| rust_ecs_step(&mut rows, DT_F64))
    });

    let jit = TesjitEcs::new(N as i32);
    group.bench_function("tesjit", |b| b.iter(|| jit.step(DT_F64)));
    let _ = jit.total();

    let mut bevy = BevyWorld::new(N);
    group.bench_function("bevy_query", |b| {
        b.iter_batched(
            || 0.016f32,
            |dt| bevy.step(dt),
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_fib, bench_ecs, bench_spike);
criterion_main!(benches);
