//! Step-2 spike (throwaway, feeds the design doc, not the product table):
//! same workload everywhere — N entities, `x += v * dt` — comparing:
//!   (a) native Bevy query
//!   (b) dynamic `FilteredEntityMut` + `get_by_id`
//!   (c1) unsafe Rust over raw column slices (memory-layout ceiling)
//!   (c1t) same + changed-tick stores (change-detection cost)
//!   (c2) hand-written Cranelift kernel over column pointers (codegen target)
//!   (d) current PlayerRow approach: snapshot + tesjit region loop +
//!       write-back, all inside the timed iteration
//!
//! Setup (world build, query states, descriptors, JIT compile) is hoisted;
//! only the hot loop is timed. Run: `cargo bench -p tesjit --bench ecs_spike`

use std::sync::Arc;
use std::time::Duration;

use bevy_ecs::prelude::*;
use bevy_ecs::component::ComponentId;
use criterion::{Criterion, criterion_group, criterion_main};

use tesjit::runtime::DEFAULT_FUEL;
use tesjit::JitBuilder;

const N: usize = 100_000;
const DT_F32: f32 = 0.016;

#[derive(Component)]
struct Pos(f32);
#[derive(Component)]
struct Vel(f32);
/// Present on every entity, never queried: widens the archetype like a real game.
#[derive(Component)]
struct Seed(f32);

fn build_world() -> World {
    let mut world = World::new();
    for i in 0..N {
        world.spawn((
            Pos(i as f32 * 0.1),
            Vel(1.0),
            Seed(i as f32),
        ));
    }
    world
}

fn component_ids(world: &World) -> (ComponentId, ComponentId) {
    (
        world.component_id::<Pos>().unwrap(),
        world.component_id::<Vel>().unwrap(),
    )
}

// ── (c1)/(c2) chunk descriptors: (len, entities, px, vx) per table ───────

struct Chunk {
    len: usize,
    px: *const f32,
    vx: *const f32,
    ticks: *mut bevy_ecs::change_detection::Tick,
}

// SAFETY: chunks borrow the world; the bench performs no structural changes
// while they are alive.
unsafe fn describe_chunks(world: &World, pos: ComponentId, vel: ComponentId) -> Vec<Chunk> {
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
    let total: usize = out.iter().map(|c| c.len).sum();
    assert_eq!(total, N, "all entities matched");
    out
}

// ── (c2) raw Cranelift kernel: kernel(n, px, vx, dt) ─────────────────────

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

// ── (d) PlayerRow replication ────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Row {
    id: u64,
    x: f32,
    v: f32,
}

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

struct PlayerRows {
    compiled: tesjit::Compiled,
    _guard: tesjit::runtime::Guard,
    rows: Vec<Row>,
}

impl PlayerRows {
    fn new() -> Self {
        let src: Arc<str> = TESJIT_STEP.into();
        let mut jit = JitBuilder::new("rows", src.clone()).build().unwrap();
        jit.compile(&src).unwrap();
        let compiled = jit.finish().unwrap();
        let guard = compiled.enter();
        Self {
            compiled,
            _guard: guard,
            rows: vec![Row::default(); N],
        }
    }
}

// ── groups ───────────────────────────────────────────────────────────────

fn bench_spike(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_100k");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    // (a) native
    {
        let mut world = build_world();
        let mut qs = world.query::<(&mut Pos, &Vel)>();
        group.bench_function("a_native", |b| {
            b.iter(|| {
                for (mut p, v) in qs.iter_mut(&mut world) {
                    p.0 += v.0 * DT_F32;
                }
            })
        });
    }

    // (b) dynamic FilteredEntityMut + get_by_id
    {
        use bevy_ecs::query::QueryBuilder;
        use bevy_ecs::world::FilteredEntityMut;
        let mut world = build_world();
        let (pos, vel) = component_ids(&world);
        let mut qb = QueryBuilder::<FilteredEntityMut>::new(&mut world);
        qb.mut_id(pos);
        qb.ref_id(vel);
        let mut qs = qb.build();
        group.bench_function("b_dynamic", |b| {
            b.iter(|| {
                for mut fem in qs.iter_mut(&mut world) {
                    // Copy out first: the read borrows `fem`, the write
                    // needs it mutably.
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
    }

    // (c1) unsafe Rust over column slices (+ticks variant)
    {
        let mut world = build_world();
        let (pos, vel) = component_ids(&world);
        let tick = world.change_tick();
        let chunks = unsafe { describe_chunks(&world, pos, vel) };
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
                        c.ticks.add(i).write(tick);
                    }
                }
            })
        });
    }

    // (c2) Cranelift kernel over the same descriptors
    {
        let world = build_world();
        let (pos, vel) = component_ids(&world);
        let chunks = unsafe { describe_chunks(&world, pos, vel) };
        let kernel = build_kernel();
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
    }

    // (d) PlayerRow: snapshot + region loop + write-back, all timed
    {
        let mut world = build_world();
        let mut pr = PlayerRows::new();
        let region = pr.compiled.register_region(
            "rows",
            pr.rows.as_mut_ptr() as *mut u8,
            N * std::mem::size_of::<Row>(),
        );
        group.bench_function("d_player_rows", |b| {
            b.iter(|| {
                // snapshot
                let mut q = world.query::<(&Pos, &Vel)>();
                for (i, (p, v)) in q.iter(&world).enumerate() {
                    pr.rows[i] = Row {
                        id: i as u64,
                        x: p.0,
                        v: v.0,
                    };
                }
                // jit loop
                pr.compiled
                    .runtime()
                    .reset_fuel(DEFAULT_FUEL);
                unsafe {
                    let step: extern "C" fn(i32, i32) =
                        pr.compiled.get("step").unwrap();
                    step(region as i32, N as i32);
                }
                // write-back
                let mut q = world.query::<(&mut Pos, &Vel)>();
                for (i, (mut p, _)) in q.iter_mut(&mut world).enumerate() {
                    p.0 = pr.rows[i].x;
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_spike);
criterion_main!(benches);
