//! Timing runner: prints one JSON doc to stdout.
//! Usage: `cargo run --release --manifest-path crates/lua-bench-luajit/Cargo.toml`

use std::time::Instant;

use lua_bench_luajit::Ecs;

const BACKEND: &str = "luajit";
const FIB_N: i64 = 22;
const FIB_ITERS: u32 = 30;
const ECS_N: usize = 10_000;
const ECS_STEPS: u32 = 200;
const DT: f64 = 0.016;

fn main() {
    // Correctness first: wrong numbers invalidate the timing.
    assert_eq!(lua_bench_luajit::fib(FIB_N), 17711);
    let ecs = Ecs::new(ECS_N);
    assert_eq!(ecs.len(), ECS_N);

    for _ in 0..5 {
        lua_bench_luajit::fib(FIB_N);
    }
    let t0 = Instant::now();
    for _ in 0..FIB_ITERS {
        lua_bench_luajit::fib(FIB_N);
    }
    let fib_ns = t0.elapsed().as_nanos() as f64 / f64::from(FIB_ITERS);

    for _ in 0..5 {
        ecs.step(DT);
    }
    let expected = 0.1 * (ECS_N * (ECS_N + 1) / 2) as f64
        + (ECS_N as f64) * f64::from(ECS_STEPS + 5) * DT;
    let t0 = Instant::now();
    for _ in 0..ECS_STEPS {
        ecs.step(DT);
    }
    let ecs_total_ns = t0.elapsed().as_nanos() as f64;
    let sum = ecs.sum();
    let rel_err = ((sum - expected) / expected).abs();
    assert!(
        rel_err < 1e-9,
        "ecs checksum drifted: got {sum}, want {expected}"
    );

    println!(
        "{{\"backend\":\"{BACKEND}\",\"fib_n\":{FIB_N},\"fib_iters\":{FIB_ITERS},\
        \"fib_ns_per_call\":{fib_ns:.1},\"ecs_n\":{ECS_N},\"ecs_steps\":{ECS_STEPS},\
        \"ecs_ns_per_row_per_step\":{:.3}}}",
        ecs_total_ns / (f64::from(ECS_STEPS) * ECS_N as f64),
    );
}
