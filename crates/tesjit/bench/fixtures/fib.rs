use std::time::Instant;

fn fibonacci(n: i64) -> i64 {
    if n <= 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut a = 0;
    let mut b = 1;
    for _ in 2..=n {
        let temp = b;
        b = a + b;
        a = temp;
    }
    b
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: i64 = if args.len() > 1 {
        args[1].parse().unwrap_or(40)
    } else {
        40
    };
    let iterations: i64 = 1_000_000;

    // Одиничний виклик
    let t0 = Instant::now();
    let res = fibonacci(n);
    let t1 = t0.elapsed();
    println!("Fibonacci({}) = {}", n, res);
    println!("Single run: {} ns\n", t1.as_nanos());

    // Бенчмарк із захистом від LICM, DCE та Black Box для гарантії виконання
    let mut accumulator = 0i64;
    let bench_start = Instant::now();

    for i in 0..iterations {
        let dynamic_n = 38 + (i % 3);
        accumulator += fibonacci(dynamic_n);
    }

    let bench_end = bench_start.elapsed();

    // Запобіжник від DCE
    if accumulator == -1 {
        println!("dummy {}", accumulator);
    }

    let total_ms = bench_end.as_secs_f64() * 1000.0;
    let avg_ns = bench_end.as_nanos() as f64 / iterations as f64;

    println!("Benchmark ({} iterations):", iterations);
    println!("  Total time: {:.3} ms", total_ms);
    println!("  Average time per call: {:.2} ns", avg_ns);
}
