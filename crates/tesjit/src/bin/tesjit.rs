//! `tesjit` CLI — compile and run a `.ts` file through the JIT.
//!
//! ```text
//! tesjit <file.ts>          compile, then call `main()` if exported (else list exports)
//! tesjit bench <file.ts>    run the fib benchmark driver
//! ```

use std::sync::Arc;
use std::time::Instant;

use tesjit::{register_host, Compiled, Jit, Ty};

extern "C" fn host_log(x: f64) {
    println!("[host] {x}");
}
register_host!(log, host_log, &[Ty::F64], Ty::Void);

fn main() -> miette::Result<()> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or_else(|| miette::miette!("usage: tesjit <file.ts> | tesjit bench <file.ts>"))?;
    let run_bench = first == "bench";

    let path = if run_bench {
        args.next()
            .ok_or_else(|| miette::miette!("usage: tesjit bench <file.ts>"))?
    } else {
        first
    };

    let src = std::fs::read_to_string(&path)
        .map_err(|e| miette::miette!("failed to read {}: {e}", path))?;
    let src: Arc<str> = src.into();

    let mut jit = Jit::new(&path, src.clone())?;
    jit.compile(&src)?;
    let compiled = jit.finish()?;

    if run_bench {
        bench(&compiled)
    } else {
        run(compiled)
    }
}

/// Compile-and-run: call `main()` when the module exports one, otherwise
/// list what it exports so `compiled.get::<F>(name)` can be used from an
/// embedding host.
fn run(compiled: Compiled) -> miette::Result<()> {
    if let Some(entry) = compiled.fns.get("main") {
        if entry.ret == Ty::Void {
            unsafe { compiled.call_void("main", &[]) }
        } else {
            unsafe { compiled.call_i64("main", &[]) };
        }
        return Ok(());
    }
    println!("no `main` export; available exports:");
    let mut names: Vec<_> = compiled.fns.keys().collect();
    names.sort();
    for name in names {
        let e = &compiled.fns[name];
        println!("  {name}({:?}) -> {:?}", e.params, e.ret);
    }
    Ok(())
}

/// Fib benchmark driver (the historical CLI behaviour).
fn bench(c: &Compiled) -> miette::Result<()> {
    let fib: extern "C" fn(i32) -> i32 = unsafe { c.get("fib").ok_or_else(no_fib)? };
    let fib2: extern "C" fn(i32, i32) -> i32 = unsafe { c.get("fib2").ok_or_else(no_fib)? };

    let inst = Instant::now();
    let f = fib(40);
    println!("{f} {} ns", inst.elapsed().as_nanos());

    let inst = Instant::now();
    let f = fib2(40, 1000000);
    let elapsed = inst.elapsed();
    println!(
        "{f} {} ms, average: {:.2} ns",
        elapsed.as_millis(),
        elapsed.as_nanos() as f64 / 1_000_000.0
    );

    let inst = Instant::now();
    let f = fib(40);
    println!("{f} {} ns", inst.elapsed().as_nanos());
    Ok(())
}

fn no_fib() -> miette::Report {
    miette::miette!("no `fib`/`fib2` exports in module")
}
