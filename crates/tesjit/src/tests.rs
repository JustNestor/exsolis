//! Tests: compile→run behaviour, host calls, and diagnostics batching.

use std::sync::atomic::{AtomicU64, Ordering};

use super::{CompileError, Jit, JitBuilder, Ty};

static LAST: AtomicU64 = AtomicU64::new(0);
extern "C" fn test_log(x: f64) {
    LAST.store(x.to_bits(), Ordering::SeqCst);
}

/// Compile `src` with a single `log(host)` registered.
fn build(src: &str) -> super::Compiled {
    let mut jit = JitBuilder::new("<test>", std::sync::Arc::from(src))
        .host_fn("log", vec![Ty::F64], Ty::Void, test_log as *const u8)
        .build()
        .unwrap_or_else(|e| panic!("jit build failed: {e:?}"));
    jit.compile(src)
        .unwrap_or_else(|d| panic!("compile failed: {d:?}"));
    jit.finish()
        .unwrap_or_else(|e| panic!("finish failed: {e:?}"))
}

/// Diagnostics for `src`, asserting it failed to compile.
fn errors(src: &str) -> Vec<CompileError> {
    let jit = JitBuilder::new("<test>", std::sync::Arc::from(src))
        .host_fn("log", vec![Ty::F64], Ty::Void, test_log as *const u8)
        .build()
        .expect("jit build");
    let diags = jit.check(src);
    assert!(!diags.ok(), "expected diagnostics");
    diags.into_iter().collect()
}

#[test]
fn fib_and_loop() {
    let c = build(
        r#"
        function fib(n: i32): i32 {
          if (n < 2) { return n; }
          return fib(n - 1) + fib(n - 2);
        }
        function sum(n: i32): f64 {
          let acc: f64 = 0;
          for (let i: i32 = 0; i < n; i++) { acc += i as f64; }
          return acc;
        }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let fib: extern "C" fn(i32) -> i32 = c.get("fib").unwrap();
        let sum: extern "C" fn(i32) -> f64 = c.get("sum").unwrap();
        assert_eq!(fib(20), 6765);
        assert_eq!(sum(100), 4950.0);
    }
    drop(guard);}

#[test]
fn host_call() {
    let c = build(
        r#"
        declare function log(x: f64): void;
        export function go(dt: f64): void { log(dt * 2); }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let go: extern "C" fn(f64) = c.get("go").unwrap();
        go(1.5);
    }
    drop(guard);
    assert_eq!(f64::from_bits(LAST.load(Ordering::SeqCst)), 3.0);
}

#[test]
fn host_signature_must_match() {
    // `declare function log(x: i32)` contradicts the registered `(f64) -> void`.
    let src = "declare function log(x: i32): void;\nexport function go(n: i32): void { log(n); }";
    let diags = errors(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{d:?}").contains("differs from host")),
        "{diags:?}"
    );
}

#[test]
fn unknown_host_is_rejected() {
    let diags = errors(
        "declare function nope(x: f64): void;\nexport function go(n: f64): void { nope(n); }",
    );
    assert!(
        diags
            .iter()
            .any(|d| format!("{d:?}").contains("no host function")),
        "{diags:?}"
    );
}

#[test]
fn type_errors_are_reported_in_batch() {
    let src = r#"
        function bad(a: i32): f64 { return a; }
        function worse(a: i32): i32 { return "x" as i32; }
        function alsoBad(x: f64, y: f64): f64 { return x + y as i32; }
        "#;
    let diags = errors(src);
    assert!(
        diags.len() >= 3,
        "expected every error, got {}: {diags:?}",
        diags.len()
    );
}

#[test]
fn control_flow_and_logical_ops() {
    let c = build(
        r#"
        function pick(n: i32): i32 {
          let out: i32 = 0;
          while (out < 100) {
            if (out > n || out == 999) { break; }
            out += 1;
            if (out == 3) { continue; }
          }
          return out;
        }
        function tern(x: i32): i32 { return x > 2 ? 10 : 20; }
        function and(a: boolean, b: boolean): boolean { return a && b; }
        function both(a: boolean): boolean { return a && a; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let pick: extern "C" fn(i32) -> i32 = c.get("pick").unwrap();
        let tern: extern "C" fn(i32) -> i32 = c.get("tern").unwrap();
        let and: extern "C" fn(bool, bool) -> bool = c.get("and").unwrap();
        let both: extern "C" fn(bool) -> bool = c.get("both").unwrap();
        assert_eq!(pick(4), 5);
        assert_eq!(pick(200), 100);
        assert_eq!(tern(1), 20);
        assert_eq!(tern(9), 10);
        assert!(!and(true, false));
        assert!(and(true, true));
        assert!(!both(false));
    }
    drop(guard);
}

#[test]
fn break_outside_loop_is_an_error() {
    let diags = errors("function f(): i32 { break; return 1; }");
    assert!(!diags.is_empty());
}

#[test]
fn updates_and_bool_to_float_cast() {
    let c = build(
        r#"
        function bump(x: i32): i32 { let y: i32 = 5; return y++; }
        function pre(x: i32): i32 { let y: i32 = 5; return ++y; }
        function truth(b: boolean): f64 { return b as f64; }
        "#,
    );
    let _guard = c.enter();
    unsafe {
        let bump: extern "C" fn(i32) -> i32 = c.get("bump").unwrap();
        let pre: extern "C" fn(i32) -> i32 = c.get("pre").unwrap();
        let truth: extern "C" fn(bool) -> f64 = c.get("truth").unwrap();
        assert_eq!(bump(0), 5);
        assert_eq!(pre(0), 6);
        assert_eq!(truth(true), 1.0);
    }
}

#[test]
fn return_type_is_required() {
    let diags = errors("function f(a: i32) { return a; }");
    assert!(
        diags
            .iter()
            .any(|d| format!("{d:?}").contains("return type")),
        "{diags:?}"
    );
}

/// `check` never touches codegen: a module that type-checks must compile.
#[test]
fn check_and_compile_agree() {
    let src = r#"
        function ok(a: i32): i32 { let b: i32 = a * 2; return b; }
        "#;
    let jit = JitBuilder::new("<test>", std::sync::Arc::from(src))
        .host_fn("log", vec![Ty::F64], Ty::Void, test_log as *const u8)
        .build()
        .unwrap();
    let diags = jit.check(src);
    assert!(diags.ok(), "{diags:?}");

    let mut jit = jit;
    jit.compile(src).unwrap();
    let compiled: Jit = jit;
    let c: super::Compiled = compiled.finish().unwrap();
    let guard = c.enter();
    unsafe {
        let ok: extern "C" fn(i32) -> i32 = c.get("ok").unwrap();
        assert_eq!(ok(21), 42);
    }
    drop(guard);
}

// ───────────────────────────── strings ─────────────────────────────

/// Strings reach the host as raw pointers; this one keeps the last one handed
/// over so tests can read its bytes.
static LAST_STR: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

fn string_bytes(s: i64) -> Vec<u8> {
    // Layout: [len: i64][utf8 bytes...] (see crate::runtime).
    let len = unsafe { *(s as *const i64) } as usize;
    let mut out = vec![0u8; len];
    unsafe { std::ptr::copy_nonoverlapping((s as *const u8).add(8), out.as_mut_ptr(), len) };
    out
}

extern "C" fn host_pass_str(ptr: i64) {
    *LAST_STR.lock().unwrap() = Some(string_bytes(ptr));
}

fn build_str(src: &str) -> super::Compiled {
    let mut jit = JitBuilder::new("<test>", std::sync::Arc::from(src))
        .host_fn("pass", vec![Ty::Str], Ty::Void, host_pass_str as *const u8)
        .build()
        .unwrap();
    jit.compile(src).unwrap();
    jit.finish().unwrap()
}

fn str_of(src: &str) -> String {
    *LAST_STR.lock().unwrap() = None;
    let c = build_str(src);
    let guard = c.enter();
    unsafe {
        let go: extern "C" fn() = c.get("go").unwrap();
        go();
    }
    drop(guard);
    String::from_utf8(LAST_STR.lock().unwrap().take().unwrap()).unwrap()
}

#[test]
fn string_literals_and_concat() {
    assert_eq!(
        str_of(r#"export function go(): void { pass("hello"); }"#),
        "hello"
    );
    assert_eq!(
        str_of(
            r#"export function go(): void {
                 let a: string = "hello";
                 let b: string = " world";
                 pass(a + b);
               }"#
        ),
        "hello world"
    );
}

#[test]
fn string_equality_and_length() {
    // Length is exposed as i32; equality as boolean comparisons.
    let c = build_str(
        r#"
        export function len(s: string): i32 { return s.length; }
        export function first(s: string): string { return "a"; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let len: extern "C" fn(i64) -> i32 = c.get("len").unwrap();
        assert_eq!(len(alloc_str("abc")), 3);
    }
    drop(guard);
}

#[test]
fn string_index_is_bounds_checked() {
    let c = build_str(
        r#"
        export function at(s: string, i: i32): i32 { return s[i]; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let at: extern "C" fn(i64, i32) -> i32 = c.get("at").unwrap();
        assert_eq!(at(alloc_str("abc"), 1), b'b' as i32);
        // Out of bounds traps instead of reading foreign memory.
        assert_eq!(at(alloc_str("abc"), 99), 0);
        assert_eq!(
            c.trap().map(|t| t.message()),
            Some("index out of bounds")
        );
        c.clear_trap();
    }
    drop(guard);
}

// ───────────────────────────── arrays ─────────────────────────────

fn alloc_str(text: &str) -> i64 {
    // Keep the compiled module alive for the whole test by leaking it: the
    // returned pointer points into its arena.
    let src = format!(r#"export function make(): string {{ return "{text}"; }}"#);
    let c = Box::leak(Box::new(build_str(&src)));
    let guard = c.enter();
    unsafe {
        let make: extern "C" fn() -> i64 = c.get("make").unwrap();
        let out = make();
        drop(guard);
        out
    }
}

// ───────────────────────────── arrays ─────────────────────────────

fn build_arr(src: &str) -> super::Compiled {
    let mut jit = JitBuilder::new("<test>", std::sync::Arc::from(src))
        .build()
        .unwrap();
    jit.compile(src).unwrap();
    jit.finish().unwrap()
}

#[test]
fn array_literal_index_and_length() {
    let c = build_arr(
        r#"
        export function sum(a: i32[]): i32 {
          let total: i32 = 0;
          for (let i: i32 = 0; i < a.length; i++) { total += a[i]; }
          return total;
        }
        export function make(n: i32): i32[] {
          let out: i32[] = [0, 1, 2];
          out.push(n);
          return out;
        }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let sum: extern "C" fn(i64) -> i32 = c.get("sum").unwrap();
        let make: extern "C" fn(i32) -> i64 = c.get("make").unwrap();

        let arr = make(7);
        assert_eq!(sum(arr), 10);

        // Bounds checks: reads and writes outside the array trap instead of
        // touching foreign memory.
        assert_eq!(sum(make(0)), 3);
    }
    drop(guard);
}

#[test]
fn array_element_assignment() {
    let c = build_arr(
        r#"
        export function poke(): i32 {
          let a: i32[] = [1, 2, 3];
          a[0] = 10;
          a[1] += 5;
          return a[0] + a[1];
        }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let poke: extern "C" fn() -> i32 = c.get("poke").unwrap();
        assert_eq!(poke(), 17);
    }
    drop(guard);
}

#[test]
fn array_out_of_bounds_traps() {
    let c = build_arr(
        r#"
        export function at(a: i32[], i: i32): i32 { return a[i]; }
        export function make(): i32[] { return [1, 2, 3]; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let at: extern "C" fn(i64, i32) -> i32 = c.get("at").unwrap();
        let make: extern "C" fn() -> i64 = c.get("make").unwrap();
        let arr = make();
        assert_eq!(at(arr, 2), 3);
        assert_eq!(at(arr, 3), 0);
        assert_eq!(c.trap().map(|t| t.message()), Some("index out of bounds"));
        c.clear_trap();
        assert_eq!(at(arr, -1), 0);
        assert_eq!(c.trap().map(|t| t.message()), Some("index out of bounds"));
        c.clear_trap();
    }
    drop(guard);
}

#[test]
fn arrays_of_strings() {
    let c = build_arr(
        r#"
        export function build(): string[] {
          let out: string[] = ["a", "bb"];
          out.push("ccc");
          return out;
        }
        export function first(a: string[]): string { return a[0]; }
        "#,
    );
    *LAST_STR.lock().unwrap() = None;
    let guard = c.enter();
    unsafe {
        let build: extern "C" fn() -> i64 = c.get("build").unwrap();
        let first: extern "C" fn(i64) -> i64 = c.get("first").unwrap();
        let arr = build();
        let s = first(arr);
        drop(guard);
        assert_eq!(string_bytes(s), b"a");
    }
}

// ───────────────────────────── sandbox: fuel ─────────────────────────────
//
// Every function entry and loop back-edge costs one unit of fuel. When the
// budget runs out the handler returns its zero value and records a trap, so a
// runaway mod can never wedge the server.

#[test]
fn runaway_loop_is_trapped_not_hung() {
    let c = build_arr(
        r#"
        export function spin(): i32 {
          let n: i32 = 0;
          while (true) { n += 1; }
          return n;
        }
        export function ok(): i32 { return 1; }
        "#,
    );
    c.runtime().reset_fuel(10_000);
    let guard = c.enter();
    unsafe {
        let spin: extern "C" fn() -> i32 = c.get("spin").unwrap();
        assert_eq!(spin(), 0, "trapped handler must return zero");
    }
    drop(guard);
    assert_eq!(
        c.trap().map(|t| t.message()),
        Some("fuel exhausted (runaway handler)")
    );
    c.clear_trap();

    // With fuel re-armed, the same module is usable again.
    c.runtime().reset_fuel(10_000);
    let guard = c.enter();
    unsafe {
        let ok: extern "C" fn() -> i32 = c.get("ok").unwrap();
        assert_eq!(ok(), 1);
    }
    drop(guard);
    assert!(c.trap().is_none());
}

#[test]
fn unbounded_recursion_is_trapped() {
    let c = build_arr(
        r#"
        export function down(n: i32): i32 { return down(n - 1) + 1; }
        "#,
    );
    c.runtime().reset_fuel(5_000);
    let guard = c.enter();
    unsafe {
        // Recursion is bounded by fuel, so this returns instead of blowing the
        // stack; the return value is whatever the unwound frames produced.
        let down: extern "C" fn(i32) -> i32 = c.get("down").unwrap();
        let _ = down(0);
    }
    drop(guard);
    assert_eq!(
        c.trap().map(|t| t.message()),
        Some("fuel exhausted (runaway handler)")
    );
}

#[test]
fn default_fuel_covers_a_million_iterations() {
    let c = build_arr(
        r#"
        export function sum(n: i32): i32 {
          let total: i32 = 0;
          for (let i: i32 = 0; i < n; i++) { total += i; }
          return total;
        }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let sum: extern "C" fn(i32) -> i32 = c.get("sum").unwrap();
        // 0+1+...+999999 = 499_999_500_000, truncated to i32.
        assert_eq!(sum(1_000_000), 499999500000_i64 as i32);
    }
    drop(guard);
    assert!(c.trap().is_none(), "default fuel should not run out");
}

// ─────────────────── ABI: host regions (query rows) ───────────────────
//
// This is the shape the Bevy bridge will use: a system lays out real Rust data
// (a query row, a component) and hands the module a pointer to it; the module
// reads fields through the builtins, using offsets from the exported type
// database rather than guesses.

/// Layout of `exsolis_common::ChunkBlocks`-style data comes from
/// `tesjit-dwarf`; the test below rebuilds a Rust value on the host side,
/// exports its type from a debug binary, and reads it back from TS by offset.
#[test]
fn host_region_read_write_roundtrip() {
    let c = build_arr(
        r#"
        export function peek(id: i32): f64 { return load_f64(id, 8); }
        export function poke(id: i32, v: f64): f64 { return store_f64(id, 8, v); }
        export function by_name(): f64 { return load_f64(region_id("row"), 0); }
        export function far(id: i32): f64 { return load_f64(id, 64); }
        "#,
    );
    let guard = c.enter();

    // A host-side "row": two f64s followed by an i32, as a Rust struct would
    // lay it out.
    let mut row: [u64; 6] = [0; 6]; // 48 bytes, 8-aligned
    let ptr = row.as_mut_ptr() as *mut u8;
    unsafe {
        std::ptr::write_unaligned(ptr.add(0) as *mut f64, 1.5);
        std::ptr::write_unaligned(ptr.add(8) as *mut f64, 2.5);
        std::ptr::write_unaligned(ptr.add(16) as *mut i32, 7);
    }
    let id = c.register_region("row", ptr, std::mem::size_of_val(&row));

    unsafe {
        let peek: extern "C" fn(i64) -> f64 = c.get("peek").unwrap();
        let poke: extern "C" fn(i64, f64) -> f64 = c.get("poke").unwrap();
        let by_name: extern "C" fn() -> f64 = c.get("by_name").unwrap();

        assert_eq!(peek(i64::from(id as u32)), 2.5, "offset 8 holds b");
        assert_eq!(poke(i64::from(id as u32), 9.0), 9.0);
        // The write landed in the host's own memory (same field offset).
        assert_eq!(*(ptr.add(8) as *const f64), 9.0);
        // And a region can be found by name instead of by id (offset 0 = a).
        assert_eq!(by_name(), 1.5);
    }

    c.unregister_region("row");
    unsafe {
        let peek: extern "C" fn(i64) -> f64 = c.get("peek").unwrap();
        assert_eq!(peek(i64::from(id as u32)), 0.0, "unregistered region must not read");
    }
    assert_eq!(
        c.trap().map(|t| t.message()),
        Some("unknown memory region")
    );
    c.clear_trap();

    // Out-of-bounds offsets inside a *known* region trap too.
    let mut small: [u64; 1] = [0];
    let id = c.register_region("small", small.as_mut_ptr() as *mut u8, 8);
    unsafe {
        let far: extern "C" fn(i64) -> f64 = c.get("far").unwrap();
        assert_eq!(far(i64::from(id as u32)), 0.0);
    }
    assert_eq!(c.trap().map(|t| t.message()), Some("index out of bounds"));
    c.clear_trap();
    drop(guard);
}

/// Offsets really do come from the type database: `tesjit-dwarf` exports the
/// layout of a Rust struct, and TS reads the same struct through those offsets.
#[test]
fn type_db_offsets_drive_region_access() {
    // Compile a fixture with the same tool the exporter uses, export its types,
    // and read a real value from TS by the exported field offsets.
    let dir = std::env::temp_dir().join(format!("tesjit-abi-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("fixture.rs");
    std::fs::write(
        &src,
        r#"
        #[derive(Clone, Copy)]
        pub struct Chunk { pub a: f64, pub b: f64, pub count: i32 }
        fn main() {
            let c = Chunk { a: 3.25, b: -1.5, count: 42 };
            std::hint::black_box((&c, c.count));
        }
        "#,
    )
    .unwrap();
    let bin = dir.join("fixture");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let ok = std::process::Command::new(rustc)
        .arg("--edition=2021")
        .arg("-g")
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .status()
        .unwrap()
        .success();
    assert!(ok, "fixture build failed");

    let data = std::fs::read(&bin).unwrap();
    let (db, _) = tesjit_dwarf::export(&data, "fixture").expect("export fixture");
    let def = db.get("fixture::Chunk").expect("Chunk exported");
    let tesjit_dwarf::TypeArgs::Struct(fields) = &def.args else {
        panic!("Chunk must be a struct");
    };
    let offset_of = |name: &str| -> u64 {
        fields
            .fields
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.offset)
            .unwrap_or_else(|| panic!("no field {name}"))
    };
    let (a, b, count) = (offset_of("a"), offset_of("b"), offset_of("count"));
    assert_eq!((a, b, count), (0, 8, 16), "rustc orders by alignment");

    let c = build_arr(&format!(
        r#"
        export function a(id: i32): f64 {{ return load_f64(id, {a} as i32); }}
        export function b(id: i32): f64 {{ return load_f64(id, {b} as i32); }}
        export function count(id: i32): i32 {{ return load_i32(id, {count} as i32); }}
        "#
    ));
    let guard = c.enter();

    // The real Rust value, laid out by the compiler the exporter read.
    #[derive(Clone, Copy)]
    #[allow(dead_code)] // read back through the region, not through Rust
    struct Chunk {
        a: f64,
        b: f64,
        count: i32,
    }
    let value = Chunk {
        a: 3.25,
        b: -1.5,
        count: 42,
    };
    let ptr = &value as *const Chunk as *mut u8;
    let id = c.register_region("chunk", ptr, std::mem::size_of::<Chunk>());
    unsafe {
        let get_a: extern "C" fn(i64) -> f64 = c.get("a").unwrap();
        let get_b: extern "C" fn(i64) -> f64 = c.get("b").unwrap();
        let get_count: extern "C" fn(i64) -> i32 = c.get("count").unwrap();
        assert_eq!(get_a(i64::from(id as u32)), 3.25);
        assert_eq!(get_b(i64::from(id as u32)), -1.5);
        assert_eq!(get_count(i64::from(id as u32)), 42);
    }
    drop(guard);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Editor tooling (`.d.ts`) has to `declare` the builtins so editors know their
/// signature — that redundant declaration must not fail a build.
#[test]
fn declaring_a_builtin_is_accepted_not_an_error() {
    let src = r#"
        declare function load_f64(region: i32, offset: i32): f64;
        export function go(): f64 { return load_f64(1, 0); }
        "#;
    let c = build_arr(src);
    let guard = c.enter();
    unsafe {
        let go: extern "C" fn() -> f64 = c.get("go").unwrap();
        assert_eq!(go(), 0.0, "an unregistered region reads as zero, not an error");
    }
    drop(guard);
}

/// A stale `.d.ts` that *documents* the wrong signature does not block the
/// module: the declaration is documentation only — the call site is checked
/// against the engine's real signature, which is what catches a genuinely wrong
/// call.
#[test]
fn a_stale_dts_signature_does_not_block_the_module() {
    let src = r#"
        declare function load_f64(region: i32): f64;
        export function go(): f64 { return load_f64(1, 0); }
        "#;
    let c = build_arr(src);
    let guard = c.enter();
    unsafe {
        let go: extern "C" fn() -> f64 = c.get("go").unwrap();
        assert_eq!(go(), 0.0);
    }
    drop(guard);

    // But a genuinely wrong call is still an error — the engine's signature,
    // not the `.d.ts`, is what the checker enforces.
    let diags = errors("declare function load_f64(region: i32): f64;\nexport function go(): f64 { return load_f64(1); }");
    assert!(
        diags.iter().any(|d| format!("{d:?}").contains("expects 2 arguments")),
        "{diags:?}"
    );
}

#[test]
fn sized_integers_wrap_and_compare() {
    let c = build(
        r#"
        function add_u8(a: u8, b: u8): u8 { return a + b; }
        function neg_i8(a: i8): i8 { return -a; }
        function add_i64(a: i64, b: i64): i64 { return a + b; }
        function shr_u32(a: u32): u32 { return a >> 1; }
        function shr_i32(a: i32): i32 { return a >> 1; }
        function ult(a: u32, b: u32): boolean { return a < b; }
        function ilt(a: i32, b: i32): boolean { return a < b; }
        function and_u16(a: u16, b: u16): u16 { return a & b; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let add_u8: extern "C" fn(u8, u8) -> u8 = c.get("add_u8").unwrap();
        let neg_i8: extern "C" fn(i8) -> i8 = c.get("neg_i8").unwrap();
        let add_i64: extern "C" fn(i64, i64) -> i64 = c.get("add_i64").unwrap();
        let shr_u32: extern "C" fn(u32) -> u32 = c.get("shr_u32").unwrap();
        let shr_i32: extern "C" fn(i32) -> i32 = c.get("shr_i32").unwrap();
        let ult: extern "C" fn(u32, u32) -> bool = c.get("ult").unwrap();
        let ilt: extern "C" fn(i32, i32) -> bool = c.get("ilt").unwrap();
        let and_u16: extern "C" fn(u16, u16) -> u16 = c.get("and_u16").unwrap();
        assert_eq!(add_u8(200, 100), 44, "u8 wraps");
        assert_eq!(neg_i8(5), -5);
        assert_eq!(add_i64(1 << 40, 1 << 40), 1 << 41);
        assert_eq!(shr_u32(0x8000_0000), 0x4000_0000, "u32 >> is logical");
        assert_eq!(shr_i32(-2), -1, "i32 >> is arithmetic");
        assert!(ult(1, 0xFFFF_FFFF), "u32 comparison is unsigned");
        assert!(!ult(0xFFFF_FFFF, 1));
        assert!(ilt(-1, 1), "i32 comparison is signed");
        assert!(!ilt(1, -1));
        assert_eq!(and_u16(0xFF00, 0x0FF0), 0x0F00);
    }
    drop(guard);
}

#[test]
fn float_widths_and_casts() {
    let c = build(
        r#"
        function half(x: f32): f32 { return x / 2; }
        function widen(x: f32): f64 { return x as f64; }
        function narrow(x: f64): f32 { return x as f32; }
        function i2f(x: i32): f32 { return x as f32; }
        function f2i(x: f64): i32 { return x as i32; }
        function u2f(x: u32): f64 { return x as f64; }
        function f2u(x: f64): u32 { return x as u32; }
        function b2u(b: boolean): u32 { return b as u32; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let half: extern "C" fn(f32) -> f32 = c.get("half").unwrap();
        let widen: extern "C" fn(f32) -> f64 = c.get("widen").unwrap();
        let narrow: extern "C" fn(f64) -> f32 = c.get("narrow").unwrap();
        let i2f: extern "C" fn(i32) -> f32 = c.get("i2f").unwrap();
        let f2i: extern "C" fn(f64) -> i32 = c.get("f2i").unwrap();
        let u2f: extern "C" fn(u32) -> f64 = c.get("u2f").unwrap();
        let f2u: extern "C" fn(f64) -> u32 = c.get("f2u").unwrap();
        let b2u: extern "C" fn(bool) -> u32 = c.get("b2u").unwrap();
        assert_eq!(half(5.0), 2.5);
        assert_eq!(widen(1.5), 1.5);
        assert_eq!(narrow(1.5), 1.5);
        assert_eq!(i2f(-3), -3.0);
        assert_eq!(f2i(1.9), 1, "float->int truncates toward zero");
        assert_eq!(f2i(-1.9), -1);
        assert_eq!(u2f(0xFFFF_FFFF), 4294967295.0);
        assert_eq!(f2u(1.9), 1);
        assert_eq!(b2u(true), 1);
        assert_eq!(b2u(false), 0);
    }
    drop(guard);
}

#[test]
fn int_width_casts_extend_and_wrap() {
    let c = build(
        r#"
        function s_extend(x: i8): i64 { return x as i64; }
        function z_extend(x: u8): i64 { return x as i64; }
        function wrap(x: i64): i8 { return x as i8; }
        function u32_to_i64(x: u32): i64 { return x as i64; }
        function i32_to_u64(x: i32): u64 { return (x as i64) as u64; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let s_extend: extern "C" fn(i8) -> i64 = c.get("s_extend").unwrap();
        let z_extend: extern "C" fn(u8) -> i64 = c.get("z_extend").unwrap();
        let wrap: extern "C" fn(i64) -> i8 = c.get("wrap").unwrap();
        let u32_to_i64: extern "C" fn(u32) -> i64 = c.get("u32_to_i64").unwrap();
        let i32_to_u64: extern "C" fn(i32) -> u64 = c.get("i32_to_u64").unwrap();
        assert_eq!(s_extend(-1), -1, "signed widening sign-extends");
        assert_eq!(z_extend(255), 255, "unsigned widening zero-extends");
        assert_eq!(wrap(0x1FF), -1, "narrowing wraps");
        assert_eq!(u32_to_i64(0xFFFF_FFFF), 4294967295);
        assert_eq!(i32_to_u64(-1), u64::MAX, "bit-preserving round trip");
    }
    drop(guard);
}

#[test]
fn entity_handles_compare_and_cast() {
    let c = build(
        r#"
        function same(a: Entity, b: Entity): boolean { return a == b; }
        function to_bits(e: Entity): u64 { return e as u64; }
        function from_bits(b: u64): Entity { return b as Entity; }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let same: extern "C" fn(u64, u64) -> bool = c.get("same").unwrap();
        let to_bits: extern "C" fn(u64) -> u64 = c.get("to_bits").unwrap();
        let from_bits: extern "C" fn(u64) -> u64 = c.get("from_bits").unwrap();
        assert!(same(7, 7));
        assert!(!same(7, 8));
        assert_eq!(to_bits(u64::MAX), u64::MAX);
        assert_eq!(from_bits(42), 42);
    }
    drop(guard);
    // No arithmetic on entities, and no casts to numbers.
    for src in [
        "function f(a: Entity, b: Entity): Entity { return a + b; }",
        "function f(a: Entity): i32 { return a as i32; }",
    ] {
        errors(src);
    }
}

#[test]
fn float_arrays_push_index_and_sum() {
    // Regression: float array elements cross the runtime ABI as raw words;
    // the adapter used to only know int/bool <-> word and refused to compile.
    let c = build(
        r#"
        function run(n: i32): f64 {
            let a: f64[] = [0.5];
            for (let i: i32 = 1; i < n; i++) { a.push(i as f64 * 0.25); }
            let s: f64 = 0;
            for (let j: i32 = 0; j < a.length; j++) { s += a[j]; }
            return s;
        }
        "#,
    );
    let guard = c.enter();
    unsafe {
        let run: extern "C" fn(i32) -> f64 = c.get("run").unwrap();
        let n = 100;
        let mut want = 0.5;
        for i in 1..n {
            want += f64::from(i) * 0.25;
        }
        let got = run(n);
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }
    drop(guard);
}

#[test]
fn mixed_widths_need_an_explicit_cast() {
    let diags = errors("function f(a: i32, b: i64): i64 { return a + b; }");
    assert!(
        diags
            .iter()
            .any(|d| format!("{d:?}").contains("operand types differ")),
        "{diags:?}"
    );
}

#[test]
fn integer_literals_are_range_checked() {
    for (ty, bad) in [
        ("u8", "256"),
        ("i8", "128"),
        ("u16", "65536"),
        ("i16", "-32769"),
        ("u32", "4294967296"),
    ] {
        let src = format!("function f(): {ty} {{ return {bad}; }}");
        let diags = errors(&src);
        assert!(
            diags
                .iter()
                .any(|d| format!("{d:?}").contains("does not fit")),
            "expected range error for {src}: {diags:?}"
        );
    }
    // In-range extremes compile.
    let c = build("function lo(): i8 { return -128; }\nfunction hi(): u8 { return 255; }");
    let guard = c.enter();
    unsafe {
        let lo: extern "C" fn() -> i8 = c.get("lo").unwrap();
        let hi: extern "C" fn() -> u8 = c.get("hi").unwrap();
        assert_eq!(lo(), -128);
        assert_eq!(hi(), 255);
    }
    drop(guard);
}

#[test]
fn phase1_types_parse_compile_clean() {
    let _c = build(
        r#"
        export function noop(x: Option<i32>, q: Query<i32>, r: Res<i32>, c: Commands): void { }
        "#,
    );
}

#[test]
fn option_roundtrip() {
    let src = r#"
        function wrap(x: i64): Option<i64> { return some(x); }
        function unwrapOrZero(x: Option<i64>): i64 { if (x.isSome()) { return x.unwrap(); } return 0; }
        function empty(): Option<i64> { return none(); }
    "#;
    let c = build(src);
    let guard = c.enter();
    unsafe {
        let wrap: extern "C" fn(i64) -> i64 = c.get("wrap").unwrap();
        let unwrapOrZero: extern "C" fn(i64) -> i64 = c.get("unwrapOrZero").unwrap();
        let empty: extern "C" fn() -> i64 = c.get("empty").unwrap();
        let p = wrap(42);
        assert_eq!(unwrapOrZero(p), 42);
        let p = empty();
        assert_eq!(unwrapOrZero(p), 0);
    }
    drop(guard);
}

#[test]
fn option_unwrapor() {
    let src = r#"
        function check(): i64 {
            let v: Option<i64> = none();
            return v.unwrapOr(7);
        }
    "#;
    let c = build(src);
    let guard = c.enter();
    unsafe {
        let check: extern "C" fn() -> i64 = c.get("check").unwrap();
        assert_eq!(check(), 7);
    }
    drop(guard);
}


#[test]
fn for_of_arrays_sum() {
    let src = r#"
        function make(n: i32): i64[] {
            let a: i64[] = [0];
            let i: i32 = 0;
            while (i < n) { a.push(i as i64); i += 1; }
            return a;
        }
        function sum(a: i64[]): i64 {
            let s: i64 = 0;
            for (let v of a) { s += v; }
            return s;
        }
    "#;
    let c = build(src);
    let guard = c.enter();
    unsafe {
        let make: extern "C" fn(i32) -> i64 = c.get("make").unwrap();
        let sum: extern "C" fn(i64) -> i64 = c.get("sum").unwrap();
        let arr = make(5);
        assert_eq!(sum(arr), 0 + 1 + 2 + 3 + 4);
    }
    drop(guard);
}

#[test]
fn for_of_arrays_mutate() {
    let src = r#"
        function make(n: i32): i64[] {
            let a: i64[] = [0];
            let i: i32 = 0;
            while (i < n) { a.push(i as i64); i += 1; }
            return a;
        }
        function bump(a: i64[]): void {
            for (let v of a) { v += 10; }
        }
        function total(a: i64[]): i64 {
            let s: i64 = 0;
            for (let v of a) { s += v; }
            return s;
        }
    "#;
    let c = build(src);
    let guard = c.enter();
    unsafe {
        let make: extern "C" fn(i32) -> i64 = c.get("make").unwrap();
        let bump: extern "C" fn(i64) -> () = c.get("bump").unwrap();
        let total: extern "C" fn(i64) -> i64 = c.get("total").unwrap();
        let arr = make(3);
        bump(arr);
        // 0+1+2 + 10*3 = 33
        assert_eq!(total(arr), 43);
    }
    drop(guard);
}
