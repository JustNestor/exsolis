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
    unsafe {
        let fib: extern "C" fn(i32) -> i32 = c.get("fib").unwrap();
        let sum: extern "C" fn(i32) -> f64 = c.get("sum").unwrap();
        assert_eq!(fib(20), 6765);
        assert_eq!(sum(100), 4950.0);
    }
}

#[test]
fn host_call() {
    let c = build(
        r#"
        declare function log(x: f64): void;
        export function go(dt: f64): void { log(dt * 2); }
        "#,
    );
    unsafe {
        let go: extern "C" fn(f64) = c.get("go").unwrap();
        go(1.5);
    }
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
    unsafe {
        let ok: extern "C" fn(i32) -> i32 = c.get("ok").unwrap();
        assert_eq!(ok(21), 42);
    }
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
        assert_eq!(sum(arr), 0 + 1 + 2 + 7);

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
