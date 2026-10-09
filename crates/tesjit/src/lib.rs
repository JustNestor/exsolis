//! tesjit — a minimal statically-typed TypeScript → Cranelift JIT.
//!
//! Pipeline: source → oxc parse → typed HIR ([`check`]) → Cranelift
//! ([`codegen`]) → machine code ([`jit`]).
//!
//! Every function boundary is fully annotated (parameters *and* return type),
//! inference is local to `let`, and there is no `any`: that is what lets the
//! compiler know the exact machine type of every value at every call boundary,
//! including host calls.
//!
//! ```ignore
//! use tesjit::{JitBuilder, Ty};
//!
//! extern "C" fn host_log(x: f64) { println!("[host] {x}"); }
//!
//! let src = r#"
//! declare function log(x: f64): void;
//! export function go(dt: f64): void { log(dt * 2); }
//! "#;
//!
//! let mut jit = JitBuilder::new("mod.ts", src)
//!     .host_fn("log", vec![Ty::F64], Ty::Void, host_log as *const u8)
//!     .build()?;
//! jit.compile(src)?;
//! let compiled = jit.finish()?;
//!
//! let go: extern "C" fn(f64) = unsafe { compiled.get("go").unwrap() };
//! go(1.5);
//! ```
//!
//! Types: `number/f64` → F64, `i32` → I32, `boolean` → I8, `void`.

pub mod check;
pub mod codegen;
pub mod diag;
pub mod hir;
pub mod host;
pub mod jit;
pub mod ty;

pub use diag::{CompileError, Diagnostics, SyntaxDiag};
pub use host::{FnKind, FnTable, HostFn, HostRegistry, HostSig};
pub use jit::{Compiled, Entry, Jit, JitBuilder};
pub use ty::Ty;

#[cfg(test)]
mod tests;
