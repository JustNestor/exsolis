//! The type universe tesjit compiles today.
//!
//! Every TS-facing type maps to exactly one Cranelift machine type, which is
//! what makes the JIT boundary safe: there is no dynamic tagging, the codegen
//! layer always knows the register class of a value.

use cranelift_codegen::ir::{self, types};

/// A statically known value type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    /// `i32` — 32-bit integer.
    I32,
    /// `number` / `f64` — IEEE double.
    F64,
    /// `boolean`.
    Bool,
    /// `void` — carries no value (only legal as a return type).
    Void,
}

impl Ty {
    /// Cranelift representation of this type.
    #[inline]
    pub fn clif(self) -> ir::Type {
        match self {
            Ty::I32 => types::I32,
            Ty::F64 => types::F64,
            Ty::Bool => types::I8,
            // `void` never materializes a value; `I32` is only a placeholder
            // so `Some(ty)` callers can stay infallible.
            Ty::Void => types::I32,
        }
    }

    /// Human-readable name, for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Ty::I32 => "i32",
            Ty::F64 => "f64",
            Ty::Bool => "boolean",
            Ty::Void => "void",
        }
    }

    /// Types a local variable may have.
    pub fn is_storable(self) -> bool {
        self != Ty::Void
    }

    /// Numeric types (`+`, `-`, ... operands).
    pub fn is_numeric(self) -> bool {
        matches!(self, Ty::I32 | Ty::F64)
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
