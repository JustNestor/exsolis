//! The type universe tesjit compiles today.
//!
//! Every TS-facing type maps to exactly one Cranelift machine type, which is
//! what makes the JIT boundary safe: there is no dynamic tagging, the codegen
//! layer always knows the register class of a value.

use cranelift_codegen::ir::{self, types};

/// A statically known value type.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    /// `i32` — 32-bit integer.
    I32,
    /// `number` / `f64` — IEEE double.
    F64,
    /// `boolean`.
    Bool,
    /// `string` — pointer to `[len: i64][utf8 bytes]` (see [`crate::runtime`]).
    Str,
    /// `T[]` — pointer to `[len: i64][cap: i64][packed T...]`.
    Array(Box<Ty>),
    /// Raw machine pointer. Not writable from TS — used only in host and
    /// runtime signatures, where it means "one machine word".
    Ptr,
    /// `void` — carries no value (only legal as a return type).
    Void,
}

impl Ty {
    /// Cranelift representation of this type.
    #[inline]
    pub fn clif(&self) -> ir::Type {
        match self {
            Ty::I32 => types::I32,
            Ty::F64 => types::F64,
            Ty::Bool => types::I8,
            // Strings, arrays and pointers are all machine words.
            Ty::Str | Ty::Array(_) | Ty::Ptr => types::I64,
            // `void` never materializes a value; `I32` is only a placeholder
            // so `Some(ty)` callers can stay infallible.
            Ty::Void => types::I32,
        }
    }

    /// Human-readable name, for diagnostics.
    pub fn name(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Ty::I32 => "i32".into(),
            Ty::F64 => "f64".into(),
            Ty::Bool => "boolean".into(),
            Ty::Str => "string".into(),
            Ty::Array(elem) => format!("{elem}[]").into(),
            Ty::Ptr => "ptr".into(),
            Ty::Void => "void".into(),
        }
    }

    /// Types a local variable may have.
    pub fn is_storable(self) -> bool {
        !matches!(self, Ty::Void)
    }

    /// Numeric types (`+`, `-`, ... operands).
    pub fn is_numeric(self) -> bool {
        matches!(self, Ty::I32 | Ty::F64)
    }

    /// Element size in bytes for arrays (scalars only, packed layout).
    pub fn elem_size(&self) -> Option<u64> {
        match self {
            Ty::I32 => Some(4),
            Ty::F64 => Some(8),
            Ty::Bool => Some(1),
            // Strings, arrays and pointers are machine words.
            Ty::Str | Ty::Array(_) | Ty::Ptr => Some(8),
            Ty::Void => None,
        }
    }

    /// Element type of an array type.
    pub fn elem(&self) -> Option<Ty> {
        match self {
            Ty::Array(elem) => Some((**elem).clone()),
            _ => None,
        }
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}
