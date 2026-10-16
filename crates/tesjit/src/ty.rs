//! The type universe tesjit compiles today.
//!
//! Every TS-facing type maps to exactly one Cranelift machine type, which is
//! what makes the JIT boundary safe: there is no dynamic tagging, the codegen
//! layer always knows the register class of a value.

use cranelift_codegen::ir::{self, types};

/// A statically known value type.
///
/// Integer widths are all distinct: `u32` and `i32` share a machine
/// representation but differ in division, remainder, right shift and
/// comparisons (unsigned vs signed). The checker keeps them apart and
/// requires an explicit `as` cast to convert, so codegen always knows
/// which instruction family to emit.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    /// `i8` — signed 8-bit integer.
    I8,
    /// `i16` — signed 16-bit integer.
    I16,
    /// `i32` — signed 32-bit integer.
    I32,
    /// `i64` — signed 64-bit integer.
    I64,
    /// `u8` — unsigned 8-bit integer.
    U8,
    /// `u16` — unsigned 16-bit integer.
    U16,
    /// `u32` — unsigned 32-bit integer.
    U32,
    /// `u64` — unsigned 64-bit integer.
    U64,
    /// `f32` — 32-bit float (`number` literals adapt to it on demand).
    F32,
    /// `number` / `f64` — IEEE double.
    F64,
    /// `boolean`.
    Bool,
    /// `Entity` — opaque 64-bit entity handle. No literals and no
    /// arithmetic; only `==`/`!=` and explicit `as` casts to/from `u64`.
    Entity,
    /// `Option<T>` — tagged pointer (heap-allocated `{tag, value}`).
    Option(Box<Ty>),
    /// `Tuple<...>` — positional product, row type of multi-component `Query`.
    Tuple(Vec<Ty>),
    /// `Query<T>` — signature-only type (never materializes).
    Query(Box<Ty>),
    /// `Res<T>` — signature-only read-only resource handle.
    Res(Box<Ty>),
    /// `Mut<T>` — signature-only mutable resource/component handle.
    Mut(Box<Ty>),
    /// `Commands` — signature-only append-only command buffer.
    Commands,
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
            Ty::I8 | Ty::U8 => types::I8,
            Ty::I16 | Ty::U16 => types::I16,
            Ty::I32 | Ty::U32 => types::I32,
            Ty::I64 | Ty::U64 | Ty::Entity | Ty::Option(_) | Ty::Tuple(_) | Ty::Query(_) | Ty::Res(_) | Ty::Mut(_) | Ty::Commands => types::I64,
            Ty::F32 => types::F32,
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
            Ty::I8 => "i8".into(),
            Ty::I16 => "i16".into(),
            Ty::I32 => "i32".into(),
            Ty::I64 => "i64".into(),
            Ty::U8 => "u8".into(),
            Ty::U16 => "u16".into(),
            Ty::U32 => "u32".into(),
            Ty::U64 => "u64".into(),
            Ty::F32 => "f32".into(),
            Ty::F64 => "f64".into(),
            Ty::Bool => "boolean".into(),
            Ty::Entity => "Entity".into(),
            Ty::Option(t) => format!("Option<{t}>").into(),
            Ty::Tuple(ts) => format!("({})", ts.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")).into(),
            Ty::Query(t) => format!("Query<{t}>").into(),
            Ty::Res(t) => format!("Res<{t}>").into(),
            Ty::Mut(t) => format!("Mut<{t}>").into(),
            Ty::Commands => "Commands".into(),
            Ty::Array(elem) => format!("{elem}[]").into(),
            Ty::Str => "string".into(),
            Ty::Ptr => "ptr".into(),
            Ty::Void => "void".into(),
        }
    }

    /// Types a local variable may have.
    pub fn is_storable(&self) -> bool {
        !matches!(
            self,
            Ty::Void | Ty::Query(_) | Ty::Res(_) | Ty::Mut(_) | Ty::Commands
        )
    }

    /// Types whose *values* never appear in JIT registers: only parameter or
    /// return slots. Codegen never allocates locals for these.
    pub fn is_signature_only(&self) -> bool {
        matches!(self, Ty::Query(_) | Ty::Res(_) | Ty::Mut(_) | Ty::Commands)
    }

    /// `Option<T>` element accessor.
    pub fn option_elem(&self) -> Option<Ty> {
        match self {
            Ty::Option(t) => Some((**t).clone()),
            _ => None,
        }
    }

    /// `Query<T>` row type accessor.
    pub fn query_row(&self) -> Option<Ty> {
        match self {
            Ty::Query(t) => Some((**t).clone()),
            _ => None,
        }
    }

    /// `Res`/`Mut` referenced component accessor.
    pub fn resource_elem(&self) -> Option<Ty> {
        match self {
            Ty::Res(t) | Ty::Mut(t) => Some((**t).clone()),
            _ => None,
        }
    }

    /// Integer types (signed and unsigned, any width).
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Ty::I8
                | Ty::I16
                | Ty::I32
                | Ty::I64
                | Ty::U8
                | Ty::U16
                | Ty::U32
                | Ty::U64
        )
    }

    /// Signed integer types (two's complement division/comparison semantics).
    pub fn is_signed(&self) -> bool {
        matches!(self, Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64)
    }

    /// Floating-point types.
    pub fn is_float(&self) -> bool {
        matches!(self, Ty::F32 | Ty::F64)
    }

    /// Numeric types (`+`, `-`, ... operands).
    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    /// Bit width of an integer type.
    pub fn int_width(&self) -> Option<u32> {
        match self {
            Ty::I8 | Ty::U8 => Some(8),
            Ty::I16 | Ty::U16 => Some(16),
            Ty::I32 | Ty::U32 => Some(32),
            Ty::I64 | Ty::U64 => Some(64),
            _ => None,
        }
    }

    /// Inclusive `(min, max)` range of an integer type, as `i128` so `u64`
    /// fits without overflow.
    pub fn int_range(&self) -> Option<(i128, i128)> {
        let width = self.int_width()?;
        if self.is_signed() {
            let max = (1i128 << (width - 1)) - 1;
            Some((-max - 1, max))
        } else {
            Some((0, (1i128 << width) - 1))
        }
    }

    /// Element size in bytes for arrays (scalars only, packed layout).
    pub fn elem_size(&self) -> Option<u64> {
        match self {
            Ty::I8 | Ty::U8 | Ty::Bool => Some(1),
            Ty::I16 | Ty::U16 => Some(2),
            Ty::I32 | Ty::U32 | Ty::F32 => Some(4),
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::Entity => Some(8),
            Ty::Option(_) | Ty::Tuple(_) | Ty::Query(_) | Ty::Res(_) | Ty::Mut(_) | Ty::Commands | Ty::Str | Ty::Array(_) | Ty::Ptr => Some(8),
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
