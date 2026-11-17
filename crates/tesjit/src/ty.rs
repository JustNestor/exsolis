//! The type universe tesjit compiles today.
//!
//! Every TS-facing type maps to exactly one Cranelift machine type, which is
//! what makes the JIT boundary safe: there is no dynamic tagging, the codegen
//! layer always knows the register class of a value.
//!
//! Structs come in two flavors: mod-defined `repr(C)` POD structs
//! (`Component<T>` / `Resource<T>`) and host-registered engine component
//! layouts (`from_parts`, used by the zero-copy query path). `Query`/`Res`/
//! `Mut`/`Commands` stay signature-only types that never materialise as locals.

use cranelift_codegen::ir::{self, types};
use std::sync::Arc;

/// A single field of a struct layout (offset is already aligned).
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct StructField {
    pub name: String,
    pub ty: Ty,
    /// Byte offset from the start of the struct.
    pub offset: u64,
}

/// Layout of a struct, either computed `repr(C)` for a mod-defined POD type
/// or verified against a host-provided engine component layout.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct StructLayout {
    pub name: String,
    pub fields: Vec<StructField>,
    pub size: u64,
    pub align: u64,
    /// Bevy component id for host-registered layouts (set at bind time).
    pub component_id: Option<u32>,
    /// Whether `Commands.spawn` may construct this component from a literal.
    pub spawnable: bool,
    /// True when every leaf is a POD scalar (no strings, arrays, Options…).
    /// Only POD structs may be mod-defined components or resources.
    pub is_pod: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum QueryFilterKind {
    With,
    Without,
    /// Only rows where the component was added since the system last ran.
    Added,
    /// Only rows where the component was added or mutably accessed since the
    /// system last ran.
    Changed,
}

impl QueryFilterKind {
    pub fn name(self) -> &'static str {
        match self {
            QueryFilterKind::With => "With",
            QueryFilterKind::Without => "Without",
            QueryFilterKind::Added => "Added",
            QueryFilterKind::Changed => "Changed",
        }
    }

    /// Whether this is a row-level change-detection filter. These cannot be
    /// resolved at archetype level; the host compares per-row change ticks.
    pub fn is_change(self) -> bool {
        matches!(self, QueryFilterKind::Added | QueryFilterKind::Changed)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct QueryFilter {
    pub kind: QueryFilterKind,
    pub component: Arc<StructLayout>,
}

impl StructLayout {
    /// Build a `repr(C)` layout from ordered `(name, Ty)` pairs.
    pub fn repr_c(name: impl Into<String>, fields: Vec<(String, Ty)>) -> Result<Self, String> {
        let mut out = Vec::with_capacity(fields.len());
        let mut offset: u64 = 0;
        let mut max_align: u64 = 1;
        let mut is_pod = true;

        for (fname, fty) in fields {
            let (sz, al) = fty
                .size_align()
                .ok_or_else(|| format!("field `{fname}` has non-sized type {}", fty))?;
            if !fty.is_pod_leaf() {
                is_pod = false;
            }
            // Align offset up to field alignment.
            let align_mask = al as u64 - 1;
            offset = (offset + align_mask) & !align_mask;
            out.push(StructField {
                name: fname,
                ty: fty,
                offset,
            });
            offset = offset.saturating_add(sz as u64);
            max_align = max_align.max(al as u64);
        }
        // Pad struct size to multiple of max alignment.
        let align_mask = max_align - 1;
        let size = (offset + align_mask) & !align_mask;

        Ok(Self {
            name: name.into(),
            fields: out,
            size,
            align: max_align,
            component_id: None,
            spawnable: true,
            is_pod,
        })
    }

    /// Build a layout only when every exposed field is naturally aligned,
    /// non-overlapping, and contained in the component's actual byte size.
    pub fn from_parts(
        name: impl Into<String>,
        fields: Vec<StructField>,
        size: u64,
        align: u64,
    ) -> Result<Self, String> {
        let name = name.into();
        if align == 0 || !align.is_power_of_two() || size % align != 0 {
            return Err(format!(
                "invalid layout for {name}: size {size} must be a multiple of non-zero power-of-two alignment {align}"
            ));
        }
        let mut fields_by_offset: Vec<_> = fields.iter().collect();
        fields_by_offset.sort_by_key(|field| field.offset);
        let mut end = 0;
        for field in fields_by_offset {
            let Some((field_size, field_align)) = field.ty.value_layout() else {
                return Err(format!(
                    "field {}.{} has no supported memory layout",
                    name, field.name
                ));
            };
            if field.offset % field_align != 0 {
                return Err(format!(
                    "field {}.{} at offset {} is not aligned to {field_align}",
                    name, field.name, field.offset
                ));
            }
            if field.offset < end {
                return Err(format!(
                    "field {}.{} overlaps a preceding field",
                    name, field.name
                ));
            }
            end = field
                .offset
                .checked_add(field_size)
                .ok_or_else(|| format!("field {}.{} overflows layout", name, field.name))?;
            if end > size {
                return Err(format!(
                    "field {}.{} ends at {end}, beyond component size {size}",
                    name, field.name
                ));
            }
        }
        Ok(Self {
            name,
            fields,
            size,
            align,
            component_id: None,
            spawnable: true,
            is_pod: true,
        })
    }

    pub fn field(&self, name: &str) -> Option<&StructField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// Stable hash of the layout (name + field names/types/offsets).
    /// Used for hot-reload reject-on-layout-change (§7).
    pub fn layout_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.name.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for f in &self.fields {
            for b in f.name.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            for b in f.ty.name().bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= f.offset;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= self.size;
        h = h.wrapping_mul(0x100000001b3);
        h
    }
}

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
    /// A host component. Values are borrowed pointers and cannot be copied.
    Struct(Arc<StructLayout>),
    /// An owned, stack-backed component value constructed by a mod.
    OwnedStruct(Arc<StructLayout>),
    /// `Tuple<...>` — positional product, row type of multi-component `Query`.
    Tuple(Vec<Ty>),
    /// `Component<T>` — marks a POD struct (or scalar) as a mod-defined
    /// component. Signature-only when used as a query term; the layout is
    /// carried for offset/size queries and hot-reload hashing.
    Component(Box<Ty>),
    /// `Resource<T>` — marks a POD type as a mod-defined resource.
    Resource(Box<Ty>),
    /// `Query<T, F>` — signature-only type (never materializes). F is an
    /// optional list of `With<Comp>` / `Without<Comp>` component filters.
    Query(Box<Ty>, Vec<QueryFilter>),
    /// `Res<T>` — signature-only read-only resource handle.
    Res(Box<Ty>),
    /// `Mut<T>` — signature-only mutable resource/component handle.
    Mut(Box<Ty>),
    /// `Added<T>` — query-only marker. In a query row it desugars to a
    /// read-only borrow of `T` plus an `Added` row filter; in the filter
    /// position it contributes the filter without a row term.
    Added(Box<Ty>),
    /// `Changed<T>` — query-only marker, same desugaring as `Added<T>`.
    Changed(Box<Ty>),
    /// `World` — signature-only read-only world handle. Its only operation is
    /// `get<T>(entity)`, which returns `Option<T>` borrowed from the world.
    World,
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
            Ty::I64
            | Ty::U64
            | Ty::Entity
            | Ty::Option(_)
            | Ty::Struct(_)
            | Ty::OwnedStruct(_)
            | Ty::Tuple(_)
            | Ty::Component(_)
            | Ty::Resource(_)
            | Ty::Query(..)
            | Ty::Res(_)
            | Ty::Mut(_)
            | Ty::Added(_)
            | Ty::Changed(_)
            | Ty::World
            | Ty::Commands => types::I64,
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
            Ty::Struct(layout) => layout.name.clone().into(),
            Ty::OwnedStruct(layout) => layout.name.clone().into(),
            Ty::Component(t) => format!("Component<{t}>").into(),
            Ty::Resource(t) => format!("Resource<{t}>").into(),
            Ty::Tuple(ts) => format!(
                "({})",
                ts.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")
            )
            .into(),
            Ty::Query(row, filters) if filters.is_empty() => format!("Query<{row}>").into(),
            Ty::Query(row, filters) => format!(
                "Query<{row}, [{}]>",
                filters
                    .iter()
                    .map(|filter| {
                        format!("{}<{}>", filter.kind.name(), filter.component.name)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into(),
            Ty::Res(t) => format!("Res<{t}>").into(),
            Ty::Mut(t) => format!("Mut<{t}>").into(),
            Ty::Added(t) => format!("Added<{t}>").into(),
            Ty::Changed(t) => format!("Changed<{t}>").into(),
            Ty::World => "World".into(),
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
            Ty::Void
                | Ty::Struct(_)
                | Ty::Query(..)
                | Ty::Res(_)
                | Ty::Mut(_)
                | Ty::Added(_)
                | Ty::Changed(_)
                | Ty::World
                | Ty::Commands
                | Ty::Component(_)
                | Ty::Resource(_)
        )
    }

    /// Types whose *values* never appear in JIT registers: only parameter or
    /// return slots. Codegen never allocates locals for these.
    pub fn is_signature_only(&self) -> bool {
        matches!(
            self,
            Ty::Query(..)
                | Ty::Res(_)
                | Ty::Mut(_)
                | Ty::Added(_)
                | Ty::Changed(_)
                | Ty::World
                | Ty::Commands
                | Ty::Component(_)
                | Ty::Resource(_)
        )
    }

    /// `Option<T>` element accessor.
    pub fn option_elem(&self) -> Option<Ty> {
        match self {
            Ty::Option(t) => Some((**t).clone()),
            _ => None,
        }
    }

    /// `Query<T, F>` row type accessor.
    pub fn query_row(&self) -> Option<Ty> {
        match self {
            Ty::Query(t, _) => Some((**t).clone()),
            _ => None,
        }
    }

    /// `Query<T, F>` component filters.
    pub fn query_filters(&self) -> Option<&[QueryFilter]> {
        match self {
            Ty::Query(_, filters) => Some(filters),
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

    /// Unwrap `Component<T>` / `Resource<T>` to the inner type.
    pub fn component_inner(&self) -> Option<Ty> {
        match self {
            Ty::Component(t) | Ty::Resource(t) => Some((**t).clone()),
            _ => None,
        }
    }

    /// Return the host layout of a struct (including `Component`/`Resource`
    /// of one, or a `Res`/`Mut` borrow of one).
    pub fn struct_layout(&self) -> Option<Arc<StructLayout>> {
        match self {
            Ty::Struct(l) | Ty::OwnedStruct(l) => Some(Arc::clone(l)),
            Ty::Component(inner) | Ty::Resource(inner) => inner.struct_layout(),
            Ty::Res(inner) | Ty::Mut(inner) => inner.struct_layout(),
            _ => None,
        }
    }

    /// Byte size and alignment for a scalar or host component value.
    pub fn value_layout(&self) -> Option<(u64, u64)> {
        match self {
            Ty::I8 | Ty::U8 | Ty::Bool => Some((1, 1)),
            Ty::I16 | Ty::U16 => Some((2, 2)),
            Ty::I32 | Ty::U32 | Ty::F32 => Some((4, 4)),
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::Entity => Some((8, 8)),
            Ty::Struct(layout) | Ty::OwnedStruct(layout) => Some((layout.size, layout.align)),
            _ => None,
        }
    }

    /// Integer types (signed and unsigned, any width).
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64
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
            Ty::Struct(layout) | Ty::OwnedStruct(layout) => Some(layout.size),
            Ty::Option(_)
            | Ty::Tuple(_)
            | Ty::Component(_)
            | Ty::Resource(_)
            | Ty::Query(..)
            | Ty::Res(_)
            | Ty::Mut(_)
            | Ty::Added(_)
            | Ty::Changed(_)
            | Ty::World
            | Ty::Commands
            | Ty::Str
            | Ty::Array(_)
            | Ty::Ptr => Some(8),
            Ty::Void => None,
        }
    }

    /// `(size, align)` in bytes for a value that can live in memory.
    /// Returns `None` for signature-only / void types.
    pub fn size_align(&self) -> Option<(u32, u32)> {
        match self {
            Ty::I8 | Ty::U8 | Ty::Bool => Some((1, 1)),
            Ty::I16 | Ty::U16 => Some((2, 2)),
            Ty::I32 | Ty::U32 | Ty::F32 => Some((4, 4)),
            Ty::I64 | Ty::U64 | Ty::F64 | Ty::Entity | Ty::Ptr | Ty::Str | Ty::Array(_) => {
                Some((8, 8))
            }
            Ty::Option(_) => Some((8, 8)),
            Ty::Struct(l) | Ty::OwnedStruct(l) => Some((l.size as u32, l.align as u32)),
            Ty::Component(inner) | Ty::Resource(inner) => inner.size_align(),
            Ty::Tuple(ts) => {
                // Treat tuple as an anonymous struct for sizing.
                let fields: Vec<(String, Ty)> = ts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (format!("_{i}"), t.clone()))
                    .collect();
                StructLayout::repr_c("", fields)
                    .ok()
                    .map(|l| (l.size as u32, l.align as u32))
            }
            Ty::Query(..)
            | Ty::Res(_)
            | Ty::Mut(_)
            | Ty::Added(_)
            | Ty::Changed(_)
            | Ty::World
            | Ty::Commands
            | Ty::Void => None,
        }
    }

    /// True for scalar POD leaves and POD structs. Used to decide whether a
    /// type may be registered as a component or resource.
    pub fn is_pod_leaf(&self) -> bool {
        match self {
            Ty::I8
            | Ty::I16
            | Ty::I32
            | Ty::I64
            | Ty::U8
            | Ty::U16
            | Ty::U32
            | Ty::U64
            | Ty::F32
            | Ty::F64
            | Ty::Bool
            | Ty::Entity => true,
            Ty::Struct(l) | Ty::OwnedStruct(l) => l.is_pod,
            Ty::Component(inner) | Ty::Resource(inner) => inner.is_pod_leaf(),
            Ty::Tuple(ts) => ts.iter().all(|t| t.is_pod_leaf()),
            _ => false,
        }
    }

    /// Ensure this type (or its inner payload) is POD; return an error string
    /// otherwise. Used when registering components/resources.
    pub fn ensure_pod(&self) -> Result<(), String> {
        if self.is_pod_leaf() {
            Ok(())
        } else {
            Err(format!(
                "type `{self}` is not POD (components/resources must be plain data)"
            ))
        }
    }

    /// Element type of an array type.
    pub fn elem(&self) -> Option<Ty> {
        match self {
            Ty::Array(elem) => Some((**elem).clone()),
            _ => None,
        }
    }

    /// Look up a field by name when this is a struct (or Component/Resource of one).
    pub fn field(&self, name: &str) -> Option<(u64, Ty)> {
        let layout = self.struct_layout()?;
        let f = layout.field(name)?;
        Some((f.offset, f.ty.clone()))
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}
