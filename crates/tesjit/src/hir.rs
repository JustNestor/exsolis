//! The typed intermediate representation.
//!
//! [`crate::check`] lowers the oxc AST into this owned, fully typed tree in one
//! pass: names are resolved to local slots, every expression carries its
//! [`Ty`], and every signature has been validated against the host registry.
//!
//! Because of that, [`crate::codegen`] is a pure structural walk — no symbol
//! tables, no type expectations, no fallible decisions. That separation is what
//! makes batch diagnostics ("report every error at once") and a future AOT
//! backend cheap to add.

use oxc_ast::ast::{BinaryOperator, LogicalOperator, UnaryOperator};
use oxc_span::Span;

use crate::ty::Ty;

/// A whole compilation unit.
pub struct Module {
    /// TS-defined functions, in source order.
    pub functions: Vec<FnDef>,
}

impl Module {
    pub fn find(&self, name: &str) -> Option<&FnDef> {
        self.functions.iter().find(|f| f.name == name)
    }
}

/// A function definition: signature, locals, and a typed body.
pub struct FnDef {
    pub name: String,
    /// Parameter types, positional.
    pub params: Vec<Ty>,
    /// Return type.
    pub ret: Ty,
    /// All named values of this function, parameters first, then `let`s in
    /// declaration order. Expressions reference them by index, never by name.
    pub locals: Vec<Local>,
    /// Typed function body.
    pub body: Vec<Stmt>,
    pub span: Span,
}

impl FnDef {
    /// Local slots occupied by parameters.
    pub fn param_len(&self) -> usize {
        self.params.len()
    }
}

/// One named value inside a function.
#[derive(Clone, Debug)]
pub struct Local {
    pub name: String,
    pub ty: Ty,
    pub span: Span,
}

/// A statement. Control flow keeps its body as a nested statement list, so
/// codegen just emits blocks in order.
pub enum Stmt {
    /// `let x: T = expr;` — `init` is `None` only when a declaration was
    /// default-initialized (the checker rejects un-annotated ones).
    Let {
        local: usize,
        init: Option<Expr>,
        span: Span,
    },
    /// Expression statement; the value is discarded.
    Expr { expr: Expr, span: Span },
    /// `return expr;`
    Return { value: Option<Expr>, span: Span },
    If {
        cond: Expr,
        then: Vec<Stmt>,
        els: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        /// `for (let i = 0; ...)`: the loop-scope declarations, in source
        /// order (the checker binds them for the whole loop).
        init: Vec<Stmt>,
        cond: Option<Expr>,
        update: Option<Expr>,
        body: Vec<Stmt>,
        span: Span,
    },
    /// A bare `{ ... }` block.
    Block { body: Vec<Stmt>, span: Span },
    /// `break` — target is the innermost loop.
    Break { span: Span },
    /// `continue` — target is the innermost loop.
    Continue { span: Span },
    ForOf {
        binding: usize,
        iter: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
}

/// A typed expression. `ty` is always the checker-computed type of the node.
#[derive(Clone, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub ty: Ty,
    pub span: Span,
}

impl Expr {
    pub fn new(kind: ExprKind, ty: Ty, span: Span) -> Self {
        Self { kind, ty, span }
    }
}

/// What a call targets: a function in this module, a registered host import,
/// or a runtime helper.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CallTarget {
    Local(usize),
    Host(String),
    /// The string/array intrinsics of [`crate::runtime`].
    Intrinsic(Intrinsic),
}

/// Builtin operations JIT code calls into [`crate::runtime`] for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intrinsic {
    /// Fuel guard: `() -> boolean` (false when the budget ran out).
    Fuel,
    /// Wrap `.rodata` bytes as an arena string: `(ptr, len) -> str`.
    StrWrap,
    StrLen,
    StrConcat,
    StrEq,
    StrByte,
    /// `(cap, elem_size) -> array`
    ArrNew,
    /// arena alloc: `(size) -> ptr`
    Alloc,
    ArrLen,
    /// `(array, elem_size, value) -> array`
    ArrPush,
    /// `(array, index, elem_size) -> value`
    ArrGet,
    /// `(region, offset) -> f64` (reads an `f32` and widens it)
    LoadF32,
    /// `(region, offset) -> i64`
    LoadI64,
    /// `(region, offset, f64) -> f64` (narrows to an `f32`)
    StoreF32,
    /// `(region, offset, i64) -> i64`
    StoreI64,
    /// `(region, offset) -> f64`
    LoadF64,
    /// `(region, offset) -> i32`
    LoadI32,
    /// `(region, offset, value) -> f64`
    StoreF64,
    /// `(region, offset, i32) -> i32`
    StoreI32,
    /// `(array, index, elem_size, value) -> value`
    ArrSet,
    /// `(string) -> i32` — resolve a region name to its id.
    RegionId,
    /// Option<T> isSome: `(ptr) -> i64`.
    OptIsSome,
    /// Option<T> unwrap (integer-ish): `(ptr) -> i64`.
    OptUnwrap,
    /// Option<T> unwrap (float): `(ptr) -> f64`.
    OptUnwrapF64,
    /// Option<T> unwrapOr (integer-ish): `(ptr, default) -> i64`.
    OptUnwrapOr,
    /// Option<T> unwrapOr (float): `(ptr, default) -> f64`.
    OptUnwrapOrF64,
}

/// Builtins callable straight from TS (they are runtime symbols, not host
/// functions, so they take no `declare`).
///
/// The names are reserved: a mod cannot define its own `load_f64`.
pub const BUILTINS: &[(&str, Intrinsic)] = &[
    ("load_f64", Intrinsic::LoadF64),
    ("load_f32", Intrinsic::LoadF32),
    ("load_i64", Intrinsic::LoadI64),
    ("store_f64", Intrinsic::StoreF64),
    ("store_f32", Intrinsic::StoreF32),
    ("store_i64", Intrinsic::StoreI64),
    ("load_i32", Intrinsic::LoadI32),
    ("store_f64", Intrinsic::StoreF64),
    ("store_i32", Intrinsic::StoreI32),
    ("region_id", Intrinsic::RegionId),
];

impl Intrinsic {
    /// Name of the runtime symbol backing this operation.
    pub fn symbol(self) -> &'static str {
        match self {
            Intrinsic::Fuel => "__tesjit_fuel",
            Intrinsic::StrWrap => "__tesjit_str_wrap",
            Intrinsic::StrLen => "__tesjit_str_len",
            Intrinsic::StrConcat => "__tesjit_str_concat",
            Intrinsic::StrEq => "__tesjit_str_eq",
            Intrinsic::StrByte => "__tesjit_str_byte",
            Intrinsic::ArrNew => "__tesjit_arr_new",
            Intrinsic::Alloc => "__tesjit_alloc",
            Intrinsic::ArrLen => "__tesjit_arr_len",
            Intrinsic::ArrPush => "__tesjit_arr_push",
            Intrinsic::LoadF64 => "__tesjit_load_f64",
            Intrinsic::LoadF32 => "__tesjit_load_f32",
            Intrinsic::LoadI64 => "__tesjit_load_i64",
            Intrinsic::LoadI32 => "__tesjit_load_i32",
            Intrinsic::StoreF64 => "__tesjit_store_f64",
            Intrinsic::StoreF32 => "__tesjit_store_f32",
            Intrinsic::StoreI64 => "__tesjit_store_i64",
            Intrinsic::StoreI32 => "__tesjit_store_i32",
            Intrinsic::RegionId => "__tesjit_region_id",
            Intrinsic::ArrGet => "__tesjit_arr_get",
            Intrinsic::ArrSet => "__tesjit_arr_set",
            Intrinsic::OptIsSome => "__tesjit_opt_is_some",
            Intrinsic::OptUnwrap => "__tesjit_opt_unwrap_i64",
            Intrinsic::OptUnwrapF64 => "__tesjit_opt_unwrap_f64",
            Intrinsic::OptUnwrapOr => "__tesjit_opt_unwrap_or_i64",
            Intrinsic::OptUnwrapOrF64 => "__tesjit_opt_unwrap_or_f64",
        }
    }
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    /// An integer literal, already narrowed to the expected type.
    Int(i64),
    /// A float literal.
    Float(f64),
    /// `true` / `false`.
    Bool(bool),
    /// A local slot (parameter or `let`).
    Local(usize),
    Unary {
        op: UnaryOperator,
        arg: Box<Expr>,
    },
    Binary {
        op: BinaryOperator,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `&&` / `||` — short-circuits, so codegen needs blocks.
    Logical {
        op: LogicalOperator,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Ternary {
        cond: Box<Expr>,
        then: Box<Expr>,
        els: Box<Expr>,
    },
    /// `x = v` or `x += v`. `op` is `None` for plain assignment.
    Assign {
        local: usize,
        op: Option<BinaryOperator>,
        value: Box<Expr>,
    },
    /// `++x` / `x++`.
    Update {
        local: usize,
        inc: bool,
        prefix: bool,
    },
    /// `callee(args...)` — arity and argument types already checked.
    Call {
        target: CallTarget,
        args: Vec<Expr>,
    },
    /// `expr as T`, including the numeric widening/narrowing conversions.
    Cast {
        expr: Box<Expr>,
        to: Ty,
    },
    /// A string literal; codegen materializes it from `.rodata` bytes.
    Str(String),
    /// `s.length` / `a.length`.
    Length {
        target: Box<Expr>,
    },
    /// `s[i]` (string byte) / `a[i]` (array element).
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    /// `a[i] = v`.
    SetIndex {
        target: Box<Expr>,
        index: Box<Expr>,
        value: Box<Expr>,
    },
    /// `[a, b, c]` — array literal with a uniform element type.
    ArrayLit {
        elems: Vec<Expr>,
        elem_ty: Ty,
    },
    PlaceRead { place: PlaceExpr },
    PlaceWrite {
        place: PlaceExpr,
        op: Option<BinaryOperator>,
        value: Box<Expr>,
    },
    Iter { query: Box<Expr> },
    OptionSome { value: Box<Expr> },
    OptionNone { ty: Ty },
}

#[derive(Clone, Debug)]
pub enum PlaceExpr {
    Slot(usize),
    Field { base: usize, name: String },
    Index { base: usize, index: usize },
}
