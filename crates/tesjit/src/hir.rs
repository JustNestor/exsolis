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

/// What a call targets: a function in this module, or a registered host import.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CallTarget {
    Local(usize),
    Host(String),
}

#[derive(Clone, Debug)]
pub enum ExprKind {
    /// An integer literal, already narrowed to the expected type.
    I32(i32),
    /// A float literal.
    F64(f64),
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
    Cast { expr: Box<Expr>, to: Ty },
}
