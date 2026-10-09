//! Lower oxc AST → typed HIR, collecting every diagnostic instead of stopping
//! at the first one.
//!
//! One pass resolves names (to local slots), checks types (against the
//! declared signatures and the host registry) and builds the HIR. On error the
//! pass records a diagnostic and substitutes a dummy expression, so a broken
//! function still lowers and its neighbours are still checked.


use oxc_allocator::Allocator;
use oxc_ast::ast::{
    AssignmentOperator, AssignmentTarget, BinaryOperator, BindingPattern, Declaration,
    Expression, ForStatementInit, Function, LogicalOperator, SimpleAssignmentTarget, Statement,
    TSType, TSTypeName, UnaryOperator, UpdateOperator, VariableDeclaration,
};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType, Span};

use crate::diag::{err, CompileError, Diagnostics, Src};
use miette::{SourceOffset, SourceSpan};
use crate::host::{FnKind, HostRegistry, HostSig};
use crate::hir::{
    CallTarget, Expr, ExprKind, FnDef, Local, Module, Stmt,
};
use crate::ty::Ty;

/// Collected pass state for one lowering run.
pub struct Lowerer<'a> {
    src: &'a Src,
    host: &'a HostRegistry,
    diags: Diagnostics,
}

/// Extract the `Function` node of a top-level statement (function declaration,
/// `export function`, `export default function`).
fn as_function<'a, 'b>(s: &'b Statement<'a>) -> Option<&'b Function<'a>> {
    match s {
        Statement::FunctionDeclaration(f) => Some(f),
        Statement::ExportDeclaration(e) => match &e.declaration {
            Declaration::FunctionDeclaration(f) => Some(f),
            _ => None,
        },
        Statement::ExportDefaultDeclaration(e) => match &e.declaration {
            oxc_ast::ast::ExportDefaultDeclarationKind::FunctionDeclaration(f) => Some(f),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> Lowerer<'a> {
    pub fn new(src: &'a Src, host: &'a HostRegistry) -> Self {
        Self {
            src,
            host,
            diags: Diagnostics::new(),
        }
    }

    // ───────────────────────── entry points ─────────────────────────

    /// Parse and lower `source`, reporting all diagnostics found.
    pub fn run(source: &str, src: &Src, host: &HostRegistry) -> (Option<Module>, Diagnostics) {
        let alloc = Allocator::default();
        let parsed = Parser::new(&alloc, source, SourceType::ts()).parse();
        if !parsed.diagnostics.is_empty() {
            let diags = parsed
                .diagnostics
                .iter()
                .map(|d| {
                    let (off, len) = d
                        .labels
                        .first()
                        .map(|l| (l.offset() as usize, l.len() as usize))
                        .unwrap_or((0, 0));
                    crate::diag::SyntaxDiag {
                        msg: d.message.to_string(),
                        span: SourceSpan::new(SourceOffset::from(off), len),
                    }
                    .into_syntax_error(src)
                })
                .collect();
            return (None, Diagnostics(diags));
        }

        let mut lowerer = Lowerer::new(src, host);
        let module = lowerer.lower(&parsed.program.body);
        let diags = std::mem::take(&mut lowerer.diags);
        if diags.is_empty() {
            (Some(module), diags)
        } else {
            (None, diags)
        }
    }

    /// Lower every top-level function of the program.
    fn lower(&mut self, body: &[Statement<'_>]) -> Module {
        // Pass 1: signatures. This is what makes forward calls legal and lets
        // the checker validate `declare function` against the host registry.
        //
        // The table is a source-ordered Vec (not a HashMap) because the HIR
        // refers to functions by *index*, and codegen builds its own table in
        // the same order.
        let mut sigs: Vec<(String, FnSig)> = Vec::new();
        for stmt in body {
            let Some(f) = as_function(stmt) else { continue };
            let Some(name) = f.id.as_ref().map(|i| i.name.as_str().to_string()) else {
                self.at(f.span, "anonymous function");
                continue;
            };
            let mut params = Vec::with_capacity(f.params.items.len());
            let mut bad = false;
            for p in &f.params.items {
                match p.type_annotation.as_ref().map(|a| ty_of(&a.type_annotation)) {
                    Some(Ok(ty)) => params.push(ty),
                    Some(Err(e)) => {
                        self.push(e);
                        bad = true;
                    }
                    None => {
                        self.at(
                            p.span,
                            format!("parameter of `{name}` needs a type annotation"),
                        );
                        bad = true;
                    }
                }
            }
            let ret = match &f.return_type {
                Some(a) => match ty_of(&a.type_annotation) {
                    Ok(ty) => ty,
                    Err(e) => {
                        self.push(e);
                        bad = true;
                        Ty::Void
                    }
                },
                None => {
                    self.at(
                        f.span,
                        format!("`{name}` needs a return type annotation (`: void` if none)"),
                    );
                    bad = true;
                    Ty::Void
                }
            };
            if f.declare || f.body.is_none() {
                // `declare function name(...)`: must match a registered host
                // function exactly.
                match self.host.get(&name) {
                    Some(host) => {
                        if host.params != params || host.ret != ret {
                            self.at(
                                f.span,
                                format!("`{name}` signature differs from host function"),
                            );
                        }
                    }
                    None => {
                        self.at(
                            f.span,
                            format!("no host function `{name}` registered"),
                        );
                    }
                }
                continue;
            }
            if bad {
                continue;
            }
            if sigs.iter().any(|(n, _)| *n == name) {
                self.at(f.span, format!("duplicate function `{name}`"));
                continue;
            }
            sigs.push((name, FnSig { params, ret }));
        }

        // Pass 2: bodies, now that every name in the module is known.
        let scope = ModuleScope {
            fns: sigs,
            host: self
                .host
                .iter()
                .map(|h| (h.name.clone(), h.sig()))
                .collect(),
        };
        let mut functions = Vec::new();
        for (name, sig) in &scope.fns {
            let Some(f) = body
                .iter()
                .filter_map(as_function)
                .find(|f| f.id.as_ref().map(|i| i.name.as_str()) == Some(name.as_str()))
            else {
                continue;
            };
            let sig = sig.clone();
            if let Some(def) = self.lower_function(f, name.clone(), sig, &scope) {
                functions.push(def);
            }
        }

        Module { functions }
    }

    fn lower_function(
        &mut self,
        f: &Function<'_>,
        name: String,
        sig: FnSig,
        scope: &ModuleScope,
    ) -> Option<FnDef> {
        let mut locals = Vec::with_capacity(f.params.items.len() + 16);
        for (i, p) in f.params.items.iter().enumerate() {
            let BindingPattern::BindingIdentifier(id) = &p.pattern else {
                self.at(p.span, "destructuring parameters are not supported");
                return None;
            };
            locals.push(Local {
                name: id.name.to_string(),
                ty: sig.params[i],
                span: id.span,
            });
        }

        let body = f.body.as_ref().expect("checked in pass 1");
        let mut ctx = FnCtx {
            lowerer: self,
            locals,
            scopes: vec![0],
            sigs: scope,
            ret: sig.ret,
            loop_depth: 0,
        };
        // The function body lives in the function's own (base) scope: locals
        // declared at its top level must stay alive for the whole function, so
        // no extra scope is pushed here.
        let body = ctx.stmts(&body.statements);
        Some(FnDef {
            name,
            params: sig.params,
            ret: sig.ret,
            locals: std::mem::take(&mut ctx.locals),
            body,
            span: f.span,
        })
    }

    fn at(&mut self, span: Span, msg: impl Into<String>) {
        self.push(err(self.src, span, msg));
    }

    fn push(&mut self, e: CompileError) {
        self.diags.push(e);
    }
}

/// Per-function lowering state.
struct FnCtx<'a, 'l> {
    lowerer: &'a mut Lowerer<'l>,
    locals: Vec<Local>,
    /// Scope boundaries as "locals length" marks; `scopes[0] == params.len()`.
    scopes: Vec<usize>,
    /// Signatures of everything callable from this unit.
    sigs: &'a ModuleScope,
    /// Function return type.
    ret: Ty,
    /// Nesting of loops, for `break` / `continue` legality.
    loop_depth: usize,
}

impl<'a, 'l> FnCtx<'a, 'l> {
    fn at(&mut self, span: Span, msg: impl Into<String>) {
        self.lowerer.at(span, msg);
    }

    fn push_scope(&mut self) {
        self.scopes.push(self.locals.len());
    }

    /// End a lexical scope.
    ///
    /// Local slots are **not** reclaimed: tesjit has no shadowing (a repeated
    /// binding name is a "duplicate binding" error), so a slot stays valid for
    /// the whole function and codegen can map it 1:1 to a Cranelift variable.
    /// The scope stack is kept so shadowing can be added without touching
    /// `Stmt::Let` indices.
    fn pop_scope(&mut self) {
        self.scopes.pop().expect("scope underflow");
    }

    /// Resolve a name to the innermost matching local slot.
    fn lookup(&self, name: &str) -> Option<usize> {
        (0..self.locals.len())
            .rev()
            .find(|&i| self.locals[i].name == name)
    }

    fn declare(&mut self, name: &str, ty: Ty, span: Span) -> Option<usize> {
        if self.lookup(name).is_some() {
            self.at(span, format!("duplicate binding `{name}`"));
            return None;
        }
        self.locals.push(Local {
            name: name.to_owned(),
            ty,
            span,
        });
        Some(self.locals.len() - 1)
    }

    fn poison(&mut self, span: Span, msg: impl Into<String>) -> Expr {
        self.at(span, msg);
        Expr::new(ExprKind::I32(0), Ty::I32, span)
    }

    // ───────────────────────── statements ─────────────────────────

    fn block(&mut self, stmts: &[Statement<'_>]) -> Vec<Stmt> {
        self.push_scope();
        let out = self.stmts(stmts);
        self.pop_scope();
        out
    }

    fn stmts(&mut self, stmts: &[Statement<'_>]) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(stmts.len());
        for s in stmts {
            if let Some(stmt) = self.stmt(s) {
                out.push(stmt);
            }
        }
        out
    }

    fn stmt(&mut self, s: &Statement<'_>) -> Option<Stmt> {
        match s {
            Statement::ExpressionStatement(e) => {
                let expr = self.expr(&e.expression, None);
                Some(Stmt::Expr {
                    expr,
                    span: e.span,
                })
            }
            Statement::VariableDeclaration(d) => {
                let decls = self.var_decl(d);
                if decls.is_empty() {
                    None
                } else {
                    Some(Stmt::Block {
                        body: decls,
                        span: s.span(),
                    })
                }
            }
            Statement::BlockStatement(bl) => Some(Stmt::Block {
                body: self.block(&bl.body),
                span: bl.span,
            }),
            Statement::ReturnStatement(r) => {
                let value = match &r.argument {
                    Some(e) => {
                        let v = self.expr(e, Some(self.ret));
                        if v.ty != self.ret {
                            self.at(
                                r.span,
                                format!("returns {}, expected {}", v.ty, self.ret),
                            );
                        }
                        Some(v)
                    }
                    None => {
                        if self.ret != Ty::Void {
                            self.at(r.span, "missing return value");
                        }
                        None
                    }
                };
                Some(Stmt::Return { value, span: r.span })
            }
            Statement::IfStatement(i) => {
                let cond = self.cond(&i.test);
                Some(Stmt::If {
                    cond,
                    then: self.stmt_block(&i.consequent),
                    els: i
                        .alternate
                        .as_ref()
                        .map(|a| self.stmt_block(a)),
                    span: i.span,
                })
            }
            Statement::WhileStatement(w) => {
                self.loop_depth += 1;
                let cond = self.cond(&w.test);
                let body = self.stmt_block(&w.body);
                self.loop_depth -= 1;
                Some(Stmt::While {
                    cond,
                    body,
                    span: w.span,
                })
            }
            Statement::ForStatement(f) => {
                self.loop_depth += 1;
                // The loop's `let` lives in a scope that outlives the body,
                // matching JS semantics (and what `update` refers to).
                self.push_scope();
                let init = match &f.init {
                    Some(ForStatementInit::VariableDeclaration(d)) => self.var_decl(d),
                    Some(i) => i
                        .as_expression()
                        .map(|e| Stmt::Expr {
                            expr: self.expr(e, None),
                            span: e.span(),
                        })
                        .into_iter()
                        .collect(),
                    None => Vec::new(),
                };
                let cond = f.test.as_ref().map(|t| self.cond(t));
                let update = f.update.as_ref().map(|u| self.expr(u, None));
                let body = self.stmt_block(&f.body);
                self.pop_scope();
                self.loop_depth -= 1;
                Some(Stmt::For {
                    init,
                    cond,
                    update,
                    body,
                    span: f.span,
                })
            }
            Statement::BreakStatement(b) => {
                if self.loop_depth == 0 {
                    self.at(b.span, "`break` outside loop");
                }
                Some(Stmt::Break { span: b.span })
            }
            Statement::ContinueStatement(c) => {
                if self.loop_depth == 0 {
                    self.at(c.span, "`continue` outside loop");
                }
                Some(Stmt::Continue { span: c.span })
            }
            Statement::EmptyStatement(_) => None,
            Statement::TSTypeAliasDeclaration(_) | Statement::TSInterfaceDeclaration(_) => None,
            other => {
                self.at(other.span(), "unsupported statement");
                None
            }
        }
    }

    /// A statement in control-flow position takes its own scope.
    fn stmt_block(&mut self, s: &Statement<'_>) -> Vec<Stmt> {
        self.push_scope();
        let mut out = Vec::new();
        if let Some(stmt) = self.stmt(s) {
            out.push(stmt);
        }
        self.pop_scope();
        out
    }

    fn var_decl(&mut self, d: &VariableDeclaration) -> Vec<Stmt> {
        let mut out = Vec::with_capacity(d.declarations.len());
        for decl in &d.declarations {
            let BindingPattern::BindingIdentifier(id) = &decl.id else {
                self.at(decl.span, "destructuring is not supported");
                continue;
            };
            let declared = match &decl.type_annotation {
                Some(a) => match ty_of(&a.type_annotation) {
                    Ok(ty) => Some(ty),
                    Err(e) => {
                        self.at(decl.span, e.to_string());
                        continue;
                    }
                },
                None => None,
            };
            let (init, ty) = match &decl.init {
                Some(init) => {
                    let v = self.expr(init, declared);
                    let ty = v.ty;
                    if let Some(t) = declared
                        && t != ty {
                            self.at(
                                decl.span,
                                format!("cannot assign {ty} to {t} (use `as`)"),
                            );
                        }
                    (Some(v), ty)
                }
                None => {
                    let Some(t) = declared else {
                        self.at(
                            decl.span,
                            "variable needs a type annotation or initializer",
                        );
                        continue;
                    };
                    (None, t)
                }
            };
            if ty == Ty::Void {
                self.at(decl.span, "variable cannot be void");
                continue;
            }
            let Some(local) = self.declare(&id.name, ty, id.span) else {
                continue;
            };
            out.push(Stmt::Let {
                local,
                init,
                span: decl.span,
            });
        }
        out
    }

    /// Boolean condition of `if` / `while` / `for`.
    fn cond(&mut self, e: &Expression<'_>) -> Expr {
        let v = self.expr(e, Some(Ty::Bool));
        if v.ty != Ty::Bool {
            self.at(e.span(), format!("condition must be boolean, found {}", v.ty));
        }
        v
    }

    // ───────────────────────── expressions ─────────────────────────

    /// `expect` is the contextual type (used to type numeric literals and to
    /// propagate an operand type into the other side of a binary op).
    fn expr(&mut self, e: &Expression<'_>, expect: Option<Ty>) -> Expr {
        match e {
            Expression::NumericLiteral(n) => {
                let span = n.span;
                if expect == Some(Ty::I32) {
                    if n.value.fract() != 0.0
                        || n.value < i32::MIN as f64
                        || n.value > i32::MAX as f64
                    {
                        return self.poison(span, "literal does not fit in i32");
                    }
                    Expr::new(ExprKind::I32(n.value as i32), Ty::I32, span)
                } else {
                    Expr::new(ExprKind::F64(n.value), Ty::F64, span)
                }
            }
            Expression::BooleanLiteral(l) => {
                Expr::new(ExprKind::Bool(l.value), Ty::Bool, l.span)
            }
            Expression::ParenthesizedExpression(p) => self.expr(&p.expression, expect),
            Expression::Identifier(id) => {
                let Some(slot) = self.lookup(id.name.as_str()) else {
                    return self.poison(id.span, format!("unknown variable `{}`", id.name));
                };
                let ty = self.locals[slot].ty;
                Expr::new(ExprKind::Local(slot), ty, id.span)
            }
            Expression::UnaryExpression(u) => {
                let a = self.expr(&u.argument, expect);
                let ty = a.ty;
                let span = u.span;
                match (u.operator, ty) {
                    (UnaryOperator::UnaryPlus, Ty::I32 | Ty::F64) => {}
                    (UnaryOperator::UnaryNegation, Ty::I32 | Ty::F64) => {}
                    (UnaryOperator::BitwiseNot, Ty::I32) => {}
                    (UnaryOperator::LogicalNot, Ty::Bool) => {}
                    _ => {
                        return self.poison(
                            span,
                            format!("unsupported unary op {:?} on {}", u.operator, ty),
                        );
                    }
                }
                Expr::new(
                    ExprKind::Unary {
                        op: u.operator,
                        arg: Box::new(a),
                    },
                    ty,
                    span,
                )
            }
            Expression::BinaryExpression(bin) => {
                let is_cmp = is_comparison(bin.operator);
                let operand_expect = if is_cmp { None } else { expect };
                let left_is_lit = matches!(&bin.left, Expression::NumericLiteral(_));
                let right_is_lit = matches!(&bin.right, Expression::NumericLiteral(_));
                let (l, r) = if left_is_lit && !right_is_lit {
                    let r = self.expr(&bin.right, operand_expect);
                    let l = self.expr(&bin.left, Some(r.ty));
                    (l, r)
                } else {
                    let l = self.expr(&bin.left, operand_expect);
                    let r = self.expr(&bin.right, Some(l.ty));
                    (l, r)
                };
                let ty = if is_cmp { Ty::Bool } else { l.ty };
                self.check_arith(bin.operator, &l, &r, bin.span);
                Expr::new(
                    ExprKind::Binary {
                        op: bin.operator,
                        lhs: Box::new(l),
                        rhs: Box::new(r),
                    },
                    ty,
                    bin.span,
                )
            }
            Expression::LogicalExpression(log) => {
                let lhs = self.expr(&log.left, Some(Ty::Bool));
                let rhs = self.expr(&log.right, Some(Ty::Bool));
                let mut ok = lhs.ty == Ty::Bool;
                if rhs.ty != Ty::Bool {
                    ok = false;
                }
                if !ok {
                    self.at(log.span, "`&&` / `||` need boolean operands");
                }
                if !matches!(log.operator, LogicalOperator::And | LogicalOperator::Or) {
                    self.at(log.span, "`??` is not supported");
                }
                Expr::new(
                    ExprKind::Logical {
                        op: log.operator,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    },
                    Ty::Bool,
                    log.span,
                )
            }
            Expression::ConditionalExpression(c) => {
                let cond = self.cond(&c.test);
                let then = self.expr(&c.consequent, expect);
                let els = self.expr(&c.alternate, Some(then.ty));
                if els.ty != then.ty {
                    self.at(c.span, "branches of `?:` have different types");
                }
                Expr::new(
                    ExprKind::Ternary {
                        cond: Box::new(cond),
                        then: Box::new(then.clone()),
                        els: Box::new(els),
                    },
                    then.ty,
                    c.span,
                )
            }
            Expression::AssignmentExpression(a) => {
                let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left else {
                    return self.poison(
                        a.span,
                        "only assignment to plain variables is supported",
                    );
                };
                let Some(slot) = self.lookup(id.name.as_str()) else {
                    return self.poison(a.span, format!("unknown variable `{}`", id.name));
                };
                let ty = self.locals[slot].ty;
                let rhs = self.expr(&a.right, Some(ty));
                let op = if a.operator == AssignmentOperator::Assign {
                    None
                } else {
                    match compound_op(a.operator) {
                        Some(op) => Some(op),
                        None => {
                            self.at(a.span, "unsupported compound assignment");
                            None
                        }
                    }
                };
                if let Some(op) = op {
                    self.check_arith(
                        op,
                        &Expr::new(ExprKind::Local(slot), ty, a.span),
                        &rhs,
                        a.span,
                    );
                } else if rhs.ty != ty {
                    self.at(
                        a.span,
                        format!("cannot assign {} to {}", rhs.ty, ty),
                    );
                }
                Expr::new(
                    ExprKind::Assign {
                        local: slot,
                        op,
                        value: Box::new(rhs),
                    },
                    ty,
                    a.span,
                )
            }
            Expression::UpdateExpression(u) => {
                let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &u.argument else {
                    return self.poison(u.span, "++/-- only on plain variables");
                };
                let Some(slot) = self.lookup(id.name.as_str()) else {
                    return self.poison(u.span, format!("unknown variable `{}`", id.name));
                };
                let ty = self.locals[slot].ty;
                if !ty.is_numeric() {
                    self.at(u.span, "++/-- need a number");
                }
                Expr::new(
                    ExprKind::Update {
                        local: slot,
                        inc: u.operator == UpdateOperator::Increment,
                        prefix: u.prefix,
                    },
                    ty,
                    u.span,
                )
            }
            Expression::CallExpression(c) => {
                let Expression::Identifier(callee) = &c.callee else {
                    return self.poison(
                        c.span,
                        "only calls to named functions are supported",
                    );
                };
                let name = callee.name.as_str();
                let Some(kind) = self.sigs.kind_of(name) else {
                    return self.poison(c.span, format!("unknown function `{name}`"));
                };
                let (params, ret) = match &kind {
                    FnKind::Local => {
                        let sig = self.sigs.sig(name).expect("local fn has a signature");
                        (sig.params.clone(), sig.ret)
                    }
                    FnKind::Host(sig) => (sig.params.clone(), sig.ret),
                };
                if c.arguments.len() != params.len() {
                    return self.poison(
                        c.span,
                        format!("`{name}` expects {} arguments", params.len()),
                    );
                }
                let mut args = Vec::with_capacity(params.len());
                let mut bad = false;
                for (a, &pty) in c.arguments.iter().zip(&params) {
                    let Some(ex) = a.as_expression() else {
                        self.at(c.span, "spread arguments are not supported");
                        bad = true;
                        continue;
                    };
                    let v = self.expr(ex, Some(pty));
                    if v.ty != pty {
                        self.at(
                            ex.span(),
                            format!("argument is {}, expected {}", v.ty, pty),
                        );
                        bad = true;
                    }
                    args.push(v);
                }
                if bad {
                    return self.poison(c.span, "bad arguments");
                }
                let target = match kind {
                    FnKind::Local => CallTarget::Local(self.sigs.index(name)),
                    FnKind::Host(_) => CallTarget::Host(name.to_owned()),
                };
                Expr::new(ExprKind::Call { target, args }, ret, c.span)
            }
            Expression::TSAsExpression(a) => {
                let target = match ty_of(&a.type_annotation) {
                    Ok(ty) => ty,
                    Err(e) => {
                        self.at(a.span, e.to_string());
                        Ty::I32
                    }
                };
                let v = self.expr(&a.expression, None);
                match (v.ty, target) {
                    (x, y) if x == y => v,
                    (Ty::I32, Ty::F64)
                    | (Ty::F64, Ty::I32)
                    | (Ty::Bool, Ty::I32)
                    | (Ty::Bool, Ty::F64) => Expr::new(
                        ExprKind::Cast {
                            expr: Box::new(v),
                            to: target,
                        },
                        target,
                        a.span,
                    ),
                    (x, y) => self.poison(
                        a.span,
                        format!("cannot cast {x} to {y}"),
                    ),
                }
            }
            other => self.poison(other.span(), "unsupported expression"),
        }
    }

    /// Operator/operand compatibility (same rules the codegen relies on).
    fn check_arith(&mut self, op: BinaryOperator, l: &Expr, r: &Expr, span: Span) {
        if l.ty != r.ty {
            self.at(
                span,
                format!(
                    "operand types differ: {} vs {} (use `as`)",
                    l.ty, r.ty
                ),
            );
            return;
        }
        let supported = match (l.ty, op) {
            (Ty::I32, _) => matches!(
                op,
                BinaryOperator::Addition
                    | BinaryOperator::Subtraction
                    | BinaryOperator::Multiplication
                    | BinaryOperator::Division
                    | BinaryOperator::Remainder
                    | BinaryOperator::BitwiseAnd
                    | BinaryOperator::BitwiseOR
                    | BinaryOperator::BitwiseXOR
                    | BinaryOperator::ShiftLeft
                    | BinaryOperator::ShiftRight
                    | BinaryOperator::ShiftRightZeroFill
            ) || is_comparison(op),
            (Ty::F64, _) => matches!(
                op,
                BinaryOperator::Addition
                    | BinaryOperator::Subtraction
                    | BinaryOperator::Multiplication
                    | BinaryOperator::Division
            ) || is_comparison(op),
            (Ty::Bool, _) => matches!(
                op,
                BinaryOperator::Equality
                    | BinaryOperator::Inequality
                    | BinaryOperator::StrictEquality
                    | BinaryOperator::StrictInequality
            ),
            (Ty::Void, _) => false,
        };
        if !supported {
            self.at(
                span,
                format!("operator {op:?} not supported on {}", l.ty),
            );
        }
    }
}

/// Compound assignment → the equivalent binary operator.
fn compound_op(op: AssignmentOperator) -> Option<BinaryOperator> {
    use AssignmentOperator as A;
    use BinaryOperator as B;
    Some(match op {
        A::Addition => B::Addition,
        A::Subtraction => B::Subtraction,
        A::Multiplication => B::Multiplication,
        A::Division => B::Division,
        A::Remainder => B::Remainder,
        A::BitwiseAnd => B::BitwiseAnd,
        A::BitwiseOR => B::BitwiseOR,
        A::BitwiseXOR => B::BitwiseXOR,
        A::ShiftLeft => B::ShiftLeft,
        A::ShiftRight => B::ShiftRight,
        _ => return None,
    })
}

fn is_comparison(op: BinaryOperator) -> bool {
    use BinaryOperator as B;
    matches!(
        op,
        B::Equality
            | B::Inequality
            | B::StrictEquality
            | B::StrictInequality
            | B::LessThan
            | B::LessEqualThan
            | B::GreaterThan
            | B::GreaterEqualThan
    )
}

/// Resolve a TS type annotation to a [`Ty`].
pub fn ty_of(t: &TSType<'_>) -> Result<Ty, CompileError> {
    match t {
        TSType::TSNumberKeyword(_) => Ok(Ty::F64),
        TSType::TSBooleanKeyword(_) => Ok(Ty::Bool),
        TSType::TSVoidKeyword(_) => Ok(Ty::Void),
        TSType::TSTypeReference(r) => match &r.type_name {
            TSTypeName::IdentifierReference(id) => match id.name.as_str() {
                "i32" => Ok(Ty::I32),
                "f64" => Ok(Ty::F64),
                other => Err(crate::diag::plain(format!("unknown type `{other}`"))),
            },
            _ => Err(crate::diag::plain("qualified type names are not supported")),
        },
        other => Err(crate::diag::plain(format!(
            "unsupported type at {:?}",
            other.span()
        ))),
    }
}

/// A function signature collected in pass 1.
#[derive(Clone, Debug)]
struct FnSig {
    params: Vec<Ty>,
    ret: Ty,
}

/// Everything callable in the unit being lowered: local functions by index,
/// host functions by name.
#[derive(Clone, Debug, Default)]
struct ModuleScope {
    /// Source-ordered local signatures; the index is what HIR calls use.
    fns: Vec<(String, FnSig)>,
    /// Host imports (order irrelevant — HIR host calls carry the name).
    host: Vec<(String, HostSig)>,
}

impl ModuleScope {
    fn kind_of(&self, name: &str) -> Option<FnKind> {
        if self.fns.iter().any(|(n, _)| n == name) {
            Some(FnKind::Local)
        } else {
            self.host
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, sig)| FnKind::Host(sig.clone()))
        }
    }

    fn sig(&self, name: &str) -> Option<&FnSig> {
        self.fns
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s)
    }

    fn index(&self, name: &str) -> usize {
        self.fns.iter().position(|(n, _)| n == name).unwrap_or(0)
    }
}
