//! Lower oxc AST → typed HIR, collecting every diagnostic instead of stopping
//! at the first one.
//!
//! One pass resolves names (to local slots), checks types (against the
//! declared signatures and the host registry) and builds the HIR. On error the
//! pass records a diagnostic and substitutes a dummy expression, so a broken
//! function still lowers and its neighbours are still checked.

use oxc_allocator::Allocator;
use oxc_ast::ast::{
    AssignmentOperator, AssignmentTarget, BinaryOperator, BindingPattern, Declaration, Expression,
    ForStatementInit, ForStatementLeft, Function, LogicalOperator, ObjectPropertyKind, PropertyKey,
    PropertyKind, SimpleAssignmentTarget, Statement, TSTupleElement, TSType, TSTypeName,
    UnaryOperator, UpdateOperator, VariableDeclaration,
};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType, Span};
use std::collections::HashSet;
use std::sync::Arc;

use crate::diag::{CompileError, Diagnostics, Src, err};
use crate::hir::{
    CallTarget, ComponentDecl, Expr, ExprKind, FnDef, FrameLoad, FrameLoadKind, Intrinsic, Local,
    Module, PlaceExpr, QueryShape, QueryTerm, Stmt, SystemExport,
};
use crate::host::{FnKind, HostRegistry, HostSig};
use crate::ty::{QueryFilter, QueryFilterKind, StructLayout, Ty};
use miette::{SourceOffset, SourceSpan};
use std::collections::HashMap;

/// Type aliases and named structs discovered during lowering.
#[derive(Clone, Debug, Default)]
struct TypeScope {
    aliases: HashMap<String, Ty>,
}

impl TypeScope {
    fn insert(&mut self, name: String, ty: Ty) {
        self.aliases.insert(name, ty);
    }
    fn get(&self, name: &str) -> Option<&Ty> {
        self.aliases.get(name)
    }
}

/// Collected pass state for one lowering run.
pub struct Lowerer<'a> {
    src: &'a Src,
    host: &'a HostRegistry,
    diags: Diagnostics,
    types: TypeScope,
    components: Vec<ComponentDecl>,
    systems: Vec<SystemExport>,
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
            types: TypeScope::default(),
            components: Vec::new(),
            systems: Vec::new(),
        }
    }


    /// Parse `export default { update: sysFn }` into provisional SystemExport entries.
    fn parse_default_export(&mut self, obj: &oxc_ast::ast::ObjectExpression<'_>) {
        for prop in &obj.properties {
            let oxc_ast::ast::ObjectPropertyKind::ObjectProperty(p) = prop else {
                self.at(obj.span, "spread in export default is not supported");
                continue;
            };
            let stage = match &p.key {
                oxc_ast::ast::PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                oxc_ast::ast::PropertyKey::StringLiteral(s) => s.value.to_string(),
                _ => {
                    self.at(obj.span, "export default keys must be plain identifiers");
                    continue;
                }
            };
            let fn_name = match &p.value {
                Expression::Identifier(id) => id.name.to_string(),
                _ => {
                    self.at(obj.span, "export default values must be function names");
                    continue;
                }
            };
            // Encode fn name until pass-2 resolution (fn_index == MAX).
            self.systems.push(SystemExport {
                stage: format!("{stage}\0{fn_name}"),
                fn_index: usize::MAX,
                fn_name: fn_name.clone(),
            });
        }
    }

    fn resolve_ty(&self, t: &TSType<'_>) -> Result<Ty, CompileError> {
        ty_of_x(t, self.host, Some(&self.types))
    }

    /// Process a `type Foo = ...` alias at the module level (Pass 0).
    fn lower_type_alias(&mut self, alias: &oxc_ast::ast::TSTypeAliasDeclaration<'_>) {
        let name = alias.id.name.to_string();
        match ty_of_x(&alias.type_annotation, self.host, Some(&self.types)) {
            Ok(ty) => {
                // Register as component/resource when wrapped, else plain alias.
                if let Ty::Component(inner) = &ty {
                    if let Some(layout) = inner.struct_layout() {
                        let mut layout = (*layout).clone();
                        layout.name = name.clone();
                        let layout = Arc::new(layout);
                        self.components.push(ComponentDecl {
                            name: name.clone(),
                            layout: layout.clone(),
                            is_resource: false,
                            span: alias.span,
                        });
                        self.types.insert(name.clone(), Ty::Struct(layout));
                    } else {
                        // Scalar component: synthesize single-field layout.
                        match StructLayout::repr_c(&name, vec![("value".into(), (**inner).clone())]) {
                            Ok(layout) => {
                                let layout = Arc::new(layout);
                                self.components.push(ComponentDecl {
                                    name: name.clone(),
                                    layout: layout.clone(),
                                    is_resource: false,
                                    span: alias.span,
                                });
                                self.types.insert(name, Ty::Struct(layout));
                            }
                            Err(e) => self.at(alias.span, e),
                        }
                    }
                } else if let Ty::Resource(inner) = &ty {
                    if let Some(layout) = inner.struct_layout() {
                        let mut layout = (*layout).clone();
                        layout.name = name.clone();
                        let layout = Arc::new(layout);
                        self.components.push(ComponentDecl {
                            name: name.clone(),
                            layout: layout.clone(),
                            is_resource: true,
                            span: alias.span,
                        });
                        self.types.insert(name.clone(), Ty::Struct(layout));
                    } else {
                        match StructLayout::repr_c(&name, vec![("value".into(), (**inner).clone())]) {
                            Ok(layout) => {
                                let layout = Arc::new(layout);
                                self.components.push(ComponentDecl {
                                    name: name.clone(),
                                    layout: layout.clone(),
                                    is_resource: true,
                                    span: alias.span,
                                });
                                self.types.insert(name, Ty::Struct(layout));
                            }
                            Err(e) => self.at(alias.span, e),
                        }
                    }
                } else if let Ty::Struct(layout) = &ty {
                    let mut layout = (**layout).clone();
                    layout.name = name.clone();
                    let layout = Arc::new(layout);
                    self.types.insert(name, Ty::Struct(layout));
                } else {
                    self.types.insert(name, ty);
                }
            }
            Err(e) => self.at(alias.span, e.to_string()),
        }
    }

    /// Process an `interface Pos { x: f32; ... }` at the module level (Pass 0).
    fn lower_interface(&mut self, iface: &oxc_ast::ast::TSInterfaceDeclaration<'_>) {
        let name = iface.id.name.to_string();
        let mut fields = Vec::new();
        for m in &iface.body.body {
            match m {
                oxc_ast::ast::TSSignature::TSPropertySignature(p) => {
                    let fname = match &p.key {
                        oxc_ast::ast::PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                        _ => {
                            self.at(iface.span, "only plain identifier fields supported");
                            continue;
                        }
                    };
                    let Some(ann) = p.type_annotation.as_ref() else {
                        self.at(iface.span, format!("field `{fname}` needs a type"));
                        continue;
                    };
                    match ty_of_x(&ann.type_annotation, self.host, Some(&self.types)) {
                        Ok(fty) => fields.push((fname, fty)),
                        Err(e) => self.at(iface.span, e.to_string()),
                    }
                }
                _ => self.at(iface.span, "only property signatures supported in interfaces"),
            }
        }
        match StructLayout::repr_c(&name, fields) {
            Ok(layout) => {
                let layout = Arc::new(layout);
                self.types.insert(name, Ty::Struct(layout));
            }
            Err(e) => self.at(iface.span, e),
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
        // Pass 0: type aliases / interfaces so later signatures can resolve
        // mod-defined component names.
        for stmt in body {
            match stmt {
                Statement::TSTypeAliasDeclaration(alias) => {
                    self.lower_type_alias(alias);
                }
                Statement::TSInterfaceDeclaration(iface) => {
                    self.lower_interface(iface);
                }
                // export default { update: foo } — collect system schedule.
                Statement::ExportDefaultDeclaration(e) => {
                    if let oxc_ast::ast::ExportDefaultDeclarationKind::ObjectExpression(obj) = &e.declaration {
                        self.parse_default_export(obj);
                    }
                }
                _ => {}
            }
        }

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
                match p
                    .type_annotation
                    .as_ref()
                    .map(|a| self.resolve_ty(&a.type_annotation))
                {
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
                Some(a) => match self.resolve_ty(&a.type_annotation) {
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
            if params.iter().any(|ty| matches!(ty, Ty::Struct(_))) {
                self.at(
                    f.span,
                    "component types must be borrowed through Query, Res, or Mut",
                );
                bad = true;
            }
            if matches!(ret, Ty::Struct(_)) {
                self.at(f.span, "component structs cannot be returned by value");
                bad = true;
            }
            if f.declare || f.body.is_none() {
                // Builtins are runtime symbols, not host functions. Declaring
                // one is redundant rather than wrong — it happens in `.d.ts`
                // editor tooling, which must describe them to the editor — so
                // it is accepted and skipped instead of failing the build.
                if crate::hir::BUILTINS
                    .iter()
                    .any(|(builtin, _)| *builtin == name)
                {
                    // Skip: the call site is lowered by the builtin path with
                    // the engine's own signature, so whatever a `.d.ts` claims
                    // here is documentation and can never affect codegen.
                    let _ = (params, ret);
                    continue;
                }
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
                        self.at(f.span, format!("no host function `{name}` registered"));
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

        // Resolve system export function names → indices.
        let mut systems = std::mem::take(&mut self.systems);
        for sys in &mut systems {
            if sys.fn_index == usize::MAX {
                if let Some((stage, fname)) = sys.stage.split_once('\0') {
                    sys.fn_name = fname.to_string();
                    if let Some(idx) = functions.iter().position(|f| f.name == fname) {
                        sys.stage = stage.to_string();
                        sys.fn_index = idx;
                    } else {
                        sys.stage = stage.to_string();
                    }
                }
            }
        }

        Module {
            functions,
            components: std::mem::take(&mut self.components),
            systems,
        }
    }

    fn lower_function(
        &mut self,
        f: &Function<'_>,
        name: String,
        sig: FnSig,
        scope: &ModuleScope,
    ) -> Option<FnDef> {
        let mut locals = Vec::with_capacity(f.params.items.len() + 16);
        let mut frame_loads = Vec::new();

        for (i, p) in f.params.items.iter().enumerate() {
            let BindingPattern::BindingIdentifier(id) = &p.pattern else {
                self.at(p.span, "destructuring parameters are not supported");
                return None;
            };
            let pty = sig.params[i].clone();
            let mut local = Local::simple(id.name.to_string(), pty.clone(), id.span);

            // Signature-only ECS params arrive via the host frame, not as
            // ordinary register arguments. Mark them and record a FrameLoad.
            match &pty {
                Ty::Query(row, _) => {
                    local.indirect = true; // holds chunks_ptr
                    let shape = query_shape_from_row(row, &self.types);
                    frame_loads.push(FrameLoad {
                        local: i,
                        kind: FrameLoadKind::Query { shape },
                    });
                    // Synthetic chunk_count local (name is stable for QueryFor).
                    // Allocated after all params so indices stay predictable.
                }
                Ty::Res(inner) => {
                    local.indirect = true;
                    let type_name = match inner.as_ref() {
                        Ty::Struct(l) => l.name.clone(),
                        other => other.name().into_owned(),
                    };
                    frame_loads.push(FrameLoad {
                        local: i,
                        kind: FrameLoadKind::Resource {
                            type_name,
                            mutable: false,
                        },
                    });
                }
                Ty::Mut(inner) => {
                    local.indirect = true;
                    local.mutable = true;
                    let type_name = match inner.as_ref() {
                        Ty::Struct(l) => l.name.clone(),
                        other => other.name().into_owned(),
                    };
                    frame_loads.push(FrameLoad {
                        local: i,
                        kind: FrameLoadKind::Resource {
                            type_name,
                            mutable: true,
                        },
                    });
                }
                Ty::Commands => {
                    frame_loads.push(FrameLoad {
                        local: i,
                        kind: FrameLoadKind::Commands,
                    });
                }
                Ty::Component(_) | Ty::Resource(_) => {
                    // These are type-level markers, not values — reject as params.
                    self.at(p.span, format!("`{pty}` is not a valid parameter type; use Query/Res/Mut"));
                    return None;
                }
                _ => {}
            }
            locals.push(local);
        }

        let body = f.body.as_ref().expect("checked in pass 1");
        let mut ctx = FnCtx {
            lowerer: self,
            locals,
            scopes: vec![0],
            sigs: scope,
            ret: sig.ret.clone(),
            loop_depth: 0,
        };
        let body = ctx.stmts(&body.statements);
        Some(FnDef {
            name,
            params: sig.params,
            ret: sig.ret,
            locals: std::mem::take(&mut ctx.locals),
            body,
            span: f.span,
            frame_loads,
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
        self.locals.push(Local::simple(name.to_owned(), ty, span));
        Some(self.locals.len() - 1)
    }

    fn poison(&mut self, span: Span, msg: impl Into<String>) -> Expr {
        self.at(span, msg);
        Expr::new(ExprKind::Int(0), Ty::I32, span)
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
                // `a.push(v)` may move the array (growth), so the statement
                // form rebinds the local to the new pointer.
                if let Expression::CallExpression(c) = &e.expression
                    && let Expression::StaticMemberExpression(m) = &c.callee
                    && m.property.name.as_str() == "push"
                    && let Expression::Identifier(id) = &m.object
                    && let Some(slot) = self.lookup(id.name.as_str())
                    && self.locals[slot].ty.elem().is_some()
                {
                    let expr = self.expr(&e.expression, None);
                    return Some(Stmt::Expr {
                        expr: Expr::new(
                            ExprKind::Assign {
                                local: slot,
                                op: None,
                                value: Box::new(expr),
                            },
                            self.locals[slot].ty.clone(),
                            e.span,
                        ),
                        span: e.span,
                    });
                }
                let expr = self.expr(&e.expression, None);
                Some(Stmt::Expr { expr, span: e.span })
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
                        let v = self.expr(e, Some(self.ret.clone()));
                        self.reject_struct_value(&v);
                        if v.ty != self.ret {
                            self.at(r.span, format!("returns {}, expected {}", v.ty, self.ret));
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
                Some(Stmt::Return {
                    value,
                    span: r.span,
                })
            }
            Statement::IfStatement(i) => {
                let cond = self.cond(&i.test);
                Some(Stmt::If {
                    cond,
                    then: self.stmt_block(&i.consequent),
                    els: i.alternate.as_ref().map(|a| self.stmt_block(a)),
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
            Statement::ForOfStatement(f) => {
                if f.r#await {
                    self.at(f.span, "`await for` is not supported");
                    return None;
                }
                let binding_decl = match &f.left {
                    ForStatementLeft::VariableDeclaration(d) if d.declarations.len() == 1 => {
                        &d.declarations[0]
                    }
                    _ => {
                        self.at(f.span, "for-of requires a single binding");
                        return None;
                    }
                };
                let iterable = self.expr(&f.right, None);
                if let Some(row_ty) = iterable.ty.query_row() {
                    let query_slot = match iterable.kind {
                        ExprKind::Local(slot) => slot,
                        _ => {
                            self.at(f.right.span(), "query iteration requires a query parameter");
                            return None;
                        }
                    };
                    let terms = match row_ty {
                        Ty::Tuple(terms) => terms,
                        term => vec![term],
                    };
                    let names = match &binding_decl.id {
                        BindingPattern::BindingIdentifier(id) if terms.len() == 1 => {
                            vec![(id.name.to_string(), id.span)]
                        }
                        BindingPattern::ArrayPattern(array) => {
                            if array.rest.is_some() || array.elements.len() != terms.len() {
                                self.at(array.span, format!("query row has {} terms", terms.len()));
                                return None;
                            }
                            let mut names = Vec::with_capacity(array.elements.len());
                            for element in &array.elements {
                                let Some(BindingPattern::BindingIdentifier(id)) = element else {
                                    self.at(
                                        array.span,
                                        "query destructuring requires simple names",
                                    );
                                    return None;
                                };
                                names.push((id.name.to_string(), id.span));
                            }
                            names
                        }
                        _ => {
                            self.at(
                                f.span,
                                "query iteration requires a single name or tuple binding",
                            );
                            return None;
                        }
                    };
                    self.push_scope();
                    let mut bindings = Vec::with_capacity(names.len());
                    for ((name, span), term) in names.iter().zip(&terms) {
                        let Some(slot) = self.declare(name, term.clone(), *span) else {
                            self.pop_scope();
                            return None;
                        };
                        bindings.push(slot);
                    }
                    self.loop_depth += 1;
                    let body = self.stmt_block(&f.body);
                    self.loop_depth -= 1;
                    self.pop_scope();
                    return Some(Stmt::QueryFor {
                        query: query_slot,
                        bindings,
                        terms,
                        body,
                        span: f.span,
                    });
                }

                let bpat = match &binding_decl.id {
                    BindingPattern::BindingIdentifier(id) => id,
                    _ => {
                        self.at(f.span, "destructuring is only supported for query rows");
                        return None;
                    }
                };

                // Compute the array iterable's elemental type.
                let Some(elem_ty) = iterable.ty.elem() else {
                    self.at(
                        f.right.span(),
                        format!("cannot iterate over {}", iterable.ty),
                    );
                    return None;
                };
                let binding_slot = match self.declare(&bpat.name, elem_ty.clone(), bpat.span) {
                    Some(s) => s,
                    None => return None,
                };

                self.loop_depth += 1;
                let index_slot = self
                    .declare("__i", Ty::I32, bpat.span)
                    .unwrap_or(binding_slot);

                let init_expr = Expr::new(ExprKind::Int(0), Ty::I32, f.span);
                let init_stmt = Stmt::Let {
                    local: index_slot,
                    init: Some(init_expr),
                    span: f.span,
                };

                let len_expr = Expr::new(
                    ExprKind::Length {
                        target: Box::new(iterable.clone()),
                    },
                    Ty::I32,
                    f.span,
                );
                let cond_lhs = Expr::new(ExprKind::Local(index_slot), Ty::I32, f.span);
                let cond = Expr::new(
                    ExprKind::Binary {
                        op: BinaryOperator::LessThan,
                        lhs: Box::new(cond_lhs),
                        rhs: Box::new(len_expr),
                    },
                    Ty::Bool,
                    f.span,
                );

                let index_val = Expr::new(ExprKind::Local(index_slot), Ty::I32, f.span);
                let slice_expr = Expr::new(
                    ExprKind::Index {
                        target: Box::new(iterable.clone()),
                        index: Box::new(index_val.clone()),
                    },
                    elem_ty.clone(),
                    f.span,
                );
                let bind_stmt = Stmt::Expr {
                    expr: Expr::new(
                        ExprKind::Assign {
                            local: binding_slot,
                            op: None,
                            value: Box::new(slice_expr),
                        },
                        Ty::Void,
                        f.span,
                    ),
                    span: f.span,
                };

                let body_stmts = self.stmt_block(&f.body);
                let mut full_body = Vec::with_capacity(body_stmts.len() + 2);
                full_body.push(bind_stmt);
                full_body.extend(body_stmts);

                let index_val2 = Expr::new(ExprKind::Local(index_slot), Ty::I32, f.span);
                let binding_val = Expr::new(ExprKind::Local(binding_slot), elem_ty.clone(), f.span);
                let write_back = Stmt::Expr {
                    expr: Expr::new(
                        ExprKind::SetIndex {
                            target: Box::new(iterable.clone()),
                            index: Box::new(index_val2),
                            value: Box::new(binding_val),
                        },
                        Ty::Void,
                        f.span,
                    ),
                    span: f.span,
                };
                full_body.push(write_back);

                let update = Expr::new(
                    ExprKind::Update {
                        local: index_slot,
                        inc: true,
                        prefix: false,
                    },
                    Ty::I32,
                    f.span,
                );

                let loop_stmt = Stmt::For {
                    init: vec![init_stmt],
                    cond: Some(cond),
                    update: Some(update),
                    body: full_body,
                    span: f.span,
                };
                self.pop_scope();
                self.loop_depth -= 1;
                Some(loop_stmt)
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
            // Type aliases and interfaces are processed at module scope (Pass 0);
            // nested ones are silently ignored.
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
                Some(a) => match self.lowerer.resolve_ty(&a.type_annotation) {
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
                    let v = self.expr(init, declared.clone());
                    if !matches!(v.ty, Ty::OwnedStruct(_)) && self.reject_struct_value(&v) {
                        continue;
                    }
                    let ty = v.ty.clone();
                    if let Some(t) = declared
                        && !compatible_value(&ty, &t)
                    {
                        self.at(decl.span, format!("cannot assign {ty} to {t} (use `as`)"));
                    }
                    (Some(v), ty)
                }
                None => {
                    let Some(t) = declared else {
                        self.at(decl.span, "variable needs a type annotation or initializer");
                        continue;
                    };
                    if matches!(t, Ty::Struct(_)) {
                        self.at(
                            decl.span,
                            "component values must be initialized with an object literal",
                        );
                        continue;
                    }
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

    /// `s.length` / `a.length` member load.
    fn member_length(&mut self, target: &Expr, span: Span) -> Expr {
        match target.ty {
            Ty::Str | Ty::Array(_) => Expr::new(
                ExprKind::Length {
                    target: Box::new(target.clone()),
                },
                Ty::I32,
                span,
            ),
            _ => self.poison(span, format!("{} has no `length`", target.ty)),
        }
    }

    fn field_place(&mut self, target: &Expr, name: &str) -> Result<(PlaceExpr, Ty, bool), String> {
        let layout = target
            .ty
            .struct_layout()
            .ok_or_else(|| format!("{} has no fields", target.ty))?;
        let field = layout
            .field(name)
            .ok_or_else(|| format!("{} has no field `{name}`", layout.name))?;
        let (base, base_offset) = match &target.kind {
            ExprKind::Local(base) => (*base, 0),
            ExprKind::PlaceRead {
                place: PlaceExpr::Field { base, offset },
            } => (*base, *offset),
            _ => return Err("component fields must be accessed through a component local".into()),
        };
        let offset = base_offset
            .checked_add(field.offset)
            .ok_or_else(|| format!("field offset overflow for {}.{name}", layout.name))?;
        if offset > i64::MAX as u64 {
            return Err(format!("field offset overflow for {}.{name}", layout.name));
        }
        let writable = matches!(&self.locals[base].ty, Ty::Mut(_) | Ty::OwnedStruct(_));
        Ok((
            PlaceExpr::Field { base, offset },
            field.ty.clone(),
            writable,
        ))
    }

    fn reject_struct_value(&mut self, expr: &Expr) -> bool {
        if matches!(&expr.ty, Ty::Struct(_) | Ty::OwnedStruct(_)) {
            self.at(
                expr.span,
                "copying a struct is not supported; assign fields individually",
            );
            true
        } else {
            false
        }
    }

    /// Lower a call to a builtin runtime helper.
    ///
    /// Argument/result types come from the intrinsic's runtime signature, so
    /// the same adaptation machinery (`i32` ↔ machine word) applies as for any
    /// other intrinsic call.
    fn builtin_call(&mut self, intrinsic: Intrinsic, c: &oxc_ast::ast::CallExpression<'_>) -> Expr {
        use Intrinsic::{
            LoadF32, LoadF64, LoadI32, LoadI64, RegionId, StoreF32, StoreF64, StoreI32, StoreI64,
        };
        // TS-facing signature; the codegen widens to the runtime's machine
        // words exactly like any other intrinsic call.
        let (params, ret) = match intrinsic {
            LoadF64 | LoadF32 => (vec![Ty::I32, Ty::I32], Ty::F64),
            LoadI64 => (vec![Ty::I32, Ty::I32], Ty::I32),
            LoadI32 => (vec![Ty::I32, Ty::I32], Ty::I32),
            StoreF64 | StoreF32 => (vec![Ty::I32, Ty::I32, Ty::F64], Ty::F64),
            StoreI64 => (vec![Ty::I32, Ty::I32, Ty::I32], Ty::I32),
            StoreI32 => (vec![Ty::I32, Ty::I32, Ty::I32], Ty::I32),
            RegionId => (vec![Ty::Str], Ty::I32),
            _ => unreachable!("builtin_call only handles region intrinsics"),
        };
        if c.arguments.len() != params.len() {
            return self.poison(
                c.span,
                format!(
                    "{} expects {} arguments",
                    crate::hir::Intrinsic::symbol(intrinsic),
                    params.len()
                ),
            );
        }
        let mut args = Vec::with_capacity(params.len());
        for (a, pty) in c.arguments.iter().zip(&params) {
            let Some(ex) = a.as_expression() else {
                return self.poison(c.span, "spread arguments are not supported");
            };
            let v = self.expr(ex, Some(pty.clone()));
            if self.reject_struct_value(&v) {
                continue;
            }
            if v.ty != *pty {
                self.at(ex.span(), format!("argument is {}, expected {}", v.ty, pty));
            }
            args.push(v);
        }
        Expr::new(
            ExprKind::Call {
                target: CallTarget::Intrinsic(intrinsic),
                args,
            },
            ret,
            c.span,
        )
    }

    /// Boolean condition of `if` / `while` / `for`.
    fn cond(&mut self, e: &Expression<'_>) -> Expr {
        let v = self.expr(e, Some(Ty::Bool));
        if v.ty != Ty::Bool {
            self.at(
                e.span(),
                format!("condition must be boolean, found {}", v.ty),
            );
        }
        v
    }

    // ───────────────────────── expressions ─────────────────────────

    /// Range-check a numeric literal against an integer type and lower it.
    /// `v` is the (possibly already negated) mathematical value.
    fn int_lit(&mut self, v: f64, expected: Ty, span: Span) -> Expr {
        // Range-checked in `f64` arithmetic: `2^w` is exact in `f64` for
        // `w <= 64`, so `v < 2^w` accepts exactly the representable in-range
        // integers (an `as i128` comparison would spuriously reject
        // `i64::MAX`, which rounds to 2^63 as an `f64`).
        let width = expected.int_width().unwrap_or(32);
        let limit = (1u128 << width) as f64;
        let fits = if expected.is_signed() {
            v >= -limit / 2.0 && v < limit / 2.0 && v.fract() == 0.0
        } else {
            v >= 0.0 && v < limit && v.fract() == 0.0
        };
        if !fits {
            return self.poison(span, format!("literal does not fit in {expected}"));
        }
        // Exact: `v` is integral and inside the type's range. Unsigned
        // values reuse the `i64` bit pattern.
        let bits = if expected.is_signed() {
            v as i64
        } else {
            (v as u64) as i64
        };
        Expr::new(ExprKind::Int(bits), expected, span)
    }

    /// `expect` is the contextual type (used to type numeric literals and to
    /// propagate an operand type into the other side of a binary op).
    fn expr(&mut self, e: &Expression<'_>, expect: Option<Ty>) -> Expr {
        match e {
            Expression::NumericLiteral(n) => {
                let span = n.span;
                // An explicitly expected integer width narrows the literal
                // (range-checked below); otherwise it is an `f64`. `f32` never
                // appears by inference — only through annotation + `as`.
                if let Some(expected) = expect.clone()
                    && expected.is_integer()
                {
                    return self.int_lit(n.value, expected, span);
                }
                if expect == Some(Ty::F32) {
                    let v = n.value;
                    if v < -(f32::MAX as f64) || v > f32::MAX as f64 {
                        return self.poison(span, "literal does not fit in f32");
                    }
                    return Expr::new(ExprKind::Float(v), Ty::F32, span);
                }
                Expr::new(ExprKind::Float(n.value), Ty::F64, span)
            }
            Expression::BooleanLiteral(l) => Expr::new(ExprKind::Bool(l.value), Ty::Bool, l.span),
            Expression::StringLiteral(l) => {
                Expr::new(ExprKind::Str(l.value.to_string()), Ty::Str, l.span)
            }
            Expression::ObjectExpression(object) => {
                let Some(layout) = expect.as_ref().and_then(Ty::struct_layout) else {
                    return self.poison(
                        object.span,
                        "component object literals need a registered component type annotation",
                    );
                };
                if !layout.spawnable {
                    self.at(
                        object.span,
                        format!(
                            "component `{}` has fields that cannot be initialized by a mod",
                            layout.name
                        ),
                    );
                }
                let mut seen = HashSet::new();
                let mut fields = Vec::with_capacity(object.properties.len());
                for property in &object.properties {
                    let ObjectPropertyKind::ObjectProperty(property) = property else {
                        self.at(property.span(), "component object spread is not supported");
                        continue;
                    };
                    if property.method || property.computed || property.kind != PropertyKind::Init {
                        self.at(property.span, "component fields must be plain properties");
                        continue;
                    }
                    let PropertyKey::StaticIdentifier(key) = &property.key else {
                        self.at(
                            property.key.span(),
                            "component field names must be identifiers",
                        );
                        continue;
                    };
                    let field_name = key.name.as_str();
                    let Some(field) = layout.field(field_name) else {
                        self.at(
                            property.key.span(),
                            format!("{} has no field `{field_name}`", layout.name),
                        );
                        continue;
                    };
                    if !seen.insert(field_name.to_owned()) {
                        self.at(
                            property.key.span(),
                            format!("duplicate field `{field_name}`"),
                        );
                        continue;
                    }
                    let value = self.expr(&property.value, Some(field.ty.clone()));
                    if !compatible_value(&value.ty, &field.ty) {
                        self.at(
                            property.value.span(),
                            format!(
                                "field {}.{} expects {}, found {}",
                                layout.name, field_name, field.ty, value.ty
                            ),
                        );
                    }
                    fields.push((field.offset, value));
                }
                for field in &layout.fields {
                    if !seen.contains(&field.name) {
                        self.at(
                            object.span,
                            format!("component literal is missing field `{}`", field.name),
                        );
                    }
                }
                Expr::new(
                    ExprKind::StructLit {
                        layout: Arc::clone(&layout),
                        fields,
                    },
                    Ty::OwnedStruct(layout),
                    object.span,
                )
            }
            Expression::ArrayExpression(arr) => {
                let span = arr.span;
                // `[1, 2]` typed against `i32[]` lowers its literals as i32.
                let declared_elem = match expect {
                    Some(Ty::Array(elem)) => Some(elem.as_ref().clone()),
                    _ => None,
                };
                let mut elems = Vec::with_capacity(arr.elements.len());
                let mut elem_ty: Option<Ty> = declared_elem.clone();
                for el in &arr.elements {
                    let Some(ex) = el.as_expression() else {
                        self.at(span, "spread elements are not supported");
                        continue;
                    };
                    let v = self.expr(ex, elem_ty.clone().or(declared_elem.clone()));
                    if self.reject_struct_value(&v) {
                        continue;
                    }
                    match &elem_ty {
                        Some(known) => {
                            if v.ty != *known {
                                self.at(ex.span(), format!("array elements must all be {known}"));
                            }
                        }
                        None => {
                            if v.ty.elem_size().is_none() {
                                self.at(ex.span(), format!("{} cannot be an array element", v.ty));
                                continue;
                            }
                            elem_ty = Some(v.ty.clone());
                        }
                    }
                    elems.push(v);
                }
                let Some(elem_ty) = elem_ty else {
                    return self.poison(span, "array literal needs at least one element");
                };
                let array_ty = Ty::Array(Box::new(elem_ty.clone()));
                Expr::new(ExprKind::ArrayLit { elems, elem_ty }, array_ty, span)
            }
            Expression::StaticMemberExpression(m) => {
                let target = self.expr(&m.object, None);
                if target.ty.struct_layout().is_some() {
                    match self.field_place(&target, m.property.name.as_str()) {
                        Ok((place, ty, _)) => Expr::new(ExprKind::PlaceRead { place }, ty, m.span),
                        Err(error) => self.poison(m.span, error),
                    }
                } else {
                    self.member_length(&target, m.span)
                }
            }
            Expression::ComputedMemberExpression(m) => {
                let target = self.expr(&m.object, None);
                let elem = target.ty.elem();
                if target.ty != Ty::Str && elem.is_none() {
                    return self.poison(
                        m.span,
                        format!("cannot index a value of type {}", target.ty),
                    );
                }
                let index = self.expr(&m.expression, Some(Ty::I32));
                if !index.ty.is_integer() {
                    self.at(m.span, "array index must be an integer");
                }
                // String indexing yields a byte (i32) — there is no char type yet.
                Expr::new(
                    ExprKind::Index {
                        target: Box::new(target),
                        index: Box::new(index),
                    },
                    elem.unwrap_or(Ty::I32),
                    m.span,
                )
            }
            Expression::ParenthesizedExpression(p) => self.expr(&p.expression, expect),
            Expression::Identifier(id) => {
                let Some(slot) = self.lookup(id.name.as_str()) else {
                    return self.poison(id.span, format!("unknown variable `{}`", id.name));
                };
                let ty = self.locals[slot].ty.clone();
                Expr::new(ExprKind::Local(slot), ty, id.span)
            }
            Expression::UnaryExpression(u) => {
                // `-(literal)` is folded before range-checking so the minimum
                // value of each signed type (`-128`, ...) is expressible: the
                // positive half alone would not fit.
                if u.operator == UnaryOperator::UnaryNegation
                    && let Expression::NumericLiteral(n) = &u.argument
                    && let Some(expected) = expect.clone()
                    && expected.is_integer()
                    && expected.is_signed()
                {
                    let negated = self.int_lit(-n.value, expected, u.span);
                    // A negation of a literal is a constant, not an `ineg` —
                    // but it still carries the unary span for diagnostics.
                    return negated;
                }
                let a = self.expr(&u.argument, expect);
                let ty = a.ty.clone();
                let span = u.span;
                match (u.operator, ty.clone()) {
                    (UnaryOperator::UnaryPlus, t) if t.is_numeric() => {}
                    (UnaryOperator::UnaryNegation, t) if t.is_numeric() => {}
                    (UnaryOperator::BitwiseNot, t) if t.is_integer() => {}
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
                    let l = self.expr(&bin.left, Some(r.ty.clone()));
                    (l, r)
                } else {
                    let l = self.expr(&bin.left, operand_expect);
                    let r = self.expr(&bin.right, Some(l.ty.clone()));
                    (l, r)
                };
                let ty = if is_cmp { Ty::Bool } else { l.ty.clone() };
                self.check_arith(bin.operator, &l, &r, bin.span);
                // Strings only concatenate and compare.
                if l.ty == Ty::Str && !is_cmp && bin.operator != BinaryOperator::Addition {
                    self.at(bin.span, "only `+` and comparisons work on strings");
                }
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
                let els = self.expr(&c.alternate, Some(then.ty.clone()));
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
                if let AssignmentTarget::StaticMemberExpression(member) = &a.left {
                    let target = self.expr(&member.object, None);
                    let (place, field_ty, writable) =
                        match self.field_place(&target, member.property.name.as_str()) {
                            Ok(place) => place,
                            Err(error) => return self.poison(a.span, error),
                        };
                    if !writable {
                        self.at(member.span, "cannot write through a read-only component");
                    }
                    if matches!(field_ty, Ty::Struct(_)) {
                        return self.poison(
                            a.span,
                            "copying a struct is not supported; assign fields individually",
                        );
                    }
                    let rhs = self.expr(&a.right, Some(field_ty.clone()));
                    if rhs.ty != field_ty {
                        self.at(a.span, format!("cannot assign {} to {}", rhs.ty, field_ty));
                    }
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
                        let current = Expr::new(
                            ExprKind::PlaceRead {
                                place: place.clone(),
                            },
                            field_ty.clone(),
                            a.span,
                        );
                        self.check_arith(op, &current, &rhs, a.span);
                    }
                    return Expr::new(
                        ExprKind::PlaceWrite {
                            place,
                            op,
                            value: Box::new(rhs),
                        },
                        field_ty,
                        a.span,
                    );
                }
                // `a[i] = v` / `a[i] += v`
                if let AssignmentTarget::ComputedMemberExpression(m) = &a.left {
                    let target = self.expr(&m.object, None);
                    let Some(elem) = target.ty.elem() else {
                        return self.poison(
                            a.span,
                            format!("cannot assign into a value of type {}", target.ty),
                        );
                    };
                    let index = self.expr(&m.expression, Some(Ty::I32));
                    if !index.ty.is_integer() {
                        self.at(a.span, "array index must be an integer");
                    }
                    let rhs = self.expr(&a.right, Some(elem.clone()));
                    if rhs.ty != elem.clone() {
                        self.at(a.span, format!("cannot assign {} to {}[]", rhs.ty, elem));
                    }
                    let op = if a.operator == AssignmentOperator::Assign {
                        None
                    } else {
                        compound_op(a.operator)
                    };
                    // `a[i] += v` must read the element first: the stored value
                    // is the combined expression, not just `v`.
                    let value = if let Some(op) = op {
                        let cur = Expr::new(
                            ExprKind::Index {
                                target: Box::new(target.clone()),
                                index: Box::new(index.clone()),
                            },
                            elem.clone(),
                            a.span,
                        );
                        self.check_arith(op, &cur, &rhs, a.span);
                        Expr::new(
                            ExprKind::Binary {
                                op,
                                lhs: Box::new(cur),
                                rhs: Box::new(rhs),
                            },
                            elem.clone(),
                            a.span,
                        )
                    } else {
                        rhs
                    };
                    return Expr::new(
                        ExprKind::SetIndex {
                            target: Box::new(target),
                            index: Box::new(index),
                            value: Box::new(value),
                        },
                        elem.clone(),
                        a.span,
                    );
                }
                let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left else {
                    return self.poison(
                        a.span,
                        "only assignment to plain variables, fields, or array elements is supported",
                    );
                };
                let Some(slot) = self.lookup(id.name.as_str()) else {
                    return self.poison(a.span, format!("unknown variable `{}`", id.name));
                };
                let ty = self.locals[slot].ty.clone();
                // Reject whole-struct copies (phase-2 rule).
                if matches!(ty, Ty::Struct(_)) {
                    return self.poison(a.span, "copying a struct is not supported; assign fields individually");
                }
                let rhs = self.expr(&a.right, Some(ty.clone()));
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
                        &Expr::new(ExprKind::Local(slot), ty.clone(), a.span),
                        &rhs,
                        a.span,
                    );
                } else if rhs.ty != ty {
                    self.at(a.span, format!("cannot assign {} to {}", rhs.ty, ty));
                }
                Expr::new(
                    ExprKind::Assign {
                        local: slot,
                        op,
                        value: Box::new(rhs),
                    },
                    ty.clone(),
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
                let ty = self.locals[slot].ty.clone();
                if !ty.clone().is_numeric() {
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
                if let Expression::StaticMemberExpression(member) = &c.callee
                    && member.property.name.as_str() == "spawn"
                {
                    let commands = self.expr(&member.object, None);
                    if commands.ty == Ty::Commands {
                        if c.arguments.is_empty()
                            || c.arguments.len() > crate::abi::MAX_SPAWN_COMPONENTS
                        {
                            return self.poison(
                                c.span,
                                format!(
                                    "commands.spawn takes 1..={} component values",
                                    crate::abi::MAX_SPAWN_COMPONENTS
                                ),
                            );
                        }
                        let mut components = Vec::with_capacity(c.arguments.len());
                        let mut ids = HashSet::new();
                        for argument in &c.arguments {
                            let Some(argument) = argument.as_expression() else {
                                return self
                                    .poison(c.span, "spread spawn arguments are unsupported");
                            };
                            let value = self.expr(argument, None);
                            let layout = match &value.ty {
                                Ty::Struct(_) | Ty::OwnedStruct(_) => value.ty.struct_layout(),
                                Ty::Mut(inner) if matches!(inner.as_ref(), Ty::Struct(_)) => {
                                    value.ty.struct_layout()
                                }
                                _ => None,
                            };
                            let Some(layout) = layout else {
                                return self.poison(
                                    argument.span(),
                                    format!(
                                        "commands.spawn expects component values, found {}",
                                        value.ty
                                    ),
                                );
                            };
                            if !layout.spawnable {
                                return self.poison(
                                    argument.span(),
                                    format!(
                                        "component `{}` cannot be copied into a spawn command",
                                        layout.name
                                    ),
                                );
                            }
                            let Some(id) = layout.component_id else {
                                return self.poison(
                                    argument.span(),
                                    format!("`{}` is not a spawnable Bevy component", layout.name),
                                );
                            };
                            if !ids.insert(id) {
                                return self.poison(
                                    argument.span(),
                                    format!(
                                        "commands.spawn contains `{}` more than once",
                                        layout.name
                                    ),
                                );
                            }
                            components.push(value);
                        }
                        return Expr::new(
                            ExprKind::CommandsSpawn {
                                commands: Box::new(commands),
                                components,
                            },
                            Ty::Void,
                            c.span,
                        );
                    }
                }
                // Option<T> constructors.
                if let Expression::Identifier(id) = &c.callee {
                    match id.name.as_str() {
                        "some" => {
                            if c.arguments.len() != 1 {
                                return self.poison(c.span, "some() takes exactly one argument");
                            }
                            let Some(arg) = c.arguments.first().and_then(|a| a.as_expression())
                            else {
                                return self.poison(c.span, "some() takes exactly one argument");
                            };
                            let v = self.expr(arg, None);
                            return Expr::new(
                                ExprKind::OptionSome {
                                    value: Box::new(v.clone()),
                                },
                                Ty::Option(Box::new(v.ty)),
                                c.span,
                            );
                        }
                        "none" => {
                            let Some(inner) = expect.clone().and_then(|t| t.option_elem()) else {
                                return self
                                    .poison(c.span, "none() requires an `Option<T>` context");
                            };
                            return Expr::new(
                                ExprKind::OptionNone { ty: inner },
                                expect.clone().unwrap(),
                                c.span,
                            );
                        }
                        _ => {}
                    }
                }
                // Option<T> methods.
                if let Expression::StaticMemberExpression(m) = &c.callee {
                    match m.property.name.as_str() {
                        "isSome" => {
                            let recv = self.expr(&m.object, None);
                            let _ = recv.ty.option_elem().unwrap();
                            if c.arguments.is_empty() {
                                return Expr::new(
                                    ExprKind::Call {
                                        target: CallTarget::Intrinsic(Intrinsic::OptIsSome),
                                        args: vec![recv],
                                    },
                                    Ty::Bool,
                                    c.span,
                                );
                            }
                        }
                        "unwrap" => {
                            let recv = self.expr(&m.object, None);
                            let inner = match recv.ty.option_elem() {
                                Some(t) => t,
                                None => {
                                    return self.poison(c.span, "unwrap() requires Option<T>");
                                }
                            };
                            if c.arguments.is_empty() {
                                return Expr::new(
                                    ExprKind::Call {
                                        target: CallTarget::Intrinsic(if inner.is_float() {
                                            Intrinsic::OptUnwrapF64
                                        } else {
                                            Intrinsic::OptUnwrap
                                        }),
                                        args: vec![recv],
                                    },
                                    inner,
                                    c.span,
                                );
                            }
                        }
                        "unwrapOr" => {
                            let recv = self.expr(&m.object, None);
                            let inner = match recv.ty.option_elem() {
                                Some(t) => t,
                                None => {
                                    return self.poison(c.span, "unwrapOr() requires Option<T>");
                                }
                            };
                            let default = match c.arguments.first().and_then(|a| a.as_expression())
                            {
                                Some(ex) => self.expr(ex, Some(inner.clone())),
                                None => {
                                    return self.poison(c.span, "unwrapOr() requires a default");
                                }
                            };
                            return Expr::new(
                                ExprKind::Call {
                                    target: CallTarget::Intrinsic(if inner.is_float() {
                                        Intrinsic::OptUnwrapOrF64
                                    } else {
                                        Intrinsic::OptUnwrapOr
                                    }),
                                    args: vec![recv, default],
                                },
                                inner,
                                c.span,
                            );
                        }
                        _ => {}
                    }
                }
                // `a.push(v)` — the only builtin method call for now.
                if let Expression::StaticMemberExpression(m) = &c.callee
                    && m.property.name.as_str() == "push"
                {
                    let target = self.expr(&m.object, None);
                    let Some(elem) = target.ty.elem() else {
                        return self.poison(c.span, "`push` is only available on arrays");
                    };
                    if c.arguments.len() != 1 {
                        return self.poison(c.span, "`push` takes exactly one argument");
                    }
                    let Some(arg) = c.arguments.first().and_then(|a| a.as_expression()) else {
                        return self.poison(c.span, "`push` takes exactly one argument");
                    };
                    let value = self.expr(arg, Some(elem.clone()));
                    if value.ty != elem.clone() {
                        self.at(
                            arg.span(),
                            format!("cannot push {} into {}[]", value.ty, elem),
                        );
                    }
                    let elem_size = Expr::new(
                        ExprKind::Int(elem.elem_size().unwrap() as i64),
                        Ty::I32,
                        c.span,
                    );
                    return Expr::new(
                        ExprKind::Call {
                            target: CallTarget::Intrinsic(Intrinsic::ArrPush),
                            args: vec![target, elem_size, value],
                        },
                        Ty::Array(Box::new(elem)),
                        c.span,
                    );
                }
                // Builtins (`load_f64`, `region_id`, ...) are runtime symbols,
                // not host functions: no `declare` needed, and the names are
                // reserved (a mod cannot define its own).
                if let Expression::Identifier(callee) = &c.callee
                    && let Some((_, intrinsic)) = crate::hir::BUILTINS
                        .iter()
                        .find(|(name, _)| *name == callee.name.as_str())
                {
                    return self.builtin_call(*intrinsic, c);
                }
                // `a.foo(b, ...)` → UFCS resolution to `foo(a, b, ...)` for
                // any function available in this unit (local or host). The
                // receiver is prepended as the first argument. This is the
                // foundation of `world.query()`, `q.iter()`, `commands.spawn()`
                // style calls. (See §3.5 of SCRIPTING_ECS_MODS_PLAN.md.)
                if let Expression::StaticMemberExpression(m) = &c.callee {
                    let method = m.property.name.as_str();
                    // `array.length` is a property, not a call — handled
                    // below (`ExprKind::Length`). Reject it explicitly here.
                    if method == "length" {
                        return self
                            .poison(c.span, "`length` is a property; use `.length` directly");
                    }
                    let kind = self.sigs.kind_of(method).map(|k| k.clone());
                    if let Some(kind) = kind {
                        // Build args: object + forwarded args.
                        let (params, ret) = match &kind {
                            FnKind::Local => {
                                let sig = self.sigs.sig(method).expect("local fn has a sig");
                                (sig.params.clone(), sig.ret.clone())
                            }
                            FnKind::Host(sig) => (sig.params.clone(), sig.ret.clone()),
                        };
                        if params.is_empty() || c.arguments.len() + 1 != params.len() {
                            // UFCS requires at least one parameter matching
                            // the receiver. Dispatch falls through; if it is
                            // still unresolved the user gets the standard
                            // "not supported" diagnostic below.
                            // Fall through to the generic rejection.
                            return self.poison(
                                c.span,
                                "only calls to named functions or `array.push` are supported",
                            );
                        }
                        let receiver = self.expr(&m.object, Some(params[0].clone()));
                        if self.reject_struct_value(&receiver) {
                            return self.poison(c.span, "component struct cannot be copied");
                        }
                        let mut args = vec![receiver.clone()];
                        let mut bad = false;
                        if receiver.ty != params[0] {
                            self.at(
                                m.object.span(),
                                format!("receiver is {}, expected {}", receiver.ty, params[0]),
                            );
                            bad = true;
                        }
                        for (a, pty) in c.arguments.iter().zip(&params[1..]) {
                            let Some(ex) = a.as_expression() else {
                                self.at(c.span, "spread arguments are not supported");
                                bad = true;
                                continue;
                            };
                            let v = self.expr(ex, Some(pty.clone()));
                            if self.reject_struct_value(&v) {
                                bad = true;
                                continue;
                            }
                            if v.ty != *pty {
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
                            FnKind::Local => CallTarget::Local(self.sigs.index(method)),
                            FnKind::Host(_) => CallTarget::Host(method.to_owned()),
                        };
                        return Expr::new(ExprKind::Call { target, args }, ret, c.span);
                    }
                }
                let Expression::Identifier(callee) = &c.callee else {
                    return self.poison(
                        c.span,
                        "only calls to named functions or `array.push` are supported",
                    );
                };
                let name = callee.name.as_str();
                let Some(kind) = self.sigs.kind_of(name) else {
                    return self.poison(c.span, format!("unknown function `{name}`"));
                };
                let (params, ret) = match &kind {
                    FnKind::Local => {
                        let sig = self.sigs.sig(name).expect("local fn has a signature");
                        (sig.params.clone(), sig.ret.clone())
                    }
                    FnKind::Host(sig) => (sig.params.clone(), sig.ret.clone()),
                };
                if c.arguments.len() != params.len() {
                    return self.poison(
                        c.span,
                        format!("`{name}` expects {} arguments", params.len()),
                    );
                }
                let mut args = Vec::with_capacity(params.len());
                let mut bad = false;
                for (a, pty) in c.arguments.iter().zip(&params) {
                    let Some(ex) = a.as_expression() else {
                        self.at(c.span, "spread arguments are not supported");
                        bad = true;
                        continue;
                    };
                    let v = self.expr(ex, Some(pty.clone()));
                    if self.reject_struct_value(&v) {
                        bad = true;
                        continue;
                    }
                    if v.ty != *pty {
                        self.at(ex.span(), format!("argument is {}, expected {}", v.ty, pty));
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
                let to = match self.lowerer.resolve_ty(&a.type_annotation) {
                    Ok(ty) => ty,
                    Err(e) => {
                        self.at(a.span, e.to_string());
                        Ty::I32
                    }
                };
                if matches!(&a.expression, Expression::ObjectExpression(_))
                    && matches!(&to, Ty::Struct(_))
                {
                    return self.expr(&a.expression, Some(to));
                }
                let v = self.expr(&a.expression, None);
                let cast_ty = to.clone();
                // Same type: no-op. Integers convert freely (wrap on
                // narrowing, extend on widening); integer/float converts use
                // saturating Cranelift conversions; `bool` converts to any
                // numeric; `Entity` converts only to/from `u64` (same bits).
                let ok = match (v.ty.clone(), cast_ty.clone()) {
                    (x, y) if x == y => true,
                    (x, y) if x.is_integer() && y.is_integer() => true,
                    (x, y) if x.is_integer() && y.is_float() => true,
                    (x, y) if x.is_float() && y.is_integer() => true,
                    (Ty::F32, Ty::F64) | (Ty::F64, Ty::F32) => true,
                    (Ty::Bool, y) if y.is_numeric() => true,
                    (Ty::Entity, Ty::U64) | (Ty::U64, Ty::Entity) => true,
                    _ => false,
                };
                if ok && v.ty == cast_ty {
                    v
                } else if ok {
                    Expr::new(
                        ExprKind::Cast {
                            expr: Box::new(v),
                            to,
                        },
                        cast_ty,
                        a.span,
                    )
                } else {
                    self.poison(a.span, format!("cannot cast {} to {cast_ty}", v.ty))
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
                format!("operand types differ: {} vs {} (use `as`)", l.ty, r.ty),
            );
            return;
        }
        let t = l.ty.clone();
        let supported = match (t, op) {
            (Ty::Str, _) => matches!(
                op,
                BinaryOperator::Addition
                    | BinaryOperator::Equality
                    | BinaryOperator::Inequality
                    | BinaryOperator::StrictEquality
                    | BinaryOperator::StrictInequality
            ),
            (Ty::Array(_) | Ty::Ptr, _) => false,
            (Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64, _) => {
                matches!(
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
                ) || is_comparison(op)
            }
            (Ty::F32 | Ty::F64, _) => {
                matches!(
                    op,
                    BinaryOperator::Addition
                        | BinaryOperator::Subtraction
                        | BinaryOperator::Multiplication
                        | BinaryOperator::Division
                ) || is_comparison(op)
            }
            // `Entity` handles compare like opaque ids; nothing else.
            (Ty::Entity, _) => matches!(
                op,
                BinaryOperator::Equality
                    | BinaryOperator::Inequality
                    | BinaryOperator::StrictEquality
                    | BinaryOperator::StrictInequality
            ),
            (Ty::Bool, _) => matches!(
                op,
                BinaryOperator::Equality
                    | BinaryOperator::Inequality
                    | BinaryOperator::StrictEquality
                    | BinaryOperator::StrictInequality
            ),
            (Ty::Void, _) => false,
            _ => false,
        };
        if !supported {
            self.at(span, format!("operator {op:?} not supported on {}", l.ty));
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
/// Build a QueryShape from a Query row type (tuple or single component).
fn query_shape_from_row(row: &Ty, types: &TypeScope) -> QueryShape {
    let terms_ty: Vec<Ty> = match row {
        Ty::Tuple(ts) => ts.clone(),
        other => vec![other.clone()],
    };
    let mut terms = Vec::with_capacity(terms_ty.len());
    for t in terms_ty {
        let (mutable, inner) = match &t {
            Ty::Mut(inner) => (true, (**inner).clone()),
            other => (false, other.clone()),
        };
        let name = match &inner {
            Ty::Struct(l) => l.name.clone(),
            Ty::Component(c) => match c.as_ref() {
                Ty::Struct(l) => l.name.clone(),
                other => other.name().into_owned(),
            },
            other => other.name().into_owned(),
        };
        // Prefer layout from the type itself, fall back to TypeScope alias.
        let layout = inner
            .struct_layout()
            .or_else(|| types.get(&name).and_then(|t| t.struct_layout()));
        terms.push(QueryTerm {
            name,
            layout,
            mutable,
        });
    }
    QueryShape { terms }
}

pub fn ty_of(t: &TSType<'_>) -> Result<Ty, CompileError> {
    ty_of_in(t, &HostRegistry::new())
}

/// Type lowering against the host component registry only (no module aliases).
fn ty_of_in(t: &TSType<'_>, host: &HostRegistry) -> Result<Ty, CompileError> {
    ty_of_x(t, host, None)
}

/// Resolve a TS type annotation: module aliases (`env`) first, then
/// host-registered components, with `Query<T, F>` filters and the
/// `Component<T>` / `Resource<T>` mod-defined markers.
fn ty_of_x(
    t: &TSType<'_>,
    host: &HostRegistry,
    env: Option<&TypeScope>,
) -> Result<Ty, CompileError> {
    match t {
        TSType::TSNumberKeyword(_) => Ok(Ty::F64),
        TSType::TSBooleanKeyword(_) => Ok(Ty::Bool),
        TSType::TSStringKeyword(_) => Ok(Ty::Str),
        TSType::TSArrayType(arr) => {
            Ok(Ty::Array(Box::new(ty_of_x(&arr.element_type, host, env)?)))
        }
        TSType::TSTupleType(tuple) => {
            let elements = tuple
                .element_types
                .iter()
                .map(|element| tuple_element_ty(element, host, env))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Ty::Tuple(elements))
        }
        TSType::TSVoidKeyword(_) => Ok(Ty::Void),
        TSType::TSTypeLiteral(lit) => {
            // Anonymous object type → anonymous struct layout.
            let mut fields = Vec::new();
            for m in &lit.members {
                match m {
                    oxc_ast::ast::TSSignature::TSPropertySignature(p) => {
                        let name = match &p.key {
                            oxc_ast::ast::PropertyKey::StaticIdentifier(id) => id.name.to_string(),
                            _ => {
                                return Err(crate::diag::plain(
                                    "only plain identifier field names are supported",
                                ));
                            }
                        };
                        let ann = p
                            .type_annotation
                            .as_ref()
                            .ok_or_else(|| crate::diag::plain(format!("field `{name}` needs a type")))?;
                        let fty = ty_of_x(&ann.type_annotation, host, env)?;
                        fields.push((name, fty));
                    }
                    _ => {
                        return Err(crate::diag::plain(
                            "only property signatures are supported in type literals",
                        ));
                    }
                }
            }
            let layout = StructLayout::repr_c("", fields)
                .map_err(crate::diag::plain)?;
            Ok(Ty::Struct(Arc::new(layout)))
        }
        TSType::TSTypeReference(r) => match &r.type_name {
            TSTypeName::IdentifierReference(id) => match id.name.as_str() {
                "i8" => Ok(Ty::I8),
                "i16" => Ok(Ty::I16),
                "i32" => Ok(Ty::I32),
                "i64" => Ok(Ty::I64),
                "u8" => Ok(Ty::U8),
                "u16" => Ok(Ty::U16),
                "u32" => Ok(Ty::U32),
                "u64" => Ok(Ty::U64),
                "f32" => Ok(Ty::F32),
                "f64" => Ok(Ty::F64),
                "Entity" => Ok(Ty::Entity),
                "Option" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Option<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Option<T> requires <T>".to_string()));
                    }
                    Ok(Ty::Option(Box::new(ty_of_x(&args.params[0], host, env)?)))
                }
                "Query" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Query<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Query<T> requires <T>".to_string()));
                    }
                    if args.params.len() > 2 {
                        return Err(crate::diag::plain(
                            "Query<T, F> accepts at most two type arguments".to_string(),
                        ));
                    }
                    let row = ty_of_x(&args.params[0], host, env)?;
                    let row = match row {
                        Ty::Tuple(mut elements) if elements.len() == 1 => elements.remove(0),
                        row => row,
                    };
                    let terms = match &row {
                        Ty::Tuple(elements) => elements.as_slice(),
                        _ => std::slice::from_ref(&row),
                    };
                    if terms.is_empty() {
                        return Err(crate::diag::plain("Query row cannot be empty".to_string()));
                    }
                    for term in terms {
                        match term {
                            Ty::Entity => {}
                            Ty::Mut(inner) if inner.struct_layout().is_some() => {}
                            Ty::Mut(inner) => {
                                return Err(crate::diag::plain(format!(
                                    "Mut<{}> is not a registered component",
                                    inner
                                )));
                            }
                            other if other.struct_layout().is_some() => {}
                            other => {
                                return Err(crate::diag::plain(format!(
                                    "query terms must be components or Entity, found {other}"
                                )));
                            }
                        }
                    }
                    let filters = args
                        .params
                        .get(1)
                        .map(|filter_ty| query_filters_in(filter_ty, host, env))
                        .transpose()?
                        .unwrap_or_default();
                    Ok(Ty::Query(Box::new(row), filters))
                }
                "Res" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Res<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Res<T> requires <T>".to_string()));
                    }
                    Ok(Ty::Res(Box::new(ty_of_x(&args.params[0], host, env)?)))
                }
                "Mut" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Mut<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Mut<T> requires <T>".to_string()));
                    }
                    Ok(Ty::Mut(Box::new(ty_of_x(&args.params[0], host, env)?)))
                }
                "Component" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Component<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Component<T> requires <T>".to_string()));
                    }
                    let inner = ty_of_x(&args.params[0], host, env)?;
                    inner.ensure_pod().map_err(crate::diag::plain)?;
                    Ok(Ty::Component(Box::new(inner)))
                }
                "Resource" => {
                    let args = r
                        .type_arguments
                        .as_ref()
                        .ok_or_else(|| crate::diag::plain("Resource<T> requires <T>".to_string()))?;
                    if args.params.is_empty() {
                        return Err(crate::diag::plain("Resource<T> requires <T>".to_string()));
                    }
                    let inner = ty_of_x(&args.params[0], host, env)?;
                    inner.ensure_pod().map_err(crate::diag::plain)?;
                    Ok(Ty::Resource(Box::new(inner)))
                }
                "Commands" => Ok(Ty::Commands),
                other => {
                    // Look up mod-defined type aliases first, then
                    // host-registered engine components.
                    if let Some(env) = env {
                        if let Some(ty) = env.get(other) {
                            return Ok(ty.clone());
                        }
                    }
                    host.component(other)
                        .map(|layout| Ty::Struct(Arc::clone(layout)))
                        .ok_or_else(|| crate::diag::plain(format!("unknown type `{other}`")))
                }
            },
            _ => Err(crate::diag::plain("qualified type names are not supported")),
        },
        other => Err(crate::diag::plain(format!(
            "unsupported type at {:?}",
            other.span()
        ))),
    }
}

fn query_filters_in(
    ty: &TSType<'_>,
    host: &HostRegistry,
    env: Option<&TypeScope>,
) -> Result<Vec<QueryFilter>, CompileError> {
    match ty {
        TSType::TSTupleType(tuple) => tuple
            .element_types
            .iter()
            .map(|element| query_filter_element(element, host, env))
            .collect(),
        _ => Ok(vec![query_filter_in(ty, host, env)?]),
    }
}

fn query_filter_element(
    element: &TSTupleElement<'_>,
    host: &HostRegistry,
    env: Option<&TypeScope>,
) -> Result<QueryFilter, CompileError> {
    match element {
        TSTupleElement::TSOptionalType(_) | TSTupleElement::TSRestType(_) => Err(
            crate::diag::plain("optional and rest query filters are not supported".to_string()),
        ),
        TSTupleElement::TSNamedTupleMember(named) => {
            query_filter_element(&named.element_type, host, env)
        }
        _ => element
            .as_ts_type()
            .ok_or_else(|| crate::diag::plain("unsupported query filter tuple element".to_string()))
            .and_then(|filter| query_filter_in(filter, host, env)),
    }
}

fn query_filter_in(
    ty: &TSType<'_>,
    host: &HostRegistry,
    env: Option<&TypeScope>,
) -> Result<QueryFilter, CompileError> {
    let TSType::TSTypeReference(reference) = ty else {
        return Err(crate::diag::plain(
            "query filters must be With<Component> or Without<Component>".to_string(),
        ));
    };
    let TSTypeName::IdentifierReference(identifier) = &reference.type_name else {
        return Err(crate::diag::plain(
            "qualified query filter names are not supported".to_string(),
        ));
    };
    let kind = match identifier.name.as_str() {
        "With" => QueryFilterKind::With,
        "Without" => QueryFilterKind::Without,
        other => {
            return Err(crate::diag::plain(format!(
                "unknown query filter `{other}`; expected With<T> or Without<T>"
            )));
        }
    };
    let arguments = reference
        .type_arguments
        .as_ref()
        .ok_or_else(|| crate::diag::plain(format!("{}<T> requires <T>", identifier.name)))?;
    if arguments.params.len() != 1 {
        return Err(crate::diag::plain(format!(
            "{}<T> requires exactly one component type",
            identifier.name
        )));
    }
    let component = ty_of_x(&arguments.params[0], host, env)?;
    let Some(component) = component.struct_layout() else {
        return Err(crate::diag::plain(format!(
            "{} filter must name a registered component",
            identifier.name
        )));
    };
    Ok(QueryFilter { kind, component })
}

fn tuple_element_ty(
    element: &TSTupleElement<'_>,
    host: &HostRegistry,
    env: Option<&TypeScope>,
) -> Result<Ty, CompileError> {
    match element {
        TSTupleElement::TSOptionalType(_) | TSTupleElement::TSRestType(_) => Err(
            crate::diag::plain("optional and rest query terms are not supported".to_string()),
        ),
        _ => match element.as_ts_type() {
            Some(TSType::TSNamedTupleMember(named)) => {
                tuple_element_ty(&named.element_type, host, env)
            }
            Some(ty) => ty_of_x(ty, host, env),
            None => Err(crate::diag::plain(
                "unsupported query tuple element".to_string(),
            )),
        },
    }
}

fn compatible_value(actual: &Ty, expected: &Ty) -> bool {
    if actual == expected {
        return true;
    }
    matches!(
        (actual, expected),
        (Ty::OwnedStruct(actual), Ty::Struct(expected))
            | (Ty::OwnedStruct(actual), Ty::OwnedStruct(expected))
            if actual == expected
    )
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
        self.fns.iter().find(|(n, _)| n == name).map(|(_, s)| s)
    }

    fn index(&self, name: &str) -> usize {
        self.fns.iter().position(|(n, _)| n == name).unwrap_or(0)
    }
}
