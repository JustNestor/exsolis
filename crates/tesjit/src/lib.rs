//! tesjit — minimal statically-typed TypeScript → Cranelift JIT compiler.
//!
//! ```ignore
//! use tesjit::Jit;
//! let mut jit = Jit::new("mod.ts", src)?;
//! jit.compile(&src)?;
//! let compiled = jit.finish()?;
//! let f: extern "C" fn(f64) = unsafe { compiled.get("run")? };
//! ```
//!
//! Types: `number/f64` → F64, `i32` → I32, `boolean` → I8, `void`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{self, AbiParam, BlockArg, InstBuilder, Value, types};
use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{Context, settings};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use linkme::distributed_slice;
use miette::{Diagnostic, NamedSource, SourceOffset, SourceSpan};
use oxc_allocator::Allocator;
use oxc_ast::ast::{
    AssignmentOperator, AssignmentTarget, BinaryOperator, BindingPattern, Declaration, Expression,
    ForStatementInit, Function, LogicalOperator, SimpleAssignmentTarget, Statement, TSType,
    TSTypeName, UnaryOperator, UpdateOperator, VariableDeclaration,
};
use oxc_parser::Parser;
use oxc_span::{GetSpan, SourceType, Span};
use thiserror::Error;

// ───────────────────────────── types ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    I32,
    F64,
    Bool,
    Void,
}

impl Ty {
    #[inline]
    fn clif(self) -> ir::Type {
        match self {
            Ty::I32 => types::I32,
            Ty::F64 => types::F64,
            Ty::Bool => types::I8,
            Ty::Void => types::I32,
        }
    }
}

// ───────────────────────────── errors ─────────────────────────────

#[derive(Error, Diagnostic, Debug)]
pub enum CompileError {
    #[error("{msg}")]
    #[diagnostic(code(tsjit::compile))]
    Spanned {
        msg: String,
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[label("{msg}")]
        span: SourceSpan,
    },

    #[error("{0}")]
    #[diagnostic(code(tsjit::plain))]
    Plain(String),

    #[error("syntax error(s)")]
    #[diagnostic(code(tsjit::syntax))]
    Syntax {
        #[source_code]
        src: NamedSource<Arc<str>>,
        #[related]
        diagnostics: Vec<SyntaxDiag>,
    },
}

#[derive(Error, Diagnostic, Debug)]
#[error("{msg}")]
#[diagnostic(code(tsjit::syntax_item))]
pub struct SyntaxDiag {
    msg: String,
    #[label]
    span: SourceSpan,
}

type R<T> = Result<T, CompileError>;

fn err(src: &NamedSource<Arc<str>>, span: Span, msg: impl Into<String>) -> CompileError {
    CompileError::Spanned {
        msg: msg.into(),
        src: src.clone(),
        span: SourceSpan::new(
            SourceOffset::from(span.start as usize),
            (span.end - span.start) as usize,
        ),
    }
}

fn plain(msg: impl Into<String>) -> CompileError {
    CompileError::Plain(msg.into())
}

fn ty_of(t: &TSType<'_>) -> R<Ty> {
    match t {
        TSType::TSNumberKeyword(_) => Ok(Ty::F64),
        TSType::TSBooleanKeyword(_) => Ok(Ty::Bool),
        TSType::TSVoidKeyword(_) => Ok(Ty::Void),
        TSType::TSTypeReference(r) => match &r.type_name {
            TSTypeName::IdentifierReference(id) => match id.name.as_str() {
                "i32" => Ok(Ty::I32),
                "f64" => Ok(Ty::F64),
                other => Err(plain(format!("unknown type `{other}`"))),
            },
            _ => Err(plain("qualified type names are not supported")),
        },
        other => Err(plain(format!("unsupported type at {:?}", other.span()))),
    }
}

// ───────────────────────────── host functions ─────────────────────────────

#[derive(Clone, Copy)]
pub struct HostFn {
    pub name: &'static str,
    pub ptr: *const u8,
    pub params: &'static [Ty],
    pub ret: Ty,
}

// SAFETY: only function pointers + static metadata
unsafe impl Send for HostFn {}
unsafe impl Sync for HostFn {}

#[distributed_slice]
pub static HOST_FNS: [HostFn] = [..];

/// Type-safe host registration.
///
/// ```ignore
/// register_host!(log, host_log, &[Ty::F64], Ty::Void);
/// ```
#[macro_export]
macro_rules! register_host {
    ($name:ident, $f:expr, $params:expr, $ret:expr) => {
        #[::linkme::distributed_slice($crate::HOST_FNS)]
        static _HOST: $crate::HostFn = $crate::HostFn {
            name: stringify!($name),
            ptr: $f as *const u8,
            params: $params,
            ret: $ret,
        };
    };
}

// ───────────────────────────── compiled output ─────────────────────────────

pub struct Entry {
    pub ptr: *const u8,
    pub params: Vec<Ty>,
    pub ret: Ty,
}

pub struct Compiled {
    module: Option<JITModule>,
    pub fns: HashMap<String, Entry>,
}

unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Compiled {
    /// Names of every exported TS function, with its TS signature.
    pub fn exports(&self) -> impl Iterator<Item = (&str, &[Ty], Ty)> {
        self.fns
            .iter()
            .map(|(name, e)| (name.as_str(), e.params.as_slice(), e.ret))
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.fns.get(name)
    }

    /// # Safety
    /// `F` must match the TS signature. Pointer invalid after `self` is dropped.
    pub unsafe fn get<F: Copy>(&self, name: &str) -> Option<F> {
        debug_assert_eq!(std::mem::size_of::<F>(), std::mem::size_of::<usize>());
        let e = self.fns.get(name)?;
        Some(unsafe { std::mem::transmute_copy::<*const u8, F>(&e.ptr) })
    }

    /// Call a TS function by name with integer/pointer arguments, returning
    /// its result as `i64`.
    ///
    /// Temporary convenience for the CLI/bindings: every argument is passed
    /// in an integer register slot, so `i32/boolean` parameters are fine but
    /// `f64` parameters are **not** (they would belong in an SSE register).
    /// Use [`Compiled::get`] with an exact signature for float parameters.
    ///
    /// # Safety
    /// `name` must exist and accept the given arguments.
    #[track_caller]
    pub unsafe fn call_i64(&self, name: &str, args: &[i64]) -> i64 {
        let e = self.entry(name).expect("unknown export");
        assert!(e.params.len() == args.len(), "argument count mismatch");
        assert!(args.len() <= 4, "call_i64 supports at most 4 arguments");
        assert!(
            e.params.iter().all(|&t| t != Ty::F64),
            "call_i64 cannot pass f64 parameters; use `get::<F>()`"
        );
        let mut buf = [0i64; 4];
        buf[..args.len()].copy_from_slice(args);
        let f: extern "C" fn(i64, i64, i64, i64) -> i64 =
            unsafe { std::mem::transmute_copy(&e.ptr) };
        f(buf[0], buf[1], buf[2], buf[3])
    }

    /// # Safety
    /// Same contract as [`Compiled::call_i64`]; the result is discarded.
    pub unsafe fn call_void(&self, name: &str, args: &[i64]) {
        unsafe { self.call_i64(name, args) };
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            unsafe { m.free_memory() };
        }
    }
}

// ───────────────────────────── scoped map ─────────────────────────────

struct ScopedMap {
    entries: Vec<(String, (Variable, Ty))>,
    depths: Vec<usize>,
}

impl ScopedMap {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(32),
            depths: vec![0],
        }
    }

    fn push(&mut self) {
        self.depths.push(self.entries.len());
    }

    fn pop(&mut self) {
        let start = self.depths.pop().expect("scope underflow");
        self.entries.truncate(start);
    }

    fn insert(&mut self, k: String, v: (Variable, Ty)) {
        self.entries.push((k, v));
    }

    fn get(&self, name: &str) -> Option<(Variable, Ty)> {
        self.entries
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| *v)
    }
}

// ───────────────────────────── the compiler ─────────────────────────────

#[derive(Clone)]
struct FnSig {
    id: FuncId,
    params: Vec<Ty>,
    ret: Ty,
}

pub struct Jit {
    module: JITModule,
    ctx: Context,
    fctx: FunctionBuilderContext,
    sigs: HashMap<String, FnSig>,
    exports: Vec<String>,
    src: NamedSource<Arc<str>>,
}

impl Jit {
    pub fn new(src_name: impl AsRef<str>, src: impl Into<Arc<str>>) -> R<Self> {
        let src = NamedSource::new(src_name.as_ref(), src.into());
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        flag_builder
            .set("preserve_frame_pointers", "false")
            .unwrap(); // Прибираємо push/pop rbp
        flag_builder.set("enable_probestack", "false").unwrap(); // Вимикаємо stack probing guards
        flag_builder.set("enable_alias_analysis", "true").unwrap();
        // 2. Отримуємо ISA з авто-детектом інструкцій вашого CPU (AVX2, SSE4.2, BMI2)
        let isa_builder = cranelift_native::builder()
            .map_err(|e| plain(format!("Native ISA detection failed: {e}")))?;

        let flags = settings::Flags::new(flag_builder);
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| plain(e.to_string()))?;

        // 3. Створюємо JITBuilder через target ISA
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());

        for h in HOST_FNS.iter() {
            jb.symbol(h.name, h.ptr);
        }

        let mut module = JITModule::new(jb);
        let mut sigs = HashMap::new();

        for h in HOST_FNS.iter() {
            let mut sig = module.make_signature();
            for &p in h.params {
                sig.params.push(AbiParam::new(p.clif()));
            }
            if h.ret != Ty::Void {
                sig.returns.push(AbiParam::new(h.ret.clif()));
            }
            let id = module
                .declare_function(h.name, Linkage::Import, &sig)
                .map_err(|e| plain(e.to_string()))?;
            sigs.insert(
                h.name.to_string(),
                FnSig {
                    id,
                    params: h.params.to_vec(),
                    ret: h.ret,
                },
            );
        }

        let ctx = module.make_context();
        Ok(Self {
            module,
            ctx,
            fctx: FunctionBuilderContext::new(),
            sigs,
            exports: Vec::new(),
            src,
        })
    }

    pub fn compile(&mut self, source: &str) -> R<()> {
        let alloc = Allocator::default();
        let parsed = Parser::new(&alloc, source, SourceType::ts()).parse();

        if !parsed.diagnostics.is_empty() {
            let diags: Vec<_> = parsed
                .diagnostics
                .iter()
                .map(|d| {
                    let (off, len) = d
                        .labels
                        .first()
                        .map(|l| (l.offset() as usize, l.len() as usize))
                        .unwrap_or((0, 0));
                    SyntaxDiag {
                        msg: d.message.to_string(),
                        span: SourceSpan::new(SourceOffset::from(off), len),
                    }
                })
                .collect();
            return Err(CompileError::Syntax {
                src: self.src.clone(),
                diagnostics: diags,
            });
        }

        let mut bodies: Vec<(&Function, FnSig)> = Vec::new();
        for stmt in &parsed.program.body {
            let Some(f) = as_function(stmt) else { continue };

            let name =
                f.id.as_ref()
                    .map(|i| i.name.as_str().to_string())
                    .ok_or_else(|| err(&self.src, f.span, "anonymous function"))?;

            let mut params = Vec::with_capacity(f.params.items.len());
            for p in &f.params.items {
                let ann = p.type_annotation.as_ref().ok_or_else(|| {
                    err(
                        &self.src,
                        p.span,
                        format!("parameter of `{name}` needs a type annotation"),
                    )
                })?;
                params.push(ty_of(&ann.type_annotation)?);
            }

            let ret = match &f.return_type {
                Some(a) => ty_of(&a.type_annotation)?,
                None => Ty::Void,
            };

            if f.declare || f.body.is_none() {
                match self.sigs.get(&name) {
                    Some(s) if s.params == params && s.ret == ret => continue,
                    Some(_) => {
                        return Err(err(
                            &self.src,
                            f.span,
                            format!("`{name}` signature differs from host"),
                        ));
                    }
                    None => {
                        return Err(err(
                            &self.src,
                            f.span,
                            format!("no host function `{name}` registered"),
                        ));
                    }
                }
            }

            if self.sigs.contains_key(&name) {
                return Err(err(
                    &self.src,
                    f.span,
                    format!("duplicate function `{name}`"),
                ));
            }

            let mut sig = self.module.make_signature();
            for p in &params {
                sig.params.push(AbiParam::new(p.clif()));
            }
            if ret != Ty::Void {
                sig.returns.push(AbiParam::new(ret.clif()));
            }
            let id = self
                .module
                .declare_function(&name, Linkage::Export, &sig)
                .map_err(|e| plain(e.to_string()))?;

            let fs = FnSig { id, params, ret };
            self.sigs.insert(name.clone(), fs.clone());
            self.exports.push(name);
            bodies.push((f, fs));
        }

        for (f, sig) in bodies {
            self.define(f, &sig)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> R<Compiled> {
        self.module
            .finalize_definitions()
            .map_err(|e| plain(e.to_string()))?;
        let mut fns = HashMap::new();
        for name in &self.exports {
            let s = &self.sigs[name];
            let ptr = self.module.get_finalized_function(s.id);
            fns.insert(
                name.clone(),
                Entry {
                    ptr,
                    params: s.params.clone(),
                    ret: s.ret,
                },
            );
        }
        Ok(Compiled {
            module: Some(self.module),
            fns,
        })
    }

    fn define(&mut self, f: &Function, sig: &FnSig) -> R<()> {
        let mut clif_sig = self.module.make_signature();
        for p in &sig.params {
            clif_sig.params.push(AbiParam::new(p.clif()));
        }
        if sig.ret != Ty::Void {
            clif_sig.returns.push(AbiParam::new(sig.ret.clif()));
        }
        self.module.clear_context(&mut self.ctx);
        self.ctx.func.signature = clif_sig;
        let target = self.module.target_config();

        {
            let mut b = FunctionBuilder::new(&mut self.ctx.func, &mut self.fctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);

            let mut scopes = ScopedMap::new();
            for (i, p) in f.params.items.iter().enumerate() {
                let BindingPattern::BindingIdentifier(id) = &p.pattern else {
                    return Err(err(
                        &self.src,
                        p.span,
                        "destructuring parameters are not supported",
                    ));
                };
                let ty = sig.params[i];
                let var = b.declare_var(ty.clif());
                let val = b.block_params(entry)[i];
                b.def_var(var, val);
                scopes.insert(id.name.as_str().to_string(), (var, ty));
            }

            let mut cg = Cg {
                b,
                module: &mut self.module,
                sigs: &self.sigs,
                scopes,
                loops: Vec::new(),
                ret: sig.ret,
                src: &self.src,
            };
            let body = f.body.as_ref().expect("checked in pass 1");
            for s in &body.statements {
                cg.stmt(s)?;
            }
            cg.default_return();
            cg.b.finalize(target);
        }

        let id = sig.id;
        let res = self.module.define_function(id, &mut self.ctx);
        self.module.clear_context(&mut self.ctx);
        res.map_err(|e| plain(format!("cranelift: {e:?}")))
    }
}

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

// ───────────────────────────── code generation ─────────────────────────────

#[derive(Clone, Copy)]
struct Val {
    v: Value,
    ty: Ty,
}

struct Cg<'a, 'f, 's> {
    b: FunctionBuilder<'f>,
    module: &'a mut JITModule,
    sigs: &'a HashMap<String, FnSig>,
    scopes: ScopedMap,
    loops: Vec<(ir::Block, ir::Block)>,
    ret: Ty,
    src: &'s NamedSource<Arc<str>>,
}

impl<'a, 'f, 's> Cg<'a, 'f, 's> {
    #[inline]
    fn zero(&mut self, ty: Ty) -> Value {
        match ty {
            Ty::F64 => self.b.ins().f64const(0.0),
            _ => self.b.ins().iconst(ty.clif(), 0),
        }
    }

    fn default_return(&mut self) {
        if self.ret == Ty::Void {
            self.b.ins().return_(&[]);
        } else {
            let z = self.zero(self.ret);
            self.b.ins().return_(&[z]);
        }
    }

    fn dead_block(&mut self) {
        let d = self.b.create_block();
        self.b.switch_to_block(d);
        self.b.seal_block(d);
    }

    fn cond(&mut self, e: &Expression) -> R<Value> {
        let v = self.expr(e, Some(Ty::Bool))?;
        if v.ty != Ty::Bool {
            return Err(err(
                self.src,
                e.span(),
                format!("condition must be boolean, found {:?}", v.ty),
            ));
        }
        Ok(v.v)
    }

    fn scoped<F: FnOnce(&mut Self) -> R<()>>(&mut self, f: F) -> R<()> {
        self.scopes.push();
        let r = f(self);
        self.scopes.pop();
        r
    }

    fn stmt(&mut self, s: &Statement) -> R<()> {
        match s {
            Statement::ExpressionStatement(e) => {
                self.expr(&e.expression, None)?;
            }
            Statement::VariableDeclaration(d) => self.var_decl(d)?,
            Statement::BlockStatement(bl) => {
                self.scoped(|this| {
                    for s in &bl.body {
                        this.stmt(s)?;
                    }
                    Ok(())
                })?;
            }
            Statement::ReturnStatement(r) => {
                match &r.argument {
                    Some(e) => {
                        let v = self.expr(e, Some(self.ret))?;
                        if v.ty != self.ret {
                            return Err(err(
                                self.src,
                                r.span,
                                format!("returns {:?}, expected {:?}", v.ty, self.ret),
                            ));
                        }
                        self.b.ins().return_(&[v.v]);
                    }
                    None => {
                        if self.ret != Ty::Void {
                            return Err(err(self.src, r.span, "missing return value"));
                        }
                        self.b.ins().return_(&[]);
                    }
                }
                self.dead_block();
            }
            Statement::IfStatement(i) => {
                let c = self.cond(&i.test)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                self.b.ins().brif(c, then_b, &[], else_b, &[]);

                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                self.scoped(|this| this.stmt(&i.consequent))?;
                self.b.ins().jump(merge, &[]);

                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                if let Some(a) = &i.alternate {
                    self.scoped(|this| this.stmt(a))?;
                }
                self.b.ins().jump(merge, &[]);

                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
            }
            Statement::WhileStatement(w) => {
                let header = self.b.create_block();
                let body = self.b.create_block();
                let exit = self.b.create_block();
                self.b.ins().jump(header, &[]);

                self.b.switch_to_block(header);
                let c = self.cond(&w.test)?;
                self.b.ins().brif(c, body, &[], exit, &[]);

                self.b.switch_to_block(body);
                self.b.seal_block(body);
                self.loops.push((header, exit));
                let r = self.scoped(|this| this.stmt(&w.body));
                self.loops.pop();
                r?;
                self.b.ins().jump(header, &[]);

                self.b.seal_block(header);
                self.b.switch_to_block(exit);
                self.b.seal_block(exit);
            }
            Statement::ForStatement(f) => {
                self.scopes.push();
                let r = self.for_stmt(f);
                self.scopes.pop();
                r?;
            }
            Statement::BreakStatement(b) => {
                let Some(&(_, brk)) = self.loops.last() else {
                    return Err(err(self.src, b.span, "`break` outside loop"));
                };
                self.b.ins().jump(brk, &[]);
                self.dead_block();
            }
            Statement::ContinueStatement(c) => {
                let Some(&(cont, _)) = self.loops.last() else {
                    return Err(err(self.src, c.span, "`continue` outside loop"));
                };
                self.b.ins().jump(cont, &[]);
                self.dead_block();
            }
            Statement::EmptyStatement(_) => {}
            Statement::TSTypeAliasDeclaration(_) | Statement::TSInterfaceDeclaration(_) => {}
            other => {
                return Err(err(self.src, other.span(), "unsupported statement"));
            }
        }
        Ok(())
    }

    fn for_stmt(&mut self, f: &oxc_ast::ast::ForStatement) -> R<()> {
        match &f.init {
            Some(ForStatementInit::VariableDeclaration(d)) => self.var_decl(d)?,
            Some(i) => {
                if let Some(e) = i.as_expression() {
                    self.expr(e, None)?;
                }
            }
            None => {}
        }

        let header = self.b.create_block();
        let body = self.b.create_block();
        let update = self.b.create_block();
        let exit = self.b.create_block();

        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);

        match &f.test {
            Some(t) => {
                let c = self.cond(t)?;
                self.b.ins().brif(c, body, &[], exit, &[]);
            }
            None => {
                self.b.ins().jump(body, &[]);
            }
        }

        self.b.switch_to_block(body);
        self.b.seal_block(body);
        self.loops.push((update, exit));
        let r = self.scoped(|this| this.stmt(&f.body));
        self.loops.pop();
        r?;
        self.b.ins().jump(update, &[]);

        self.b.switch_to_block(update);
        self.b.seal_block(update);
        if let Some(u) = &f.update {
            self.expr(u, None)?;
        }
        self.b.ins().jump(header, &[]);

        self.b.seal_block(header);
        self.b.switch_to_block(exit);
        self.b.seal_block(exit);
        Ok(())
    }

    fn var_decl(&mut self, d: &VariableDeclaration) -> R<()> {
        for decl in &d.declarations {
            let BindingPattern::BindingIdentifier(id) = &decl.id else {
                return Err(err(self.src, decl.span, "destructuring is not supported"));
            };
            let declared = match &decl.type_annotation {
                Some(a) => Some(ty_of(&a.type_annotation)?),
                None => None,
            };
            let (val, ty) = match &decl.init {
                Some(init) => {
                    let v = self.expr(init, declared)?;
                    if let Some(t) = declared {
                        if t != v.ty {
                            return Err(err(
                                self.src,
                                decl.span,
                                format!("cannot assign {:?} to {:?} (use `as`)", v.ty, t),
                            ));
                        }
                    }
                    (v.v, v.ty)
                }
                None => {
                    let Some(t) = declared else {
                        return Err(err(
                            self.src,
                            decl.span,
                            "variable needs a type annotation or initializer",
                        ));
                    };
                    (self.zero(t), t)
                }
            };
            if ty == Ty::Void {
                return Err(err(self.src, decl.span, "variable cannot be void"));
            }
            let var = self.b.declare_var(ty.clif());
            self.b.def_var(var, val);
            self.scopes.insert(id.name.as_str().to_string(), (var, ty));
        }
        Ok(())
    }

    fn expr(&mut self, e: &Expression, expect: Option<Ty>) -> R<Val> {
        match e {
            Expression::NumericLiteral(n) => {
                if expect == Some(Ty::I32) {
                    if n.value.fract() != 0.0
                        || n.value < i32::MIN as f64
                        || n.value > i32::MAX as f64
                    {
                        return Err(err(self.src, n.span, "literal does not fit in i32"));
                    }
                    let v = self.b.ins().iconst(types::I32, n.value as i64);
                    Ok(Val { v, ty: Ty::I32 })
                } else {
                    let v = self.b.ins().f64const(n.value);
                    Ok(Val { v, ty: Ty::F64 })
                }
            }
            Expression::BooleanLiteral(l) => {
                let v = self.b.ins().iconst(types::I8, l.value as i64);
                Ok(Val { v, ty: Ty::Bool })
            }
            Expression::ParenthesizedExpression(p) => self.expr(&p.expression, expect),
            Expression::Identifier(id) => {
                let Some((var, ty)) = self.scopes.get(id.name.as_str()) else {
                    return Err(err(
                        self.src,
                        id.span,
                        format!("unknown variable `{}`", id.name),
                    ));
                };
                Ok(Val {
                    v: self.b.use_var(var),
                    ty,
                })
            }
            Expression::UnaryExpression(u) => {
                let a = self.expr(&u.argument, expect)?;
                let v = match (u.operator, a.ty) {
                    (UnaryOperator::UnaryPlus, Ty::I32 | Ty::F64) => a.v,
                    (UnaryOperator::UnaryNegation, Ty::I32) => self.b.ins().ineg(a.v),
                    (UnaryOperator::UnaryNegation, Ty::F64) => self.b.ins().fneg(a.v),
                    (UnaryOperator::BitwiseNot, Ty::I32) => self.b.ins().bnot(a.v),
                    (UnaryOperator::LogicalNot, Ty::Bool) => {
                        // modern cranelift: prefer icmp_imm_s / icmp_imm_u
                        let v = self.b.ins().icmp_imm(IntCC::Equal, a.v, 0);
                        return Ok(Val { v, ty: Ty::Bool });
                    }
                    _ => {
                        return Err(err(
                            self.src,
                            u.span,
                            format!("unsupported unary op on {:?}", a.ty),
                        ));
                    }
                };
                Ok(Val { v, ty: a.ty })
            }
            Expression::BinaryExpression(bin) => {
                use BinaryOperator as B;
                let is_cmp = matches!(
                    bin.operator,
                    B::Equality
                        | B::Inequality
                        | B::StrictEquality
                        | B::StrictInequality
                        | B::LessThan
                        | B::LessEqualThan
                        | B::GreaterThan
                        | B::GreaterEqualThan
                );
                let operand_expect = if is_cmp { None } else { expect };
                let lit = |e: &Expression| matches!(e, Expression::NumericLiteral(_));
                let (l, r) = if lit(&bin.left) && !lit(&bin.right) {
                    let r = self.expr(&bin.right, operand_expect)?;
                    let l = self.expr(&bin.left, Some(r.ty))?;
                    (l, r)
                } else {
                    let l = self.expr(&bin.left, operand_expect)?;
                    let r = self.expr(&bin.right, Some(l.ty))?;
                    (l, r)
                };
                self.arith(bin.operator, l, r, bin.span)
            }
            Expression::LogicalExpression(l) => {
                let lhs = self.expr(&l.left, Some(Ty::Bool))?;
                if lhs.ty != Ty::Bool {
                    return Err(err(self.src, l.span, "`&&` / `||` need boolean operands"));
                }
                let rhs_b = self.b.create_block();
                let merge = self.b.create_block();
                let res = self.b.append_block_param(merge, types::I8);
                match l.operator {
                    LogicalOperator::And => {
                        self.b
                            .ins()
                            .brif(lhs.v, rhs_b, &[], merge, &[BlockArg::from(lhs.v)]);
                    }
                    LogicalOperator::Or => {
                        self.b
                            .ins()
                            .brif(lhs.v, merge, &[BlockArg::from(lhs.v)], rhs_b, &[]);
                    }
                    _ => return Err(err(self.src, l.span, "`??` is not supported")),
                }
                self.b.switch_to_block(rhs_b);
                self.b.seal_block(rhs_b);
                let rhs = self.expr(&l.right, Some(Ty::Bool))?;
                if rhs.ty != Ty::Bool {
                    return Err(err(self.src, l.span, "`&&` / `||` need boolean operands"));
                }
                self.b.ins().jump(merge, &[BlockArg::from(rhs.v)]);
                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Ok(Val {
                    v: res,
                    ty: Ty::Bool,
                })
            }
            Expression::ConditionalExpression(c) => {
                let t = self.cond(&c.test)?;
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                self.b.ins().brif(t, then_b, &[], else_b, &[]);

                self.b.switch_to_block(then_b);
                self.b.seal_block(then_b);
                let a = self.expr(&c.consequent, expect)?;
                let merge = self.b.create_block();
                let res = self.b.append_block_param(merge, a.ty.clif());
                self.b.ins().jump(merge, &[BlockArg::from(a.v)]);

                self.b.switch_to_block(else_b);
                self.b.seal_block(else_b);
                let o = self.expr(&c.alternate, Some(a.ty))?;
                if o.ty != a.ty {
                    return Err(err(
                        self.src,
                        c.span,
                        "branches of `?:` have different types",
                    ));
                }
                self.b.ins().jump(merge, &[BlockArg::from(o.v)]);

                self.b.switch_to_block(merge);
                self.b.seal_block(merge);
                Ok(Val { v: res, ty: a.ty })
            }
            Expression::AssignmentExpression(a) => {
                let AssignmentTarget::AssignmentTargetIdentifier(id) = &a.left else {
                    return Err(err(
                        self.src,
                        a.span,
                        "only assignment to plain variables is supported",
                    ));
                };
                let Some((var, ty)) = self.scopes.get(id.name.as_str()) else {
                    return Err(err(
                        self.src,
                        id.span,
                        format!("unknown variable `{}`", id.name),
                    ));
                };
                let rhs = self.expr(&a.right, Some(ty))?;
                let new = if a.operator == AssignmentOperator::Assign {
                    rhs
                } else {
                    use AssignmentOperator as A;
                    use BinaryOperator as B;
                    let op = match a.operator {
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
                        _ => return Err(err(self.src, a.span, "unsupported compound assignment")),
                    };
                    let cur = Val {
                        v: self.b.use_var(var),
                        ty,
                    };
                    self.arith(op, cur, rhs, a.span)?
                };
                if new.ty != ty {
                    return Err(err(
                        self.src,
                        a.span,
                        format!("cannot assign {:?} to {:?}", new.ty, ty),
                    ));
                }
                self.b.def_var(var, new.v);
                Ok(new)
            }
            Expression::UpdateExpression(u) => {
                let SimpleAssignmentTarget::AssignmentTargetIdentifier(id) = &u.argument else {
                    return Err(err(self.src, u.span, "++/-- only on plain variables"));
                };
                let Some((var, ty)) = self.scopes.get(id.name.as_str()) else {
                    return Err(err(
                        self.src,
                        id.span,
                        format!("unknown variable `{}`", id.name),
                    ));
                };
                let old = self.b.use_var(var);
                let inc = u.operator == UpdateOperator::Increment;
                let new = match ty {
                    Ty::I32 => {
                        let one = self.b.ins().iconst(types::I32, 1);
                        if inc {
                            self.b.ins().iadd(old, one)
                        } else {
                            self.b.ins().isub(old, one)
                        }
                    }
                    Ty::F64 => {
                        let one = self.b.ins().f64const(1.0);
                        if inc {
                            self.b.ins().fadd(old, one)
                        } else {
                            self.b.ins().fsub(old, one)
                        }
                    }
                    _ => return Err(err(self.src, u.span, "++/-- need a number")),
                };
                self.b.def_var(var, new);
                Ok(Val {
                    v: if u.prefix { new } else { old },
                    ty,
                })
            }
            Expression::CallExpression(c) => {
                let Expression::Identifier(callee) = &c.callee else {
                    return Err(err(
                        self.src,
                        c.span,
                        "only calls to named functions are supported",
                    ));
                };
                let name = callee.name.as_str();
                let Some(sig) = self.sigs.get(name).cloned() else {
                    return Err(err(self.src, c.span, format!("unknown function `{name}`")));
                };
                if c.arguments.len() != sig.params.len() {
                    return Err(err(
                        self.src,
                        c.span,
                        format!("`{name}` expects {} arguments", sig.params.len()),
                    ));
                }
                let mut args = Vec::with_capacity(sig.params.len());
                for (a, &pty) in c.arguments.iter().zip(&sig.params) {
                    let Some(ex) = a.as_expression() else {
                        return Err(err(self.src, c.span, "spread arguments are not supported"));
                    };
                    let v = self.expr(ex, Some(pty))?;
                    if v.ty != pty {
                        return Err(err(
                            self.src,
                            ex.span(),
                            format!("argument is {:?}, expected {:?}", v.ty, pty),
                        ));
                    }
                    args.push(v.v);
                }
                let fref = self.module.declare_func_in_func(sig.id, self.b.func);
                let call = self.b.ins().call(fref, &args);
                if sig.ret == Ty::Void {
                    let v = self.b.ins().iconst(types::I32, 0);
                    Ok(Val { v, ty: Ty::Void })
                } else {
                    Ok(Val {
                        v: self.b.inst_results(call)[0],
                        ty: sig.ret,
                    })
                }
            }
            Expression::TSAsExpression(a) => {
                let target = ty_of(&a.type_annotation)?;
                let v = self.expr(&a.expression, None)?;
                let out = match (v.ty, target) {
                    (x, y) if x == y => v.v,
                    (Ty::I32, Ty::F64) => self.b.ins().fcvt_from_sint(types::F64, v.v),
                    (Ty::F64, Ty::I32) => self.b.ins().fcvt_to_sint_sat(types::I32, v.v),
                    (Ty::Bool, Ty::I32) => self.b.ins().uextend(types::I32, v.v),
                    (x, y) => {
                        return Err(err(self.src, a.span, format!("cannot cast {x:?} to {y:?}")));
                    }
                };
                Ok(Val { v: out, ty: target })
            }
            other => Err(err(self.src, other.span(), "unsupported expression")),
        }
    }

    fn arith(&mut self, op: BinaryOperator, l: Val, r: Val, span: Span) -> R<Val> {
        use BinaryOperator as B;
        if l.ty != r.ty {
            return Err(err(
                self.src,
                span,
                format!("operand types differ: {:?} vs {:?} (use `as`)", l.ty, r.ty),
            ));
        }
        let (a, c) = (l.v, r.v);
        let bad = || {
            err(
                self.src,
                span,
                format!("operator {op:?} not supported on {:?}", l.ty),
            )
        };
        let (v, ty) = match l.ty {
            Ty::I32 => {
                let cc = match op {
                    B::Equality | B::StrictEquality => Some(IntCC::Equal),
                    B::Inequality | B::StrictInequality => Some(IntCC::NotEqual),
                    B::LessThan => Some(IntCC::SignedLessThan),
                    B::LessEqualThan => Some(IntCC::SignedLessThanOrEqual),
                    B::GreaterThan => Some(IntCC::SignedGreaterThan),
                    B::GreaterEqualThan => Some(IntCC::SignedGreaterThanOrEqual),
                    _ => None,
                };
                let ins = self.b.ins();
                if let Some(cc) = cc {
                    (ins.icmp(cc, a, c), Ty::Bool)
                } else {
                    let v = match op {
                        B::Addition => ins.iadd(a, c),
                        B::Subtraction => ins.isub(a, c),
                        B::Multiplication => ins.imul(a, c),
                        B::Division => ins.sdiv(a, c),
                        B::Remainder => ins.srem(a, c),
                        B::BitwiseAnd => ins.band(a, c),
                        B::BitwiseOR => ins.bor(a, c),
                        B::BitwiseXOR => ins.bxor(a, c),
                        B::ShiftLeft => ins.ishl(a, c),
                        B::ShiftRight => ins.sshr(a, c),
                        B::ShiftRightZeroFill => ins.ushr(a, c),
                        _ => return Err(bad()),
                    };
                    (v, Ty::I32)
                }
            }
            Ty::F64 => {
                let cc = match op {
                    B::Equality | B::StrictEquality => Some(FloatCC::Equal),
                    B::Inequality | B::StrictInequality => Some(FloatCC::NotEqual),
                    B::LessThan => Some(FloatCC::LessThan),
                    B::LessEqualThan => Some(FloatCC::LessThanOrEqual),
                    B::GreaterThan => Some(FloatCC::GreaterThan),
                    B::GreaterEqualThan => Some(FloatCC::GreaterThanOrEqual),
                    _ => None,
                };
                let ins = self.b.ins();
                if let Some(cc) = cc {
                    (ins.fcmp(cc, a, c), Ty::Bool)
                } else {
                    let v = match op {
                        B::Addition => ins.fadd(a, c),
                        B::Subtraction => ins.fsub(a, c),
                        B::Multiplication => ins.fmul(a, c),
                        B::Division => ins.fdiv(a, c),
                        _ => return Err(bad()),
                    };
                    (v, Ty::F64)
                }
            }
            Ty::Bool => {
                let cc = match op {
                    B::Equality | B::StrictEquality => IntCC::Equal,
                    B::Inequality | B::StrictInequality => IntCC::NotEqual,
                    _ => return Err(bad()),
                };
                (self.b.ins().icmp(cc, a, c), Ty::Bool)
            }
            Ty::Void => return Err(bad()),
        };
        Ok(Val { v, ty })
    }
}
// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static LAST: AtomicU64 = AtomicU64::new(0);
    extern "C" fn test_log(x: f64) {
        LAST.store(x.to_bits(), Ordering::SeqCst);
    }
    register_host!(log, test_log, &[Ty::F64], Ty::Void);

    fn build(src: &str) -> Compiled {
        let mut jit = Jit::new("<test>", Arc::from(src)).unwrap();
        jit.compile(src).unwrap();
        jit.finish().unwrap()
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
    fn type_errors_are_reported() {
        let mut jit = Jit::new("<test>", Arc::from("")).unwrap();
        let e = jit
            .compile("function f(a: i32): f64 { return a; }")
            .unwrap_err();
        assert!(matches!(e, CompileError::Spanned { .. }));
    }
}
