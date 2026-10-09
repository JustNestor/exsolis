//! Cranelift codegen for the typed HIR.
//!
//! This is a pure structural translation: the checker already resolved every
//! name, type and signature, so nothing here can fail and nothing consults the
//! source. Each function becomes one Cranelift function; each TS local becomes
//! one Cranelift `Variable` (function-scoped, like SSA `def_var`s).
//!
//! The [`FunctionBuilder`] is threaded through every method rather than stored
//! in the state struct, because `finalize` consumes it at the end of a
//! function.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{self, AbiParam, BlockArg, InstBuilder, Value, types};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use oxc_ast::ast::{BinaryOperator, LogicalOperator, UnaryOperator};

use crate::hir::{CallTarget, Expr, ExprKind, FnDef, Stmt};
use crate::ty::Ty;

/// Cranelift function id plus the TS signature it was declared with.
#[derive(Clone, Debug)]
pub struct FnSig {
    pub id: FuncId,
    pub params: Vec<Ty>,
    pub ret: Ty,
}

/// State of the function currently being lowered.
///
/// The builder is passed separately (see module docs); `sigs` is the whole
/// unit's table, so calls resolve without touching the module state.
pub struct FnLowerer<'s, 'm> {
    module: &'m mut JITModule,
    sigs: &'s [(String, FnSig)],
    /// Local slot → Cranelift variable.
    vars: Vec<Variable>,
    /// Local slot → TS type (constants/masks only; no lookup needed).
    slot_tys: Vec<Ty>,
    /// Innermost `(continue_target, break_target)` loop headers.
    loops: Vec<(ir::Block, ir::Block)>,
    ret: Ty,
}

impl<'s, 'm> FnLowerer<'s, 'm> {
    /// Declare an exported function's signature in the module, without a body.
    pub fn declare(
        module: &mut JITModule,
        name: &str,
        params: &[Ty],
        ret: Ty,
    ) -> Result<FuncId, String> {
        let mut sig = module.make_signature();
        for p in params {
            sig.params.push(AbiParam::new(p.clif()));
        }
        if ret != Ty::Void {
            sig.returns.push(AbiParam::new(ret.clif()));
        }
        module
            .declare_function(name, Linkage::Export, &sig)
            .map_err(|e| e.to_string())
    }

    /// Declare an imported host function.
    pub fn declare_import(
        module: &mut JITModule,
        name: &str,
        params: &[Ty],
        ret: Ty,
    ) -> Result<FuncId, String> {
        let mut sig = module.make_signature();
        for p in params {
            sig.params.push(AbiParam::new(p.clif()));
        }
        if ret != Ty::Void {
            sig.returns.push(AbiParam::new(ret.clif()));
        }
        module
            .declare_function(name, Linkage::Import, &sig)
            .map_err(|e| e.to_string())
    }

    /// Emit the body of `f` into the function already declared as `sig.id`.
    pub fn define(
        module: &mut JITModule,
        sigs: &[(String, FnSig)],
        fctx: &mut FunctionBuilderContext,
        ctx: &mut cranelift_codegen::Context,
        target: TargetFrontendConfig,
        f: &FnDef,
        sig: FnSig,
    ) -> Result<(), String> {
        module.clear_context(ctx);
        let mut clif_sig = module.make_signature();
        for p in &f.params {
            clif_sig.params.push(AbiParam::new(p.clif()));
        }
        if f.ret != Ty::Void {
            clif_sig.returns.push(AbiParam::new(f.ret.clif()));
        }
        ctx.func.signature = clif_sig;

        let mut builder = FunctionBuilder::new(&mut ctx.func, fctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let mut vars = Vec::with_capacity(f.locals.len());
        for (i, local) in f.locals.iter().enumerate() {
            let var = builder.declare_var(local.ty.clif());
            if i < f.params.len() {
                let val = builder.block_params(entry)[i];
                builder.def_var(var, val);
            }
            vars.push(var);
        }

        let mut lowerer = FnLowerer {
            module,
            sigs,
            vars,
            slot_tys: f.locals.iter().map(|l| l.ty).collect(),
            loops: Vec::new(),
            ret: f.ret,
        };
        for stmt in &f.body {
            lowerer.stmt(&mut builder, stmt);
        }
        lowerer.default_return(&mut builder);
        builder.seal_all_blocks();
        builder.finalize(target);

        // NOTE: `clear_context` must run *after* `define_function` — the
        // function lives in `ctx`, and clearing it first would wipe it.
        module
            .define_function(sig.id, ctx)
            .map_err(|e| format!("cranelift: {e:?}"))
    }

    fn default_return(&self, b: &mut FunctionBuilder<'_>) {
        if self.ret == Ty::Void {
            b.ins().return_(&[]);
        } else {
            let z = self.zero(b, self.ret);
            b.ins().return_(&[z]);
        }
    }

    fn zero(&self, b: &mut FunctionBuilder<'_>, ty: Ty) -> Value {
        match ty {
            Ty::F64 => b.ins().f64const(0.0),
            _ => b.ins().iconst(ty.clif(), 0),
        }
    }

    fn sig_of(&self, name: &str) -> Option<FnSig> {
        self.sigs
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s.clone())
    }

    /// Code after `return` is unreachable but must stay in a valid block.
    fn dead_block(b: &mut FunctionBuilder<'_>) {
        let d = b.create_block();
        b.switch_to_block(d);
        b.seal_block(d);
    }

    // ───────────────────────── statements ─────────────────────────

    fn stmt(&mut self, b: &mut FunctionBuilder<'_>, s: &Stmt) {
        match s {
            Stmt::Let { local, init, .. } => {
                let var = self.vars[*local];
                // The checker guarantees a declared type; a `let` without
                // initializer starts at the type's zero value.
                let v = match init {
                    Some(init) => self.expr(b, init),
                    None => self.zero(b, self.locals_ty(*local)),
                };
                b.def_var(var, v);
            }
            Stmt::Expr { expr, .. } => {
                self.expr(b, expr);
            }
            Stmt::Return { value, .. } => {
                match value {
                    Some(v) => {
                        let val = self.expr(b, v);
                        b.ins().return_(&[val]);
                    }
                    None => {
                        b.ins().return_(&[]);
                    }
                }
                // The current block is now filled; switch to a fresh
                // (unreachable) block so later statements stay well-formed.
                Self::dead_block(b)
            }
            Stmt::If {
                cond,
                then,
                els,
                span,
            } => {
                let _ = span;
                let c = self.expr(b, cond);
                let then_b = b.create_block();
                let else_b = b.create_block();
                let merge = b.create_block();
                b.ins().brif(c, then_b, &[], else_b, &[]);

                b.switch_to_block(then_b);
                b.seal_block(then_b);
                for s in then {
                    self.stmt(b, s);
                }
                b.ins().jump(merge, &[]);

                b.switch_to_block(else_b);
                b.seal_block(else_b);
                if let Some(els) = els {
                    for s in els {
                        self.stmt(b, s);
                    }
                }
                b.ins().jump(merge, &[]);

                b.switch_to_block(merge);
                b.seal_block(merge);
            }
            Stmt::While { cond, body, .. } => {
                let header = b.create_block();
                let body_b = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);

                b.switch_to_block(header);
                let c = self.expr(b, cond);
                b.ins().brif(c, body_b, &[], exit, &[]);

                b.switch_to_block(body_b);
                b.seal_block(body_b);
                self.loops.push((header, exit));
                for s in body {
                    self.stmt(b, s);
                }
                self.loops.pop();
                b.ins().jump(header, &[]);

                b.seal_block(header);
                b.switch_to_block(exit);
                b.seal_block(exit);
            }
            Stmt::For {
                init,
                cond,
                update,
                body,
                ..
            } => {
                for s in init {
                    self.stmt(b, s);
                }
                let header = b.create_block();
                let body_b = b.create_block();
                let update_b = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);

                b.switch_to_block(header);
                match cond {
                    Some(c) => {
                        let v = self.expr(b, c);
                        b.ins().brif(v, body_b, &[], exit, &[]);
                    }
                    None => {
                        b.ins().jump(body_b, &[]);
                    }
                }

                b.switch_to_block(body_b);
                b.seal_block(body_b);
                self.loops.push((update_b, exit));
                for s in body {
                    self.stmt(b, s);
                }
                self.loops.pop();
                b.ins().jump(update_b, &[]);

                b.switch_to_block(update_b);
                b.seal_block(update_b);
                if let Some(u) = update {
                    self.expr(b, u);
                }
                b.ins().jump(header, &[]);

                b.seal_block(header);
                b.switch_to_block(exit);
                b.seal_block(exit);
            }
            Stmt::Block { body, .. } => {
                // Names are already resolved; a block only carries statements.
                for s in body {
                    self.stmt(b, s);
                }
            }
            Stmt::Break { .. } => {
                let Some(&(_, brk)) = self.loops.last() else {
                    return;
                };
                b.ins().jump(brk, &[]);
                Self::dead_block(b);
            }
            Stmt::Continue { .. } => {
                let Some(&(cont, _)) = self.loops.last() else {
                    return;
                };
                b.ins().jump(cont, &[]);
                Self::dead_block(b);
            }
        }
    }

    /// Type of local `slot` — the checker already validated it, and the JIT
    /// never sees an ill-typed module.
    fn locals_ty(&self, slot: usize) -> Ty {
        self.slot_tys[slot]
    }

    // ───────────────────────── expressions ─────────────────────────

    /// Emit `e`, returning its machine value. `e.ty` decides the instruction
    /// family, and `Ty::Void` never reaches here.
    fn expr(&mut self, b: &mut FunctionBuilder<'_>, e: &Expr) -> Value {
        let ins = b.ins();
        match &e.kind {
            ExprKind::I32(v) => ins.iconst(types::I32, i64::from(*v)),
            ExprKind::F64(v) => ins.f64const(*v),
            ExprKind::Bool(v) => ins.iconst(types::I8, i64::from(*v)),
            ExprKind::Local(slot) => b.use_var(self.vars[*slot]),
            ExprKind::Unary { op, arg } => {
                let a = self.expr(b, arg);
                match (op, arg.ty) {
                    (UnaryOperator::UnaryPlus, _) => a,
                    (UnaryOperator::UnaryNegation, Ty::I32) => b.ins().ineg(a),
                    (UnaryOperator::UnaryNegation, Ty::F64) => b.ins().fneg(a),
                    (UnaryOperator::BitwiseNot, Ty::I32) => b.ins().bnot(a),
                    (UnaryOperator::LogicalNot, Ty::Bool) => {
                        b.ins().icmp_imm_s(IntCC::Equal, a, 0)
                    }
                    _ => unreachable!("checker rejected unary {op:?} on {}", arg.ty),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(b, lhs);
                let r = self.expr(b, rhs);
                self.arith(b, *op, l, r, lhs.ty)
            }
            ExprKind::Logical { op, lhs, rhs } => match op {
                LogicalOperator::And => {
                    let rhs_b = b.create_block();
                    let merge = b.create_block();
                    let res = b.append_block_param(merge, types::I8);
                    let l = self.expr(b, lhs);
                    b.ins()
                        .brif(l, rhs_b, &[], merge, &[BlockArg::from(l)]);
                    b.switch_to_block(rhs_b);
                    b.seal_block(rhs_b);
                    let r = self.expr(b, rhs);
                    b.ins().jump(merge, &[BlockArg::from(r)]);
                    b.switch_to_block(merge);
                    b.seal_block(merge);
                    res
                }
                LogicalOperator::Or => {
                    let rhs_b = b.create_block();
                    let merge = b.create_block();
                    let res = b.append_block_param(merge, types::I8);
                    let l = self.expr(b, lhs);
                    b.ins().brif(l, merge, &[BlockArg::from(l)], rhs_b, &[]);
                    b.switch_to_block(rhs_b);
                    b.seal_block(rhs_b);
                    let r = self.expr(b, rhs);
                    b.ins().jump(merge, &[BlockArg::from(r)]);
                    b.switch_to_block(merge);
                    b.seal_block(merge);
                    res
                }
                _ => unreachable!("checker rejected logical {op:?}"),
            },
            ExprKind::Ternary { cond, then, els } => {
                let t = self.expr(b, cond);
                let then_b = b.create_block();
                let else_b = b.create_block();
                b.ins().brif(t, then_b, &[], else_b, &[]);

                b.switch_to_block(then_b);
                b.seal_block(then_b);
                let a = self.expr(b, then);
                let merge = b.create_block();
                let res = b.append_block_param(merge, then.ty.clif());
                b.ins().jump(merge, &[BlockArg::from(a)]);

                b.switch_to_block(else_b);
                b.seal_block(else_b);
                let bb = self.expr(b, els);
                b.ins().jump(merge, &[BlockArg::from(bb)]);

                b.switch_to_block(merge);
                b.seal_block(merge);
                res
            }
            ExprKind::Assign { local, op, value } => {
                let var = self.vars[*local];
                let rhs = self.expr(b, value);
                let new = match op {
                    None => rhs,
                    Some(op) => {
                        let cur = b.use_var(var);
                        self.arith(b, *op, cur, rhs, value.ty)
                    }
                };
                b.def_var(var, new);
                new
            }
            ExprKind::Update { local, inc, prefix } => {
                let var = self.vars[*local];
                let old = b.use_var(var);
                let new = if e.ty == Ty::F64 {
                    let one = b.ins().f64const(1.0);
                    if *inc {
                        b.ins().fadd(old, one)
                    } else {
                        b.ins().fsub(old, one)
                    }
                } else {
                    let one = b.ins().iconst(types::I32, 1);
                    if *inc {
                        b.ins().iadd(old, one)
                    } else {
                        b.ins().isub(old, one)
                    }
                };
                b.def_var(var, new);
                if *prefix { new } else { old }
            }
            ExprKind::Call { target, args } => {
                let sig = match target {
                    CallTarget::Local(index) => self.sigs[*index].1.clone(),
                    CallTarget::Host(name) => self
                        .sig_of(name)
                        .expect("checker resolved every call target"),
                };
                let mut arg_vals = Vec::with_capacity(args.len());
                for (a, &pty) in args.iter().zip(&sig.params) {
                    let v = self.expr(b, a);
                    debug_assert_eq!(a.ty, pty, "call argument type");
                    arg_vals.push(v);
                }
                let fref = self.module.declare_func_in_func(sig.id, b.func);
                let call = b.ins().call(fref, &arg_vals);
                if sig.ret == Ty::Void {
                    b.ins().iconst(types::I32, 0)
                } else {
                    b.inst_results(call)[0]
                }
            }
            ExprKind::Cast { expr, to } => {
                let v = self.expr(b, expr);
                match (expr.ty, *to) {
                    (Ty::I32, Ty::F64) => b.ins().fcvt_from_sint(types::F64, v),
                    (Ty::F64, Ty::I32) => b.ins().fcvt_to_sint_sat(types::I32, v),
                    (Ty::Bool, Ty::I32) => b.ins().uextend(types::I32, v),
                    (Ty::Bool, Ty::F64) => {
                        let wide = b.ins().uextend(types::I32, v);
                        b.ins().fcvt_from_sint(types::F64, wide)
                    }
                    _ => unreachable!("checker rejected cast {} -> {}", expr.ty, to),
                }
            }
        }
    }

    /// Binary op on values of a known operand type.
    fn arith(
        &mut self,
        b: &mut FunctionBuilder<'_>,
        op: BinaryOperator,
        l: Value,
        r: Value,
        ty: Ty,
    ) -> Value {
        let ins = b.ins();
        match ty {
            Ty::I32 => {
                let cc = match op {
                    BinaryOperator::Equality | BinaryOperator::StrictEquality => {
                        Some(IntCC::Equal)
                    }
                    BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                        Some(IntCC::NotEqual)
                    }
                    BinaryOperator::LessThan => Some(IntCC::SignedLessThan),
                    BinaryOperator::LessEqualThan => Some(IntCC::SignedLessThanOrEqual),
                    BinaryOperator::GreaterThan => Some(IntCC::SignedGreaterThan),
                    BinaryOperator::GreaterEqualThan => Some(IntCC::SignedGreaterThanOrEqual),
                    _ => None,
                };
                if let Some(cc) = cc {
                    ins.icmp(cc, l, r)
                } else {
                    match op {
                        BinaryOperator::Addition => ins.iadd(l, r),
                        BinaryOperator::Subtraction => ins.isub(l, r),
                        BinaryOperator::Multiplication => ins.imul(l, r),
                        BinaryOperator::Division => ins.sdiv(l, r),
                        BinaryOperator::Remainder => ins.srem(l, r),
                        BinaryOperator::BitwiseAnd => ins.band(l, r),
                        BinaryOperator::BitwiseOR => ins.bor(l, r),
                        BinaryOperator::BitwiseXOR => ins.bxor(l, r),
                        BinaryOperator::ShiftLeft => ins.ishl(l, r),
                        BinaryOperator::ShiftRight => ins.sshr(l, r),
                        BinaryOperator::ShiftRightZeroFill => ins.ushr(l, r),
                        _ => unreachable!("checker rejected {op:?} on i32"),
                    }
                }
            }
            Ty::F64 => {
                let cc = match op {
                    BinaryOperator::Equality | BinaryOperator::StrictEquality => {
                        Some(FloatCC::Equal)
                    }
                    BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                        Some(FloatCC::NotEqual)
                    }
                    BinaryOperator::LessThan => Some(FloatCC::LessThan),
                    BinaryOperator::LessEqualThan => Some(FloatCC::LessThanOrEqual),
                    BinaryOperator::GreaterThan => Some(FloatCC::GreaterThan),
                    BinaryOperator::GreaterEqualThan => Some(FloatCC::GreaterThanOrEqual),
                    _ => None,
                };
                if let Some(cc) = cc {
                    ins.fcmp(cc, l, r)
                } else {
                    match op {
                        BinaryOperator::Addition => ins.fadd(l, r),
                        BinaryOperator::Subtraction => ins.fsub(l, r),
                        BinaryOperator::Multiplication => ins.fmul(l, r),
                        BinaryOperator::Division => ins.fdiv(l, r),
                        _ => unreachable!("checker rejected {op:?} on f64"),
                    }
                }
            }
            Ty::Bool => {
                let cc = match op {
                    BinaryOperator::Equality | BinaryOperator::StrictEquality => IntCC::Equal,
                    BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                        IntCC::NotEqual
                    }
                    _ => unreachable!("checker rejected {op:?} on boolean"),
                };
                ins.icmp(cc, l, r)
            }
            Ty::Void => unreachable!("void has no operands"),
        }
    }
}
