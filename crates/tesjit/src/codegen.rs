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

use cranelift_codegen::ir::MachMemFlags;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    self, AbiParam, BlockArg, InstBuilder, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Linkage, Module};

use oxc_ast::ast::{BinaryOperator, LogicalOperator, UnaryOperator};
use std::collections::HashMap;

/// Plain memory access (no alias-region hints — the arrays are freshly
/// allocated and never alias a Rust object).
const fn mem_flags() -> MachMemFlags {
    MachMemFlags::new()
}

use crate::hir::{CallTarget, Expr, ExprKind, FnDef, Intrinsic, PlaceExpr, Stmt};
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
    /// Already-materialized string literals, keyed by their Rust text.
    rodata: std::collections::HashMap<String, cranelift_module::DataId>,
    /// Local slot → TS type (constants/masks only; no lookup needed).
    slot_tys: Vec<Ty>,
    /// Innermost `(continue_target, break_target)` loop headers.
    loops: Vec<(ir::Block, ir::Block)>,
    /// Block the function jumps to when a runtime guard trips (fuel, traps):
    /// it returns the function's zero value instead of unwinding into the host.
    trap_block: ir::Block,
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

        // Sandbox guard: every function starts with a fuel check and jumps to a
        // dedicated block that returns the zero value. A runaway mod therefore
        // stays inside its own frame — the host sees a normal return plus a
        // trap, never an unwind into Rust.
        //
        // `trap_block` is only *created* here and filled afterwards: blocks are
        // laid out in fill order, and the entry block (with the parameters)
        // must stay first.
        let trap_block = builder.create_block();
        let fuel = Self::fuel_check(module, sigs, &mut builder);
        let body_block = builder.create_block();
        builder.ins().brif(fuel, body_block, &[], trap_block, &[]);
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);

        builder.switch_to_block(trap_block);
        if f.ret != Ty::Void {
            let zero = match f.ret.clone() {
                Ty::F32 => builder.ins().f32const(0.0),
                Ty::F64 => builder.ins().f64const(0.0),
                other => builder.ins().iconst(other.clif(), 0),
            };
            builder.ins().return_(&[zero]);
        } else {
            builder.ins().return_(&[]);
        }
        // NOT sealed: loops branch into `trap_block` (declaring predecessors)
        // long after this point; `seal_all_blocks` at the end covers it.
        builder.switch_to_block(body_block);

        let mut lowerer = FnLowerer {
            module,
            sigs,
            vars,
            rodata: HashMap::new(),
            trap_block,
            slot_tys: f.locals.iter().map(|l| l.ty.clone()).collect(),
            loops: Vec::new(),
            ret: f.ret.clone(),
        };

        for stmt in &f.body {
            lowerer.stmt(&mut builder, stmt);
        }
        lowerer.default_return(&mut builder);
        builder.seal_all_blocks();
        builder.finalize(target);

        if std::env::var_os("TESJIT_DUMP_IR").is_some() {
            eprintln!("--- {} ---\n{}", f.name, ctx.func.display());
        }
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
            let z = self.zero(b, self.ret.clone());
            b.ins().return_(&[z]);
        }
    }

    fn zero(&self, b: &mut FunctionBuilder<'_>, ty: Ty) -> Value {
        match ty {
            Ty::F32 => b.ins().f32const(0.0),
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
                let cond_b = b.create_block();
                let exit = b.create_block();
                b.ins().jump(header, &[]);

                b.switch_to_block(header);
                // Each iteration costs fuel; when it runs out the loop jumps to
                // the function's trap block instead of spinning.
                let ok = self.guard_check(b);
                b.ins().brif(ok, cond_b, &[], self.trap_block, &[]);
                b.switch_to_block(cond_b);
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
                // Same fuel accounting as `while`: the guard branches to a
                // fresh test block, which then decides the next iteration.
                let ok = self.guard_check(b);
                let test_b = b.create_block();
                b.ins().brif(ok, test_b, &[], self.trap_block, &[]);
                b.switch_to_block(test_b);
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
            Stmt::ForOf { .. } => {}
            Stmt::QueryFor {
                query,
                bindings,
                terms,
                body,
                ..
            } => self.query_for(b, *query, bindings, terms, body),
        }
    }

    fn query_for(
        &mut self,
        b: &mut FunctionBuilder<'_>,
        query: usize,
        bindings: &[usize],
        terms: &[Ty],
        body: &[Stmt],
    ) {
        let input = b.use_var(self.vars[query]);
        let chunks = b.ins().load(types::I64, mem_flags(), input, 0);
        let chunk_count_ptr = b.ins().iadd_imm_s(input, 8);
        let chunk_count = b.ins().load(types::I64, mem_flags(), chunk_count_ptr, 0);

        let chunk_index = b.declare_var(types::I64);
        let row_index = b.declare_var(types::I64);
        let zero = b.ins().iconst(types::I64, 0);
        b.def_var(chunk_index, zero);

        let chunk_header = b.create_block();
        let chunk_body = b.create_block();
        let row_header = b.create_block();
        let row_body = b.create_block();
        let row_next = b.create_block();
        let chunk_next = b.create_block();
        let exit = b.create_block();
        b.ins().jump(chunk_header, &[]);

        b.switch_to_block(chunk_header);
        let chunk_i = b.use_var(chunk_index);
        let has_chunk = b.ins().icmp(IntCC::UnsignedLessThan, chunk_i, chunk_count);
        b.ins().brif(has_chunk, chunk_body, &[], exit, &[]);

        b.switch_to_block(chunk_body);
        b.seal_block(chunk_body);
        let chunk_i = b.use_var(chunk_index);
        let desc_stride = b
            .ins()
            .iconst(types::I64, (crate::abi::desc_words(terms.len()) * 8) as i64);
        let desc_offset = b.ins().imul(chunk_i, desc_stride);
        let desc = b.ins().iadd(chunks, desc_offset);
        let rows = b.ins().load(types::I64, mem_flags(), desc, 0);
        let fuel_ok = self.fuel_check_n(b, rows);

        let mut column_vars = Vec::with_capacity(terms.len());
        let mut strides = Vec::with_capacity(terms.len());
        for (index, term) in terms.iter().enumerate() {
            let col_offset = b.ins().iconst(types::I64, ((index + 1) * 8) as i64);
            let col_addr = b.ins().iadd(desc, col_offset);
            let col_ptr = b.ins().load(types::I64, mem_flags(), col_addr, 0);
            let col_var = b.declare_var(types::I64);
            b.def_var(col_var, col_ptr);
            column_vars.push(col_var);
            strides.push(if matches!(term, Ty::Entity) {
                8
            } else {
                term.struct_layout()
                    .expect("query component has a registered layout")
                    .size
            });
        }
        let zero = b.ins().iconst(types::I64, 0);
        b.def_var(row_index, zero);
        b.ins().brif(fuel_ok, row_header, &[], self.trap_block, &[]);

        b.switch_to_block(row_header);
        let row_i = b.use_var(row_index);
        let has_row = b.ins().icmp(IntCC::UnsignedLessThan, row_i, rows);
        b.ins().brif(has_row, row_body, &[], chunk_next, &[]);

        b.switch_to_block(row_body);
        b.seal_block(row_body);
        for ((binding, term), col_var) in bindings.iter().zip(terms).zip(&column_vars) {
            let ptr = b.use_var(*col_var);
            let value = if matches!(term, Ty::Entity) {
                b.ins().load(types::I64, mem_flags(), ptr, 0)
            } else {
                ptr
            };
            b.def_var(self.vars[*binding], value);
        }
        self.loops.push((row_next, exit));
        for stmt in body {
            self.stmt(b, stmt);
        }
        self.loops.pop();
        b.ins().jump(row_next, &[]);

        b.switch_to_block(row_next);
        b.seal_block(row_next);
        let row_i = b.use_var(row_index);
        let one = b.ins().iconst(types::I64, 1);
        let next_row = b.ins().iadd(row_i, one);
        b.def_var(row_index, next_row);
        for (col_var, stride) in column_vars.iter().zip(strides) {
            let ptr = b.use_var(*col_var);
            let increment = b.ins().iconst(types::I64, stride as i64);
            let next_ptr = b.ins().iadd(ptr, increment);
            b.def_var(*col_var, next_ptr);
        }
        b.ins().jump(row_header, &[]);

        b.switch_to_block(chunk_next);
        b.seal_block(chunk_next);
        let chunk_i = b.use_var(chunk_index);
        let one = b.ins().iconst(types::I64, 1);
        let next_chunk = b.ins().iadd(chunk_i, one);
        b.def_var(chunk_index, next_chunk);
        b.ins().jump(chunk_header, &[]);

        b.seal_block(row_header);
        b.seal_block(chunk_header);
        b.switch_to_block(exit);
        b.seal_block(exit);
    }

    /// Type of local `slot` — the checker already validated it, and the JIT
    /// never sees an ill-typed module.
    fn locals_ty(&self, slot: usize) -> Ty {
        self.slot_tys[slot].clone()
    }

    // ───────────────────────── expressions ─────────────────────────

    /// Emit `e`, returning its machine value. `e.ty` decides the instruction
    /// family, and `Ty::Void` never reaches here.
    fn expr(&mut self, b: &mut FunctionBuilder<'_>, e: &Expr) -> Value {
        let ins = b.ins();
        match &e.kind {
            ExprKind::Int(v) => ins.iconst(e.ty.clif(), *v),
            ExprKind::Float(v) => {
                if e.ty == Ty::F32 {
                    ins.f32const(*v as f32)
                } else {
                    ins.f64const(*v)
                }
            }
            ExprKind::Bool(v) => ins.iconst(types::I8, i64::from(*v)),
            ExprKind::Str(text) => self.str_literal(b, text),
            ExprKind::Local(slot) => b.use_var(self.vars[*slot]),
            ExprKind::Unary { op, arg } => {
                let a = self.expr(b, arg);
                let ty = arg.ty.clone();
                match op {
                    UnaryOperator::UnaryPlus => a,
                    UnaryOperator::UnaryNegation if ty.is_float() => b.ins().fneg(a),
                    UnaryOperator::UnaryNegation if ty.is_integer() => b.ins().ineg(a),
                    UnaryOperator::BitwiseNot if ty.is_integer() => b.ins().bnot(a),
                    UnaryOperator::LogicalNot if ty == Ty::Bool => {
                        b.ins().icmp_imm_s(IntCC::Equal, a, 0)
                    }
                    _ => unreachable!("checker rejected unary {op:?} on {ty}"),
                }
            }
            ExprKind::Binary { op, lhs, rhs } => {
                let l = self.expr(b, lhs);
                let r = self.expr(b, rhs);
                self.arith(b, *op, l, r, lhs.ty.clone())
            }
            ExprKind::Logical { op, lhs, rhs } => match op {
                LogicalOperator::And => {
                    let rhs_b = b.create_block();
                    let merge = b.create_block();
                    let res = b.append_block_param(merge, types::I8);
                    let l = self.expr(b, lhs);
                    b.ins().brif(l, rhs_b, &[], merge, &[BlockArg::from(l)]);
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
                        self.arith(b, *op, cur, rhs, value.ty.clone())
                    }
                };
                b.def_var(var, new);
                new
            }
            ExprKind::Update { local, inc, prefix } => {
                let var = self.vars[*local];
                let old = b.use_var(var);
                let new = if e.ty.is_float() {
                    let one = if e.ty == Ty::F32 {
                        b.ins().f32const(1.0)
                    } else {
                        b.ins().f64const(1.0)
                    };
                    if *inc {
                        b.ins().fadd(old, one)
                    } else {
                        b.ins().fsub(old, one)
                    }
                } else {
                    let one = b.ins().iconst(e.ty.clif(), 1);
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
                    // Intrinsics are declared as runtime imports by `Jit`.
                    CallTarget::Intrinsic(intrinsic) => self
                        .sig_of(intrinsic.symbol())
                        .expect("runtime helper declared by Jit"),
                };
                // The runtime ABI is machine words (`i64`: pointers, lengths,
                // sizes), so arguments and results are adapted between the TS
                // type at the call site and the helper signature.
                let intrinsic = matches!(target, CallTarget::Intrinsic(_));
                let mut arg_vals = Vec::with_capacity(args.len());
                for (a, pty) in args.iter().zip(&sig.params) {
                    let v = self.expr(b, a);
                    if intrinsic {
                        arg_vals.push(Self::adapt(b, &a.ty, pty, v));
                    } else {
                        debug_assert_eq!(&a.ty, pty, "call argument type");
                        arg_vals.push(v);
                    }
                }
                let fref = self.module.declare_func_in_func(sig.id, b.func);
                let call = b.ins().call(fref, &arg_vals);
                let result = if sig.ret == Ty::Void {
                    b.ins().iconst(types::I32, 0)
                } else {
                    b.inst_results(call)[0]
                };
                if intrinsic {
                    Self::adapt(b, &sig.ret, &e.ty, result)
                } else {
                    result
                }
            }
            ExprKind::Cast { expr, to } => {
                let v = self.expr(b, expr);
                let from = expr.ty.clone();
                if &from == to {
                    v
                } else {
                    self.convert(b, v, &from, to)
                }
            }
            ExprKind::Length { target } => match target.ty.clone() {
                Ty::Str => {
                    let s = self.expr(b, target);
                    self.call_intrinsic(b, Intrinsic::StrLen, &[(&Ty::Str, s)], &Ty::I32)
                }
                Ty::Array(_) => {
                    let a = self.expr(b, target);
                    self.call_intrinsic(b, Intrinsic::ArrLen, &[(&Ty::Ptr, a)], &Ty::I32)
                }
                other => unreachable!("no length on {other}"),
            },
            ExprKind::Index { target, index } => match target.ty.elem() {
                Some(elem) => {
                    // Bounds-checked read through the runtime: a bad index
                    // traps instead of touching foreign memory.
                    let base = self.expr(b, target);
                    let raw = self.expr(b, index);
                    let i = Self::adapt(b, &index.ty, &Ty::Ptr, raw);
                    let size = elem.elem_size().expect("element size") as i64;
                    let size_v = b.ins().iconst(types::I64, size);
                    self.call_intrinsic(
                        b,
                        Intrinsic::ArrGet,
                        &[(&Ty::Ptr, base), (&Ty::Ptr, i), (&Ty::Ptr, size_v)],
                        &elem,
                    )
                }
                None if target.ty == Ty::Str => {
                    // Strings expose bytes; `str_byte` traps on out-of-bounds.
                    let s = self.expr(b, target);
                    let i = self.expr(b, index);
                    self.call_intrinsic(
                        b,
                        Intrinsic::StrByte,
                        &[(&Ty::Str, s), (&index.ty, i)],
                        &Ty::I32,
                    )
                }
                None => unreachable!("cannot index {}", target.ty),
            },
            ExprKind::SetIndex {
                target,
                index,
                value,
            } => {
                let elem = target
                    .ty
                    .elem()
                    .expect("checker only allows indexed assignment to arrays");
                let base = self.expr(b, target);
                let raw = self.expr(b, index);
                let i = Self::adapt(b, &index.ty, &Ty::Ptr, raw);
                let v = self.expr(b, value);
                let size = elem.elem_size().expect("element size") as i64;
                let size_v = b.ins().iconst(types::I64, size);
                self.call_intrinsic(
                    b,
                    Intrinsic::ArrSet,
                    &[
                        (&Ty::Ptr, base),
                        (&Ty::Ptr, i),
                        (&Ty::Ptr, size_v),
                        (&value.ty, v),
                    ],
                    &elem,
                )
            }
            ExprKind::ArrayLit { elems, elem_ty } => {
                let size = elem_ty.elem_size().expect("element size");
                // Allocate room for every element, then push them one by one:
                // the runtime maintains len/cap and grows on demand.
                let cap_v = b.ins().iconst(types::I64, elems.len().max(1) as i64);
                let size_v = b.ins().iconst(types::I64, size as i64);
                let mut arr = self.call_intrinsic(
                    b,
                    Intrinsic::ArrNew,
                    &[(&Ty::Ptr, cap_v), (&Ty::Ptr, size_v)],
                    &Ty::Array(Box::new(Ty::I32)),
                );
                for (i, el) in elems.iter().enumerate() {
                    let v = self.expr(b, el);
                    let index_v = b.ins().iconst(types::I64, i as i64);
                    let elem_ptr = self.elem_addr(b, arr, index_v, size);
                    b.ins().store(mem_flags(), v, elem_ptr, 0);
                    // Keep len in sync with the stored slots.
                    let len_v = b.ins().iconst(types::I64, (i + 1) as i64);
                    let header = b.ins().iadd_imm_s(arr, 0);
                    b.ins().store(mem_flags(), len_v, header, 0);
                }
                let _ = &mut arr;
                arr
            }
            ExprKind::StructLit { layout, fields } => {
                let slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    layout.size.max(1) as u32,
                    layout.align.trailing_zeros() as u8,
                ));
                let ptr = b.ins().stack_addr(types::I64, slot, 0);
                for (offset, value) in fields {
                    let value_ty = value.ty.clone();
                    let value_ptr = self.offset_pointer(b, ptr, *offset);
                    let value = self.expr(b, value);
                    if let Some(nested) = value_ty.struct_layout() {
                        self.copy_struct_fields(b, value, value_ptr, nested);
                    } else {
                        b.ins().store(mem_flags(), value, value_ptr, 0);
                    }
                }
                ptr
            }
            ExprKind::CommandsSpawn {
                commands,
                components,
            } => {
                let descriptor_size =
                    (components.len() * std::mem::size_of::<crate::abi::SpawnComponent>()) as u32;
                let descriptor_slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    descriptor_size.max(1),
                    3,
                ));
                let descriptor_ptr = b.ins().stack_addr(types::I64, descriptor_slot, 0);
                for (index, component) in components.iter().enumerate() {
                    let layout = component
                        .ty
                        .struct_layout()
                        .expect("checker validated spawn component");
                    let source_ptr = self.expr(b, component);
                    let component_slot = b.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        layout.size.max(1) as u32,
                        layout.align.trailing_zeros() as u8,
                    ));
                    let component_ptr = b.ins().stack_addr(types::I64, component_slot, 0);
                    self.zero_struct_padding(b, component_ptr, layout);
                    self.copy_struct_fields(b, source_ptr, component_ptr, layout);
                    let base_offset =
                        (index * std::mem::size_of::<crate::abi::SpawnComponent>()) as u64;
                    let fields = [
                        layout.component_id.expect("checker validated component id") as u64,
                        layout.size,
                        layout.align,
                    ];
                    for (field_index, value) in fields.into_iter().enumerate() {
                        let offset = base_offset + (field_index * 8) as u64;
                        let field_ptr = self.offset_pointer(b, descriptor_ptr, offset);
                        let value = b.ins().iconst(types::I64, value as i64);
                        b.ins().store(mem_flags(), value, field_ptr, 0);
                    }
                    let data_ptr = self.offset_pointer(b, descriptor_ptr, base_offset + 24);
                    b.ins().store(mem_flags(), component_ptr, data_ptr, 0);
                }
                let input_slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    std::mem::size_of::<crate::abi::SpawnInput>() as u32,
                    3,
                ));
                let input_ptr = b.ins().stack_addr(types::I64, input_slot, 0);
                let count = b.ins().iconst(types::I64, components.len() as i64);
                b.ins().store(mem_flags(), count, input_ptr, 0);
                let components_addr = self.offset_pointer(b, input_ptr, 8);
                b.ins()
                    .store(mem_flags(), descriptor_ptr, components_addr, 0);
                let commands = self.expr(b, commands);
                let sig = self
                    .sig_of("__tesjit_commands_spawn")
                    .expect("Commands host function registered");
                let fref = self.module.declare_func_in_func(sig.id, b.func);
                b.ins().call(fref, &[commands, input_ptr]);
                b.ins().iconst(types::I32, 0)
            }
            ExprKind::OptionNone { ty: _ } => {
                use crate::hir::Intrinsic;
                let size = b.ins().iconst(types::I64, 16);
                self.call_intrinsic(b, Intrinsic::Alloc, &[(&Ty::Ptr, size)], &Ty::Ptr)
            }
            ExprKind::OptionSome { value } => {
                use crate::hir::Intrinsic;
                let is_float = value.ty.is_float();
                let ptr = {
                    let size = b.ins().iconst(types::I64, 16);
                    self.call_intrinsic(b, Intrinsic::Alloc, &[(&Ty::Ptr, size)], &Ty::Ptr)
                };
                let tag = b.ins().iconst(types::I64, 1);
                b.ins().store(mem_flags(), tag, ptr, 0);
                let v = self.expr(b, value);
                if is_float {
                    let v = if value.ty == Ty::F64 {
                        v
                    } else {
                        self.convert(b, v, &Ty::F32, &Ty::F64)
                    };
                    b.ins().store(mem_flags(), v, ptr, 8);
                } else {
                    let v = if value.ty == Ty::I64 {
                        v
                    } else {
                        self.convert(b, v, &value.ty, &Ty::I64)
                    };
                    b.ins().store(mem_flags(), v, ptr, 8);
                }
                ptr
            }
            ExprKind::PlaceRead { place } => {
                let ptr = self.place_pointer(b, place);
                if matches!(e.ty, Ty::Struct(_) | Ty::OwnedStruct(_)) {
                    ptr
                } else {
                    b.ins().load(e.ty.clif(), mem_flags(), ptr, 0)
                }
            }
            ExprKind::PlaceWrite { place, op, value } => {
                let ptr = self.place_pointer(b, place);
                let mut value = self.expr(b, value);
                if let Some(op) = op {
                    let current = b.ins().load(e.ty.clif(), mem_flags(), ptr, 0);
                    value = self.arith(b, *op, current, value, e.ty.clone());
                }
                b.ins().store(mem_flags(), value, ptr, 0);
                value
            }
            ExprKind::Iter { .. } => {
                unreachable!("query iteration is not lowered yet")
            }
        }
    }

    fn place_pointer(&self, b: &mut FunctionBuilder<'_>, place: &PlaceExpr) -> Value {
        let (base, offset) = match place {
            PlaceExpr::Field { base, offset } => (*base, *offset),
            PlaceExpr::Slot(_) | PlaceExpr::Index { .. } => {
                unreachable!("unsupported component place")
            }
        };
        let ptr = b.use_var(self.vars[base]);
        if offset == 0 {
            ptr
        } else {
            let byte_offset = b.ins().iconst(types::I64, offset as i64);
            b.ins().iadd(ptr, byte_offset)
        }
    }

    fn offset_pointer(&self, b: &mut FunctionBuilder<'_>, ptr: Value, offset: u64) -> Value {
        if offset == 0 {
            ptr
        } else {
            let offset = b.ins().iconst(types::I64, offset as i64);
            b.ins().iadd(ptr, offset)
        }
    }

    fn copy_struct_fields(
        &mut self,
        b: &mut FunctionBuilder<'_>,
        source: Value,
        target: Value,
        layout: &crate::ty::StructLayout,
    ) {
        for field in &layout.fields {
            let source_ptr = self.offset_pointer(b, source, field.offset);
            let target_ptr = self.offset_pointer(b, target, field.offset);
            if let Some(nested) = field.ty.struct_layout() {
                self.copy_struct_fields(b, source_ptr, target_ptr, nested);
            } else {
                let value = b.ins().load(field.ty.clif(), mem_flags(), source_ptr, 0);
                b.ins().store(mem_flags(), value, target_ptr, 0);
            }
        }
    }

    fn zero_struct_padding(
        &self,
        b: &mut FunctionBuilder<'_>,
        ptr: Value,
        layout: &crate::ty::StructLayout,
    ) {
        let mut fields: Vec<_> = layout.fields.iter().collect();
        fields.sort_by_key(|field| field.offset);
        let mut end = 0;
        for field in fields {
            self.zero_bytes(b, ptr, end, field.offset);
            let field_ptr = self.offset_pointer(b, ptr, field.offset);
            if let Some(nested) = field.ty.struct_layout() {
                self.zero_struct_padding(b, field_ptr, nested);
            }
            let (size, _) = field
                .ty
                .value_layout()
                .expect("checker validated component field layout");
            end = field.offset + size;
        }
        self.zero_bytes(b, ptr, end, layout.size);
    }

    fn zero_bytes(&self, b: &mut FunctionBuilder<'_>, ptr: Value, start: u64, end: u64) {
        let zero = b.ins().iconst(types::I8, 0);
        for offset in start..end {
            let byte_ptr = self.offset_pointer(b, ptr, offset);
            b.ins().store(mem_flags(), zero, byte_ptr, 0);
        }
    }

    /// Emit the fuel guard call; true while budget remains.
    ///
    /// Called on function entry (before the `FnLowerer` exists) and on every
    /// loop back-edge, so it does not depend on instance state.
    fn fuel_check(
        module: &mut JITModule,
        sigs: &[(String, FnSig)],
        b: &mut FunctionBuilder<'_>,
    ) -> Value {
        let sig = sigs
            .iter()
            .find(|(name, _)| name == Intrinsic::Fuel.symbol())
            .map(|(_, sig)| sig.clone())
            .expect("fuel helper declared by Jit");
        let fref = module.declare_func_in_func(sig.id, b.func);
        let call = b.ins().call(fref, &[]);
        Self::adapt(b, &sig.ret, &Ty::Bool, b.inst_results(call)[0])
    }

    /// Loop/recursion back-edge guard: branch to the function's trap block
    /// when the budget is spent.
    fn guard_check(&mut self, b: &mut FunctionBuilder<'_>) -> Value {
        Self::fuel_check(self.module, self.sigs, b)
    }

    fn fuel_check_n(&mut self, b: &mut FunctionBuilder<'_>, rows: Value) -> Value {
        let sig = self
            .sigs
            .iter()
            .find(|(name, _)| name == Intrinsic::FuelN.symbol())
            .map(|(_, sig)| sig.clone())
            .expect("chunk fuel helper declared by Jit");
        let fref = self.module.declare_func_in_func(sig.id, b.func);
        let call = b.ins().call(fref, &[rows]);
        Self::adapt(b, &sig.ret, &Ty::Bool, b.inst_results(call)[0])
    }

    /// Explicit `as` conversion. The checker guarantees the pair is legal;
    /// every combination below is total (wrapping integers, saturating
    /// float↔int, zero-extension from `bool`).
    fn convert(&mut self, b: &mut FunctionBuilder<'_>, v: Value, from: &Ty, to: &Ty) -> Value {
        debug_assert_ne!(from, to);
        match (from, to) {
            // Integer ↔ integer: extend or truncate. Signedness only matters
            // for widening (sign- vs zero-extend); narrowing wraps.
            (f, t) if f.is_integer() && t.is_integer() => {
                let (fw, tw) = (f.int_width().unwrap_or(32), t.int_width().unwrap_or(32));
                if fw == tw {
                    v
                } else if tw < fw {
                    b.ins().ireduce(to.clif(), v)
                } else if f.is_signed() {
                    b.ins().sextend(to.clif(), v)
                } else {
                    b.ins().uextend(to.clif(), v)
                }
            }
            // Integer → float. Narrower widths extend to 64 bits first;
            // the conversion itself always runs from `i64`.
            (f, t) if f.is_integer() && t.is_float() => {
                let wide = match f.int_width().unwrap_or(32) {
                    64 => v,
                    _ if f.is_signed() => b.ins().sextend(types::I64, v),
                    _ => b.ins().uextend(types::I64, v),
                };
                if f.is_signed() {
                    if *to == Ty::F32 {
                        b.ins().fcvt_from_sint(types::F32, wide)
                    } else {
                        b.ins().fcvt_from_sint(types::F64, wide)
                    }
                } else if *to == Ty::F32 {
                    b.ins().fcvt_from_uint(types::F32, wide)
                } else {
                    b.ins().fcvt_from_uint(types::F64, wide)
                }
            }
            // Float → integer. Saturates to the 32/64-bit lane (`_sat`,
            // NaN → 0), then wraps on narrowing — documented, not silent:
            // `1000.0 as i8` is `-24`, matching integer narrowing.
            (f, t) if f.is_float() && t.is_integer() => {
                if t.is_signed() {
                    if t.int_width().unwrap_or(32) <= 32 {
                        let w = b.ins().fcvt_to_sint_sat(types::I32, v);
                        if *to == Ty::I32 {
                            w
                        } else {
                            b.ins().ireduce(to.clif(), w)
                        }
                    } else {
                        let w = b.ins().fcvt_to_sint_sat(types::I64, v);
                        if *to == Ty::I64 {
                            w
                        } else {
                            b.ins().ireduce(to.clif(), w)
                        }
                    }
                } else if t.int_width().unwrap_or(32) <= 32 {
                    let w = b.ins().fcvt_to_uint_sat(types::I32, v);
                    if *to == Ty::U32 {
                        w
                    } else {
                        b.ins().ireduce(to.clif(), w)
                    }
                } else {
                    let w = b.ins().fcvt_to_uint_sat(types::I64, v);
                    if *to == Ty::U64 {
                        w
                    } else {
                        b.ins().ireduce(to.clif(), w)
                    }
                }
            }

            // `f32` ↔ `f64`.
            (Ty::F32, Ty::F64) => b.ins().fpromote(types::F64, v),
            (Ty::F64, Ty::F32) => b.ins().fdemote(types::F32, v),
            // `bool` → numeric zero-extends.
            (Ty::Bool, t) if t.is_numeric() => {
                let wide = b.ins().uextend(types::I32, v);
                if t.is_integer() {
                    match t.int_width().unwrap_or(32) {
                        32 => wide,
                        w if w < 32 => b.ins().ireduce(to.clif(), wide),
                        _ => b.ins().uextend(to.clif(), wide),
                    }
                } else if *to == Ty::F32 {
                    b.ins().fcvt_from_uint(types::F32, wide)
                } else {
                    b.ins().fcvt_from_uint(types::F64, wide)
                }
            }
            // `Entity` ↔ `u64`: identical bits.
            (Ty::Entity, Ty::U64) | (Ty::U64, Ty::Entity) => v,
            _ => unreachable!("checker rejected cast {from} -> {to}"),
        }
    }

    /// Convert a value between a TS type and the machine-word runtime ABI
    /// (used only for intrinsic calls): any integer/boolean/float ↔ `i64`.
    ///
    /// Array/string helpers move values as raw little-endian words
    /// (`i64::from_le_bytes` on the Rust side), so floats cross by *bitcast*,
    /// never by numeric conversion: an `f64` keeps its bits, an `f32` is
    /// zero-extended like every other 4-byte element.
    fn adapt(b: &mut FunctionBuilder<'_>, from: &Ty, to: &Ty, v: Value) -> Value {
        if from == to || from.clif() == to.clif() {
            return v;
        }
        match (from, to) {
            // Widening: zero-extend, so a negative index stays a huge unsigned
            // value and the runtime bounds check traps on it.
            (f, Ty::Ptr) if f.is_integer() || *f == Ty::Bool => b.ins().uextend(types::I64, v),
            (Ty::Ptr, t) if t.is_integer() || *t == Ty::Bool => b.ins().ireduce(to.clif(), v),
            // Floats cross as words: exact-size bitcast, zero-extended words
            // for `f32` (matching `arr_get`'s zero-padded read).
            (Ty::F64, Ty::Ptr) => b.ins().bitcast(types::I64, mem_flags(), v),
            (Ty::Ptr, Ty::F64) => b.ins().bitcast(types::F64, mem_flags(), v),
            (Ty::F32, Ty::Ptr) => {
                let w = b.ins().bitcast(types::I32, mem_flags(), v);
                b.ins().uextend(types::I64, w)
            }
            (Ty::Ptr, Ty::F32) => {
                let w = b.ins().ireduce(types::I32, v);
                b.ins().bitcast(types::F32, mem_flags(), w)
            }
            _ => unreachable!("no adaptation between {from} and {to}"),
        }
    }

    /// `base + 16 + i * size` — element address inside an array value.
    fn elem_addr(
        &mut self,
        b: &mut FunctionBuilder<'_>,
        base: Value,
        i: Value,
        size: u64,
    ) -> Value {
        let data = b.ins().iadd_imm_s(base, 16);
        if size == 1 {
            b.ins().iadd(data, i)
        } else {
            let offset = b.ins().imul_imm_s(i, size as i64);
            b.ins().iadd(data, offset)
        }
    }

    /// Call a runtime helper, importing it into this function on demand.
    ///
    /// Arguments and the result are given as `(TS type, value)` pairs so the
    /// machine-word runtime ABI (`i64` pointers/sizes) can adapt on the way in
    /// and out.
    fn call_intrinsic(
        &mut self,
        b: &mut FunctionBuilder<'_>,
        intrinsic: Intrinsic,
        args: &[(&Ty, Value)],
        result: &Ty,
    ) -> Value {
        let name = intrinsic.symbol();
        let sig = self.sig_of(name).expect("intrinsic declared by Jit");
        let adapted: Vec<Value> = args
            .iter()
            .zip(&sig.params)
            .map(|((ty, v), pty)| {
                if ty.clif() == pty.clif() {
                    *v
                } else {
                    Self::adapt(b, ty, pty, *v)
                }
            })
            .collect();
        let fref = self.module.declare_func_in_func(sig.id, b.func);
        let call = b.ins().call(fref, &adapted);
        let out = if sig.ret == Ty::Void {
            b.ins().iconst(types::I32, 0)
        } else {
            b.inst_results(call)[0]
        };
        if sig.ret.clif() == result.clif() {
            out
        } else {
            Self::adapt(b, &sig.ret, result, out)
        }
    }

    /// Materialize a string literal: `.rodata` bytes wrapped by the runtime.
    fn str_literal(&mut self, b: &mut FunctionBuilder<'_>, text: &str) -> Value {
        let bytes = text.as_bytes();
        // One `.rodata` blob per literal, wrapped into the module arena at
        // runtime; deduplicated by text so repeated literals share memory.
        if !self.rodata.contains_key(text) {
            let mut desc = cranelift_module::DataDescription::new();
            desc.define(bytes.to_vec().into_boxed_slice());
            let id = self
                .module
                .declare_data(
                    &format!("__tesjit_str_{}", self.rodata.len()),
                    Linkage::Local,
                    false,
                    false,
                )
                .expect("declare_data");
            self.module.define_data(id, &desc).expect("define_data");
            self.rodata.insert(text.to_owned(), id);
        }
        let data_id = self.rodata[text];
        let gv = self.module.declare_data_in_func(data_id, b.func);
        let ptr = b.ins().symbol_value(types::I64, gv);
        let len = b.ins().iconst(types::I64, bytes.len() as i64);
        self.call_intrinsic(
            b,
            Intrinsic::StrWrap,
            &[(&Ty::Ptr, ptr), (&Ty::Ptr, len)],
            &Ty::Str,
        )
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
            Ty::Str => match op {
                BinaryOperator::Addition => self.call_intrinsic(
                    b,
                    Intrinsic::StrConcat,
                    &[(&Ty::Str, l), (&Ty::Str, r)],
                    &Ty::Str,
                ),
                BinaryOperator::Equality | BinaryOperator::StrictEquality => self.call_intrinsic(
                    b,
                    Intrinsic::StrEq,
                    &[(&Ty::Str, l), (&Ty::Str, r)],
                    &Ty::Bool,
                ),
                BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                    let eq = self.call_intrinsic(
                        b,
                        Intrinsic::StrEq,
                        &[(&Ty::Str, l), (&Ty::Str, r)],
                        &Ty::Bool,
                    );
                    b.ins().icmp_imm_s(IntCC::Equal, eq, 1)
                }
                _ => unreachable!("checker rejected {op:?} on string"),
            },
            Ty::Array(_) | Ty::Ptr => unreachable!("no operators on arrays/pointers in v1"),
            Ty::I8 | Ty::I16 | Ty::I32 | Ty::I64 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 => {
                // Signedness follows the operand type: unsigned division,
                // remainder, right shift and comparisons use the `_U` family.
                let unsigned = matches!(ty, Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64);
                let cc = match op {
                    BinaryOperator::Equality | BinaryOperator::StrictEquality => Some(IntCC::Equal),
                    BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                        Some(IntCC::NotEqual)
                    }
                    BinaryOperator::LessThan => Some(if unsigned {
                        IntCC::UnsignedLessThan
                    } else {
                        IntCC::SignedLessThan
                    }),
                    BinaryOperator::LessEqualThan => Some(if unsigned {
                        IntCC::UnsignedLessThanOrEqual
                    } else {
                        IntCC::SignedLessThanOrEqual
                    }),
                    BinaryOperator::GreaterThan => Some(if unsigned {
                        IntCC::UnsignedGreaterThan
                    } else {
                        IntCC::SignedGreaterThan
                    }),
                    BinaryOperator::GreaterEqualThan => Some(if unsigned {
                        IntCC::UnsignedGreaterThanOrEqual
                    } else {
                        IntCC::SignedGreaterThanOrEqual
                    }),
                    _ => None,
                };
                if let Some(cc) = cc {
                    ins.icmp(cc, l, r)
                } else {
                    match op {
                        BinaryOperator::Addition => ins.iadd(l, r),
                        BinaryOperator::Subtraction => ins.isub(l, r),
                        BinaryOperator::Multiplication => ins.imul(l, r),
                        BinaryOperator::Division => {
                            if unsigned {
                                ins.udiv(l, r)
                            } else {
                                ins.sdiv(l, r)
                            }
                        }
                        BinaryOperator::Remainder => {
                            if unsigned {
                                ins.urem(l, r)
                            } else {
                                ins.srem(l, r)
                            }
                        }
                        BinaryOperator::BitwiseAnd => ins.band(l, r),
                        BinaryOperator::BitwiseOR => ins.bor(l, r),
                        BinaryOperator::BitwiseXOR => ins.bxor(l, r),
                        BinaryOperator::ShiftLeft => ins.ishl(l, r),
                        // `>>` is arithmetic for signed, logical for unsigned;
                        // `>>>` is always logical.
                        BinaryOperator::ShiftRight => {
                            if unsigned {
                                ins.ushr(l, r)
                            } else {
                                ins.sshr(l, r)
                            }
                        }
                        BinaryOperator::ShiftRightZeroFill => ins.ushr(l, r),
                        _ => unreachable!("checker rejected {op:?} on {ty}"),
                    }
                }
            }
            Ty::F32 | Ty::F64 => {
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
                        _ => unreachable!("checker rejected {op:?} on {ty}"),
                    }
                }
            }
            // `Entity` handles compare by identity, like opaque 64-bit ids.
            Ty::Entity => {
                let cc = match op {
                    BinaryOperator::Equality | BinaryOperator::StrictEquality => IntCC::Equal,
                    BinaryOperator::Inequality | BinaryOperator::StrictInequality => {
                        IntCC::NotEqual
                    }
                    _ => unreachable!("checker rejected {op:?} on Entity"),
                };
                ins.icmp(cc, l, r)
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
            Ty::Struct(_)
            | Ty::OwnedStruct(_)
            | Ty::Option(_)
            | Ty::Tuple(_)
            | Ty::Query(..)
            | Ty::Res(_)
            | Ty::Mut(_)
            | Ty::Commands => {
                unreachable!("no operators on {}", ty)
            }
            Ty::Void => unreachable!("void has no operands"),
        }
    }
}
