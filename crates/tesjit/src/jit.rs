//! The embeddable compiler entry point.
//!
//! A [`JitBuilder`] configures one compilation unit (source + host functions),
//! [`Jit`] compiles it, and [`Compiled`] owns the finished machine code.

use std::collections::HashMap;
use std::sync::Arc;

use cranelift_codegen::settings::Configurable;
use cranelift_codegen::{Context, settings};
use cranelift_frontend::FunctionBuilderContext;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Module, default_libcall_names};
use miette::NamedSource;

use crate::check;
use crate::codegen::{FnLowerer, FnSig};
use crate::diag::{CompileError, Diagnostics, Src};
use crate::hir::{ComponentDecl, Module as HirModule, SystemExport};
use crate::host::{HostFn, HostRegistry};
use crate::runtime::TesjitRuntime;
use crate::ty::{StructLayout, Ty};

/// A finished, callable compilation unit.
pub struct Compiled {
    module: Option<JITModule>,
    /// Heap for strings/arrays created by the compiled code.
    runtime: Arc<TesjitRuntime>,
    pub fns: HashMap<String, Entry>,
    /// Systems registered via `export default { stage: fn }`.
    pub systems: Vec<SystemExport>,
    /// Mod-defined component/resource layouts (for hot-reload hash checks).
    pub components: Vec<ComponentDecl>,
    /// Aggregate layout hash of `components` at compile time.
    pub component_layout_hash: u64,
}

unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

/// One exported function's address and TS signature.
pub struct Entry {
    pub ptr: *const u8,
    pub params: Vec<Ty>,
    pub ret: Ty,
    /// Host-frame loads for system functions (empty for ordinary exports).
    pub frame_loads: Vec<crate::hir::FrameLoad>,
    /// Number of machine i64 slots the function actually takes after frame
    /// ABI expansion (Query → 2 slots). Host calls must pass this many args.
    pub machine_arity: usize,
}

impl Compiled {
    /// Names of every exported TS function, with its TS signature.
    pub fn exports(&self) -> impl Iterator<Item = (&str, &[Ty], Ty)> {
        self.fns
            .iter()
            .map(|(name, e)| (name.as_str(), e.params.as_slice(), e.ret.clone()))
    }

    pub fn entry(&self, name: &str) -> Option<&Entry> {
        self.fns.get(name)
    }

    /// Systems scheduled by this mod (`export default`).
    pub fn systems(&self) -> &[SystemExport] {
        &self.systems
    }

    /// Component/resource layouts declared by this mod.
    pub fn components(&self) -> &[ComponentDecl] {
        &self.components
    }

    /// Pending runtime trap (out-of-bounds index, out-of-memory, ...).
    ///
    /// Callers check this after invoking into compiled code and should
    /// [`Compiled::clear_trap`] once the error has been logged: a trap leaves
    /// the function returning normally with zero values, so one buggy handler
    /// never unwinds into the host.
    pub fn trap(&self) -> Option<crate::runtime::TrapKind> {
        self.runtime.trap()
    }

    pub fn clear_trap(&self) {
        self.runtime.clear_trap();
    }

    /// The module's arena: bytes handed out for strings/arrays.
    pub fn runtime(&self) -> &TesjitRuntime {
        &self.runtime
    }

    /// Expose host memory (a query row, a component, ...) to the module.
    ///
    /// Returns the opaque id JIT code passes to the `load_*`/`store_*`
    /// intrinsics. The region stays valid until [`Compiled::unregister_region`]
    /// or until the pointer's owner goes away — bevy hands those over inside a
    /// system that holds the data for the duration of the call.
    pub fn register_region(&self, name: impl Into<String>, ptr: *mut u8, len: usize) -> u64 {
        self.runtime.register_region(name, ptr, len)
    }

    pub fn unregister_region(&self, name: &str) {
        self.runtime.unregister_region(name);
    }

    /// Enter this module's runtime for the duration of `f`.
    ///
    /// Required for hand-written `extern "C"` wrappers obtained through
    /// [`Compiled::get`] (which cannot take a guard themselves): without it the
    /// string/array builtins would panic on a missing runtime, and host
    /// functions could not build strings.
    pub fn enter(&self) -> crate::runtime::Guard {
        crate::runtime::enter(&self.runtime)
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
    /// Convenience for bindings, the CLI and the host bridge: every argument
    /// is passed in an integer register slot, so `i32`/`boolean`/pointer
    /// parameters are fine but `f64` parameters are **not** (they belong in an
    /// SSE register). Use [`Compiled::get`] with an exact signature for float
    /// parameters.
    ///
    /// # Safety
    /// `name` must exist and accept the given arguments.
    #[track_caller]
    pub unsafe fn call_i64(&self, name: &str, args: &[i64]) -> i64 {
        // String/array builtins resolve the module heap through a thread-local,
        // so the module must be entered for the duration of the call — and host
        // functions need it too, to build strings they hand back.
        let _guard = crate::runtime::enter(&self.runtime);
        let e = self.entry(name).expect("unknown export");
        assert!(e.params.len() == args.len(), "argument count mismatch");
        assert!(
            args.len() <= MAX_ARGS,
            "call_i64 supports at most {MAX_ARGS} arguments"
        );
        assert!(
            e.params.iter().all(|t| !t.is_float()),
            "call_i64 cannot pass f32/f64 parameters; use `get::<F>()`"
        );
        let mut buf = [0i64; MAX_ARGS];
        buf[..args.len()].copy_from_slice(args);
        // Extra argument slots are simply ignored by the callee (System V,
        // caller-cleaned), which keeps this single trampoline for every arity.
        let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
            unsafe { std::mem::transmute_copy(&e.ptr) };
        f(
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        )
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

/// Configures a compilation unit.
///
/// ```ignore
/// let mut jit = JitBuilder::new("mod.ts", src)
///     .host_fn("log", vec![Ty::F64], Ty::Void, host_log as *const u8)
///     .build()?;
/// jit.compile(&src)?;
/// let compiled = jit.finish()?;
/// ```
/// Maximum arguments [`Compiled::call_i64`] can pass.
pub const MAX_ARGS: usize = 8;

pub struct JitBuilder {
    src_name: String,
    src: Arc<str>,
    host: HostRegistry,
}

impl JitBuilder {
    pub fn new(src_name: impl Into<String>, src: impl Into<Arc<str>>) -> Self {
        Self {
            src_name: src_name.into(),
            src: src.into(),
            host: HostRegistry::new(),
        }
    }

    /// Register a host function the module may call.
    pub fn host_fn(
        mut self,
        name: impl Into<String>,
        params: Vec<Ty>,
        ret: Ty,
        ptr: *const u8,
    ) -> Self {
        self.host.insert(HostFn::new(name, params, ret, ptr));
        self
    }

    /// Register a host borrow (useful when a table of API functions is built
    /// at runtime).
    pub fn host_fn_ref(&mut self, f: HostFn) {
        self.host.insert(f);
    }

    /// Register a host-provided component type for query and field checking.
    pub fn component(
        mut self,
        name: impl Into<String>,
        layout: Arc<StructLayout>,
    ) -> Result<Self, CompileError> {
        self.host
            .insert_engine_component(name, layout)
            .map_err(crate::diag::plain)?;
        Ok(self)
    }

    pub fn build(self) -> Result<Jit, CompileError> {
        Jit::new(self.src_name, self.src, self.host)
    }
}

/// One live compilation unit: owns the JIT machinery plus everything the unit
/// can call.
pub struct Jit {
    module: JITModule,
    ctx: Context,
    fctx: FunctionBuilderContext,
    /// Heap for strings/arrays created by the compiled code.
    runtime: Arc<TesjitRuntime>,
    /// Every declared function, in declaration order (locals first, then
    /// host imports — indices are what HIR calls refer to).
    sigs: Vec<(String, FnSig)>,
    host: HostRegistry,
    src: Src,
    /// Filled by the last successful `compile`.
    systems: Vec<SystemExport>,
    components: Vec<ComponentDecl>,
    /// Per-function frame loads (systems only).
    frame_loads: HashMap<String, Vec<crate::hir::FrameLoad>>,
}

impl Jit {
    pub fn new(
        src_name: impl AsRef<str>,
        src: Arc<str>,
        host: HostRegistry,
    ) -> Result<Self, CompileError> {
        let src = NamedSource::new(src_name.as_ref(), src);
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| crate::diag::plain(format!("cranelift config: {e}")))?;
        flag_builder
            .set("preserve_frame_pointers", "false")
            .map_err(|e| crate::diag::plain(format!("cranelift config: {e}")))?;
        // Stack probes: with them off, a function with a large frame can skip
        // the guard page and corrupt memory below the stack. Bounded recursion
        // is handled by fuel, but "deep and legal" still has to be safe.
        flag_builder
            .set("enable_probestack", "true")
            .map_err(|e| crate::diag::plain(format!("cranelift config: {e}")))?;
        flag_builder
            .set("enable_alias_analysis", "true")
            .map_err(|e| crate::diag::plain(format!("cranelift config: {e}")))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| crate::diag::plain(format!("Native ISA detection failed: {e}")))?;
        let flags = settings::Flags::new(flag_builder);
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| crate::diag::plain(e.to_string()))?;

        let runtime = Arc::new(TesjitRuntime::new());
        let mut jb = JITBuilder::with_isa(isa, default_libcall_names());
        for h in host.iter() {
            assert!(
                !crate::runtime::RESERVED.contains(&h.name.as_str()),
                "`{}` is a reserved runtime symbol",
                h.name
            );
            jb.symbol(&h.name, h.ptr);
        }
        // Runtime helpers: one set of thunks per module, each finding the
        // module's own arena through the thread-local entered on call.
        for (name, ptr) in crate::runtime::symbols() {
            jb.symbol(name, ptr);
        }
        let module = JITModule::new(jb);

        let ctx = module.make_context();
        Ok(Self {
            module,
            ctx,
            fctx: FunctionBuilderContext::new(),
            sigs: Vec::new(),
            runtime,
            host,
            src,
            systems: Vec::new(),
            components: Vec::new(),
            frame_loads: HashMap::new(),
        })
    }

    /// Type-check and compile `source`. All diagnostics are reported at once;
    /// nothing is emitted if any error was found.
    pub fn compile(&mut self, source: &str) -> Result<(), Diagnostics> {
        let (module, diags) = check::Lowerer::run(source, &self.src, &self.host);
        if !diags.ok() {
            return Err(diags);
        }
        let module = module.expect("no diagnostics implies a module");

        self.systems = module.systems.clone();
        self.components = module.components.clone();
        self.frame_loads.clear();
        for f in &module.functions {
            if !f.frame_loads.is_empty() {
                self.frame_loads.insert(f.name.clone(), f.frame_loads.clone());
            }
        }
        // Register mod component layouts on the host registry so signature
        // checks and reload hashing see them.
        for c in &module.components {
            self.host.insert_component(c.name.clone(), Arc::clone(&c.layout));
        }

        self.sigs.clear();
        // Local functions first (source order — this is the order HIR call
        // indices refer to), host imports after: host calls carry names, so
        // their position does not matter.
        for f in &module.functions {
            let id = FnLowerer::declare_framed(
                &mut self.module,
                &f.name,
                &f.params,
                f.ret.clone(),
                &f.frame_loads,
            )
            .map_err(|e| Diagnostics::from(CompileError::Plain(e)))?;
            self.sigs.push((
                f.name.clone(),
                FnSig {
                    id,
                    params: f.params.clone(),
                    ret: f.ret.clone(),
                },
            ));
        }
        for h in self.host.iter() {
            let id = FnLowerer::declare_import(&mut self.module, &h.name, &h.params, h.ret.clone())
                .map_err(|e| Diagnostics::from(CompileError::Plain(e)))?;
            self.sigs.push((
                h.name.clone(),
                FnSig {
                    id,
                    params: h.params.clone(),
                    ret: h.ret.clone(),
                },
            ));
        }
        // Runtime helpers (string/array intrinsics): imported symbols with the
        // signatures declared in `runtime_signatures`.
        for (name, params, ret) in Self::runtime_signatures() {
            let id = FnLowerer::declare_import(&mut self.module, name, &params, ret.clone())
                .map_err(|e| Diagnostics::from(CompileError::Plain(e)))?;
            self.sigs.push((name.to_owned(), FnSig { id, params, ret }));
        }

        let mut errors = Diagnostics::new();
        for f in &module.functions {
            let index = self
                .sigs
                .iter()
                .position(|(name, _)| *name == f.name)
                .expect("declared above");
            let target = self.module.target_config();
            let sig = self.sigs[index].1.clone();
            if let Err(e) = FnLowerer::define(
                &mut self.module,
                &self.sigs,
                &mut self.fctx,
                &mut self.ctx,
                target,
                f,
                sig,
            ) {
                errors.push(CompileError::Plain(e));
            }
        }
        errors.into_result(())
    }

    /// Type-check `source` without emitting code. Returns every diagnostic.
    pub fn check(&self, source: &str) -> Diagnostics {
        let (_, diags) = check::Lowerer::run(source, &self.src, &self.host);
        diags
    }

    /// Finalize the unit and hand out the callable entry points.
    ///
    /// Only TS functions can be called: host imports stay symbols inside the
    /// module and are linked at call time by name.
    pub fn finish(mut self) -> Result<Compiled, CompileError> {
        self.module
            .finalize_definitions()
            .map_err(|e| CompileError::Plain(e.to_string()))?;
        let mut fns = HashMap::new();
        let runtime = Arc::clone(&self.runtime);
        for (name, sig) in &self.sigs {
            // Host imports and runtime helpers are symbols, not compiled code.
            if self.host.contains(name) || name.starts_with("__tesjit_") {
                continue;
            }
            let ptr = self.module.get_finalized_function(sig.id);
            let frame_loads = self.frame_loads.remove(name).unwrap_or_default();
            // Unified frame ABI: every param is one machine word (Query params
            // receive a `QueryInput` pointer), framed or not.
            let machine_arity = sig.params.len();
            fns.insert(
                name.clone(),
                Entry {
                    ptr,
                    params: sig.params.clone(),
                    ret: sig.ret.clone(),
                    frame_loads,
                    machine_arity,
                },
            );
        }
        let component_layout_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            let mut entries: Vec<_> = self.components.iter().map(|c| (&c.name, c.layout.layout_hash())).collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            for (name, lh) in entries {
                for b in name.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h ^= lh;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        Ok(Compiled {
            module: Some(self.module),
            runtime,
            fns,
            systems: self.systems,
            components: self.components,
            component_layout_hash,
        })
    }

    /// Exported names seen so far (`finish` splits `self`, so ask before).
    pub fn exports(&self) -> impl Iterator<Item = &str> {
        self.sigs.iter().map(|(name, _)| name.as_str())
    }

    /// Signatures of the runtime helpers, mirroring [`crate::runtime::symbols`].
    fn runtime_signatures() -> Vec<(&'static str, Vec<Ty>, Ty)> {
        use Ty::{Ptr, Str};
        vec![
            ("__tesjit_fuel", vec![], Ty::Ptr),
            ("__tesjit_fuel_n", vec![Ty::I64], Ty::Ptr),
            ("__tesjit_alloc", vec![Ptr], Ptr),
            ("__tesjit_str_wrap", vec![Ptr, Ptr], Str),
            ("__tesjit_str_len", vec![Str], Ptr),
            ("__tesjit_str_concat", vec![Str, Str], Str),
            ("__tesjit_str_eq", vec![Str, Str], Ptr),
            ("__tesjit_str_byte", vec![Str, Ptr], Ptr),
            ("__tesjit_arr_new", vec![Ptr, Ptr], Ptr),
            ("__tesjit_arr_len", vec![Ptr], Ptr),
            ("__tesjit_arr_push", vec![Ptr, Ptr, Ptr], Ptr),
            ("__tesjit_load_f64", vec![Ptr, Ptr], Ty::F64),
            ("__tesjit_load_f32", vec![Ptr, Ptr], Ty::F64),
            ("__tesjit_load_i64", vec![Ptr, Ptr], Ty::Ptr),
            ("__tesjit_load_i32", vec![Ptr, Ptr], Ty::I32),
            ("__tesjit_store_f64", vec![Ptr, Ptr, Ty::F64], Ty::F64),
            ("__tesjit_store_f32", vec![Ptr, Ptr, Ty::F64], Ty::F64),
            ("__tesjit_store_i64", vec![Ptr, Ptr, Ty::Ptr], Ty::Ptr),
            ("__tesjit_store_i32", vec![Ptr, Ptr, Ty::I32], Ty::I32),
            ("__tesjit_region_id", vec![Ty::Str], Ty::I32),
            ("__tesjit_opt_is_some", vec![Ty::Ptr], Ty::Ptr),
            ("__tesjit_opt_unwrap_i64", vec![Ty::Ptr], Ty::Ptr),
            ("__tesjit_opt_unwrap_f64", vec![Ty::Ptr], Ty::F64),
            (
                "__tesjit_opt_unwrap_or_i64",
                vec![Ty::Ptr, Ty::Ptr],
                Ty::Ptr,
            ),
            (
                "__tesjit_opt_unwrap_or_f64",
                vec![Ty::Ptr, Ty::F64],
                Ty::F64,
            ),
            ("__tesjit_arr_get", vec![Ptr, Ptr, Ptr], Ptr),
            ("__tesjit_arr_set", vec![Ptr, Ptr, Ptr, Ptr], Ptr),
        ]
    }

    /// The lowered HIR of the last successful compile, for tooling/tests.
    pub fn lower(source: &str, src: &Src, host: &HostRegistry) -> (Option<HirModule>, Diagnostics) {
        check::Lowerer::run(source, src, host)
    }
}
