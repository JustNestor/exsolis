//! Runtime support for heap-shaped values (strings, arrays).
//!
//! JIT code calls these `extern "C"` helpers for anything that cannot be a
//! machine value: allocation, string concat/compare, array growth, bounds
//! checks. They are registered per compilation unit like host functions (see
//! [`crate::host`]), so a module never reaches another module's memory.
//!
//! ## Value layouts
//!
//! ```text
//! string: [len: i64][utf8 bytes...]              (pointer is an i64)
//! array:  [len: i64][cap: i64][packed elems...]  (pointer is an i64)
//! ```
//!
//! Both live in the module's arena (bump-allocated, freed when the `Jit`/`Compiled`
//! handle is dropped — no GC), and the arena is capped, so runaway allocation
//! becomes a trap instead of an OOM kill.
//!
//! ## Traps
//!
//! A trap (out-of-bounds index, arena exhausted, ...) records a code in the
//! module runtime and returns a zero value, so the function returns normally.
//! The host then inspects [`crate::Compiled::trap`] and logs it: one buggy
//! handler never takes the server down.

use std::cell::RefCell;
use std::sync::Arc;

/// Largest allocation the arena will serve before trapping.
pub const ARENA_CAP: usize = 64 * 1024 * 1024;

/// Default per-call fuel budget: function entries and loop back-edges each
/// cost one unit, so this bounds both total work and recursion depth.
pub const DEFAULT_FUEL: u64 = 10_000_000;

/// Trap codes, surfaced to the host through [`crate::Compiled::trap`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrapKind {
    /// Index was negative or >= length.
    IndexOutOfBounds,
    /// Array `push` could not grow the array.
    ArrayOverflow,
    /// The arena ran out of memory.
    OutOfMemory,
    /// The fuel budget ran out (runaway loop or unbounded recursion).
    FuelExhausted,
}

impl TrapKind {
    pub fn message(self) -> &'static str {
        match self {
            TrapKind::IndexOutOfBounds => "index out of bounds",
            TrapKind::ArrayOverflow => "array capacity exceeded",
            TrapKind::OutOfMemory => "out of memory",
            TrapKind::FuelExhausted => "fuel exhausted (runaway handler)",
        }
    }
}

thread_local! {
    /// The runtime of the module currently executing on this thread.
    ///
    /// Execution is single-threaded per module (same model as the current Lua
    /// runtime), so one slot is enough and keeps the JIT ABI flat: no hidden
    /// arguments threaded through every call.
    static CURRENT: RefCell<Option<Arc<TesjitRuntime>>> = const { RefCell::new(None) };
}

/// A bump arena with a *stable* backing buffer.
///
/// JIT code holds pointers into this arena for the whole lifetime of a module
/// (strings, arrays, ...), so it must never reallocate: the buffer is reserved
/// once (virtual memory only — pages fault in on first touch) and handed out
/// in aligned slices.
struct Arena {
    ptr: *mut u8,
    used: usize,
}

impl std::fmt::Debug for Arena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arena")
            .field("used", &self.used)
            .field("cap", &ARENA_CAP)
            .finish_non_exhaustive()
    }
}

// SAFETY: the arena is only touched from the module's execution thread, and
// the pointer is valid for the lifetime of the `TesjitRuntime` that owns it.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    fn new() -> Self {
        // SAFETY: zeroed allocation of `ARENA_CAP` bytes, 64-byte aligned.
        let ptr = unsafe {
            std::alloc::alloc_zeroed(
                std::alloc::Layout::from_size_align(ARENA_CAP, 64).expect("arena layout"),
            )
        };
        assert!(!ptr.is_null(), "tesjit arena allocation failed");
        Self { ptr, used: 0 }
    }

    #[inline]
    fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        // Every value carries an `i64` header (string length, array len/cap),
        // and the header is read with an *aligned* load, so allocations are
        // rounded up to the header alignment.
        let size = (size + 7) & !7;
        let end = self.used.checked_add(size)?;
        if end > ARENA_CAP {
            return None;
        }
        // SAFETY: `end <= ARENA_CAP` and the buffer is zeroed.
        let out = unsafe { self.ptr.add(self.used) };
        self.used = end;
        Some(out)
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // SAFETY: allocated by `alloc_zeroed` with the same layout.
        unsafe {
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::from_size_align(ARENA_CAP, 64).expect("arena layout"),
            );
        }
    }
}

/// Per-module heap + trap state. Shared with the `extern "C"` thunks that are
/// registered as module symbols.
#[derive(Debug)]
pub struct TesjitRuntime {
    arena: RefCell<Arena>,
    trap: RefCell<Option<TrapKind>>,
    /// Remaining fuel for the current call.
    fuel: std::cell::Cell<u64>,
}

impl Default for TesjitRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// SAFETY: see the type docs — one execution thread per module.
unsafe impl Sync for TesjitRuntime {}

impl TesjitRuntime {
    pub fn new() -> Self {
        Self {
            arena: RefCell::new(Arena::new()),
            trap: RefCell::new(None),
            fuel: std::cell::Cell::new(DEFAULT_FUEL),
        }
    }

    /// Arm (or re-arm) the fuel budget for the next call into module code.
    ///
    /// Hosts call this before invoking a handler so every entry starts fresh.
    pub fn reset_fuel(&self, fuel: u64) {
        self.fuel.set(fuel);
    }

    /// Fuel still available.
    pub fn fuel(&self) -> u64 {
        self.fuel.get()
    }

    /// Consume one unit; returns false when the budget is spent.
    fn spend_fuel(&self) -> bool {
        let left = self.fuel.get();
        if left == 0 {
            *self.trap.borrow_mut() = Some(TrapKind::FuelExhausted);
            return false;
        }
        self.fuel.set(left - 1);
        true
    }

    /// Bytes handed out so far (diagnostics/benchmarks).
    pub fn used(&self) -> usize {
        self.arena.borrow().used
    }

    /// Pending trap, if any.
    pub fn trap(&self) -> Option<TrapKind> {
        *self.trap.borrow()
    }

    /// Clear a pending trap (hosts do this after logging it).
    pub fn clear_trap(&self) {
        *self.trap.borrow_mut() = None;
    }

    fn trap_hit(&self, kind: TrapKind) -> i64 {
        *self.trap.borrow_mut() = Some(kind);
        0
    }

    fn alloc(&self, size: usize) -> Option<*mut u8> {
        self.arena.borrow_mut().alloc(size)
    }

    /// Mark this runtime as the module executing on this thread.
    ///
    /// The guard restores the previous runtime on drop, so nested or reentrant
    /// modules each keep their own heap.
    pub fn enter(self: &Arc<Self>) -> RuntimeGuard {
        let previous = CURRENT.with(|c| c.borrow_mut().replace(self.clone()));
        RuntimeGuard { previous }
    }
}

/// Restores the previous runtime on drop.
pub struct RuntimeGuard {
    previous: Option<Arc<TesjitRuntime>>,
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| *c.borrow_mut() = self.previous.take());
    }
}

/// The runtime executing on this thread.
///
/// # Panics
/// If no runtime is entered — all JIT code runs inside [`TesjitRuntime::enter`].
fn current() -> &'static TesjitRuntime {
    let raw: *const TesjitRuntime = CURRENT.with(|c| {
        c.borrow()
            .as_ref()
            .map(Arc::as_ptr)
            .unwrap_or(std::ptr::null())
    });
    assert!(!raw.is_null(), "tesjit runtime not entered");
    unsafe { &*raw }
}

fn trap(kind: TrapKind) -> i64 {
    current().trap_hit(kind)
}

// ───────────────────────────── extern "C" ABI ─────────────────────────────
//
// These are what JIT code actually calls. The `__tesjit_` prefix means they can
// never collide with a mod's host API names.

extern "C" fn alloc(size: i64) -> i64 {
    if size < 0 || size > ARENA_CAP as i64 {
        return trap(TrapKind::OutOfMemory);
    }
    match current().alloc(size as usize) {
        Some(ptr) => ptr as i64,
        None => trap(TrapKind::OutOfMemory),
    }
}

/// Wrap raw bytes (a string literal in .rodata) as an arena string.
extern "C" fn str_wrap(ptr: *const u8, len: i64) -> i64 {
    if len < 0 {
        return trap(TrapKind::IndexOutOfBounds);
    }
    let len = len as usize;
    let Some(dst) = current().alloc(8 + len) else {
        return trap(TrapKind::OutOfMemory);
    };
    unsafe {
        std::ptr::write_unaligned(dst as *mut i64, len as i64);
        std::ptr::copy_nonoverlapping(ptr, dst.add(8), len);
    }
    dst as i64
}

extern "C" fn str_len(s: i64) -> i64 {
    unsafe { *(s as *const i64) }
}

extern "C" fn str_concat(a: i64, b: i64) -> i64 {
    let (la, lb) = (str_len(a) as usize, str_len(b) as usize);
    let Some(total) = la.checked_add(lb) else {
        return trap(TrapKind::OutOfMemory);
    };
    let Some(dst) = current().alloc(8 + total) else {
        return trap(TrapKind::OutOfMemory);
    };
    unsafe {
        std::ptr::write_unaligned(dst as *mut i64, total as i64);
        std::ptr::copy_nonoverlapping((a as *const u8).add(8), dst.add(8), la);
        std::ptr::copy_nonoverlapping((b as *const u8).add(8), dst.add(8 + la), lb);
    }
    dst as i64
}

/// Byte-wise equality (returns 1/0 so it can be used as a boolean).
extern "C" fn str_eq(a: i64, b: i64) -> i64 {
    let (la, lb) = (str_len(a) as usize, str_len(b) as usize);
    if la != lb {
        return 0;
    }
    let left = unsafe { std::slice::from_raw_parts((a as *const u8).add(8), la) };
    let right = unsafe { std::slice::from_raw_parts((b as *const u8).add(8), lb) };
    i64::from(left == right)
}

/// Byte at `i`; traps when out of bounds.
extern "C" fn str_byte(s: i64, i: i64) -> i64 {
    let len = str_len(s);
    if i < 0 || i >= len {
        return trap(TrapKind::IndexOutOfBounds);
    }
    unsafe { *((s as *const u8).add(8 + i as usize)) as i64 }
}

/// Allocate an array with `cap` slots of `elem_size` bytes and length 0.
extern "C" fn arr_new(cap: i64, elem_size: i64) -> i64 {
    if cap < 0 || elem_size <= 0 {
        return trap(TrapKind::ArrayOverflow);
    }
    let elems = match (cap as usize).checked_mul(elem_size as usize) {
        Some(elems) => elems,
        None => return trap(TrapKind::OutOfMemory),
    };
    let Some(bytes) = 16usize.checked_add(elems) else {
        return trap(TrapKind::OutOfMemory);
    };
    match current().alloc(bytes) {
        Some(ptr) => {
            unsafe {
                std::ptr::write_unaligned(ptr as *mut i64, 0);
                std::ptr::write_unaligned(ptr.add(8) as *mut i64, cap);
            }
            ptr as i64
        }
        None => trap(TrapKind::OutOfMemory),
    }
}

extern "C" fn arr_len(a: i64) -> i64 {
    unsafe { *(a as *const i64) }
}

/// Append `value` to `a`, growing (copying into a fresh arena block) when full.
/// Returns the (possibly moved) array pointer.
extern "C" fn arr_push(a: i64, elem_size: i64, value: i64) -> i64 {
    let elem_size = elem_size as usize;
    let a = a as *mut u8;
    let (len, cap) = unsafe {
        (
            *(a as *const i64) as usize,
            *(a.add(8) as *const i64) as usize,
        )
    };
    if len >= cap {
        let new_cap = (cap.saturating_mul(2)).max(4);
        let Some(bytes) = 16usize.checked_add(new_cap * elem_size) else {
            return trap(TrapKind::OutOfMemory);
        };
        let Some(dst) = current().alloc(bytes) else {
            return trap(TrapKind::OutOfMemory);
        };
        unsafe {
            std::ptr::write_unaligned(dst as *mut i64, (len + 1) as i64);
            std::ptr::write_unaligned(dst.add(8) as *mut i64, new_cap as i64);
            std::ptr::copy_nonoverlapping(a.add(16), dst.add(16), len * elem_size);
            write_elem(dst.add(16), len, elem_size, value);
        }
        return dst as i64;
    }
    unsafe {
        std::ptr::write_unaligned(a as *mut i64, (len + 1) as i64);
        write_elem(a.add(16), len, elem_size, value);
    }
    a as i64
}

unsafe fn write_elem(base: *mut u8, index: usize, elem_size: usize, value: i64) {
    let dst = unsafe { base.add(index * elem_size) };
    // Elements are machine scalars: copy the low bytes of the value.
    let bytes = (value as u64).to_le_bytes();
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, elem_size.min(8)) };
}

/// Element address after a bounds check, or `None` when out of range (with the
/// trap recorded).
fn arr_elem(a: i64, i: i64, elem_size: i64) -> Option<*mut u8> {
    let a = a as *const u8;
    let len = unsafe { *(a as *const i64) };
    if i < 0 || i >= len {
        trap(TrapKind::IndexOutOfBounds);
        return None;
    }
    let elem_size = elem_size.clamp(1, 8) as usize;
    Some(unsafe { a.add(16 + (i as usize) * elem_size) as *mut u8 })
}

/// Read one element as an integer (only `elem_size` bytes are touched, so the
/// read never crosses into a neighbouring element).
extern "C" fn arr_get(a: i64, i: i64, elem_size: i64) -> i64 {
    let Some(ptr) = arr_elem(a, i, elem_size) else {
        return 0;
    };
    let elem_size = elem_size.clamp(1, 8) as usize;
    let mut bytes = [0u8; 8];
    unsafe { std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), elem_size) };
    i64::from_le_bytes(bytes)
}

/// Write one element from the low bytes of `value`.
extern "C" fn arr_set(a: i64, i: i64, elem_size: i64, value: i64) -> i64 {
    let Some(ptr) = arr_elem(a, i, elem_size) else {
        return 0;
    };
    let elem_size = elem_size.clamp(1, 8) as usize;
    let bytes = (value as u64).to_le_bytes();
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, elem_size) };
    value
}

/// Fuel check on function entry and loop back-edges.
///
/// Returns 1 while fuel remains, 0 when it ran out (with the trap recorded) —
/// the caller branches to the function's trap block, so the handler returns
/// immediately instead of spinning forever.
extern "C" fn fuel() -> i64 {
    i64::from(current().spend_fuel())
}

/// Named runtime symbols to register for one module.
pub fn symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("__tesjit_fuel", fuel as *const u8),
        ("__tesjit_alloc", alloc as *const u8),
        ("__tesjit_str_wrap", str_wrap as *const u8),
        ("__tesjit_str_len", str_len as *const u8),
        ("__tesjit_str_concat", str_concat as *const u8),
        ("__tesjit_str_eq", str_eq as *const u8),
        ("__tesjit_str_byte", str_byte as *const u8),
        ("__tesjit_arr_new", arr_new as *const u8),
        ("__tesjit_arr_len", arr_len as *const u8),
        ("__tesjit_arr_push", arr_push as *const u8),
        ("__tesjit_arr_get", arr_get as *const u8),
        ("__tesjit_arr_set", arr_set as *const u8),
    ]
}

/// Reserved names: host modules cannot define these.
pub const RESERVED: &[&str] = &[
    "__tesjit_alloc",
    "__tesjit_str_wrap",
    "__tesjit_str_len",
    "__tesjit_str_concat",
    "__tesjit_str_eq",
    "__tesjit_str_byte",
    "__tesjit_arr_new",
    "__tesjit_arr_len",
    "__tesjit_arr_push",
    "__tesjit_arr_get",
    "__tesjit_arr_set",
];
