//! Host function registration and lookup.
//!
//! Host functions are the *only* way JIT code reaches the embedding program
//! (logging, game API, ...). Unlike the previous `linkme` global slice, a
//! registry belongs to one [`crate::Jit`], so two embedders in the same binary
//! can expose different `api.*` surfaces without symbol collisions, and the
//! compiler can check every call site against the exact registered signature.

use std::collections::HashMap;

use crate::ty::Ty;

/// A registered host function: how to call it from JIT code.
#[derive(Clone, Debug)]
pub struct HostFn {
    /// Name TS sources call it by.
    pub name: String,
    /// `extern "C"` function pointer.
    pub ptr: *const u8,
    /// Parameter types, in order.
    pub params: Vec<Ty>,
    /// Return type (`Ty::Void` when it returns nothing).
    pub ret: Ty,
}

// SAFETY: only a function pointer plus plain metadata.
unsafe impl Send for HostFn {}
unsafe impl Sync for HostFn {}

/// Signature of a host function, for `declare function` compatibility checks.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HostSig {
    pub params: Vec<Ty>,
    pub ret: Ty,
}

impl HostFn {
    pub fn new(name: impl Into<String>, params: Vec<Ty>, ret: Ty, ptr: *const u8) -> Self {
        Self {
            name: name.into(),
            ptr,
            params,
            ret,
        }
    }

    pub fn sig(&self) -> HostSig {
        HostSig {
            params: self.params.clone(),
            ret: self.ret.clone(),
        }
    }
}

/// Host functions of one compilation unit, by name.
#[derive(Clone, Debug, Default)]
pub struct HostRegistry {
    fns: HashMap<String, HostFn>,
}

impl HostRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a host function. Registration order is irrelevant; a duplicate
    /// name replaces the previous entry.
    pub fn insert(&mut self, f: HostFn) {
        self.fns.insert(f.name.clone(), f);
    }

    pub fn get(&self, name: &str) -> Option<&HostFn> {
        self.fns.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.fns.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &HostFn> {
        self.fns.values()
    }

    pub fn len(&self) -> usize {
        self.fns.len()
    }

    /// Cheap stable hash of this registry's signatures. Used as the *shape*
    /// component of hot-reload's reject-on-layout-change rule (§7 of
    /// `docs/SCRIPTING_ECS_MODS_PLAN.md`). Two registries with the same
    /// hash present an identical mod surface to the JIT.
    pub fn layout_hash(&self) -> u64 {
        let mut entries: Vec<String> = self
            .fns
            .values()
            .map(|f| {
                let ps: Vec<String> = f.params.iter().map(|p| p.name().into_owned()).collect();
                format!("{}({}) -> {}", f.name, ps.join(","), f.ret.name())
            })
            .collect();
        entries.sort();
        let mut h: u64 = 0xcbf29ce484222325;
        for entry in &entries {
            for b in entry.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h = h.wrapping_mul(0x100000001b3); // separator
        }
        h
    }
}

/// A function the compiler knows about: either TS-defined or host-provided.
#[derive(Clone, Debug)]
pub enum FnKind {
    /// Defined in the module being compiled; callable directly.
    Local,
    /// Imported from the host; called through an import symbol.
    Host(HostSig),
}

/// Compile-time view of everything callable in a unit.
#[derive(Clone, Debug, Default)]
pub struct FnTable {
    entries: HashMap<String, FnKind>,
}

impl FnTable {
    pub fn insert(&mut self, name: String, kind: FnKind) {
        self.entries.insert(name, kind);
    }

    pub fn get(&self, name: &str) -> Option<&FnKind> {
        self.entries.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
}
