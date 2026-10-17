//! Host function registration and lookup.
//!
//! Host functions are the *only* way JIT code reaches the embedding program
//! (logging, game API, ...). Unlike the previous `linkme` global slice, a
//! registry belongs to one [`crate::Jit`], so two embedders in the same binary
//! can expose different `api.*` surfaces without symbol collisions, and the
//! compiler can check every call site against the exact registered signature.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::ty::{StructLayout, Ty};

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
    /// Component / resource layouts registered by the host (engine allowlist)
    /// or discovered from the mod. Keyed by type name.
    components: HashMap<String, Arc<StructLayout>>,
    engine_components: HashSet<String>,
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

    /// Register a component layout supplied by the embedding engine.
    pub fn insert_engine_component(
        &mut self,
        name: impl Into<String>,
        layout: Arc<StructLayout>,
    ) -> Result<(), String> {
        let name = name.into();
        if self.components.contains_key(&name) {
            return Err(format!("component `{name}` is already registered"));
        }
        self.components.insert(name.clone(), layout);
        self.engine_components.insert(name);
        Ok(())
    }

    pub fn component(&self, name: &str) -> Option<&Arc<StructLayout>> {
        self.components.get(name)
    }

    pub fn components(&self) -> impl Iterator<Item = (&str, &Arc<StructLayout>)> {
        self.components
            .iter()
            .map(|(name, layout)| (name.as_str(), layout))
    }

    pub fn is_engine_component(&self, name: &str) -> bool {
        self.engine_components.contains(name)
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

    /// Register (or replace) a component/resource layout by name.
    pub fn insert_component(&mut self, name: impl Into<String>, layout: Arc<StructLayout>) {
        self.components.insert(name.into(), layout);
    }

    /// Cheap stable hash of this registry's *function* signatures.
    /// Renamed from `layout_hash` in phase 2: the old name collided with
    /// per-component layout hashing (§7).
    pub fn signature_hash(&self) -> u64 {
        let mut entries: Vec<String> = self
            .fns
            .values()
            .map(|f| {
                let ps: Vec<String> = f.params.iter().map(|p| p.name().into_owned()).collect();
                format!("{}({}) -> {}", f.name, ps.join(","), f.ret.name())
            })
            .collect();
        entries.extend(self.components.iter().map(|(name, layout)| {
            let fields = layout
                .fields
                .iter()
                .map(|field| format!("{}:{}@{}", field.name, field.ty.name(), field.offset))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "component:{name}:{}:{}:{}",
                layout.size, layout.align, fields
            )
        }));
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

    /// Aggregate hash of every registered component layout. Used by the
    /// hot-reload path to reject mods whose component shapes changed.
    pub fn component_layout_hash(&self) -> u64 {
        let mut entries: Vec<(&String, u64)> = self
            .components
            .iter()
            .map(|(n, l)| (n, l.layout_hash()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        let mut h: u64 = 0xcbf29ce484222325;
        for (name, lh) in entries {
            for b in name.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= lh;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Backward-compatible alias — prefer [`signature_hash`].
    #[deprecated(note = "renamed to signature_hash in phase 2")]
    pub fn layout_hash(&self) -> u64 {
        self.signature_hash()
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