//! On-disk / in-memory format of an exported type database.
//!
//! A `TypeDb` is a flat, versioned map from fully-qualified Rust type name to
//! its layout (size, alignment, fields or variants). It is produced once from
//! DWARF (see [`crate::export`]) and loaded by the server at startup, so
//! generated JIT code and host structs agree on offsets/padding.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use serde::{Deserialize, Serialize};

/// A reference to a type by qualified name (`exsolis_common::ChunkBlocks`,
/// `u8`, `&str`, `[u8; 32]`, ...).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TypeName(pub String);

impl TypeName {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TypeName {
    fn from(s: &str) -> Self {
        TypeName(s.to_owned())
    }
}

impl From<String> for TypeName {
    fn from(s: String) -> Self {
        TypeName(s)
    }
}

impl std::fmt::Display for TypeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// One struct/union field: offset in bytes from the start of the value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: TypeName,
    pub offset: u64,
}

/// Struct/union payload (identical shape; `union` fields all offset 0).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructType {
    pub is_union: bool,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayType {
    pub len: u64,
    pub elem: TypeName,
}

/// Discriminant storage of an enum (niche or explicit tag).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Discriminant {
    pub name: Option<String>,
    pub ty: TypeName,
    pub offset: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    /// Value to compare the discriminant against.
    pub discr: i128,
    /// Payload fields laid out from the start of the enum.
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnumType {
    pub discriminant: Discriminant,
    pub variants: Vec<Variant>,
}

/// Types whose layout is not expressible as fields/variants but still matter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpaqueKind {
    /// fn(A, B) -> R with the `fn(A,B)->R` name of the referenced type.
    Function { monomorphized: bool },
    /// A `dyn Trait` fat pointer.
    DynFn,
    /// Non-Rust type we keep by name only.
    Foreign,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeArgs {
    Struct(StructType),
    Enum(EnumType),
    Array(ArrayType),
    Pointer { pointee: TypeName },
    Opaque(OpaqueKind),
}

/// A single recorded type: identity + layout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeDef {
    pub name: TypeName,
    pub size: u64,
    pub align: u64,
    pub args: TypeArgs,
}

/// Format metadata. Bump [`FORMAT_VERSION`] whenever the layout of [`TypeDef`]
/// or this header changes in an incompatible way.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Head {
    /// `TSDB`.
    pub magic: [u8; 4],
    /// Ever-incrementing format revision.
    pub version: u32,
    /// DWARF direction convention of the copied structs:
    /// `true` = structs are packed little-endian.
    pub packed: bool,
    /// Names which are file types. Always empty for now.
    pub file_types: Vec<TypeName>,
    /// Bytes free after de-duplicating repeated DWARF names.
    pub dedup_bytes: u32,
    /// Implementation-defined `<Trait>::Assoc` style info. Always empty for now.
    pub defined_types: Vec<String>,
    /// Producer of the exported types (the DWARF `DW_AT_producer` string).
    pub producer: String,
    /// Binary the types were exported from (informational).
    pub source: String,
    /// Extra `key=value` metadata.
    pub attributes: Vec<(String, String)>,
}

const MAGIC: &[u8; 4] = b"TSDB";

pub const FORMAT_VERSION: u32 = 1;

impl Head {
    pub fn new() -> Self {
        Head {
            magic: *MAGIC,
            version: FORMAT_VERSION,
            ..Default::default()
        }
    }

    pub fn is_valid(&self) -> bool {
        self.magic == *MAGIC && self.version <= FORMAT_VERSION
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeDb {
    pub head: Head,
    /// Canonical types by qualified name.
    pub types: BTreeMap<String, TypeDef>,
    /// Legacy/demangled aliases → canonical name (`&T`, `*mut T`, typedefs).
    pub aliases: BTreeMap<String, String>,
}

/// Errors returned by [`TypeDb`] queries/layout walks.
#[derive(Debug, thiserror::Error)]
pub enum TypeDbError {
    /// The database referenced a type name that is not present.
    #[error("unknown type `{0}`")]
    UnknownType(String),
    /// A struct is (indirectly) a member of itself.
    #[error("recursive type `{0}` has no stable layout")]
    RecursiveType(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    Struct(StructType),
    Enum(EnumType),
    Array(ArrayType),
    Pointer { pointee: TypeName },
    /// A type with no further structure of interest.
    Opaque,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub name: TypeName,
    pub size: u64,
    pub align: u64,
    pub body: Resolution,
}

impl TypeDb {
    pub fn new() -> Self {
        Self {
            head: Head::new(),
            types: BTreeMap::new(),
            aliases: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Look up a type definition, following one alias hop.
    pub fn get(&self, name: &str) -> Option<&TypeDef> {
        self.get_raw(name).or_else(|| {
            let canonical = self.aliases.get(name)?;
            self.types.get(canonical)
        })
    }

    /// Look up a type definition by canonical name only (no aliases).
    pub fn get_raw(&self, name: &str) -> Option<&TypeDef> {
        self.types.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TypeDef)> {
        self.types.iter()
    }

    /// Canonical name for a queried name, resolving `<T as Trait>::Assoc`,
    /// `<T>::Assoc` and legacy aliases.
    pub fn canonicalize(&self, name: &str) -> String {
        let mut current = name.to_owned();
        // At most two normalizations are ever needed: one alias hop plus one
        // trait-qualification strip.
        for _ in 0..2 {
            let Some(def) = self.types.get(&current) else {
                if let Some(next) = self.aliases.get(&current)
                    && *next != current {
                        current = next.clone();
                        continue;
                    }
                return current;
            };
            if let TypeArgs::Struct(_) | TypeArgs::Enum(_) = &def.args {
                let normalized = strip_trait_qualification(&current);
                if normalized != current && self.types.contains_key(&normalized) {
                    current = normalized;
                    continue;
                }
            }
            break;
        }
        current
    }

    /// Every alias that resolves to `canonical`.
    pub fn aliases_of(&self, canonical: &str) -> Vec<&str> {
        self.aliases
            .iter()
            .filter(|(_, target)| target.as_str() == canonical)
            .map(|(alias, _)| alias.as_str())
            .collect()
    }

    /// Remove a type and everything that (transitively) references it,
    /// returning how many definitions were dropped.
    pub fn remove_recursive(&mut self, name: &str) -> usize {
        let mut removed: HashSet<String> = HashSet::new();
        removed.insert(self.canonicalize(name));

        // Fixed point: keep dropping types whose members point at a removed
        // type. Structural members can only be structs/enums.
        loop {
            let dead: Vec<String> = self
                .types
                .iter()
                .filter(|(_, def)| {
                    referenced_names(def)
                        .iter()
                        .any(|r| removed.contains(&self.canonicalize(r.as_str())))
                })
                .map(|(name, _)| name.clone())
                .collect();
            let mut changed = false;
            for name in dead {
                if removed.insert(name.clone()) {
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        removed.iter().filter(|n| self.types.remove(n.as_str()).is_some()).count()
    }

    /// Resolve a type and (recursively) its nested members/variants.
    ///
    /// Pointers/arrays stop at the next named type: the caller resolves those.
    pub fn resolve(&self, name: &str) -> Result<Resolved, TypeDbError> {
        let canonical = self.canonicalize(name);
        let def = self
            .types
            .get(&canonical)
            .ok_or_else(|| TypeDbError::UnknownType(canonical.clone()))?;
        let body = match &def.args {
            TypeArgs::Struct(s) => Resolution::Struct(s.clone()),
            TypeArgs::Enum(e) => Resolution::Enum(e.clone()),
            TypeArgs::Array(a) => Resolution::Array(a.clone()),
            TypeArgs::Pointer { pointee } => Resolution::Pointer {
                pointee: TypeName(self.canonicalize(pointee.as_str())),
            },
            TypeArgs::Opaque(_) => Resolution::Opaque,
        };
        Ok(Resolved {
            name: TypeName(canonical),
            size: def.size,
            align: def.align,
            body,
        })
    }

    /// Flatten a struct/enum into a list of leaf fields with absolute offsets.
    ///
    /// Nested structs are inlined with their offsets shifted; pointers stop the
    /// walk (JIT codegen treats `&T`/`*mut T` as opaque slots).
    pub fn layout(&self, name: &str) -> Result<Vec<LeafField>, TypeDbError> {
        self.layout_inner(name, &mut Vec::new())
    }

    fn layout_inner(
        &self,
        name: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec<LeafField>, TypeDbError> {
        let canonical = self.canonicalize(name);
        if stack.iter().any(|s| s == &canonical) {
            return Err(TypeDbError::RecursiveType(canonical));
        }
        stack.push(canonical.clone());
        let out = self.layout_resolved(&canonical, stack);
        stack.pop();
        out
    }

    fn layout_resolved(
        &self,
        canonical: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec<LeafField>, TypeDbError> {
        let resolved = self.resolve(canonical)?;
        Ok(match resolved.body {
            Resolution::Struct(s) => {
                let mut leaves = Vec::new();
                for field in &s.fields {
                    let base = if s.is_union { 0 } else { field.offset };
                    let nested = self.layout_inner(field.ty.as_str(), stack)?;
                    if nested.is_empty() {
                        leaves.push(LeafField {
                            name: field.name.clone(),
                            ty: field.ty.clone(),
                            offset: base,
                        });
                    } else {
                        leaves.extend(nested.into_iter().map(|mut l| {
                            l.offset += base;
                            l.prefix(&field.name);
                            l
                        }));
                    }
                }
                leaves
            }
            Resolution::Enum(e) => {
                let mut leaves = Vec::new();
                for variant in &e.variants {
                    for field in &variant.fields {
                        let nested = self.layout_inner(field.ty.as_str(), stack)?;
                        let path = format!("{}::{}", variant.name, field.name);
                        if nested.is_empty() {
                            leaves.push(LeafField {
                                name: path,
                                ty: field.ty.clone(),
                                offset: field.offset,
                            });
                        } else {
                            leaves.extend(nested.into_iter().map(|mut l| {
                                l.offset += field.offset;
                                l.prefix(&path);
                                l
                            }));
                        }
                    }
                }
                leaves
            }
            Resolution::Array(_) | Resolution::Pointer { .. } | Resolution::Opaque => {
                vec![LeafField {
                    name: String::new(),
                    ty: resolved.name.clone(),
                    offset: 0,
                }]
            }
        })
    }

    /// Serialize to the versioned on-disk format (bincode).
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Load from the versioned on-disk format (bincode).
    pub fn from_bytes(bytes: &[u8]) -> Result<TypeDb, TypeDbDecodeError> {
        let db: TypeDb = bincode::deserialize(bytes).map_err(TypeDbDecodeError::Decode)?;
        if !db.head.is_valid() {
            return Err(TypeDbDecodeError::BadHead);
        }
        if db.head.version > FORMAT_VERSION {
            return Err(TypeDbDecodeError::VersionTooNew(db.head.version));
        }
        Ok(db)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TypeDbDecodeError {
    #[error("failed to decode type database: {0}")]
    Decode(bincode::Error),
    #[error("type database header is missing or from an incompatible version")]
    BadHead,
    #[error("type database does not support this format version ({0} > {FORMAT_VERSION})")]
    VersionTooNew(u32),
}

/// A leaf of a flattened layout: offset from the start of the root value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafField {
    /// `Struct::field` path (empty for the root itself).
    pub name: String,
    pub ty: TypeName,
    pub offset: u64,
}

impl LeafField {
    fn prefix(&mut self, path: &str) {
        if self.name.is_empty() {
            self.name = path.to_owned();
        } else {
            self.name = format!("{path}::{}", self.name);
        }
    }
}

pub fn referenced_names(def: &TypeDef) -> BTreeSet<TypeName> {
    let mut out = BTreeSet::new();
    match &def.args {
        TypeArgs::Struct(s) => out.extend(s.fields.iter().map(|f| f.ty.clone())),
        TypeArgs::Enum(e) => {
            out.insert(e.discriminant.ty.clone());
            for v in &e.variants {
                out.extend(v.fields.iter().map(|f| f.ty.clone()));
            }
        }
        TypeArgs::Array(a) => {
            out.insert(a.elem.clone());
        }
        TypeArgs::Pointer { pointee } => {
            out.insert(pointee.clone());
        }
        TypeArgs::Opaque(_) => {}
    }
    out
}

/// `<T as Trait>::Assoc` → `<T>::Assoc` (our databases never store the
/// trait-qualified form, so this is a pure lookup key).
fn strip_trait_qualification(name: &str) -> String {
    let Some(rest) = name.strip_prefix('<') else {
        return name.to_owned();
    };
    let Some(end) = rest.find(" as ") else {
        return name.to_owned();
    };
    format!("<{}::{}", &rest[..end], &rest[end + 4..])
}
