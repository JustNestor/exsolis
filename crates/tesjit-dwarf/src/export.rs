//! DWARF → [`TypeDb`] exporter.
//!
//! Walks the DWARF debug info of an object file (the host binary, a debug
//! build of the server, ...), collects every Rust type it can describe with a
//! struct/enum/array/pointer layout, and returns a [`TypeDb`] ready to be
//! saved and loaded at server start.
//!
//! Demangling is applied to every DWARF name (`rustc-demangle`), so type keys
//! read like `exsolis_common::ChunkBlocks` and `Vec<T>` instead of `_ZN...`.
//!
//! Known limitations (documented for the ABI spike):
//! - generic instantiations collapse onto their monomorphized DWARF name
//! - niche-encoded enums are exported as a plain struct with an artificial
//!   `__field0` tag member
//! - `.debug_types` (DWARF 4 type units) is not walked; only `.debug_info`

use std::collections::HashMap;

use gimli::RunTimeEndian;
use object::{Object, ObjectSection};

use crate::schema::{
    ArrayType, Discriminant, EnumType, Field, Head, OpaqueKind, StructType, TypeArgs, TypeDb,
    TypeDef, TypeName, Variant,
};

/// What the exporter did, for CLI/CI reporting.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportStats {
    pub compilation_units: usize,
    pub dies: usize,
    pub types: usize,
    pub structs: usize,
    pub unions: usize,
    pub enums: usize,
    pub arrays: usize,
    pub pointers: usize,
    pub aliases: usize,
    pub opaque: usize,
    /// DWARF tags the exporter ignored (not layout-bearing).
    pub skipped_dies: usize,
    /// Bytes of the source binary (informational).
    pub file_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("failed to parse object file: {0}")]
    Object(#[from] object::read::Error),
    #[error("failed to read DWARF: {0}")]
    Dwarf(#[from] gimli::Error),
    #[error("`{path}` has no exported types (no debug info?)")]
    NoDebugInfo { path: String },
}

type R<'a> = gimli::EndianSlice<'a, RunTimeEndian>;

/// Export the type database from the DWARF in `data`.
///
/// `data` is only borrowed while parsing: the returned database owns plain
/// strings and does not keep the binary alive.
pub fn export(data: &[u8], source_name: &str) -> Result<(TypeDb, ExportStats), ExportError> {
    let file = object::File::parse(data)?;
    let endian = if file.is_little_endian() {
        RunTimeEndian::Little
    } else {
        RunTimeEndian::Big
    };
    // Borrow each DWARF section straight out of the parsed file: uncompressed
    // debug sections (the norm) borrow the caller's buffer, so nothing has to
    // be leaked or copied into the gimli readers.
    let mut load_section = |id: gimli::SectionId| -> Result<R<'_>, gimli::Error> {
        let bytes = match file.section_by_name(id.name()) {
            Some(section) => section.data().unwrap_or_default(),
            None => b"",
        };
        Ok(gimli::EndianSlice::new(bytes, endian))
    };
    let dwarf = gimli::Dwarf::load(&mut load_section)?;

    let mut builder = Builder::new(&dwarf);
    builder.walk()?;
    let db = builder.finish(source_name, data.len() as u64);
    if db.types.is_empty() {
        return Err(ExportError::NoDebugInfo {
            path: source_name.to_owned(),
        });
    }
    Ok((db, builder.stats))
}

// ───────────────────────────── DIE recording ─────────────────────────────

/// A recorded DIE: just enough to reconstruct Rust layouts.
struct DieRec {
    tag: gimli::DwTag,
    name: Option<String>,
    byte_size: Option<u64>,
    alignment: Option<u64>,
    /// `DW_AT_type`: `(unit index, offset)` for in-unit refs, or an absolute
    /// debug-info offset for `DW_FORM_ref_addr`.
    type_ref: Option<TypeRef>,
    location: Option<u64>,
    count: Option<u64>,
    const_value: Option<i128>,
}

/// Where a `DW_AT_type`-style reference points to.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TypeRef {
    /// Some(unit) for in-unit refs; `None` means `offset` is an absolute
    /// `.debug_info` offset (resolved after all units are collected).
    unit: Option<usize>,
    offset: u64,
}

impl TypeRef {
    fn in_unit(unit: usize, offset: gimli::UnitOffset) -> Self {
        Self {
            unit: Some(unit),
            offset: offset.0 as u64,
        }
    }

    fn absolute(offset: gimli::DebugInfoOffset) -> Self {
        Self {
            unit: None,
            offset: offset.0 as u64,
        }
    }
}

/// Position of one DIE inside the collected units.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DieIndex {
    unit: usize,
    /// Index into `UnitDies::order` / `children` / `parents`.
    die: usize,
}

/// All DIEs of one compilation unit.
struct UnitDies {
    /// Insertion-ordered (deterministic) list of DIE offsets.
    order: Vec<gimli::UnitOffset>,
    dies: HashMap<gimli::UnitOffset, DieRec>,
    /// For each DIE index: indices of its direct children, in DIE order.
    children: Vec<Vec<usize>>,
    /// For each DIE index: parent index, or `usize::MAX` for the unit root.
    parents: Vec<usize>,
    /// Absolute `.debug_info` offset of this unit.
    start: u64,
}

impl UnitDies {
    fn rec(&self, index: DieIndex) -> &DieRec {
        &self.dies[&self.order[index.die]]
    }
}

struct Builder<'a, 'b> {
    dwarf: &'a gimli::Dwarf<R<'b>>,
    units: Vec<UnitDies>,
    /// Absolute offset → (unit index, DIE index) for `DW_FORM_ref_addr`.
    abs_index: HashMap<u64, DieIndex>,
    stats: ExportStats,
    producer: String,
}

impl<'a, 'b> Builder<'a, 'b> {
    fn new(dwarf: &'a gimli::Dwarf<R<'b>>) -> Self {
        Self {
            dwarf,
            units: Vec::new(),
            abs_index: HashMap::new(),
            stats: ExportStats::default(),
            producer: String::new(),
        }
    }

    // ─────────────────────────── pass 1: collect ───────────────────────────

    fn walk(&mut self) -> Result<(), gimli::Error> {
        let mut headers = self.dwarf.units();
        while let Some(header) = headers.next()? {
            let unit = self.dwarf.unit(header)?;
            let _abbrev = &unit.abbreviations;
            let unit_index = self.units.len();
            let mut collected = UnitDies {
                order: Vec::new(),
                dies: HashMap::new(),
                children: Vec::new(),
                parents: Vec::new(),
                start: unit
                    .header
                    .offset()
                    .as_debug_info_offset()
                    .map(|o| o.0 as u64)
                    .unwrap_or(0),
            };
            let mut cursor = unit.entries();
            // `next_dfs` yields deltas; the root DIE comes out at depth 1.
            let mut depth: isize = 0;
            // `stack[k]` is the DIE index at depth `k + 1`.
            let mut stack: Vec<usize> = Vec::new();
            while let Some((delta, entry)) = cursor.next_dfs()? {
                depth += delta;
                let rec = self.record(&unit, unit_index, entry)?;
                let die = collected.order.len();
                collected.parents.push(usize::MAX);
                collected.children.push(Vec::new());
                collected.dies.insert(entry.offset(), rec);
                collected.order.push(entry.offset());
                if depth >= 2 {
                    if let Some(&parent) = stack.get(depth as usize - 2) {
                        collected.parents[die] = parent;
                        collected.children[parent].push(die);
                    }
                }
                stack.resize(depth.max(1) as usize, usize::MAX);
                stack[depth as usize - 1] = die;
            }
            for (die, offset) in collected.order.iter().enumerate() {
                self.abs_index
                    .insert(collected.start + offset.0 as u64, DieIndex { unit: unit_index, die });
            }
            self.stats.dies += collected.order.len();
            self.stats.compilation_units += 1;
            self.units.push(collected);
        }
        Ok(())
    }

    fn record(
        &mut self,
        unit: &gimli::Unit<R<'b>>,
        unit_index: usize,
        entry: &gimli::DebuggingInformationEntry<R<'b>>,
    ) -> Result<DieRec, gimli::Error> {
        let mut rec = DieRec {
            tag: entry.tag(),
            name: None,
            byte_size: None,
            alignment: None,
            type_ref: None,
            location: None,
            count: None,
            const_value: None,
        };
        let mut attrs = entry.attrs();
        while let Some(attr) = attrs.next()? {
            match attr.name() {
                gimli::DW_AT_name => rec.name = self.attr_string(unit, attr.value())?,
                gimli::DW_AT_byte_size => rec.byte_size = attr.udata_value(),
                gimli::DW_AT_alignment => rec.alignment = attr.udata_value(),
                gimli::DW_AT_producer if self.producer.is_empty() => {
                    self.producer = self
                        .attr_string(unit, attr.value())?
                        .unwrap_or_default();
                }
                gimli::DW_AT_type => {
                    rec.type_ref = match attr.value() {
                        gimli::AttributeValue::UnitRef(offset) => {
                            Some(TypeRef::in_unit(unit_index, offset))
                        }
                        gimli::AttributeValue::DebugInfoRef(offset) => {
                            Some(TypeRef::absolute(offset))
                        }
                        _ => None,
                    }
                }
                gimli::DW_AT_data_member_location => {
                    rec.location = attr
                        .udata_value()
                        .or_else(|| attr.sdata_value().and_then(|v| u64::try_from(v).ok()));
                }
                gimli::DW_AT_count => rec.count = attr.udata_value(),
                gimli::DW_AT_const_value => {
                    let value: Option<i64> = attr
                        .sdata_value()
                        .or_else(|| attr.udata_value().map(|v| v as i64));
                    rec.const_value = value.map(i128::from);
                }
                _ => {}
            }
        }
        Ok(rec)
    }

    fn attr_string(
        &self,
        unit: &gimli::Unit<R<'b>>,
        value: gimli::AttributeValue<R<'b>>,
    ) -> Result<Option<String>, gimli::Error> {
        match self.dwarf.attr_string(unit, value) {
            Ok(reader) => Ok(Some(demangle(&reader.to_string_lossy()))),
            Err(gimli::Error::UnsupportedAttributeForm) => Ok(None),
            Err(e) => Err(e),
        }
    }

    // ────────────────────────── pass 2: emit ──────────────────────────

    fn finish(&mut self, source_name: &str, file_bytes: u64) -> TypeDb {
        let units = std::mem::take(&mut self.units);
        let mut db = TypeDb::new();
        let mut seen: HashMap<DieIndex, TypeName> = HashMap::new();

        for unit in 0..units.len() {
            let order_len = units[unit].order.len();
            for die in 0..order_len {
                let idx = DieIndex { unit, die };
                emit_die(&units, idx, &mut db, &mut seen, &mut self.stats);
            }
        }
        infer_alignments(&mut db);

        self.stats.types = db.types.len();
        self.stats.aliases = db.aliases.len();
        self.stats.file_bytes = file_bytes;
        db.head = Head {
            magic: *b"TSDB",
            version: crate::schema::FORMAT_VERSION,
            packed: true,
            file_types: Vec::new(),
            dedup_bytes: 0,
            defined_types: Vec::new(),
            producer: std::mem::take(&mut self.producer),
            source: source_name.to_owned(),
            attributes: vec![
                ("types".to_owned(), db.types.len().to_string()),
                ("structs".to_owned(), self.stats.structs.to_string()),
                ("enums".to_owned(), self.stats.enums.to_string()),
                ("arrays".to_owned(), self.stats.arrays.to_string()),
                ("pointers".to_owned(), self.stats.pointers.to_string()),
                ("aliases".to_owned(), db.aliases.len().to_string()),
            ],
        };
        db
    }
}

/// Rust-flavoured demangling of a DWARF name (no-op for plain names).
fn demangle(name: &str) -> String {
    rustc_demangle::demangle(name).to_string()
}

// ───────────────────────────── type emission ─────────────────────────────

/// Emit the type for one DIE (recursively emitting referents), returning its
/// canonical name so field types can reference it.
fn emit_die(
    units: &[UnitDies],
    idx: DieIndex,
    db: &mut TypeDb,
    emitted: &mut HashMap<DieIndex, TypeName>,
    stats: &mut ExportStats,
) -> Option<TypeName> {
    if let Some(name) = emitted.get(&idx) {
        return Some(name.clone());
    }
    let rec = units[idx.unit].rec(idx);
    let name = match rec.tag {
        gimli::DW_TAG_structure_type | gimli::DW_TAG_union_type => {
            let name = name_for(units, idx)?;
            let size = rec.byte_size?;
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            let body = if let Some(vp) = child_with_tag(units, idx, gimli::DW_TAG_variant_part) {
                let (discriminant, variants) = emit_enum(units, idx, vp, db, emitted, stats);
                TypeArgs::Enum(EnumType {
                    discriminant,
                    variants,
                })
            } else {
                let mut fields = Vec::new();
                for (ordinal, child) in children_of(units, idx).iter().enumerate() {
                    let crec = units[child.unit].rec(*child);
                    if crec.tag != gimli::DW_TAG_member {
                        continue;
                    }
                    let Some(tref) = crec.type_ref.and_then(|r| resolve_index(units, r)) else {
                        continue;
                    };
                    let Some(ty) = emit_die(units, tref, db, emitted, stats) else {
                        continue;
                    };
                    fields.push(Field {
                        name: crec
                            .name
                            .clone()
                            .unwrap_or_else(|| format!("__field{ordinal}")),
                        ty,
                        offset: crec.location.unwrap_or(0),
                    });
                }
                if rec.tag == gimli::DW_TAG_union_type {
                    stats.unions += 1;
                } else {
                    stats.structs += 1;
                }
                TypeArgs::Struct(StructType {
                    is_union: rec.tag == gimli::DW_TAG_union_type,
                    fields,
                })
            };
            if matches!(body, TypeArgs::Enum(_)) {
                stats.enums += 1;
            }
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size,
                    align: rec.alignment.unwrap_or(0),
                    args: body,
                },
            );
            name
        }
        gimli::DW_TAG_enumeration_type => {
            let name = name_for(units, idx)?;
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            let size = rec.byte_size.unwrap_or(0);
            let discriminant = rec
                .type_ref
                .and_then(|r| resolve_index(units, r))
                .and_then(|d| emit_die(units, d, db, emitted, stats))
                .unwrap_or_else(|| TypeName::from("i32"));
            let mut variants = Vec::new();
            for child in children_of(units, idx) {
                let crec = units[child.unit].rec(child);
                if crec.tag != gimli::DW_TAG_enumerator {
                    continue;
                }
                variants.push(Variant {
                    name: crec.name.clone().unwrap_or_else(|| format!("__v{}", variants.len())),
                    discr: crec.const_value.unwrap_or(0),
                    fields: Vec::new(),
                });
            }
            stats.enums += 1;
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size,
                    align: rec.alignment.unwrap_or(0),
                    args: TypeArgs::Enum(EnumType {
                        discriminant: Discriminant {
                            name: None,
                            ty: discriminant,
                            offset: 0,
                        },
                        variants,
                    }),
                },
            );
            name
        }
        gimli::DW_TAG_array_type => {
            let name = name_for(units, idx)?;
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            let elem = rec
                .type_ref
                .and_then(|r| resolve_index(units, r))
                .and_then(|e| emit_die(units, e, db, emitted, stats))?;
            let len = children_of(units, idx)
                .iter()
                .map(|c| units[c.unit].rec(*c))
                .find(|c| c.tag == gimli::DW_TAG_subrange_type)
                .and_then(|c| c.count)
                .unwrap_or(0);
            stats.arrays += 1;
            let name = if name.0.is_empty() {
                TypeName::from(format!("[{elem}; {len}]"))
            } else {
                TypeName::from(format!("[{}; {len}]", name.0))
            };
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size: rec.byte_size.unwrap_or(0),
                    align: rec.alignment.unwrap_or(0),
                    args: TypeArgs::Array(ArrayType { len, elem }),
                },
            );
            name
        }
        gimli::DW_TAG_pointer_type | gimli::DW_TAG_reference_type => {
            let pointee = rec
                .type_ref
                .and_then(|r| resolve_index(units, r))
                .and_then(|p| emit_die(units, p, db, emitted, stats));
            let name = name_for(units, idx)
                .unwrap_or_else(|| TypeName::from(format!("&{}", pointee.as_ref().map(|p| p.0.as_str()).unwrap_or("?"))));
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            stats.pointers += 1;
            let Some(pointee) = pointee else {
                // Opaque pointer: size unknown but pointer-sized.
                db.types.insert(
                    name.0.clone(),
                    TypeDef {
                        name: name.clone(),
                        size: 0,
                        align: 8,
                        args: TypeArgs::Opaque(OpaqueKind::Foreign),
                    },
                );
                return Some(name);
            };
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size: 8,
                    align: 8,
                    args: TypeArgs::Pointer { pointee },
                },
            );
            name
        }
        gimli::DW_TAG_base_type => {
            let name = name_for(units, idx)?;
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            stats.opaque += 1;
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size: rec.byte_size.unwrap_or(0),
                    align: rec.alignment.unwrap_or(0),
                    args: TypeArgs::Opaque(OpaqueKind::Foreign),
                },
            );
            name
        }
        gimli::DW_TAG_subroutine_type => {
            // `fn(A, B) -> R` — recorded for name/signature checking only.
            let name = name_for(units, idx)?;
            emitted.insert(idx, name.clone());
            if db.types.contains_key(name.as_str()) {
                return Some(name);
            }
            stats.opaque += 1;
            db.types.insert(
                name.0.clone(),
                TypeDef {
                    name: name.clone(),
                    size: 8,
                    align: 8,
                    args: TypeArgs::Opaque(OpaqueKind::Function { monomorphized: true }),
                },
            );
            name
        }
        gimli::DW_TAG_typedef
        | gimli::DW_TAG_const_type
        | gimli::DW_TAG_volatile_type
        | gimli::DW_TAG_restrict_type
        | gimli::DW_TAG_atomic_type => {
            let Some(name) = name_for(units, idx) else {
                return None;
            };
            emitted.insert(idx, name.clone());
            let Some(target) = rec
                .type_ref
                .and_then(|r| resolve_index(units, r))
                .and_then(|t| emit_die(units, t, db, emitted, stats))
            else {
                return None;
            };
            if name != target && !db.aliases.contains_key(name.as_str()) {
                db.aliases.insert(name.0.clone(), target.0.clone());
                stats.aliases += 1;
            }
            name
        }
        _ => {
            stats.skipped_dies += 1;
            TypeName::from("")
        }
    };
    if name.0.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Best-guess alignment for a scalar from its size (used when DWARF omitted
/// `DW_AT_alignment` and the type is opaque).
fn size_align(size: u64) -> u64 {
    let size = size.max(1);
    if size.is_power_of_two() {
        size
    } else {
        1 << (u64::BITS - 1 - size.leading_zeros())
    }
}

/// Build the `EnumType` for a struct with a `DW_TAG_variant_part` child.
fn emit_enum(
    units: &[UnitDies],
    parent: DieIndex,
    variant_part: DieIndex,
    db: &mut TypeDb,
    emitted: &mut HashMap<DieIndex, TypeName>,
    stats: &mut ExportStats,
) -> (Discriminant, Vec<Variant>) {
    let vp_children = children_of(units, variant_part);

    // The discriminant lives in the first `DW_TAG_member` of the variant part.
    let mut discriminant = Discriminant {
        name: None,
        ty: TypeName::from("i32"),
        offset: 0,
    };
    if let Some(first) = vp_children.first() {
        let rec = units[first.unit].rec(*first);
        if rec.tag == gimli::DW_TAG_member {
            discriminant.offset = rec.location.unwrap_or(0);
            if let Some(tty) = rec
                .type_ref
                .and_then(|r| resolve_index(units, r))
                .and_then(|t| emit_die(units, t, db, emitted, stats))
            {
                discriminant.ty = tty;
            }
        }
    }

    let mut variants = Vec::new();
    for child in vp_children.iter().skip(1) {
        let rec = units[child.unit].rec(*child);
        if rec.tag != gimli::DW_TAG_variant {
            continue;
        }
        let mut fields = Vec::new();
        for (ordinal, member) in children_of(units, *child).iter().enumerate() {
            let crec = units[member.unit].rec(*member);
            if crec.tag != gimli::DW_TAG_member {
                continue;
            }
            let Some(tref) = crec.type_ref.and_then(|r| resolve_index(units, r)) else {
                continue;
            };
            let Some(ty) = emit_die(units, tref, db, emitted, stats) else {
                continue;
            };
            fields.push(Field {
                name: crec
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("__field{ordinal}")),
                ty,
                offset: crec.location.unwrap_or(0),
            });
        }
        variants.push(Variant {
            name: rec
                .name
                .clone()
                .unwrap_or_else(|| format!("__v{}", variants.len())),
            discr: rec.const_value.unwrap_or(0),
            fields,
        });
    }
    let _ = parent;
    (discriminant, variants)
}

/// Direct children of a DIE.
fn children_of(units: &[UnitDies], idx: DieIndex) -> Vec<DieIndex> {
    units[idx.unit].children[idx.die]
        .iter()
        .map(|&die| DieIndex { unit: idx.unit, die })
        .collect()
}

/// First direct child with `tag`.
fn child_with_tag(units: &[UnitDies], idx: DieIndex, tag: gimli::DwTag) -> Option<DieIndex> {
    children_of(units, idx)
        .into_iter()
        .find(|child| units[child.unit].rec(*child).tag == tag)
}

/// Name of a DIE, synthesized from the parent when the DIE is anonymous.
fn name_for(units: &[UnitDies], idx: DieIndex) -> Option<TypeName> {
    let rec = units[idx.unit].rec(idx);
    if let Some(name) = &rec.name {
        if !name.is_empty() {
            return Some(TypeName::from(name.clone()));
        }
    }
    // Anonymous: derive from the enclosing declaration.
    let mut parent = units[idx.unit].parents[idx.die];
    let mut suffix = idx.die.to_string();
    while parent != usize::MAX {
        let prec = units[idx.unit].rec(DieIndex { unit: idx.unit, die: parent });
        if let Some(name) = &prec.name {
            if !name.is_empty() {
                return Some(TypeName::from(format!("{name}::{{anon{suffix}}}")));
            }
        }
        suffix = format!("{}{}", parent, idx.die);
        parent = units[idx.unit].parents[parent];
    }
    None
}

/// Resolve a recorded type reference into a DIE index.
fn resolve_index(units: &[UnitDies], tref: TypeRef) -> Option<DieIndex> {
    match tref.unit {
        Some(unit) => {
            let unit_dies = units.get(unit)?;
            let offset = gimli::UnitOffset(tref.offset as usize);
            if unit_dies.dies.contains_key(&offset) {
                let die = unit_dies.order.iter().position(|o| o.0 as u64 == tref.offset)?;
                Some(DieIndex { unit, die })
            } else {
                // Cross-unit absolute reference recorded as in-unit: search all.
                units
                    .iter()
                    .position(|u| u.dies.contains_key(&offset))
                    .and_then(|unit| {
                        let die = units[unit]
                            .order
                            .iter()
                            .position(|o| o.0 as u64 == tref.offset)?;
                        Some(DieIndex { unit, die })
                    })
            }
        }
        None => {
            // Absolute `.debug_info` offset: find the unit that starts at or
            // before this offset, then the DIE inside it.
            let mut containing = None;
            for (index, unit) in units.iter().enumerate() {
                if unit.start <= tref.offset {
                    if let Some(die) = unit
                        .order
                        .iter()
                        .position(|o| unit.start + o.0 as u64 == tref.offset)
                    {
                        containing = Some(DieIndex { unit: index, die });
                    }
                }
            }
            containing
        }
    }
}

/// Fill in `align` for every type where DWARF did not record one.
///
/// Structs derive it from their members, arrays from their element, pointers
/// keep the (x86-64) 8-byte placeholder, and scalars fall back to the largest
/// power of two that fits in their size.
fn infer_alignments(db: &mut TypeDb) {
    for _ in 0..8 {
        let mut updates: Vec<(String, u64)> = Vec::new();
        for def in db.types.values() {
            if def.align != 0 {
                continue;
            }
            let align = match &def.args {
                TypeArgs::Struct(s) => s
                    .fields
                    .iter()
                    .filter_map(|f| db.get(f.ty.as_str()))
                    .map(|f| f.align)
                    .max()
                    .unwrap_or(0),
                TypeArgs::Enum(e) => db.get(e.discriminant.ty.as_str()).map(|d| d.align).unwrap_or(0),
                TypeArgs::Array(a) => db.get(a.elem.as_str()).map(|a| a.align).unwrap_or(0),
                TypeArgs::Pointer { .. } => 8,
                TypeArgs::Opaque(_) => size_align(def.size),
            };
            if align > 0 {
                updates.push((def.name.0.clone(), align));
            }
        }
        if updates.is_empty() {
            break;
        }
        for (name, align) in updates {
            if let Some(def) = db.types.get_mut(&name) {
                if def.align == 0 {
                    def.align = align;
                }
            }
        }
    }
}
