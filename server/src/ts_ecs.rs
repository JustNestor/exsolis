//! Bevy table access for tesjit's zero-copy query ABI.

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use bevy_ecs::component::{ComponentId, StorageType};
use bevy_ecs::prelude::World;
use bevy_ecs::storage::{TableId, TableRow};
use nonmax::NonMaxU32;
use tesjit::{QueryFilterKind, StructField, StructLayout, Ty, abi::QueryInput};
use tesjit_dwarf::{TypeArgs, TypeDb, TypeDef};

pub(crate) trait TsSchema: 'static {
    fn ts_schema() -> Result<StructLayout, String>;
}

pub(crate) trait TsSchemaField {
    fn ts_field() -> Result<(Ty, bool), String>;
}

macro_rules! scalar_schema_fields {
    ($($type:ty => $ty:expr),* $(,)?) => {
        $(
            impl TsSchemaField for $type {
                fn ts_field() -> Result<(Ty, bool), String> {
                    Ok(($ty, true))
                }
            }
        )*
    };
}

scalar_schema_fields! {
    bool => Ty::Bool,
    i8 => Ty::I8,
    i16 => Ty::I16,
    i32 => Ty::I32,
    i64 => Ty::I64,
    u8 => Ty::U8,
    u16 => Ty::U16,
    u32 => Ty::U32,
    u64 => Ty::U64,
    f32 => Ty::F32,
    f64 => Ty::F64,
}

impl TsSchemaField for char {
    fn ts_field() -> Result<(Ty, bool), String> {
        Ok((Ty::U32, false))
    }
}

impl TsSchemaField for glam::Vec3 {
    fn ts_field() -> Result<(Ty, bool), String> {
        Ok((Ty::Struct(Arc::new(vec3_schema()?)), true))
    }
}

fn vec3_schema() -> Result<StructLayout, String> {
    let mut layout = StructLayout::from_parts(
        "Vec3",
        vec![
            StructField {
                name: "x".to_owned(),
                ty: Ty::F32,
                offset: std::mem::offset_of!(glam::Vec3, x) as u64,
            },
            StructField {
                name: "y".to_owned(),
                ty: Ty::F32,
                offset: std::mem::offset_of!(glam::Vec3, y) as u64,
            },
            StructField {
                name: "z".to_owned(),
                ty: Ty::F32,
                offset: std::mem::offset_of!(glam::Vec3, z) as u64,
            },
        ],
        std::mem::size_of::<glam::Vec3>() as u64,
        std::mem::align_of::<glam::Vec3>() as u64,
    )?;
    layout.spawnable = true;
    Ok(layout)
}

#[derive(Default)]
pub(crate) struct TsSchemaRegistry {
    by_type: HashMap<TypeId, Arc<StructLayout>>,
}

impl TsSchemaRegistry {
    pub(crate) fn register<T: TsSchema>(&mut self) -> Result<(), String> {
        let layout = T::ts_schema()?;
        if layout.size != std::mem::size_of::<T>() as u64
            || layout.align != std::mem::align_of::<T>() as u64
        {
            return Err(format!(
                "TypeScript schema for `{}` has layout {}:{}, expected {}:{}",
                std::any::type_name::<T>(),
                layout.size,
                layout.align,
                std::mem::size_of::<T>(),
                std::mem::align_of::<T>()
            ));
        }
        let type_id = TypeId::of::<T>();
        if self.by_type.contains_key(&type_id) {
            return Err(format!(
                "TypeScript schema for `{}` is already registered",
                std::any::type_name::<T>()
            ));
        }
        self.by_type.insert(type_id, Arc::new(layout));
        Ok(())
    }
}

pub(crate) fn server_ts_schemas() -> Result<TsSchemaRegistry, String> {
    let mut schemas = TsSchemaRegistry::default();
    schemas.register::<crate::movement::PlayerLook>()?;
    schemas.register::<crate::chunks::PlayerPosition>()?;
    Ok(schemas)
}

impl TsSchema for crate::chunks::PlayerPosition {
    fn ts_schema() -> Result<StructLayout, String> {
        let mut layout = StructLayout::from_parts(
            "PlayerPosition",
            vec![StructField {
                name: "position".to_owned(),
                ty: Ty::Struct(Arc::new(vec3_schema()?)),
                offset: std::mem::offset_of!(Self, 0) as u64,
            }],
            std::mem::size_of::<Self>() as u64,
            std::mem::align_of::<Self>() as u64,
        )?;
        layout.spawnable = true;
        Ok(layout)
    }
}

#[derive(Clone)]
struct EngineComponent {
    id: ComponentId,
    layout: Arc<StructLayout>,
    table_stored: bool,
}

/// Component names and layouts available to one mod compilation.
#[derive(Default)]
pub(crate) struct ComponentRegistry {
    by_name: HashMap<String, EngineComponent>,
    ambiguous: HashSet<String>,
    resources: HashSet<ComponentId>,
}

pub(crate) fn layout_hash(params: &[Ty]) -> u64 {
    let mut layouts = Vec::new();
    for param in params {
        collect_layouts(param, &mut layouts);
    }
    layouts.sort();
    layouts.dedup();
    let mut hash = 0xcbf29ce484222325u64;
    let mut add = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash = hash.wrapping_mul(0x100000001b3);
    };
    add(&tesjit::abi::QUERY_DESC_VERSION.to_le_bytes());
    for layout in layouts {
        add(layout.as_bytes());
    }
    hash
}

fn collect_layouts(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::Struct(layout) => {
            out.push(format!(
                "{}:{}:{}:{}",
                layout.name,
                layout.size,
                layout.align,
                layout
                    .fields
                    .iter()
                    .map(|field| format!("{}:{}@{}", field.name, field.ty.name(), field.offset))
                    .collect::<Vec<_>>()
                    .join(",")
            ));
            for field in &layout.fields {
                collect_layouts(&field.ty, out);
            }
        }
        Ty::Query(inner, filters) => {
            collect_layouts(inner, out);
            for filter in filters {
                collect_layouts(&Ty::Struct(Arc::clone(&filter.component)), out);
            }
        }
        Ty::Mut(inner) | Ty::Res(inner) => collect_layouts(inner, out),
        Ty::Tuple(elements) => {
            for element in elements {
                collect_layouts(element, out);
            }
        }
        _ => {}
    }
}

impl ComponentRegistry {
    pub(crate) fn from_world_with_schemas(
        world: &World,
        db: Option<&TypeDb>,
        schemas: &TsSchemaRegistry,
    ) -> Self {
        let mut registry = Self::default();
        registry
            .resources
            .extend(world.iter_resources().map(|(info, _)| info.id()));
        for info in world.components().iter_registered() {
            let Some(type_id) = info.type_id() else {
                continue;
            };
            let rust_name = info.name().to_string();
            let registered_layout = schemas
                .by_type
                .get(&type_id)
                .map(|layout| (**layout).clone());
            let (name, mut layout) = if let Some(layout) = registered_layout {
                (layout.name.clone(), layout)
            } else {
                let Some(db) = db else {
                    continue;
                };
                let Some(def) = find_type(db, &rust_name) else {
                    continue;
                };
                let name = short_name(&def.name.0);
                let Some(layout) = component_layout(db, def, &name) else {
                    continue;
                };
                (name, layout)
            };
            if !valid_ts_identifier(&name) {
                continue;
            };
            if !registry.resources.contains(&info.id()) {
                let Ok(component_id) = u32::try_from(info.id().index()) else {
                    bevy_log::warn!(
                        "ts ECS: ignored `{rust_name}` because its component id does not fit u32"
                    );
                    continue;
                };
                layout.component_id = Some(component_id);
            }
            let actual = info.layout();
            if layout.size != actual.size() as u64 || layout.align != actual.align() as u64 {
                bevy_log::warn!(
                    "ts ECS: ignored `{}` because registered layout {}:{} differs from Bevy layout {}:{}",
                    rust_name,
                    layout.size,
                    layout.align,
                    actual.size(),
                    actual.align()
                );
                continue;
            }
            if registry.ambiguous.contains(&name) {
                continue;
            }
            let component = EngineComponent {
                id: info.id(),
                layout: Arc::new(layout),
                table_stored: info.storage_type() == StorageType::Table,
            };
            if registry.by_name.insert(name.clone(), component).is_some() {
                registry.by_name.remove(&name);
                registry.ambiguous.insert(name);
            }
        }
        registry
    }

    pub(crate) fn layouts(&self) -> impl Iterator<Item = (&str, &Arc<StructLayout>)> {
        self.by_name
            .iter()
            .map(|(name, component)| (name.as_str(), &component.layout))
    }

    pub(crate) fn write_declarations(&self, root: &Path) -> std::io::Result<()> {
        let path = root.join("exsolis.generated.d.ts");
        let mut layouts = BTreeMap::new();
        let mut components: Vec<_> = self.by_name.iter().collect();
        components.sort_by_key(|(name, _)| *name);
        for (name, component) in components {
            collect_declarations(&component.layout, &mut layouts);
            layouts.insert(name.clone(), component.layout.clone());
        }
        let mut contents =
            String::from("// Generated from the current server component layouts; do not edit.\n");
        for (name, layout) in layouts {
            if !valid_ts_identifier(&name) {
                continue;
            }
            contents.push_str(&format!("interface {name} {{\n"));
            for field in &layout.fields {
                if valid_ts_identifier(&field.name) {
                    contents.push_str(&format!(
                        "  {}: {};\n",
                        field.name,
                        declaration_type(&field.ty)
                    ));
                }
            }
            contents.push_str("}\n\n");
        }
        if std::fs::read_to_string(&path).ok().as_deref() != Some(contents.as_str()) {
            std::fs::write(path, contents)?;
        }
        Ok(())
    }

    fn get(&self, name: &str) -> Option<&EngineComponent> {
        self.by_name.get(name)
    }

    fn resource(&self, name: &str) -> Option<&EngineComponent> {
        self.by_name
            .get(name)
            .filter(|component| self.resources.contains(&component.id))
    }
}

fn collect_declarations(layout: &StructLayout, out: &mut BTreeMap<String, Arc<StructLayout>>) {
    for field in &layout.fields {
        if let Ty::Struct(nested) = &field.ty {
            out.entry(nested.name.clone())
                .or_insert_with(|| Arc::clone(nested));
            collect_declarations(nested, out);
        }
    }
}

fn declaration_type(ty: &Ty) -> String {
    match ty {
        Ty::Tuple(elements) => format!(
            "[{}]",
            elements
                .iter()
                .map(declaration_type)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Struct(layout) => layout.name.clone(),
        _ => ty.name().into_owned(),
    }
}

fn valid_ts_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
}

fn find_type<'a>(db: &'a TypeDb, rust_name: &str) -> Option<&'a TypeDef> {
    if let Some(def) = db.get(rust_name) {
        return Some(def);
    }
    let leaf = rust_name.rsplit("::").next()?;
    let mut matching = db
        .iter()
        .filter(|(name, _)| name.rsplit("::").next() == Some(leaf));
    let (name, _) = matching.next()?;
    if matching.next().is_some() {
        return None;
    }
    db.get(name)
}

fn short_name(name: &str) -> String {
    name.rsplit("::").next().unwrap_or(name).to_owned()
}

fn component_layout(db: &TypeDb, def: &TypeDef, name: &str) -> Option<StructLayout> {
    let TypeArgs::Struct(body) = &def.args else {
        return None;
    };
    if body.is_union {
        return None;
    }
    let mut seen = HashSet::new();
    let fields: Vec<_> = body
        .fields
        .iter()
        .filter_map(|field| {
            let ty = type_layout(db, field.ty.as_str(), &mut seen)?;
            Some(StructField {
                name: field.name.clone(),
                ty,
                offset: field.offset,
            })
        })
        .collect();
    let spawnable = fields.len() == body.fields.len()
        && body
            .fields
            .iter()
            .all(|field| spawnable_type(db, field.ty.as_str(), &mut HashSet::new()));
    let mut layout = StructLayout::from_parts(name, fields, def.size, def.align).ok()?;
    layout.spawnable = spawnable;
    Some(layout)
}

fn type_layout(db: &TypeDb, name: &str, seen: &mut HashSet<String>) -> Option<Ty> {
    let primitive = match name {
        "bool" => Some(Ty::Bool),
        "i8" => Some(Ty::I8),
        "i16" => Some(Ty::I16),
        "i32" => Some(Ty::I32),
        "i64" => Some(Ty::I64),
        "u8" => Some(Ty::U8),
        "u16" => Some(Ty::U16),
        "u32" | "char" => Some(Ty::U32),
        "u64" => Some(Ty::U64),
        "f32" => Some(Ty::F32),
        "f64" => Some(Ty::F64),
        _ => None,
    };
    if primitive.is_some() {
        return primitive;
    }

    let def = db.get(name)?;
    let leaf = short_name(&def.name.0);
    if leaf == "Entity" && def.size == 8 {
        return Some(Ty::Entity);
    }
    let TypeArgs::Struct(body) = &def.args else {
        return None;
    };
    if body.is_union || !seen.insert(def.name.0.clone()) {
        return None;
    }
    let fields: Vec<_> = body
        .fields
        .iter()
        .filter_map(|field| {
            let ty = type_layout(db, field.ty.as_str(), seen)?;
            Some(StructField {
                name: field.name.clone(),
                ty,
                offset: field.offset,
            })
        })
        .collect();
    let spawnable = fields.len() == body.fields.len()
        && body
            .fields
            .iter()
            .all(|field| spawnable_type(db, field.ty.as_str(), &mut HashSet::new()));
    seen.remove(&def.name.0);
    let mut layout = StructLayout::from_parts(leaf, fields, def.size, def.align).ok()?;
    layout.spawnable = spawnable;
    Some(Ty::Struct(Arc::new(layout)))
}

fn spawnable_type(db: &TypeDb, name: &str, seen: &mut HashSet<String>) -> bool {
    match name {
        "bool" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64" => {
            return true;
        }
        "char" => return false,
        _ => {}
    }
    let Some(def) = db.get(name) else {
        return false;
    };
    if short_name(&def.name.0) == "Entity" && def.size == 8 {
        return true;
    }
    let TypeArgs::Struct(body) = &def.args else {
        return false;
    };
    if body.is_union || !seen.insert(def.name.0.clone()) {
        return false;
    }
    let spawnable = body
        .fields
        .iter()
        .all(|field| spawnable_type(db, field.ty.as_str(), seen));
    seen.remove(&def.name.0);
    spawnable
}

#[derive(Clone)]
struct QueryTerm {
    component: Option<ComponentId>,
    mutable: bool,
}

/// Linked query shape with a cache of tables matching its component set.
pub(crate) struct QueryPlan {
    terms: Vec<QueryTerm>,
    filters: Vec<(ComponentId, QueryFilterKind)>,
    archetypes_seen: usize,
    tables: Vec<TableId>,
    table_ids: HashSet<u32>,
}

impl QueryPlan {
    fn link(
        row: &Ty,
        query_filters: &[tesjit::ty::QueryFilter],
        registry: &ComponentRegistry,
    ) -> Result<Self, String> {
        let terms = match row {
            Ty::Tuple(terms) => terms.as_slice(),
            term => std::slice::from_ref(term),
        };
        if terms.is_empty() {
            return Err("query row cannot be empty".to_owned());
        }
        let mut linked = Vec::with_capacity(terms.len());
        let mut used_components = HashSet::new();
        for term in terms {
            match term {
                Ty::Entity => linked.push(QueryTerm {
                    component: None,
                    mutable: false,
                }),
                Ty::Struct(layout) => {
                    let component = registry.get(&layout.name).ok_or_else(|| {
                        format!(
                            "query component `{}` is not linked to a Bevy component",
                            layout.name
                        )
                    })?;
                    if component.layout.size == 0 {
                        return Err(format!(
                            "zero-sized query component `{}` is unsupported",
                            layout.name
                        ));
                    }
                    if !component.table_stored {
                        return Err(format!(
                            "query component `{}` is not table-stored",
                            layout.name
                        ));
                    }
                    if component.layout.size != layout.size
                        || component.layout.align != layout.align
                    {
                        return Err(format!(
                            "query component `{}` layout changed since compilation",
                            layout.name
                        ));
                    }
                    if !used_components.insert(component.id) {
                        return Err(format!(
                            "query contains component `{}` more than once",
                            layout.name
                        ));
                    }
                    linked.push(QueryTerm {
                        component: Some(component.id),
                        mutable: false,
                    });
                }
                Ty::Mut(inner) => {
                    let Ty::Struct(layout) = inner.as_ref() else {
                        return Err(format!("Mut<{}> is not a linked component", inner));
                    };
                    let component = registry.get(&layout.name).ok_or_else(|| {
                        format!(
                            "query component `{}` is not linked to a Bevy component",
                            layout.name
                        )
                    })?;
                    if component.layout.size == 0 {
                        return Err(format!(
                            "zero-sized query component `{}` is unsupported",
                            layout.name
                        ));
                    }
                    if !component.table_stored {
                        return Err(format!(
                            "query component `{}` is not table-stored",
                            layout.name
                        ));
                    }
                    if component.layout.size != layout.size
                        || component.layout.align != layout.align
                    {
                        return Err(format!(
                            "query component `{}` layout changed since compilation",
                            layout.name
                        ));
                    }
                    if !used_components.insert(component.id) {
                        return Err(format!(
                            "query contains component `{}` more than once",
                            layout.name
                        ));
                    }
                    linked.push(QueryTerm {
                        component: Some(component.id),
                        mutable: true,
                    });
                }
                other => return Err(format!("unsupported query term `{other}`")),
            }
        }
        let mut filters = Vec::with_capacity(query_filters.len());
        let mut filter_kinds = HashMap::new();
        for filter in query_filters {
            let component = registry.get(&filter.component.name).ok_or_else(|| {
                format!(
                    "query filter component `{}` is not linked to a Bevy component",
                    filter.component.name
                )
            })?;
            if !component.table_stored {
                return Err(format!(
                    "query filter component `{}` is not table-stored",
                    filter.component.name
                ));
            }
            if component.layout.size != filter.component.size
                || component.layout.align != filter.component.align
            {
                return Err(format!(
                    "query filter component `{}` layout changed since compilation",
                    filter.component.name
                ));
            }
            if let Some(previous) = filter_kinds.insert(component.id, filter.kind)
                && previous != filter.kind
            {
                return Err(format!(
                    "query filter component `{}` is both With and Without",
                    filter.component.name
                ));
            }
            if !filters
                .iter()
                .any(|(id, kind)| *id == component.id && *kind == filter.kind)
            {
                filters.push((component.id, filter.kind));
            }
        }
        Ok(Self {
            terms: linked,
            filters,
            archetypes_seen: 0,
            tables: Vec::new(),
            table_ids: HashSet::new(),
        })
    }

    fn refresh(&mut self, world: &World) {
        let archetypes = world.archetypes();
        for archetype in archetypes.iter().skip(self.archetypes_seen) {
            self.archetypes_seen += 1;
            if self.terms.iter().all(|term| {
                term.component
                    .is_none_or(|component| archetype.contains(component))
            }) && self.filters.iter().all(|(component, kind)| match kind {
                QueryFilterKind::With => archetype.contains(*component),
                QueryFilterKind::Without => !archetype.contains(*component),
            }) {
                let id = archetype.table_id();
                if self.table_ids.insert(id.as_u32()) {
                    self.tables.push(id);
                }
            }
        }
    }

    pub(crate) fn fill(&mut self, world: &mut World, buffer: &mut QueryBuffer) {
        self.refresh(world);
        let needs_tick = self.terms.iter().any(|term| term.mutable);
        let tick = needs_tick.then(|| world.change_tick());
        let has_entity_term = self.terms.iter().any(|term| term.component.is_none());
        let total_rows = if has_entity_term {
            self.tables
                .iter()
                .filter_map(|id| world.storages().tables.get(*id))
                .map(|table| table.entity_count() as usize)
                .sum()
        } else {
            0
        };

        buffer.chunks.clear();
        buffer.entity_bits.clear();
        buffer.columns.clear();
        buffer.chunk_count = 0;
        buffer.entity_bits.reserve(total_rows);

        for table_id in &self.tables {
            let Some(table) = world.storages().tables.get(*table_id) else {
                continue;
            };
            let rows = table.entity_count() as usize;
            if rows == 0 {
                continue;
            }
            let entity_addr = if has_entity_term {
                let start = buffer.entity_bits.len();
                buffer
                    .entity_bits
                    .extend(table.entities().iter().map(|entity| entity.to_bits()));
                unsafe { buffer.entity_bits.as_mut_ptr().add(start) as usize as u64 }
            } else {
                0
            };
            buffer.columns.clear();
            for term in &self.terms {
                if let Some(component) = term.component {
                    let Some(column) = table.get_column(component) else {
                        buffer.columns.clear();
                        break;
                    };
                    let ptr = unsafe {
                        column
                            .get_data_unchecked(TableRow::new(
                                NonMaxU32::new(0).expect("zero is not the NonMaxU32 sentinel"),
                            ))
                            .as_ptr()
                    };
                    buffer.columns.push(ptr as usize as u64);
                    if term.mutable {
                        if let Some(tick) = tick {
                            if let Some(changed) = table.get_changed_ticks_slice_for(component) {
                                for cell in changed {
                                    unsafe { *cell.get() = tick };
                                }
                            }
                        }
                    }
                } else {
                    buffer.columns.push(entity_addr);
                }
            }
            if buffer.columns.len() == self.terms.len() {
                tesjit::abi::write_chunk_addrs(&mut buffer.chunks, rows as u64, &buffer.columns);
                buffer.chunk_count += 1;
            }
        }
        buffer.input = QueryInput {
            chunks_addr: if buffer.chunks.is_empty() {
                0
            } else {
                buffer.chunks.as_ptr() as usize as u64
            },
            chunk_count: buffer.chunk_count as u64,
        };
    }
}

pub(crate) struct QueryBuffer {
    chunks: Vec<u64>,
    entity_bits: Vec<u64>,
    columns: Vec<u64>,
    chunk_count: usize,
    input: QueryInput,
}

impl Default for QueryBuffer {
    fn default() -> Self {
        Self {
            chunks: Vec::new(),
            entity_bits: Vec::new(),
            columns: Vec::new(),
            chunk_count: 0,
            input: QueryInput {
                chunks_addr: 0,
                chunk_count: 0,
            },
        }
    }
}

#[derive(Clone, Copy)]
struct SpawnableComponent {
    id: ComponentId,
    size: usize,
    align: usize,
}

struct SpawnValue {
    component: ComponentId,
    bytes: Vec<u8>,
    align: usize,
}

pub(crate) struct CommandBuffer {
    components: HashMap<u32, SpawnableComponent>,
    spawns: Vec<Vec<SpawnValue>>,
    error: Option<String>,
}

impl CommandBuffer {
    fn new(registry: &ComponentRegistry) -> Self {
        let components = registry
            .by_name
            .values()
            .filter_map(|component| {
                let id = component.layout.component_id?;
                Some((
                    id,
                    SpawnableComponent {
                        id: component.id,
                        size: component.layout.size as usize,
                        align: component.layout.align as usize,
                    },
                ))
            })
            .collect();
        Self {
            components,
            spawns: Vec::new(),
            error: None,
        }
    }

    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(message.into());
        }
    }

    fn apply(&mut self, world: &mut World) {
        if let Some(error) = self.error.take() {
            bevy_log::error!("ts ECS command buffer rejected a spawn: {error}");
        }
        for spawn in self.spawns.drain(..) {
            let mut allocated = Vec::with_capacity(spawn.len());
            let mut allocation_failed = false;
            for value in spawn {
                let Ok(layout) =
                    std::alloc::Layout::from_size_align(value.bytes.len().max(1), value.align)
                else {
                    bevy_log::error!("ts ECS command buffer has an invalid component layout");
                    allocation_failed = true;
                    break;
                };
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    bevy_log::error!("ts ECS command buffer could not allocate component data");
                    allocation_failed = true;
                    break;
                }
                if !value.bytes.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(value.bytes.as_ptr(), ptr, value.bytes.len());
                    }
                }
                allocated.push((value.component, ptr, layout));
            }
            if allocation_failed {
                for (_, ptr, layout) in allocated {
                    unsafe { std::alloc::dealloc(ptr, layout) };
                }
                continue;
            }

            let mut entity = world.spawn_empty();
            for (component, ptr, layout) in allocated {
                let owning = unsafe {
                    bevy_ecs::ptr::OwningPtr::new(
                        std::ptr::NonNull::new(ptr).expect("component allocation is non-null"),
                    )
                };
                unsafe { entity.insert_by_id(component, owning) };
                unsafe { std::alloc::dealloc(ptr, layout) };
            }
        }
    }
}

pub(crate) extern "C" fn host_commands_spawn(buffer: i64, input: i64) {
    if buffer == 0 || input == 0 {
        bevy_log::error!("ts ECS spawn received a null command buffer or descriptor");
        return;
    }
    let buffer = unsafe { &mut *(buffer as usize as *mut CommandBuffer) };
    let input = unsafe { &*(input as usize as *const tesjit::abi::SpawnInput) };
    let Ok(count) = usize::try_from(input.component_count) else {
        buffer.fail("component count does not fit usize");
        return;
    };
    if count == 0 || count > tesjit::abi::MAX_SPAWN_COMPONENTS || input.components_addr == 0 {
        buffer.fail("invalid spawn descriptor size");
        return;
    }
    let descriptors = unsafe {
        std::slice::from_raw_parts(
            input.components_addr as usize as *const tesjit::abi::SpawnComponent,
            count,
        )
    };
    let mut values = Vec::with_capacity(count);
    let mut used = HashSet::new();
    for descriptor in descriptors {
        let Ok(id) = u32::try_from(descriptor.component_id) else {
            buffer.fail("component id does not fit u32");
            return;
        };
        let Some(component) = buffer.components.get(&id).copied() else {
            buffer.fail(format!("component id {id} is not spawnable"));
            return;
        };
        let (Ok(size), Ok(align)) = (
            usize::try_from(descriptor.size),
            usize::try_from(descriptor.align),
        ) else {
            buffer.fail("component layout does not fit usize");
            return;
        };
        if size != component.size
            || align != component.align
            || !used.insert(id)
            || descriptor.data_addr == 0
            || descriptor.data_addr % descriptor.align.max(1) != 0
        {
            buffer.fail("component layout or address does not match the registered type");
            return;
        }
        let bytes = if size == 0 {
            Vec::new()
        } else {
            unsafe {
                std::slice::from_raw_parts(descriptor.data_addr as usize as *const u8, size)
                    .to_vec()
            }
        };
        values.push(SpawnValue {
            component: component.id,
            bytes,
            align,
        });
    }
    buffer.spawns.push(values);
}

pub(crate) struct QuerySystem {
    pub(crate) name: String,
    params: Vec<LinkedParam>,
    args: Vec<i64>,
}

enum LinkedParam {
    Query {
        plan: QueryPlan,
        buffer: QueryBuffer,
    },
    Resource {
        component: ComponentId,
        mutable: bool,
    },
    Commands {
        buffer: CommandBuffer,
    },
}

impl QuerySystem {
    pub(crate) fn link(
        name: &str,
        params: &[Ty],
        ret: &Ty,
        registry: &ComponentRegistry,
    ) -> Result<Self, String> {
        if *ret != Ty::Void {
            return Err(format!("query system `{name}` must return void"));
        }
        if params.len() > tesjit::jit::MAX_ARGS {
            return Err(format!("query system `{name}` exceeds the argument limit"));
        }
        let mut linked = Vec::with_capacity(params.len());
        let mut access = HashMap::new();
        for param in params {
            match param {
                Ty::Query(row, filters) => {
                    let plan = QueryPlan::link(row, filters, registry)?;
                    for term in &plan.terms {
                        if let Some(component) = term.component
                            && let Some(previous_mutable) = access.insert(component, term.mutable)
                            && (previous_mutable || term.mutable)
                        {
                            return Err(format!(
                                "query system `{name}` has conflicting access to one component"
                            ));
                        }
                    }
                    linked.push(LinkedParam::Query {
                        plan,
                        buffer: QueryBuffer::default(),
                    });
                }
                Ty::Res(inner) | Ty::Mut(inner) => {
                    let Ty::Struct(layout) = inner.as_ref() else {
                        return Err(format!(
                            "resource system parameter `{}` must name a registered struct",
                            param.name()
                        ));
                    };
                    let component = registry.resource(&layout.name).ok_or_else(|| {
                        format!(
                            "resource `{}` is not registered or is not currently present",
                            layout.name
                        )
                    })?;
                    if component.layout.size != layout.size
                        || component.layout.align != layout.align
                    {
                        return Err(format!(
                            "resource `{}` layout changed since compilation",
                            layout.name
                        ));
                    }
                    let mutable = matches!(param, Ty::Mut(_));
                    if let Some(previous_mutable) = access.insert(component.id, mutable)
                        && (previous_mutable || mutable)
                    {
                        return Err(format!(
                            "query system `{name}` has conflicting access to one component or resource"
                        ));
                    }
                    linked.push(LinkedParam::Resource {
                        component: component.id,
                        mutable,
                    });
                }
                Ty::Commands => linked.push(LinkedParam::Commands {
                    buffer: CommandBuffer::new(registry),
                }),
                other => {
                    return Err(format!(
                        "query system `{name}` parameters must be Query<T>, Res<T>, or Mut<T>; found {other}"
                    ));
                }
            }
        }
        Ok(Self {
            name: name.to_owned(),
            params: linked,
            args: Vec::new(),
        })
    }

    pub(crate) fn args(&mut self, world: &mut World) -> Option<&[i64]> {
        self.args.clear();
        for param in &mut self.params {
            match param {
                LinkedParam::Query { plan, buffer } => {
                    plan.fill(world, buffer);
                    self.args
                        .push(&buffer.input as *const QueryInput as usize as i64);
                }
                LinkedParam::Resource {
                    component,
                    mutable: false,
                } => {
                    let value = world.get_resource_by_id(*component)?;
                    self.args.push(value.as_ptr() as usize as i64);
                }
                LinkedParam::Resource {
                    component,
                    mutable: true,
                } => {
                    let mut value = world.get_resource_mut_by_id(*component)?;
                    let ptr = value.as_mut().as_ptr();
                    self.args.push(ptr as usize as i64);
                }
                LinkedParam::Commands { buffer } => {
                    self.args.push(buffer as *mut CommandBuffer as usize as i64);
                }
            }
        }
        Some(&self.args)
    }

    pub(crate) fn apply_commands(&mut self, world: &mut World) {
        for param in &mut self.params {
            if let LinkedParam::Commands { buffer } = param {
                buffer.apply(world);
            }
        }
    }
}
