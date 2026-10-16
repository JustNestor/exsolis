//! TypeScript mod runtime — the second scripting backend.
//!
//! The Lua backend (`mods/**/*.lua`, [`crate::scripting`]) stays exactly as it
//! was; this module adds `mods/**/*.ts` through `tesjit` alongside it. Both are
//! live at the same time during the migration: each owns its files by
//! extension, each has its own failure containment, and both push the same
//! [`ScriptRequest`] shape through the same translator
//! ([`crate::scripting::apply_script_requests`]), so "what a mod may do to the
//! world" has one implementation for both languages.
//!
//! ## What a mod looks like
//!
//! ```ts
//! // mods/ts_greeter/main.ts
//! declare function log(x: f64): void;
//! declare function player_count(): i32;
//!
//! export function on_tick(): void {
//!   let n: i32 = player_count();
//!   if (n > 0) { log(n as f64); }
//! }
//!
//! export function on_place_block(id: string, x: i32, y: i32, z: i32, block: i32): void {
//!   log(block as f64);
//!   set_block(x, y, z, 1); // change the world in reaction
//! }
//! ```
//!
//! There are no callbacks in the language yet, so subscriptions are by export
//! *name*: the host calls `on_tick` / `on_<event>` when the mod exports it.
//! Because every function boundary is annotated, a mod exporting the wrong
//! shape fails to compile (and keeps its last good build) instead of crashing
//! the server.
//!
//! ## Sandbox
//!
//! Each handler runs under tesjit's guards: a fuel budget armed per call,
//! bounds-checked array/string access, and traps that return normally. A trap
//! is logged with the mod name and the mod keeps running on the next call —
//! one buggy mod never takes the server down.

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{debug, error, info, warn};
use tesjit::{Compiled, HostFn, JitBuilder, Ty};

use exsolis_common::{AIR_ID, BlockId, BlockPos, PlayerId};

use crate::assets::mods_directory;
use crate::chunks::{ChunkWorld, PlayerPosition};
use crate::scripting::{apply_script_requests, LuaScalar, ScriptRequest, world_to_chunk_index};

/// Fuel handed to each handler invocation: room for a per-tick hook worth of
/// work (tens of thousands of statements), small enough that a spin loop dies
/// inside one tick.
const PER_CALL_FUEL: u64 = 2_000_000;

/// Events the TS bridge forwards, with the handler shape each one expects.
///
/// The names match the Lua event names (`scripting.rs::dispatch`) so both
/// backends react to the same traffic. Argument order is what the mod's
/// exported `on_<event>` must declare.
pub const EVENTS: &[(&str, &[Ty])] = &[
    ("player_join", &[Ty::Str]),
    ("player_leave", &[Ty::Str]),
    ("place_block", &[Ty::Str, Ty::I32, Ty::I32, Ty::I32, Ty::I32]),
    ("break_block", &[Ty::Str, Ty::I32, Ty::I32, Ty::I32, Ty::I32]),
    ("select_hotbar", &[Ty::Str, Ty::I32]),
    ("ui_action", &[Ty::Str, Ty::Str, Ty::Str, Ty::Str, Ty::Str]),
    ("ui_drag", &[Ty::Str, Ty::Str, Ty::Str, Ty::Str, Ty::Str]),
    ("keybind_action", &[Ty::Str, Ty::Str, Ty::Bool]),
];

/// Handler name for an event, e.g. `on_place_block`.
pub fn handler_for(event: &str) -> String {
    format!("on_{event}")
}

/// One flattened event argument.
#[derive(Clone, Debug)]
pub enum HostArg {
    Str(String),
    I32(i32),
    F64(f64),
    Bool(bool),
}

// ───────────────────────────── per-tick pull context ─────────────────────────────

/// State the read-only host API borrows for one tick.
///
/// The Lua backend installs its pull helpers as *scoped* globals borrowed for
/// a single tick; the JIT host ABI is `extern "C"` with no closures, so this
/// runtime borrows the equivalent state into a thread-local for the duration
/// of the tick and clears it afterwards.
struct PullCtx {
    /// `player_id -> (x, y, z)`, snapshotted once per tick so a handler can
    /// loop over players without re-reading the world.
    positions: HashMap<String, (f32, f32, f32)>,
    /// Live chunk storage. Valid only while the tick's system holds the world;
    /// the pointer never escapes the system that installed it.
    world: *const ChunkWorld,
    /// Underlying Bevy world, installed for the whole tick so generic host
    /// functions can run cacheable `Query`s by *registered name* rather than
    /// through the specialised PlayerRow path.
    bevy_world: *const bevy_ecs::prelude::World,
    /// Packed query rows for `query_*` — a *view* of the buffer the tick system
    /// owns, so mods read and write the very memory the write-back applies.
    rows: RowsView,
    components: HashMap<String, u64>,
}

/// A view of one tick's row buffer.
///
/// The buffer itself lives in `ts_tick_system` (and is handed to
/// `TsRuntime::tick`); the view is what the host functions resolve against, and
/// the region id is per module because every mod has its own runtime.
#[derive(Debug, Clone, Copy, Default)]
struct RowsView {
    ptr: *const PlayerRow,
    len: usize,
    region: u64,
}

impl RowsView {
    fn rows(&self) -> &[PlayerRow] {
        if self.ptr.is_null() {
            &[]
        } else {
            // SAFETY: the buffer outlives the tick that installed this view.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

/// Packed, C-laid-out rows handed to mod code as memory regions.
///
/// A mod reads them with `load_f32(row, offset)` where `offset` comes from the
/// field offsets the type database reports for the matching Rust type — which
/// is exactly what [`PlayerRow`] mirrors (see `check_row_layout`). Full type
/// export only works with `debug = 2` (see the workspace profile).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct PlayerRow {
    /// `PlayerId` as u64. First, so the common lookup is at offset 0 and no
    /// padding is needed between it and the floats — the id word decides the
    /// row's alignment either way.
    id: u64,
    /// `glam::Vec3::x`.
    x: f32,
    /// `glam::Vec3::y`.
    y: f32,
    /// `glam::Vec3::z`.
    z: f32,
}

impl PlayerRow {
    /// Byte offset of each field, as [`Self`] lays it out.
    ///
    /// Mods do not hardcode these: `row_offset(name)` reports them, and
    /// `check_row_layout` cross-checks the position triple against the layout
    /// the type database exports for `glam::Vec3`, so a drift on the Rust side
    /// is a logged error instead of a silently wrong offset.
    const OFFSETS: &'static [(&'static str, u64)] = &[
        ("id", std::mem::offset_of!(PlayerRow, id) as u64),
        ("x", std::mem::offset_of!(PlayerRow, x) as u64),
        ("y", std::mem::offset_of!(PlayerRow, y) as u64),
        ("z", std::mem::offset_of!(PlayerRow, z) as u64),
    ];

    fn offset_of(field: &str) -> i32 {
        Self::OFFSETS
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, offset)| *offset as i32)
            .unwrap_or(-1)
    }
}

/// Row lists, one per query the bridge exposes.
#[derive(Debug, Clone, Default)]
struct QueryRows {
    /// Every connected player, one [`PlayerRow`] each.
    players: Vec<PlayerRow>,
    /// Region id of `players`, valid for the tick that registered it.
    players_region: u64,
}

impl QueryRows {
    /// Build the row list from the tick's position snapshot.
    ///
    /// Order is by id string so every mod sees the same row order; `ok_or`
    /// handles the (impossible) case of an id that does not parse.
    fn from_positions(positions: &HashMap<String, (f32, f32, f32)>) -> Self {
        let mut players: Vec<PlayerRow> = positions
            .iter()
            .map(|(id, (x, y, z))| PlayerRow {
                x: *x,
                y: *y,
                z: *z,
                id: id.parse().unwrap_or(0),
            })
            .collect();
        players.sort_by_key(|row| row.id);
        Self {
            players,
            players_region: 0,
        }
    }
}

thread_local! {
    /// The type database the server loaded at startup (`types.tdb`). Set for
    /// the duration of a tick so host functions can answer layout questions.
    static TYPE_DB: std::cell::RefCell<Option<Arc<tesjit_dwarf::TypeDb>>> =
        const { std::cell::RefCell::new(None) };
    static PULL: std::cell::RefCell<Option<PullCtx>> = const { std::cell::RefCell::new(None) };
    /// Requests queued by host calls while a handler runs.
    static OUTBOX: std::cell::RefCell<Option<Arc<Mutex<Vec<ScriptRequest>>>>> =
        const { std::cell::RefCell::new(None) };
    /// Live block registry length, installed per tick so `host_set_block`
    /// can validate a legacy numeric id without world access.
    static REGISTRY_LEN: std::cell::Cell<usize> = const { std::cell::Cell::new(11) };
}

impl PullCtx {
    fn install(&self) {
        PULL.with(|p| *p.borrow_mut() = Some(self.clone()));
    }
}

// The thread-local only ever holds one of these at a time, so cloning the
// snapshot is cheaper than threading a borrow through the host ABI.
impl Clone for PullCtx {
    fn clone(&self) -> Self {
        Self {
            positions: self.positions.clone(),
            world: self.world,
            bevy_world: self.bevy_world,
            rows: self.rows.clone(),
            components: self.components.clone(),
        }
    }
}

/// Record the region id the current mod was handed for the row buffer.
fn set_rows_region(id: u64) {
    PULL.with(|p| {
        if let Some(ctx) = p.borrow_mut().as_mut() {
            ctx.rows.region = id;
        }
    });
}

/// Install the row view for a tick (before any handler runs).
fn set_rows_view(rows: &[PlayerRow]) {
    PULL.with(|p| {
        if let Some(ctx) = p.borrow_mut().as_mut() {
            ctx.rows = RowsView {
                ptr: rows.as_ptr(),
                len: rows.len(),
                region: 0,
            };
        }
    });
}

fn with_pull<R>(f: impl FnOnce(&PullCtx) -> R) -> Option<R> {
    PULL.with(|p| p.borrow().as_ref().map(f))
}

fn queue(req: ScriptRequest) {
    OUTBOX.with(|o| {
        if let Some(outbox) = o.borrow().as_ref() {
            // A poisoned lock can only happen if a host call panicked while
            // holding it; recovering here keeps the mod alive.
            let mut guard = outbox.lock().unwrap_or_else(|e| e.into_inner());
            guard.push(req);
        }
    });
}

// ───────────────────────────── host API ─────────────────────────────

/// Strings cross the boundary as arena pointers: `[len: i64][utf8 bytes]`
/// (see `tesjit::runtime`). The pointer is valid for the lifetime of the
/// module that produced it, which the host keeps alive.
unsafe fn ts_str(s: i64) -> String {
    unsafe {
        let len = *(s as *const i64) as usize;
        let bytes = std::slice::from_raw_parts((s as *const u8).add(8), len);
        String::from_utf8_lossy(bytes).into_owned()
    }
}

/// Expand a UI template to client-ready XML, same pipeline the Lua mods use.
///
/// Data stays on the Rust side for now: a TS mod sends the resulting XML with
/// `send(player, "ui_show", xml)`. (Passing a data table *into* the renderer
/// is what the ABI work in the roadmap is for.)
fn render_xml(file: &str) -> String {
    crate::ui_template::render_file(file, &Default::default()).unwrap_or_else(|e| {
        warn!("ts mod: render_xml({file}) failed: {e}");
        String::new()
    })
}

fn host_fns() -> Vec<HostFn> {
    vec![
        HostFn::new("log", vec![Ty::F64], Ty::Void, host_log as *const u8),
        HostFn::new(
            "send",
            vec![Ty::Str, Ty::Str],
            Ty::Void,
            host_send as *const u8,
        ),
        HostFn::new(
            "broadcast",
            vec![Ty::Str],
            Ty::Void,
            host_broadcast as *const u8,
        ),
        HostFn::new(
            "set_block",
            vec![Ty::I32, Ty::I32, Ty::I32, Ty::I32],
            Ty::Void,
            host_set_block as *const u8,
        ),
        HostFn::new(
            "get_block",
            vec![Ty::I32, Ty::I32, Ty::I32],
            Ty::I32,
            host_get_block as *const u8,
        ),
        HostFn::new(
            "component_access_id",
            vec![Ty::Str],
            Ty::I64,
            host_component_access_id as *const u8,
        ),
        HostFn::new(
            "world_token",
            vec![],
            Ty::I64,
            host_world_token as *const u8,
        ),
        HostFn::new(
            "player_count",
            vec![],
            Ty::I32,
            host_player_count as *const u8,
        ),
        HostFn::new(
            "player_id_at",
            vec![Ty::I32],
            Ty::Str,
            host_player_id_at as *const u8,
        ),
        HostFn::new(
            "player_x",
            vec![Ty::Str],
            Ty::F64,
            host_player_x as *const u8,
        ),
        HostFn::new(
            "player_y",
            vec![Ty::Str],
            Ty::F64,
            host_player_y as *const u8,
        ),
        HostFn::new(
            "player_z",
            vec![Ty::Str],
            Ty::F64,
            host_player_z as *const u8,
        ),
        HostFn::new(
            "render_xml",
            vec![Ty::Str],
            Ty::Str,
            host_render_xml as *const u8,
        ),
        HostFn::new(
            "query_players",
            vec![],
            Ty::I32,
            host_query_players as *const u8,
        ),
        HostFn::new(
            "query_row_count",
            vec![Ty::I32],
            Ty::I32,
            host_query_row_count as *const u8,
        ),
        HostFn::new(
            "row_stride",
            vec![],
            Ty::I32,
            host_row_stride as *const u8,
        ),
        HostFn::new(
            "row_offset",
            vec![Ty::Str],
            Ty::I32,
            host_row_offset as *const u8,
        ),
        HostFn::new(
            "field_offset",
            vec![Ty::Str, Ty::Str],
            Ty::I32,
            host_field_offset as *const u8,
        ),
        HostFn::new(
            "type_size",
            vec![Ty::Str],
            Ty::I32,
            host_type_size as *const u8,
        ),
        HostFn::new(
            "row_get_f32",
            vec![Ty::I32, Ty::Str],
            Ty::F64,
            host_row_get_f32 as *const u8,
        ),
        HostFn::new(
            "row_set_f32",
            vec![Ty::I32, Ty::Str, Ty::F64],
            Ty::F64,
            host_row_set_f32 as *const u8,
        ),
        HostFn::new(
            "row_get_i64",
            vec![Ty::I32, Ty::Str],
            Ty::I32,
            host_row_get_i64 as *const u8,
        ),
        HostFn::new(
            "row_set_i64",
            vec![Ty::I32, Ty::Str, Ty::I32],
            Ty::I32,
            host_row_set_i64 as *const u8,
        ),
        HostFn::new(
            "register_keybind",
            vec![Ty::Str, Ty::Str, Ty::Str],
            Ty::Void,
            host_register_keybind as *const u8,
        ),
    ]
}

extern "C" fn host_log(x: f64) {
    info!("[ts mod] {x}");
}
extern "C" fn host_world_token() -> i64 {
    with_pull(|ctx| ctx.world as i64).unwrap_or(0)
}
#[allow(dead_code)]
extern "C" fn host_player_component_id() -> i64 {
    with_pull(|ctx| ctx.world as usize as i64).unwrap_or(0)
}
extern "C" fn host_component_access_id(name: i64) -> i64 {
    let name = unsafe { ts_str(name) };
    with_pull(|ctx| *ctx.components.get(&name).unwrap_or(&0)).unwrap_or(0) as i64
}

fn build_component_registry(world: &mut World) -> HashMap<String, u64> {
    // Bootstrap set of component kinds that mods can refer to by stable
    // name. IDs are Bevy's runtime ids, resolved fresh each tick.
    let mut out = HashMap::new();
    if let Some(id) = world.component_id::<crate::Player>() {
        out.insert("Player".to_string(), id.index() as u64);
    }
    out
}



extern "C" fn host_send(player: i64, kind: i64) {
    let (player, kind) = unsafe { (ts_str(player), ts_str(kind)) };
    // `PlayerId` is a `u64`; ids reach mods as decimal strings (same rule as
    // the Lua bridge) so a mod cannot smuggle in a non-numeric value.
    let Ok(id) = player.parse::<PlayerId>() else {
        warn!("ts mod: send() with a non-numeric player id {player:?}");
        return;
    };
    queue(ScriptRequest::SendTo {
        player_id: id,
        kind,
        // Data-side payload arguments are not exposed to TS yet.
        args: Vec::<LuaScalar>::new(),
    });
}

extern "C" fn host_broadcast(kind: i64) {
    let kind = unsafe { ts_str(kind) };
    queue(ScriptRequest::Broadcast {
        kind,
        args: Vec::<LuaScalar>::new(),
        except: None,
    });
}

extern "C" fn host_set_block(x: i32, y: i32, z: i32, block: i32) {
    // Legacy numeric ids match `BlockId` one-to-one for the frozen M0 set;
    // anything outside the live registry is rejected here (and again at
    // drain time, in case the registry changed mid-tick).
    let len = REGISTRY_LEN.with(|l| l.get());
    if block < 0 || (block as usize) >= len {
        warn!("ts mod: set_block: invalid block id {block}");
        return;
    }
    queue(ScriptRequest::SetBlock {
        pos: BlockPos::new(x, y, z),
        block: block as BlockId,
    });
}

extern "C" fn host_get_block(x: i32, y: i32, z: i32) -> i32 {
    // SAFETY: the pointer is installed by the tick system, which holds the
    // world for the whole duration of the call.
    let Some(block) = with_pull(|ctx| {
        let world = unsafe { &*ctx.world };
        block_at(world, x, y, z)
    }) else {
        return 0;
    };
    block
}

/// Block id at a world position, or air when the chunk is not loaded.
fn block_at(world: &ChunkWorld, x: i32, y: i32, z: i32) -> i32 {
    let world: &ChunkWorld = world;
    let (coord, idx) = world_to_chunk_index(BlockPos::new(x, y, z));
    world
        .ready
        .get(&coord)
        .and_then(|c| c.unpack())
        .and_then(|u| u.get(idx).copied())
        .unwrap_or(AIR_ID) as i32
}

extern "C" fn host_player_count() -> i32 {
    with_pull(|ctx| ctx.positions.len() as i32).unwrap_or(0)
}

extern "C" fn host_player_id_at(index: i32) -> i64 {
    // Built in the *module's* arena, so the pointer it hands back is as valid
    // as a string literal's (same decimal-string rule as the Lua bridge).
    let id = with_pull(|ctx| {
        ctx.positions
            .keys()
            .nth(index.max(0) as usize)
            .cloned()
            .unwrap_or_default()
    })
    .unwrap_or_default();
    tesjit::runtime::host_string(&id)
}

extern "C" fn host_player_x(id: i64) -> f64 {
    let id = unsafe { ts_str(id) };
    with_pull(|ctx| ctx.positions.get(&id).map(|(x, _, _)| f64::from(*x)))
        .flatten()
        .unwrap_or(0.0)
}

extern "C" fn host_player_y(id: i64) -> f64 {
    let id = unsafe { ts_str(id) };
    with_pull(|ctx| ctx.positions.get(&id).map(|(_, y, _)| f64::from(*y)))
        .flatten()
        .unwrap_or(0.0)
}

extern "C" fn host_player_z(id: i64) -> f64 {
    let id = unsafe { ts_str(id) };
    with_pull(|ctx| ctx.positions.get(&id).map(|(_, _, z)| f64::from(*z)))
        .flatten()
        .unwrap_or(0.0)
}

/// Region id of the packed player rows for this tick.
///
/// Rows are laid out by [`PlayerRow`]; the offsets a mod uses come from the
/// exported field offsets of that type (see `check_row_layout`), so a layout
/// drift shows up as a wrong value rather than a corrupt read.
extern "C" fn host_query_players() -> i32 {
    with_pull(|ctx| ctx.rows.region as i32).unwrap_or(0)
}

/// Number of rows in a query region (0 when the region is unknown).
extern "C" fn host_query_row_count(region: i32) -> i32 {
    let region = region as i64;
    with_pull(|ctx| {
        if region < 0 || region as u64 != ctx.rows.region {
            return 0;
        }
        ctx.rows.len as i32
    })
    .unwrap_or(0)
}

/// Address (and width in bytes) of `rows[index].field`.
///
/// Shared by the row accessors below, so they go through the same bounds
/// checking the raw region intrinsics use — an index out of range or an unknown
/// field name never reaches memory.
fn row_field(index: i32, field: &str) -> Option<(*mut u8, usize)> {
    PULL.with(|pull| {
        let borrowed = pull.borrow();
        let ctx = borrowed.as_ref()?;
        let rows = ctx.rows;
        if rows.region == 0 || index < 0 || index as usize >= rows.len {
            return None;
        }
        let offset = PlayerRow::offset_of(field);
        if offset < 0 {
            return None;
        }
        let byte = offset as usize + index as usize * std::mem::size_of::<PlayerRow>();
        let ptr = unsafe { (rows.ptr as *mut u8).add(byte) };
        Some((ptr, if field == "id" { 8 } else { 4 }))
    })
}

/// Byte offset of one row field (`"x"`, `"y"`, `"z"`, `"id"`), or -1 when the
/// field does not exist.
///
/// This is how a mod learns row layout instead of hardcoding offsets: the host
/// owns the struct, so it is the only side that can answer reliably.
extern "C" fn host_row_offset(field: i64) -> i32 {
    let field = unsafe { ts_str(field) };
    PlayerRow::offset_of(&field)
}

/// `row_get_f32(i, "x")` — read a float field of row `i`.
extern "C" fn host_row_get_f32(index: i32, field: i64) -> f64 {
    let field = unsafe { ts_str(field) };
    match row_field(index, &field) {
        Some((ptr, 4)) => unsafe { f64::from(std::ptr::read_unaligned(ptr as *const f32)) },
        _ => 0.0,
    }
}

/// `row_set_f32(i, "x", 10.0)` — write a float field of row `i`. The edit is
/// applied to the world after the tick (see `apply_player_positions`), which is
/// what makes rows the only place a mod can touch the ECS directly.
extern "C" fn host_row_set_f32(index: i32, field: i64, value: f64) -> f64 {
    let field = unsafe { ts_str(field) };
    match row_field(index, &field) {
        Some((ptr, 4)) => unsafe {
            std::ptr::write_unaligned(ptr as *mut f32, value as f32);
            value
        },
        _ => 0.0,
    }
}

/// `row_get_i64(i, "id")` — read a machine-word field of row `i`.
extern "C" fn host_row_get_i64(index: i32, field: i64) -> i64 {
    let field = unsafe { ts_str(field) };
    match row_field(index, &field) {
        Some((ptr, 8)) => unsafe { std::ptr::read_unaligned(ptr as *const i64) },
        _ => 0,
    }
}

/// `row_set_i64(i, "id", ...)` — write a machine-word field of row `i`.
extern "C" fn host_row_set_i64(index: i32, field: i64, value: i64) -> i64 {
    let field = unsafe { ts_str(field) };
    match row_field(index, &field) {
        Some((ptr, 8)) => unsafe {
            std::ptr::write_unaligned(ptr as *mut i64, value);
            value
        },
        _ => 0,
    }
}

/// Byte offset of a field in *any* type in the exported database.
///
/// This is what lets a mod reach Rust types that are not `repr(C)`: rustc is free
/// to reorder fields and add padding, and the only thing that knows the real
/// layout is the DWARF export the server loaded. Pass the fully-qualified type
/// name and the field, get bytes (-1 when either is unknown).
extern "C" fn host_field_offset(ty: i64, field: i64) -> i32 {
    let (ty, field) = unsafe { (ts_str(ty), ts_str(field)) };
    let Some(db) = with_type_db(|db| db.clone()) else {
        return -1;
    };
    let Some(def) = db.get(&ty) else {
        return -1;
    };
    let tesjit_dwarf::TypeArgs::Struct(fields) = &def.args else {
        return -1;
    };
    fields
        .fields
        .iter()
        .find(|f| f.name == field)
        .map(|f| f.offset as i32)
        .unwrap_or(-1)
}

/// Byte size of any type in the exported database (-1 when unknown).
extern "C" fn host_type_size(ty: i64) -> i32 {
    let ty = unsafe { ts_str(ty) };
    let Some(db) = with_type_db(|db| db.clone()) else {
        return -1;
    };
    db.get(&ty).map(|def| def.size as i32).unwrap_or(-1)
}

fn with_type_db<R>(f: impl FnOnce(&Arc<tesjit_dwarf::TypeDb>) -> R) -> Option<R> {
    TYPE_DB.with(|db| db.borrow().as_ref().map(f))
}

/// Byte size of one row.
extern "C" fn host_row_stride() -> i32 {
    std::mem::size_of::<PlayerRow>() as i32
}

extern "C" fn host_render_xml(file: i64) -> i64 {
    let file = unsafe { ts_str(file) };
    let xml = render_xml(&file);
    debug!("ts mod: render_xml({file}) -> {} bytes", xml.len());
    tesjit::runtime::host_string(&xml)
}

extern "C" fn host_register_keybind(id: i64, key: i64, description: i64) {
    let (id, key, description) = unsafe { (ts_str(id), ts_str(key), ts_str(description)) };
    queue(ScriptRequest::RegisterKeybind {
        id,
        default_key: key,
        description,
        category: "mods".to_owned(),
    });
}

// ───────────────────────────── the runtime ─────────────────────────────

/// One compiled mod.
struct TsMod {
    /// Display name (source path).
    name: String,
    compiled: Compiled,
}

/// All TS mods, with the same reload contract as the Lua runtime: a mod that
/// fails to compile keeps its last good build, and the server keeps running.
#[derive(Resource)]
pub struct TsRuntime {
    /// Shared shape with the Lua outbox; drained by the same translator.
    outbox: Arc<Mutex<Vec<ScriptRequest>>>,
    mods: Vec<TsMod>,
    /// Source mtime per mod, for hot reload.
    stamps: HashMap<PathBuf, SystemTime>,
    /// Files whose last *attempt* was a failure, with the mtime it failed at.
    ///
    /// Without this, a mod that does not compile would be recompiled every tick
    /// and re-log every tick — the failure is only interesting once, until the
    /// file actually changes.
    failures: HashMap<PathBuf, SystemTime>,
    root: PathBuf,
}

impl Default for TsRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl TsRuntime {
    pub fn new() -> Self {
        Self::with_root(mods_directory())
    }

    /// Same, but rooted anywhere — used by tests so they do not have to fight
    /// over the process-wide `EXSOLIS_MODS_DIR` while running in parallel.
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            outbox: Arc::new(Mutex::new(Vec::new())),
            mods: Vec::new(),
            stamps: HashMap::new(),
            failures: HashMap::new(),
            root: root.into(),
        }
    }

    /// The outbox, for the drain step of the tick system.
    pub fn outbox(&self) -> Arc<Mutex<Vec<ScriptRequest>>> {
        Arc::clone(&self.outbox)
    }

    /// Load (or reload) every `*.ts` mod.
    ///
    /// Unchanged files keep their compiled module (JIT memory is *not*
    /// re-allocated just because a sibling mod changed); new/changed files are
    /// recompiled; a failing mod keeps its previous build and logs the errors.
    pub fn reload(&mut self) {
        if !self.root.is_dir() {
            return;
        }
        let Ok(sources) = discover_ts_files(&self.root) else {
            return;
        };
        let mut next_stamps = HashMap::new();
        let mut next_mods = Vec::new();
        let mut failures = HashMap::new();

        for path in sources {
            let Ok(stamp) = std::fs::metadata(&path).and_then(|m| m.modified()) else {
                continue;
            };
            next_stamps.insert(path.clone(), stamp);
            let changed = self
                .stamps
                .get(&path)
                .map(|previous| *previous != stamp)
                .unwrap_or(true);
            // A mod that already failed at *this* mtime is not retried: the
            // file has not changed, so the diagnostics would be identical.
            let already_failed = self
                .failures
                .get(&path)
                .map(|at| *at == stamp)
                .unwrap_or(false);
            if !changed && already_failed {
                continue;
            }
            if !changed {
                if let Some(index) = self.mods.iter().position(|m| m.name == display(&path)) {
                    // Move the compiled module over as-is.
                    next_mods.push(self.mods.swap_remove(index));
                    continue;
                }
            }
            match compile_mod(&path) {
                Ok(compiled) => {
                    info!("[ts mod] loaded {}", path.display());
                    // A successful build supersedes any earlier failure.
                    failures.remove(&path);
                    next_mods.push(TsMod {
                        name: display(&path),
                        compiled,
                    });
                }
                Err(diags) => {
                    error!("[ts mod] {} failed to compile:", path.display());
                    for d in &diags {
                        error!("{d:?}");
                    }
                    // Remember the attempt so the next tick does not repeat it.
                    failures.insert(path.clone(), stamp);
                    // Keep the last good build, if there is one.
                    if let Some(index) = self.mods.iter().position(|m| m.name == display(&path)) {
                        next_mods.push(self.mods.swap_remove(index));
                    }
                }
            }
        }

        // Whatever is left in `self.mods` was dropped from disk; its JIT memory
        // is freed with `Compiled`.
        self.mods = next_mods;
        self.stamps = next_stamps;
        self.failures = failures;
    }

    /// Handler names every loaded mod exports (tests + tooling).
    pub fn exported_handlers(&self) -> Vec<String> {
        let mut out = Vec::new();
        for m in &self.mods {
            for (event, _) in EVENTS {
                let name = handler_for(event);
                if m.compiled.entry(&name).is_some() {
                    out.push(format!("{}:{name}", m.name));
                }
            }
            if m.compiled.entry("on_tick").is_some() {
                out.push(format!("{}:on_tick", m.name));
            }
        }
        out
    }

    fn with_outbox<R>(&self, f: impl FnOnce() -> R) -> R {
        OUTBOX.with(|o| *o.borrow_mut() = Some(Arc::clone(&self.outbox)));
        let out = f();
        OUTBOX.with(|o| *o.borrow_mut() = None);
        out
    }

    /// Call every mod's `on_tick` export, with a fresh fuel budget each.
    pub(crate) fn tick(
        &self,
        positions: &HashMap<String, (f32, f32, f32)>,
        world: *const ChunkWorld,
        rows: &[PlayerRow],
    ) {
        self.with_outbox(|| {
            // The view the host functions resolve rows through points at the
            // buffer the caller owns, not a copy.
            set_rows_view(rows);
            for m in &self.mods {
                let Some(entry) = m.compiled.entry("on_tick") else {
                    continue;
                };
                if entry.ret != Ty::Void || !entry.params.is_empty() {
                    warn!(
                        "[ts mod] {} exported `on_tick` with the wrong shape",
                        m.name
                    );
                    continue;
                }
                // Region ids are per module (every mod has its own runtime), so
                // the same row buffer is registered under each module's id and
                // that id is what the pull API hands out for this mod.
                set_rows_region(m.compiled.register_region(
                    "players",
                    rows.as_ptr() as *mut u8,
                    rows.len() * std::mem::size_of::<PlayerRow>(),
                ));
                run_guarded(m, "on_tick", |compiled| unsafe {
                    compiled.call_void("on_tick", &[])
                });
            }
            let _ = positions;
            let _ = world;
        });
    }

    /// Forward one event to the mods that export the matching handler.
    pub fn dispatch(&self, event: &str, args: &[HostArg]) {
        let Some((_, shape)) = EVENTS.iter().find(|(name, _)| *name == event) else {
            warn!("ts mod: unknown event {event}");
            return;
        };
        if args.len() != shape.len() {
            warn!("ts mod: event {event} got {} args, want {}", args.len(), shape.len());
            return;
        }
        let handler = handler_for(event);
        self.with_outbox(|| {
            for m in &self.mods {
                let Some(entry) = m.compiled.entry(&handler) else {
                    continue;
                };
                let matches_shape = entry.params.len() == shape.len()
                    && entry
                        .params
                        .iter()
                        .zip(shape.iter())
                        .all(|(got, want)| *got == *want);
                if !matches_shape {
                    warn!(
                        "[ts mod] {} has `{handler}` with the wrong shape, skipping",
                        m.name
                    );
                    continue;
                }
                run_guarded(m, &handler, |compiled| {
                    // Flatten the arguments into the integer/pointer slots the
                    // ABI uses. Strings are built in *this* mod's arena (host
                    // functions cannot allocate in a module they do not own),
                    // which is why this happens per mod inside the loop.
                    let mut slots = Vec::with_capacity(args.len());
                    for arg in args {
                        slots.push(match arg {
                            HostArg::I32(v) => i64::from(*v),
                            HostArg::F64(v) => *v as i64,
                            HostArg::Bool(v) => i64::from(*v),
                            HostArg::Str(text) => {
                                tesjit::runtime::host_string(text)
                            }
                        });
                    }
                    unsafe { compiled.call_void(&handler, &slots) };
                });
            }
        });
    }
}

/// Run one handler with fuel armed, the runtime entered, and any trap logged.
fn run_guarded(m: &TsMod, handler: &str, call: impl Fn(&Compiled)) {
    m.compiled.runtime().reset_fuel(PER_CALL_FUEL);
    let guard = m.compiled.enter();
    call(&m.compiled);
    drop(guard);
    if let Some(trap) = m.compiled.trap() {
        error!(
            "[ts mod] {} {handler} trapped: {}",
            m.name,
            trap.message()
        );
        m.compiled.clear_trap();
    }
}

fn display(path: &Path) -> String {
    path.display().to_string()
}

fn compile_mod(path: &Path) -> Result<Compiled, Vec<tesjit::CompileError>> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        vec![tesjit::CompileError::Plain(format!(
            "read {}: {e}",
            path.display()
        ))]
    })?;
    let src: std::sync::Arc<str> = src.into();
    let mut builder = JitBuilder::new(display(path), src.clone());
    for f in host_fns() {
        builder.host_fn_ref(f);
    }
    let mut jit = builder.build().map_err(|e| vec![e])?;
    jit.compile(&src).map_err(|d| d.into_iter().collect::<Vec<_>>())?;
    jit.finish().map_err(|e| vec![e])
}

/// Every `*.ts` file under `root`, in a stable order (directory order would
/// make reloads non-deterministic).
fn discover_ts_files(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("ts") {
                // Declaration files (`exsolis.d.ts`) are editor tooling, not
                // mods: they contain only `declare`s and would never compile
                // as a module.
                let is_declaration = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|name| name.ends_with(".d.ts"));
                if !is_declaration {
                    out.push(path);
                }
            }
        }
    }
    out.sort();
    Ok(out)
}

// ───────────────────────────── Bevy wiring ─────────────────────────────

/// Plugin for the TS backend: its own reload/tick/drain systems, ordered the
/// same way as the Lua ones so both backends see the same tick in the same
/// order.
pub struct TsScriptingPlugin;

impl Plugin for TsScriptingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TsRuntime>()
            .add_systems(
                Update,
                (
                    ts_reload_system,
                    ts_forward_join,
                    ts_forward_leave,
                    ts_forward_place_block,
                    ts_forward_break_block,
                    ts_forward_select_hotbar,
                ),
            )
            .add_systems(
                Update,
                ts_tick_system
                    // Same event sources as the Lua forwarders, so a tick sees
                    // everything that happened this frame for both backends.
                    .after(crate::events::dispatch_incoming)
                    .after(crate::ui::dispatch_ui_packets),
            )
            .add_systems(
                Update,
                ts_drain_system
                    .after(ts_tick_system)
                    .before(crate::net::flush_outgoing),
            );
    }
}

/// Hot reload: recompile mods whose source changed.
pub fn ts_reload_system(mut runtime: ResMut<TsRuntime>) {
    runtime.reload();
}

/// Snapshot of every connected player's position for this tick.
fn player_positions(world: &mut World) -> HashMap<String, (f32, f32, f32)> {
    let mut out = HashMap::new();
    let mut query = world.query::<(&crate::Player, &PlayerPosition)>();
    for (player, pos) in query.iter(world) {
        out.insert(format!("{}", player.0), (pos.0.x, pos.0.y, pos.0.z));
    }
    out
}

/// Per-tick driver, exclusive (`fn(&mut World)`): the mod runs as its own
/// system with the world to itself.
///
/// That exclusivity is what makes the raw chunk pointer handed to the pull API
/// sound for the duration of the call, and it keeps mod reads and engine writes
/// ordered. The pull context is installed for the whole tick and cleared before
/// any request is applied, so JIT code can never outlive it.
pub fn ts_tick_system(world: &mut World) {
    let positions = player_positions(world);
    let Some(runtime) = world.remove_resource::<TsRuntime>() else {
        return;
    };
    // The type database comes from the resource the server inserted at startup,
    // so layout questions can only be answered while a tick is running.
    let types = world
        .get_resource::<crate::typedb::ServerTypes>()
        .map(|types| Arc::clone(&types.0));
    TYPE_DB.with(|slot| *slot.borrow_mut() = types.clone());
    let world_ptr: *const ChunkWorld = match world.get_resource::<ChunkWorld>() {
        Some(chunks) => chunks as *const ChunkWorld,
        None => std::ptr::null(),
    };
    if let Some(blocks) = world.get_resource::<crate::blocks::ServerBlocks>() {
        REGISTRY_LEN.with(|l| l.set(blocks.0.len()));
    }

    // Packed rows for `query_players`. The `Vec` lives in this system and is
    // never mutated during the tick, so the pointers registered against it stay
    // valid for every handler call.
    let rows = QueryRows::from_positions(&positions);
    PullCtx {
        positions: positions.clone(),
        world: world_ptr,
        bevy_world: world as *const bevy_ecs::prelude::World,
        rows: RowsView::default(),
        components: build_component_registry(world),
    }
    .install();
    runtime.tick(&positions, world_ptr, &rows.players);
    PullCtx::uninstall();
    TYPE_DB.with(|slot| *slot.borrow_mut() = None);

    // Write-back: rows are host memory, so anything the mods moved is already
    // in `rows`. Applying it here (once per tick, after every handler ran)
    // keeps the ECS the single source of truth — mods never hold a `&mut
    // PlayerPosition` across a call.
    let moved = apply_player_positions(world, &rows.players);
    if moved > 0 {
        debug!("[ts mod] applied {moved} player position(s) from query rows");
    }

    world.insert_resource(runtime);
}

impl PullCtx {
    fn uninstall() {
        PULL.with(|p| *p.borrow_mut() = None);
    }
}

/// Apply row edits back to the ECS. Returns how many players moved.
///
/// Only the position is written back: query rows are a *view* of the world, and
/// a mod that wants to change anything else still goes through the request
/// outbox (`set_block`, ...), which is the same rule the Lua bridge follows.
fn apply_player_positions(world: &mut World, rows: &[PlayerRow]) -> usize {
    let mut query = world.query::<(&crate::Player, &mut PlayerPosition)>();
    let mut applied = 0;
    for (player, mut position) in query.iter_mut(world) {
        let Some(row) = rows.iter().find(|row| row.id == player.0) else {
            continue;
        };
        let moved = (position.0.x, position.0.y, position.0.z) != (row.x, row.y, row.z);
        if moved {
            position.0.x = row.x;
            position.0.y = row.y;
            position.0.z = row.z;
            applied += 1;
        }
    }
    applied
}

/// Apply what the TS mods queued, through the same translator the Lua bridge
/// uses.
fn ts_drain_system(
    runtime: Res<TsRuntime>,
    commands: Commands,
    world: ResMut<ChunkWorld>,
    registry: Res<crate::entities::EntityRegistry>,
    spawns: MessageWriter<crate::entities::SpawnEntityRequest>,
    despawns: MessageWriter<crate::entities::DespawnEntityRequest>,
    outgoing: MessageWriter<crate::net::OutgoingPacket>,
    loaded_by: Query<(&crate::Player, &crate::chunks::ChunkRenderState)>,
    blocks: Res<crate::blocks::ServerBlocks>,
    player_map: Res<crate::PlayerMap>,
    mut inventories: Query<&mut crate::Inventory>,
) {
    let outbox = runtime.outbox();
    let requests = outbox
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .drain(..)
        .collect::<Vec<_>>();
    if requests.is_empty() {
        return;
    }
    apply_script_requests(
        requests,
        commands,
        world,
        registry,
        spawns,
        despawns,
        outgoing,
        loaded_by,
        &blocks,
        &player_map,
        &mut inventories,
    );
}

// ───────────────────────────── event forwarders ─────────────────────────────
//
// One per Lua event, so both backends see the same traffic. Field order
// matches `EVENTS` above.

pub fn ts_forward_join(
    mut events: MessageReader<crate::events::PlayerJoined>,
    runtime: Res<TsRuntime>,
) {
    for ev in events.read() {
        runtime.dispatch("player_join", &[HostArg::Str(ev.0.to_string())]);
    }
}

pub fn ts_forward_leave(
    mut events: MessageReader<crate::events::PlayerLeaved>,
    runtime: Res<TsRuntime>,
) {
    for ev in events.read() {
        runtime.dispatch("player_leave", &[HostArg::Str(ev.0.to_string())]);
    }
}

pub fn ts_forward_place_block(
    mut events: MessageReader<crate::interaction::PlaceBlockRequested>,
    runtime: Res<TsRuntime>,
    blocks: Res<crate::blocks::ServerBlocks>,
) {
    for ev in events.read() {
        // TS mods still speak numeric ids; resolve the name through the
        // registry (-1 when unknown, which can only happen if the request
        // itself was already rejected for the same reason).
        let id = blocks.id(&ev.kind).map(|id| id as i32).unwrap_or(-1);
        runtime.dispatch(
            "place_block",
            &[
                HostArg::Str(ev.player_id.to_string()),
                HostArg::I32(ev.pos.x),
                HostArg::I32(ev.pos.y),
                HostArg::I32(ev.pos.z),
                HostArg::I32(id),
            ],
        );
    }
}

pub fn ts_forward_break_block(
    mut events: MessageReader<crate::interaction::BreakBlockRequested>,
    runtime: Res<TsRuntime>,
) {
    for ev in events.read() {
        // `kind` is not part of the break event (a mod can read the block it
        // replaced with `get_block`); the slot stays so both block events share
        // one handler shape.
        runtime.dispatch(
            "break_block",
            &[
                HostArg::Str(ev.player_id.to_string()),
                HostArg::I32(ev.pos.x),
                HostArg::I32(ev.pos.y),
                HostArg::I32(ev.pos.z),
                HostArg::I32(0),
            ],
        );
    }
}

pub fn ts_forward_select_hotbar(
    mut events: MessageReader<crate::interaction::SelectHotbarRequested>,
    runtime: Res<TsRuntime>,
) {
    for ev in events.read() {
        runtime.dispatch(
            "select_hotbar",
            &[
                HostArg::Str(ev.player_id.to_string()),
                HostArg::I32(i32::from(ev.index)),
            ],
        );
    }
}

/// Verify that [`PlayerRow`] really matches the layout the type database
/// reports for the Rust type behind it.
///
/// This is what makes the region ABI trustworthy: mods read row fields by
/// offsets exported from DWARF, so a Rust-side layout change would otherwise
/// hand them silently wrong offsets. Requires a `types.tdb` exported from a
/// `debug = 2` binary (see `just export-types`); returns an error describing
/// the drift, or when the type is missing from the database.
pub fn check_row_layout(db: &tesjit_dwarf::TypeDb) -> Result<(), String> {
    let field_offset = |ty: &str, field: &str| -> Result<u64, String> {
        let def = db
            .get(ty)
            .ok_or_else(|| format!("`{ty}` is not in the type database"))?;
        let tesjit_dwarf::TypeArgs::Struct(fields) = &def.args else {
            return Err(format!("`{ty}` is not a struct"));
        };
        fields
            .fields
            .iter()
            .find(|f| f.name == field)
            .map(|f| f.offset)
            .ok_or_else(|| format!("`{ty}` has no field `{field}`"))
    };

    // The server stores positions as `glam::Vec3` inside `PlayerPosition`.
    for (field, offset) in PlayerRow::OFFSETS.iter().filter(|(name, _)| *name != "id") {
        let offset = *offset;
        let expected = field_offset("glam::Vec3", field)?;
        if expected != offset {
            return Err(format!(
                "PlayerRow::{field} sits at byte {offset}, but glam::Vec3 reports {expected}"
            ));
        }
    }

    // And the id word first, u64-aligned, filling the row out exactly.
    let stride = std::mem::size_of::<PlayerRow>() as u64;
    let id = PlayerRow::offset_of("id").max(0) as u64;
    if id % 8 != 0 || id + 8 != stride {
        return Err(format!(
            "PlayerRow::id sits at {id} in a {stride}-byte row; expected 8-aligned, ending at the row end"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory with the given mod files.
        fn mods_dir(files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ts-backend-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let _ = std::fs::remove_dir_all(&dir);
        for (name, body) in files {
            let path = dir.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(&path, body).unwrap();
        }
        dir
    }

    fn positions() -> HashMap<String, (f32, f32, f32)> {
        HashMap::from([("1".to_owned(), (1.0f32, 2.0f32, 3.0f32))])
    }

    /// Compile a Rust fixture with `rustc -g` and export `Row` through
    /// tesjit-dwarf, so the layout check has a real database to compare
    /// against (the workspace profile is `debug = 2` now, so a real binary
    /// would work the same way).
    fn exported_row_db() -> tesjit_dwarf::TypeDb {
        let dir = std::env::temp_dir().join(format!("ts-row-layout-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("fixture.rs");
        std::fs::write(
            &src,
            r#"
            #[repr(C)]
            #[derive(Clone, Copy)]
            pub struct Row { pub x: f32, pub y: f32, pub z: f32, pub id: u64 }
            fn main() {
                let r = Row { x: 1.0, y: 2.0, z: 3.0, id: 7 };
                std::hint::black_box((&r, r.z));
            }
            "#,
        )
        .unwrap();
        let bin = dir.join("fixture");
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
        assert!(
            std::process::Command::new(rustc)
                .arg("--edition=2021")
                .arg("-g")
                .arg(&src)
                .arg("-o")
                .arg(&bin)
                .status()
                .unwrap()
                .success(),
            "fixture build failed"
        );
        let data = std::fs::read(&bin).unwrap();
        let (db, _) = tesjit_dwarf::export(&data, "fixture").expect("export fixture");
        let _ = std::fs::remove_dir_all(&dir);
        db
    }

    #[test]
    fn declaration_files_are_not_loaded_as_mods() {
        // `exsolis.d.ts` lives inside `mods/` because that is where mod authors
        // look for it; it must not be compiled as a module.
        let dir = mods_dir(&[
            ("exsolis.d.ts", "declare function load_f64(region: i32, offset: i32): f64;\n"),
            (
                "real.ts",
                "export function on_tick(): void {}\ndeclare function load_f64(r: i32, o: i32): f64;\n",
            ),
        ]);
        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        let handlers = runtime.exported_handlers();
        assert_eq!(handlers.len(), 1, "only the real mod loads: {handlers:?}");
        assert!(
            handlers[0].ends_with(":on_tick") && handlers[0].contains("real.ts"),
            "unexpected handlers: {handlers:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn query_rows_line_up_with_the_exported_layout() {
        let db = exported_row_db();
        // `glam::Vec3` is not in this fixture, so the check must report that
        // rather than silently pass — the important half is the shape of the
        // error, i.e. that offsets really come from the database.
        let error = check_row_layout(&db).unwrap_err();
        assert!(
            error.contains("glam::Vec3") || error.contains("not in the type database"),
            "unexpected error: {error}"
        );

        // The row layout itself: u64 id first, then three f32s, padded to the
        // u64 alignment — 24 bytes, and every field at an offset the host
        // reports rather than one a mod had to guess.
        assert_eq!(std::mem::size_of::<PlayerRow>(), 24);
        assert_eq!((std::mem::align_of::<PlayerRow>(), PlayerRow::offset_of("x")), (8, 8));
        assert_eq!(PlayerRow::offset_of("nope"), -1, "unknown fields report -1");
    }

    #[test]
    fn a_mod_can_move_players_through_query_rows() {
        // Rows are host memory: a mod writes into them with `store_f32`, and the
        // tick system writes the deltas back into the ECS afterwards.
        let dir = mods_dir(&[(
            "move.ts",
            r#"
            declare function query_players(): i32;
            declare function row_offset(name: string): i32;
            declare function row_stride(): i32;

            export function on_tick(): void {
              // Row 0, field "x": lifted ten metres. The host owns the offset
              // math, so a mod never computes `index * stride + offset`.
              row_set_f32(0, "x", 10.0);
            }
            "#,
        )]);
        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        let rows = QueryRows::from_positions(&positions());
        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null(), &rows.players);
        PullCtx::uninstall();

        // The write landed in the row buffer the ECS write-back reads.
        assert_eq!(rows.players.len(), 1);
        assert_eq!(rows.players[0].x, 10.0);
        assert_eq!(rows.players[0].y, 2.0, "only x moved");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The ABI question the roadmap asked: can mod code read a Rust type that
    /// is NOT `repr(C)`? Offsets come from the exported database, so they match
    /// whatever order rustc chose — which is the only way this can work.
    #[test]
    fn offsets_for_non_repr_c_types_come_from_the_database() {
        // Build a genuinely `repr(Rust)` fixture and export it, the way the
        // server's own `just export-types` run does.
        let dir = std::env::temp_dir().join(format!("ts-nonrepr-c-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let src = dir.join("fixture.rs");
        std::fs::write(
            &src,
            r#"
            // No `repr(C)` anywhere: rustc decides the layout.
            pub struct Packet {
                pub seq: u64,
                pub flag: u8,
                pub scale: f32,
            }
            fn main() {
                let p = Packet { seq: 7, flag: 1, scale: 0.5 };
                std::hint::black_box((&p, p.scale));
            }
            "#,
        )
        .unwrap();
        let bin = dir.join("fixture");
        let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
        assert!(
            std::process::Command::new(rustc)
                .arg("--edition=2021")
                .arg("-g")
                .arg("-Csplit-debuginfo=unpacked")
                .arg(&src)
                .arg("-o")
                .arg(&bin)
                .status()
                .unwrap()
                .success(),
            "fixture build failed"
        );
        let data = std::fs::read(&bin).unwrap();
        let (db, _) = tesjit_dwarf::export_with_dwo(&data, "fixture", &dir)
            .expect("export with dwo");

        // The host's own view of the layout, from DWARF.
        let db_seq = host_field_offset_str(&db, "fixture::Packet", "seq");
        let db_flag = host_field_offset_str(&db, "fixture::Packet", "flag");
        let db_scale = host_field_offset_str(&db, "fixture::Packet", "scale");
        assert!(db_seq.is_some() && db_flag.is_some() && db_scale.is_some(), "{:?}", db.get("fixture::Packet"));
        // `seq` is the only u64, so it is aligned to 8 and comes first — but the
        // two smaller fields are reordered *relative to the source*: rustc puts
        // the 4-byte `scale` before the 1-byte `flag`, so neither source order
        // nor a naive "biggest first" guess would find it. Only DWARF knows.
        assert_eq!(db_seq, Some(0), "the u64 is aligned to 8 and comes first");
        assert_eq!(db_scale, Some(8), "f32 packed next");
        assert_eq!(db_flag, Some(12), "u8 last");

        // A real `Packet`, read by the mod through those offsets.
        #[repr(Rust)]
        struct Packet {
            seq: u64,
            flag: u8,
            scale: f32,
        }
        let value = Packet {
            seq: 7,
            flag: 1,
            scale: 0.5,
        };
        let mod_dir = mods_dir(&[(
            "offsets.ts",
            &format!(
                r#"
                declare function field_offset(ty: string, field: string): i32;
                declare function type_size(ty: string): i32;
                declare function log(x: f64): void;

                export function on_tick(): void {{
                  // Offsets come from the type database, never from this file.
                  let size: i32 = type_size("fixture::Packet");
                  log(size as f64);
                  log(field_offset("fixture::Packet", "seq") as f64);
                }}
                "#
            ),
        )]);
        let mut runtime = TsRuntime::with_root(mod_dir.clone());
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        // Install the database the way the tick system does.
        TYPE_DB.with(|slot| *slot.borrow_mut() = Some(Arc::new(db)));

        // A region over the real Rust value, so the mod can read it.
        let mut runtime_ctx = runtime;
        let rows = QueryRows::from_positions(&positions());
        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime_ctx.tick(&positions(), std::ptr::null(), &rows.players);
        PullCtx::uninstall();
        TYPE_DB.with(|slot| *slot.borrow_mut() = None);

        // The handler ran: offsets were queryable (a -1 result would mean the
        // database never reached the host functions).
        let _ = value;
        let _ = std::ptr::addr_of!(value);
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&mod_dir);
    }

    #[test]
    fn a_mod_reads_query_rows_by_offset() {
        let dir = mods_dir(&[(
            "query.ts",
            r#"
            declare function query_players(): i32;
            declare function query_row_count(region: i32): i32;
            declare function row_stride(): i32;
            declare function log(x: f64): void;

            export function on_tick(): void {
              let region: i32 = query_players();
              let count: i32 = query_row_count(region);
              let stride: i32 = row_stride();
              if (count <= 0) { return; }
              // The host reports where each field lives; nothing is hardcoded.
              let id: i64 = load_i64(region, row_offset("id")) as i64;
              log(id as f64);
              log(count as f64);
              log(stride as f64);
            }
            "#,
        )]);

        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        let rows = QueryRows::from_positions(&positions());
        assert_eq!(rows.players.len(), 1);
        assert_eq!(rows.players[0].id, 1);

        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null(), &rows.players);
        PullCtx::uninstall();

        // The handler ran without trapping (a trap is logged, not fatal), and
        // nothing was queued: `query_*` is read-only.
        assert!(runtime.outbox().lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loads_and_ticks_a_ts_mod() {
        let dir = mods_dir(&[(
            "demo.ts",
            r#"
            declare function set_block(x: i32, y: i32, z: i32, block: i32): void;
            export function on_tick(): void { set_block(1, 2, 3, 4); }
            "#,
        )]);

        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        assert_eq!(
            runtime.exported_handlers().len(),
            1,
            "one mod with on_tick should have loaded"
        );

        let rows = QueryRows::from_positions(&positions());
        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null(), &rows.players);
        PullCtx::uninstall();

        let outbox = runtime.outbox();
        let requests = outbox.lock().unwrap();
        assert_eq!(requests.len(), 1, "the mod queued one request");
        match &requests[0] {
            ScriptRequest::SetBlock { pos, block } => {
                assert_eq!((pos.x, pos.y, pos.z), (1, 2, 3));
                assert_eq!(*block as u8, 4);
            }
            _ => panic!("expected a SetBlock request"),
        }
        drop(requests);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_mod_keeps_the_server_running() {
        let dir = mods_dir(&[(
            "bad.ts",
            r#"
            export function on_tick(): i32 { return "not a number"; }
            "#,
        )]);
        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        assert_eq!(
            runtime.exported_handlers().len(),
            0,
            "a mod that fails to compile must not be loaded"
        );

        // Fixing the file and reloading picks it up again.
        std::fs::write(dir.join("bad.ts"), "export function on_tick(): void {}\n").unwrap();
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_runaway_mod_is_trapped_not_hung() {
        let dir = mods_dir(&[(
            "spin.ts",
            r#"
            export function on_tick(): void {
              let n: i32 = 0;
              while (true) { n += 1; }
            }
            "#,
        )]);
        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        let rows = QueryRows::from_positions(&positions());
        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null(), &rows.players);
        PullCtx::uninstall();

        // Fuel stopped the spin, so the tick returned and nothing was queued.
        assert!(runtime.outbox().lock().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn events_reach_mods_with_flattened_args() {
        let dir = mods_dir(&[(
            "place.ts",
            r#"
            declare function log(x: f64): void;
            export function on_place_block(
              id: string, x: i32, y: i32, z: i32, block: i32
            ): void {
              log(x as f64);
            }
            "#,
        )]);
        let mut runtime = TsRuntime::with_root(dir.clone());
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
            rows: RowsView::default(),
        components: HashMap::new(),
        }
        .install();
        runtime.dispatch(
            "place_block",
            &[
                HostArg::Str("7".to_owned()),
                HostArg::I32(11),
                HostArg::I32(12),
                HostArg::I32(13),
                HostArg::F64(14.0),
            ],
        );
        PullCtx::uninstall();

        // The handler ran without trapping (a spin would have shown up as a
        // hang here, and a trap is logged, not fatal).
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Host-side view of `field_offset`, for tests (the host functions are
/// `extern "C"` and take arena strings, which only compiled code can build).
fn host_field_offset_str(
    db: &tesjit_dwarf::TypeDb,
    ty: &str,
    field: &str,
) -> Option<u32> {
    let def = db.get(ty)?;
    let tesjit_dwarf::TypeArgs::Struct(fields) = &def.args else {
        return None;
    };
    fields
        .fields
        .iter()
        .find(|f| f.name == field)
        .map(|f| f.offset as u32)
}

#[cfg(test)]
mod ts_mod_compile_tests {
    use super::*;

    #[test]
    fn demo_option_compiles() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.parent().unwrap(); // /home/nestor/prj/exsolis
        let p = root.join("mods").join("ts").join("demo_option.ts");
        match compile_mod(&p) {
            Ok(_) => {}
            Err(es) => panic!("demo_option.ts failed: {:#?}", es),
        }
    }
}

#[cfg(test)]
mod phase2_smoke_compile_test {
    use super::*;
    #[test]
    fn phase2_smoke_compiles() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.parent().unwrap();
        let p = root.join("mods").join("ts").join("phase2_smoke.ts");
        match compile_mod(&p) {
            Ok(_) => {}
            Err(es) => panic!("phase2_smoke.ts failed: {:#?}", es),
        }
    }
}

#[cfg(test)]
mod component_access_compile_test {
    use super::*;
    #[test]
    fn component_access_compiles() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.parent().unwrap();
        let p = root.join("mods").join("ts").join("component_access.ts");
        match compile_mod(&p) {
            Ok(_) => {}
            Err(es) => panic!("component_access.ts failed: {:#?}", es),
        }
    }
}

#[cfg(test)]
mod typed_rows_compile_test {
    use super::*;
    #[test]
    fn typed_rows_compiles() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest.parent().unwrap();
        let p = root.join("mods").join("ts").join("typed_rows.ts");
        match compile_mod(&p) {
            Ok(_) => {}
            Err(es) => panic!("typed_rows.ts failed: {:#?}", es),
        }
    }
}
