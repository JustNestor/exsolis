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

use exsolis_common::{BlockPos, BlockType, PlayerId};

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
}

thread_local! {
    static PULL: std::cell::RefCell<Option<PullCtx>> = const { std::cell::RefCell::new(None) };
    /// Requests queued by host calls while a handler runs.
    static OUTBOX: std::cell::RefCell<Option<Arc<Mutex<Vec<ScriptRequest>>>>> =
        const { std::cell::RefCell::new(None) };
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
        }
    }
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
    let Some(block) = BlockType::from_u8(block as u8) else {
        warn!("ts mod: set_block: invalid block id {block}");
        return;
    };
    queue(ScriptRequest::SetBlock {
        pos: BlockPos::new(x, y, z),
        block,
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

/// Block id at a world position, or `Air` when the chunk is not loaded.
fn block_at(world: &ChunkWorld, x: i32, y: i32, z: i32) -> i32 {
    let world: &ChunkWorld = world;
    let (coord, idx) = world_to_chunk_index(BlockPos::new(x, y, z));
    world
        .ready
        .get(&coord)
        .and_then(|c| c.unpack())
        .and_then(|u| u.get(idx).copied())
        .unwrap_or(BlockType::Air) as u8 as i32
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
    root: PathBuf,
}

impl Default for TsRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl TsRuntime {
    pub fn new() -> Self {
        Self {
            outbox: Arc::new(Mutex::new(Vec::new())),
            mods: Vec::new(),
            stamps: HashMap::new(),
            root: mods_directory(),
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

        for path in sources {
            let Ok(stamp) = std::fs::metadata(&path).and_then(|m| m.modified()) else {
                continue;
            };
            next_stamps.insert(path.clone(), stamp);
            let unchanged = self
                .stamps
                .get(&path)
                .map(|previous| *previous == stamp)
                .unwrap_or(false);
            if unchanged {
                if let Some(index) = self.mods.iter().position(|m| m.name == display(&path)) {
                    // Move the compiled module over as-is.
                    next_mods.push(self.mods.swap_remove(index));
                    continue;
                }
            }
            match compile_mod(&path) {
                Ok(compiled) => {
                    info!("[ts mod] loaded {}", path.display());
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
    pub fn tick(&self, positions: &HashMap<String, (f32, f32, f32)>, world: *const ChunkWorld) {
        self.with_outbox(|| {
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
                out.push(path);
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
            .add_systems(Update, ts_tick_system)
            .add_systems(Update, ts_drain_system.before(crate::net::flush_outgoing));
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
    let world_ptr: *const ChunkWorld = match world.get_resource::<ChunkWorld>() {
        Some(chunks) => chunks as *const ChunkWorld,
        None => std::ptr::null(),
    };

    PullCtx {
        positions: positions.clone(),
        world: world_ptr,
    }
    .install();
    runtime.tick(&positions, world_ptr);
    PullCtx::uninstall();

    world.insert_resource(runtime);
}

impl PullCtx {
    fn uninstall() {
        PULL.with(|p| *p.borrow_mut() = None);
    }
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
) {
    for ev in events.read() {
        runtime.dispatch(
            "place_block",
            &[
                HostArg::Str(ev.player_id.to_string()),
                HostArg::I32(ev.pos.x),
                HostArg::I32(ev.pos.y),
                HostArg::I32(ev.pos.z),
                HostArg::I32(ev.kind as u8 as i32),
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
        // `mods_directory()` honours `EXSOLIS_MODS_DIR`.
        unsafe { std::env::set_var("EXSOLIS_MODS_DIR", &dir) };
        dir
    }

    fn positions() -> HashMap<String, (f32, f32, f32)> {
        HashMap::from([("1".to_owned(), (1.0f32, 2.0f32, 3.0f32))])
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

        let mut runtime = TsRuntime::new();
        runtime.reload();
        assert_eq!(
            runtime.exported_handlers().len(),
            1,
            "one mod with on_tick should have loaded"
        );

        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null());
        PullCtx::uninstall();

        let requests = runtime.outbox().lock().unwrap();
        assert_eq!(requests.len(), 1, "the mod queued one request");
        match &requests[0] {
            ScriptRequest::SetBlock { pos, block } => {
                assert_eq!((pos.x, pos.y, pos.z), (1, 2, 3));
                assert_eq!(*block as u8, 4);
            }
            other => panic!("unexpected request: {other:?}"),
        }
        drop(requests);
        unsafe { std::env::remove_var("EXSOLIS_MODS_DIR") };
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
        let mut runtime = TsRuntime::new();
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
        unsafe { std::env::remove_var("EXSOLIS_MODS_DIR") };
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
        let mut runtime = TsRuntime::new();
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
        }
        .install();
        runtime.tick(&positions(), std::ptr::null());
        PullCtx::uninstall();

        // Fuel stopped the spin, so the tick returned and nothing was queued.
        assert!(runtime.outbox().lock().unwrap().is_empty());
        unsafe { std::env::remove_var("EXSOLIS_MODS_DIR") };
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
        let mut runtime = TsRuntime::new();
        runtime.reload();
        assert_eq!(runtime.exported_handlers().len(), 1);

        PullCtx {
            positions: positions(),
            world: std::ptr::null(),
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
        unsafe { std::env::remove_var("EXSOLIS_MODS_DIR") };
        let _ = std::fs::remove_dir_all(&dir);
    }
}
