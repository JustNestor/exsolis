//! Lua modding layer (mlua, LuaJIT).
//!
//! Scope, deliberately kept small for v1:
//!   - one long-lived `Lua` VM, ticked once per `Update`
//!   - mods subscribe to a fixed set of Bevy `Message` types by name
//!     (`api.on("place_block", fn(ev) ... end)`) and get called with a
//!     plain Lua table per event, once per tick, in the order the engine
//!     emitted them
//!   - mods call back into the engine via a small, explicit `api.*`
//!     surface (`api.send_chat`, `api.set_block`, `api.broadcast`, ...)
//!     that queues a request; a Bevy system drains the queue after the
//!     VM tick and turns it into real ECS messages / `OutgoingPacket`s
//!
//! Everything here is written assuming the *interpreter* changes out from
//! under it later (mlua std Lua -> LuaJIT, or a full rewrite of this
//! bridge), but the *shape* of the Rust-side API does not:
//!   - never hand a mod a live reference into the `World`, an `Entity`,
//!     or any Rust type. Every value crossing the boundary is a number,
//!     string, bool, or a flat table of those. That's what makes it
//!     portable to a differently-embedded Lua later, and it's also what
//!     keeps a single call cheap: no reflection, no trait-object
//!     dispatch, no per-call table-of-tables allocation unless the event
//!     shape genuinely needs it.
//!   - a mod callback NEVER gets `&mut World` / `Commands` / anything
//!     ECS-shaped. It reads a table, it returns a value or calls `api.*`.
//!     The translation from "mod asked for X" to "ECS mutation" happens
//!     entirely in `drain_script_requests`, in plain Rust, after the Lua
//!     call has already returned. This is what lets this whole bridge be
//!     swapped for a differently-shaped one later without mods noticing.
//!   - hot paths (anything that could fire every tick, e.g. movement)
//!     are exposed as *pull* (`api.get_player_pos(id)`), not *push*
//!     (there is no `on("player_moved", ...)` firing 20x/sec to every
//!     mod whether it cares or not). Only discrete, low-frequency events
//!     (join/leave, place/break, chat, hotbar select) are push/`on`.
//!     A mod that wants per-tick behavior registers `api.on_tick(fn)`
//!     and pulls whatever state it needs from inside that, instead of
//!     the engine pushing per-tick data at every mod whether or not it
//!     asked for it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{error, info, warn};
use mlua::{Lua, MultiValue, Table, Value};

use exsolis_common::{BlockPos, BlockType, PlayerId, ServerMessage};

use crate::Player;
use crate::chunks::{ChunkWorld, PlayerPosition};
use crate::interaction::{BreakBlockRequested, PlaceBlockRequested, SelectHotbarRequested};
use crate::net::OutgoingPacket;

// ---------------------------------------------------------------------
// Public plugin
// ---------------------------------------------------------------------

pub struct ScriptingPlugin;

impl Plugin for ScriptingPlugin {
    fn build(&self, app: &mut App) {
        let runtime = ScriptRuntime::new().expect("failed to initialize Lua runtime");

        app.insert_non_send_resource(runtime)
            .add_systems(
                Update,
                (
                    forward_place_block,
                    forward_break_block,
                    forward_select_hotbar,
                )
                    .after(crate::events::dispatch_incoming)
                    .before(run_on_tick),
            )
            .add_systems(Update, run_on_tick.after(forward_select_hotbar))
            .add_systems(Update, drain_script_requests.after(run_on_tick));
    }
}

// ---------------------------------------------------------------------
// The runtime itself
// ---------------------------------------------------------------------

/// `Lua` (and therefore this) is `!Sync`, so it's a non-send resource —
/// Bevy still schedules its systems on the main thread for us, same as
/// any other `!Send`/`!Sync` resource, it just can't be accessed from
/// worker threads. That's fine: script ticking is one call per frame,
/// not something we want parallelized anyway.
pub struct ScriptRuntime {
    lua: Lua,
    /// Requests queued by `api.*` calls during this tick's Lua callbacks.
    /// `Rc<RefCell<..>>` because the closures registered as Lua callbacks
    /// are 'static and can't borrow the `ScriptRuntime` itself; they
    /// close over a clone of this handle instead. Drained once per tick
    /// by `drain_script_requests`, right after the VM runs, so nothing
    /// mutates the ECS from inside a Lua call.
    outbox: Rc<RefCell<Vec<ScriptRequest>>>,
}

/// One request a mod made this tick, translated into ECS terms by
/// `drain_script_requests`. Kept as a flat enum of plain data (no
/// `Entity`, no `World` reference) so building this list never needs
/// scheduler access and so this type is trivial to keep unchanged if the
/// Lua side is later swapped for LuaJIT talking over the same shape.
enum ScriptRequest {
    /// `api.send(player_id, kind, ...)` — a raw `ServerMessage` a mod
    /// wants delivered to one player. `kind` is matched against a small
    /// fixed set of message kinds a mod is allowed to send (see
    /// `build_server_message`); anything else is a no-op + a warning,
    /// never a panic, since the string came from a mod.
    SendTo {
        player_id: PlayerId,
        kind: String,
        args: Vec<LuaScalar>,
    },
    Broadcast {
        kind: String,
        args: Vec<LuaScalar>,
        except: Option<PlayerId>,
    },
    SetBlock {
        pos: BlockPos,
        block: BlockType,
    },
}

/// A Lua value narrowed to the handful of primitive shapes the bridge
/// actually needs to move across the boundary. Kept separate from
/// `mlua::Value` so `ScriptRequest` (and therefore the outbox) has no
/// lifetime tied to the `Lua` instance and no `mlua` types leak past
/// this module.
#[derive(Clone, Debug)]
#[allow(dead_code)] // Bool/as_u8 have no caller yet; kept for the next
// message kind a mod needs (e.g. a bool flag arg).
enum LuaScalar {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

impl LuaScalar {
    fn from_value(v: &Value) -> Option<Self> {
        match v {
            Value::Integer(i) => Some(LuaScalar::Int(*i)),
            Value::Number(n) => Some(LuaScalar::Float(*n)),
            Value::String(s) => s.to_str().ok().map(|s| LuaScalar::Str(s.to_string())),
            Value::Boolean(b) => Some(LuaScalar::Bool(*b)),
            _ => None,
        }
    }
    fn as_f32(&self) -> Option<f32> {
        match self {
            LuaScalar::Int(i) => Some(*i as f32),
            LuaScalar::Float(f) => Some(*f as f32),
            _ => None,
        }
    }
    fn as_i32(&self) -> Option<i32> {
        match self {
            LuaScalar::Int(i) => Some(*i as i32),
            LuaScalar::Float(f) => Some(*f as i32),
            _ => None,
        }
    }
    fn as_u8(&self) -> Option<u8> {
        self.as_i32().map(|i| i.clamp(0, u8::MAX as i32) as u8)
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            LuaScalar::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl ScriptRuntime {
    pub fn new() -> mlua::Result<Self> {
        let lua = Lua::new();
        let outbox: Rc<RefCell<Vec<ScriptRequest>>> = Rc::new(RefCell::new(Vec::new()));

        install_api(&lua, outbox.clone())?;
        load_mods(&lua)?;

        Ok(Self { lua, outbox })
    }

    /// Calls every Lua handler registered for `event`, in registration
    /// order, with `table` as the sole argument. Errors from an
    /// individual mod are logged and do not stop other handlers or the
    /// server; a bug in one mod's script must never take the game down.
    fn dispatch(&self, event: &str, table: Table) {
        let handlers: Table = match self.lua.globals().get("__handlers") {
            Ok(t) => t,
            Err(e) => {
                error!("scripting: __handlers table missing: {e}");
                return;
            }
        };
        let list: Option<Table> = handlers.get(event).ok();
        let Some(list) = list else { return };

        let len = list.raw_len();
        for i in 1..=len {
            let f: mlua::Function = match list.raw_get(i) {
                Ok(f) => f,
                Err(_) => continue,
            };
            if let Err(e) = f.call::<()>(table.clone()) {
                warn!("scripting: handler for '{event}' errored: {e}");
            }
        }
    }

    /// Calls every `api.on_tick(fn)` handler once, with no arguments.
    /// This is the ONLY per-tick push callback; everything else a mod
    /// wants per-tick state for, it pulls via `api.get_*` from inside
    /// this callback instead of the engine calling it out separately.
    fn tick(&self) {
        let handlers: Table = match self.lua.globals().get("__tick_handlers") {
            Ok(t) => t,
            Err(_) => return,
        };
        let len = handlers.raw_len();
        for i in 1..=len {
            let f: mlua::Function = match handlers.raw_get(i) {
                Ok(f) => f,
                Err(_) => continue,
            };
            if let Err(e) = f.call::<()>(()) {
                warn!("scripting: on_tick handler errored: {e}");
            }
        }
    }

    fn take_requests(&self) -> Vec<ScriptRequest> {
        std::mem::take(&mut self.outbox.borrow_mut())
    }
}

// ---------------------------------------------------------------------
// Lua-facing API surface
// ---------------------------------------------------------------------

/// Installs the `api` global table and the internal `__handlers` /
/// `__tick_handlers` registries. Called once at startup; nothing here
/// runs per-tick.
fn install_api(lua: &Lua, outbox: Rc<RefCell<Vec<ScriptRequest>>>) -> mlua::Result<()> {
    let globals = lua.globals();

    let handlers = lua.create_table()?;
    globals.set("__handlers", handlers)?;
    let tick_handlers = lua.create_table()?;
    globals.set("__tick_handlers", tick_handlers)?;

    let api = lua.create_table()?;

    // --- api.on(event_name, fn(ev)) : subscribe to a discrete event ----
    api.set(
        "on",
        lua.create_function(|lua, (event, f): (String, mlua::Function)| {
            let handlers: Table = lua.globals().get("__handlers")?;
            let list: Table = match handlers.get::<Option<Table>>(event.clone())? {
                Some(t) => t,
                None => {
                    let t = lua.create_table()?;
                    handlers.set(event.clone(), t.clone())?;
                    t
                }
            };
            let len = list.raw_len();
            list.raw_set(len + 1, f)?;
            Ok(())
        })?,
    )?;

    // --- api.on_tick(fn()) : called once per server tick --------------
    api.set(
        "on_tick",
        lua.create_function(|lua, f: mlua::Function| {
            let list: Table = lua.globals().get("__tick_handlers")?;
            let len = list.raw_len();
            list.raw_set(len + 1, f)?;
            Ok(())
        })?,
    )?;

    // --- api.send(player_id, kind, ...) : queue a message to one player
    {
        let outbox = outbox.clone();
        api.set(
            "send",
            lua.create_function(move |_, args: MultiValue| {
                let mut it = args.into_iter();
                let player_id = pop_player_id(&mut it)?;
                let kind = pop_string(&mut it)?;
                let rest = collect_scalars(it);
                outbox.borrow_mut().push(ScriptRequest::SendTo {
                    player_id,
                    kind,
                    args: rest,
                });
                Ok(())
            })?,
        )?;
    }

    // --- api.broadcast(kind, ...) : queue a message to every player ----
    {
        let outbox = outbox.clone();
        api.set(
            "broadcast",
            lua.create_function(move |_, args: MultiValue| {
                let mut it = args.into_iter();
                let kind = pop_string(&mut it)?;
                let rest = collect_scalars(it);
                outbox.borrow_mut().push(ScriptRequest::Broadcast {
                    kind,
                    args: rest,
                    except: None,
                });
                Ok(())
            })?,
        )?;
    }

    // --- api.broadcast_except(player_id, kind, ...) --------------------
    {
        let outbox = outbox.clone();
        api.set(
            "broadcast_except",
            lua.create_function(move |_, args: MultiValue| {
                let mut it = args.into_iter();
                let except = pop_player_id(&mut it)?;
                let kind = pop_string(&mut it)?;
                let rest = collect_scalars(it);
                outbox.borrow_mut().push(ScriptRequest::Broadcast {
                    kind,
                    args: rest,
                    except: Some(except),
                });
                Ok(())
            })?,
        )?;
    }

    // --- api.set_block(x, y, z, block_id) : queue a world edit ---------
    {
        let outbox = outbox.clone();
        api.set(
            "set_block",
            lua.create_function(move |_, (x, y, z, block_id): (i32, i32, i32, u8)| {
                let Some(block) = BlockType::from_u8(block_id) else {
                    return Err(mlua::Error::RuntimeError(format!(
                        "set_block: invalid block id {block_id}"
                    )));
                };
                outbox.borrow_mut().push(ScriptRequest::SetBlock {
                    pos: BlockPos::new(x, y, z),
                    block,
                });
                Ok(())
            })?,
        )?;
    }

    // Pull-based world-read helpers (`get_player_pos`, `get_block`,
    // `list_players`) are NOT installed here, because they need to
    // borrow live ECS state (`Query`, `Res<ChunkWorld>`) that only
    // exists for the duration of one system call. Registering a
    // `'static` global function that reads that state safely isn't
    // possible without either cloning the whole world every tick or
    // reaching for unsafe. Instead they're installed for the scope of a
    // single tick, borrowed, via `mlua::Lua::scope` in `run_on_tick`
    // below — the safe way to lend Lua a callback that closes over
    // short-lived Rust references. `dispatch`/`tick` still work at
    // global scope for the always-available `on`/`on_tick`/`send`/etc,
    // since those don't need to borrow anything ECS-shaped.
    globals.set("api", api)?;
    Ok(())
}

/// `PlayerId` is `u64`, and standard Lua numbers are `f64` (53 bits of
/// exact integer precision) — a real client id like
/// `4174018466291290165` needs 63 bits and silently rounds to the
/// nearest representable double the instant it becomes a Lua number,
/// coming back out as a DIFFERENT id that's a few hundred off (this is
/// exactly what produced "Tried to send a message to invalid client
/// ...112" for an id ending in ...165: the double couldn't hold it).
/// So `PlayerId` crosses the boundary as a decimal *string* instead,
/// both directions — `player_id_to_lua` here, and the string branch in
/// `pop_player_id` below. Never put a `PlayerId` into a Lua table or
/// function argument as a bare integer/number again; always route it
/// through this pair.
fn player_id_to_lua(id: PlayerId) -> String {
    id.to_string()
}

fn pop_player_id(it: &mut impl Iterator<Item = Value>) -> mlua::Result<PlayerId> {
    match it.next() {
        Some(Value::String(s)) => {
            let s = s.to_str()?;
            s.parse::<PlayerId>()
                .map_err(|_| mlua::Error::RuntimeError(format!("'{s}' is not a valid player_id")))
        }
        // Tolerated for convenience (e.g. a mod hardcoding a small test
        // id), but anything beyond 2^53 silently lost precision before
        // it ever reached us — there's no way to recover it here, so
        // this path only works for small ids. Prefer the string form.
        Some(Value::Integer(i)) => Ok(i as PlayerId),
        Some(Value::Number(n)) => Ok(n as PlayerId),
        other => Err(mlua::Error::RuntimeError(format!(
            "expected player_id (string, e.g. from ev.player_id), got {other:?}"
        ))),
    }
}

fn pop_string(it: &mut impl Iterator<Item = Value>) -> mlua::Result<String> {
    match it.next() {
        Some(Value::String(s)) => Ok(s.to_str()?.to_string()),
        other => Err(mlua::Error::RuntimeError(format!(
            "expected string, got {other:?}"
        ))),
    }
}

fn collect_scalars(it: impl Iterator<Item = Value>) -> Vec<LuaScalar> {
    it.filter_map(|v| LuaScalar::from_value(&v)).collect()
}

/// Where mod files live. v1: load every `.lua` file in this directory,
/// in filename order, at startup. No hot reload yet.
const MODS_DIR: &str = "mods";

fn load_mods(lua: &Lua) -> mlua::Result<()> {
    let dir = std::path::Path::new(MODS_DIR);
    if !dir.is_dir() {
        warn!("scripting: no '{MODS_DIR}' directory found, no mods loaded");
        return Ok(());
    }
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| mlua::Error::RuntimeError(e.to_string()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "lua"))
        .collect();
    paths.sort();

    for path in paths {
        let src =
            std::fs::read_to_string(&path).map_err(|e| mlua::Error::RuntimeError(e.to_string()))?;
        let name = path.display().to_string();
        if let Err(e) = lua.load(&src).set_name(&name).exec() {
            error!("scripting: failed to load {name}: {e}");
        }
        info!("[mod] loading {}", name);
    }
    Ok(())
}

/// Converts a world position to `(ChunkCoord, index-into-chunk)`. Shared
/// by `get_block` (read) and `set_block_and_broadcast` (write) so the two
/// can never disagree about which cell a coordinate lands on.
fn world_to_chunk_index(pos: BlockPos) -> (exsolis_common::ChunkCoord, usize) {
    const S: i32 = exsolis_common::CHUNK_SIZE as i32;
    let coord = exsolis_common::ChunkCoord::new(
        pos.x.div_euclid(S),
        pos.y.div_euclid(S),
        pos.z.div_euclid(S),
    );
    let (lx, ly, lz) = (
        pos.x.rem_euclid(S) as usize,
        pos.y.rem_euclid(S) as usize,
        pos.z.rem_euclid(S) as usize,
    );
    let idx = lx
        + ly * exsolis_common::CHUNK_SIZE
        + lz * exsolis_common::CHUNK_SIZE * exsolis_common::CHUNK_SIZE;
    (coord, idx)
}

// ---------------------------------------------------------------------
// Bevy systems: bridge ECS <-> Lua
// ---------------------------------------------------------------------

/// Forwards `PlaceBlockRequested` to any Lua `on("place_block", ...)`
/// handlers as an observational hook — this fires in ADDITION to (not
/// instead of) `interaction.rs`'s own server-authoritative handling; a
/// mod cannot override or veto placement in v1, only react to it (e.g.
/// for stats, chat announcements, custom drops via `api.set_block`).
fn forward_place_block(
    mut events: MessageReader<PlaceBlockRequested>,
    runtime: NonSend<ScriptRuntime>,
) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("x", ev.pos.x);
        let _ = t.set("y", ev.pos.y);
        let _ = t.set("z", ev.pos.z);
        let _ = t.set("block_id", ev.kind as u8);
        runtime.dispatch("place_block", t);
    }
}

fn forward_break_block(
    mut events: MessageReader<BreakBlockRequested>,
    runtime: NonSend<ScriptRuntime>,
) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("x", ev.pos.x);
        let _ = t.set("y", ev.pos.y);
        let _ = t.set("z", ev.pos.z);
        runtime.dispatch("break_block", t);
    }
}

fn forward_select_hotbar(
    mut events: MessageReader<SelectHotbarRequested>,
    runtime: NonSend<ScriptRuntime>,
) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("index", ev.index);
        runtime.dispatch("select_hotbar", t);
    }
}

/// Runs one tick of scripting: installs the pull-style read helpers
/// (`get_player_pos`, `get_block`, `list_players`) as *scoped* globals —
/// borrowed from this system's own `Query`/`Res` params for the
/// duration of the closure passed to `lua.scope`, then automatically
/// un-registered when the closure returns — and then fires every
/// `api.on_tick` handler.
///
/// `Lua::scope` is mlua's safe answer to "let a Lua callback close over
/// a Rust reference that doesn't live for `'static`": no unsafe, no
/// snapshot-cloning of the whole chunk world every tick, and the
/// borrow checker guarantees a mod can't stash the closure somewhere
/// and call it after this system returns (mlua enforces the scoped
/// function is dropped/invalidated when the scope ends).
///
/// Building a small hashmap of positions once per tick (not once per
/// `get_player_pos` call) is what keeps this cheap even if an `on_tick`
/// handler calls it in a loop over every player.
fn run_on_tick(
    runtime: NonSend<ScriptRuntime>,
    positions: Query<(&Player, &PlayerPosition)>,
    world: Res<ChunkWorld>,
) {
    let positions: HashMap<PlayerId, (f32, f32, f32)> = positions
        .iter()
        .map(|(player, pos)| (player.0, (pos.0.x, pos.0.y, pos.0.z)))
        .collect();
    let world: &ChunkWorld = &world;

    let result = runtime.lua.scope(|scope| {
        let api: Table = runtime.lua.globals().get("api")?;

        let get_player_pos = scope.create_function(|lua, player_id: String| {
            let Ok(player_id) = player_id.parse::<PlayerId>() else {
                return Ok(Value::Nil);
            };
            match positions.get(&player_id) {
                Some((x, y, z)) => {
                    let t = lua.create_table()?;
                    t.set("x", *x)?;
                    t.set("y", *y)?;
                    t.set("z", *z)?;
                    Ok(Value::Table(t))
                }
                None => Ok(Value::Nil),
            }
        })?;
        api.set("get_player_pos", get_player_pos)?;

        // Same precision issue as everywhere else PlayerId crosses into
        // Lua: each id goes out as a decimal string, not a number.
        let list_players = scope.create_function(|lua, ()| {
            let t = lua.create_table()?;
            for (i, id) in positions.keys().enumerate() {
                t.raw_set(i + 1, player_id_to_lua(*id))?;
            }
            Ok(t)
        })?;
        api.set("list_players", list_players)?;

        let get_block = scope.create_function(|_, (x, y, z): (i32, i32, i32)| {
            let (coord, idx) = world_to_chunk_index(BlockPos::new(x, y, z));
            let block = world
                .ready
                .get(&coord)
                .and_then(|c| c.unpack())
                .and_then(|u| u.get(idx).copied())
                .unwrap_or(BlockType::Air);
            Ok(block as u8)
        })?;
        api.set("get_block", get_block)?;

        runtime.tick();
        Ok(())
    });

    if let Err(e) = result {
        error!("scripting: on_tick scope errored: {e}");
    }
}

/// Drains everything mods queued via `api.send` / `api.broadcast` /
/// `api.set_block` this tick and turns each into a real ECS effect:
/// world edits go through `ChunkWorld` directly (same mutation
/// `interaction.rs::apply_and_broadcast` does), messages go out as
/// `OutgoingPacket`. This is the only place `ScriptRequest` is
/// interpreted, and the only place scripting touches gameplay state —
/// keeping it in one system means a future rewrite of the Lua bridge
/// only has to keep producing this same `ScriptRequest` shape, not
/// understand gameplay itself.
fn drain_script_requests(
    runtime: NonSend<ScriptRuntime>,
    mut world: ResMut<ChunkWorld>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    loaded_by: Query<(&Player, &crate::chunks::ChunkRenderState)>,
) {
    for req in runtime.take_requests() {
        match req {
            ScriptRequest::SendTo {
                player_id,
                kind,
                args,
            } => match build_server_message(&kind, &args) {
                Ok(msg) => {
                    outgoing.write(OutgoingPacket::to(player_id, msg));
                }
                Err(e) => warn!("scripting: api.send: {e}"),
            },
            ScriptRequest::Broadcast { kind, args, except } => {
                match build_server_message(&kind, &args) {
                    Ok(msg) => {
                        outgoing.write(match except {
                            Some(id) => OutgoingPacket::broadcast_except(id, msg),
                            None => OutgoingPacket::broadcast(msg),
                        });
                    }
                    Err(e) => warn!("scripting: api.broadcast: {e}"),
                }
            }
            ScriptRequest::SetBlock { pos, block } => {
                set_block_and_broadcast(pos, block, &mut world, &mut outgoing, &loaded_by);
            }
        }
    }
}

/// The fixed set of `ServerMessage` kinds a mod is allowed to construct.
/// Deliberately a small allow-list, not a generic "serialize whatever
/// table the mod gives you" — mods are untrusted-ish content, and the
/// wire protocol is something clients hard-code assumptions about
/// (`lib.rs`'s `#[non_exhaustive]` comment), so letting a mod emit
/// arbitrary `ServerMessage` variants with malformed fields is a client
/// crash waiting to happen. Extend this list as mods need more, same as
/// `ClientMessage`/`ServerMessage` themselves get extended: append,
/// don't renumber.
fn build_server_message(kind: &str, args: &[LuaScalar]) -> Result<ServerMessage, String> {
    match kind {
        // args: [entity_id, kind_id, x, y, z]
        "show_entity" => {
            let entity_id = args.first().and_then(LuaScalar::as_i32).unwrap_or(0) as u64;
            let x = args.get(2).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            let y = args.get(3).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            let z = args.get(4).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            Ok(ServerMessage::ShowEntity {
                entity_id,
                kind: exsolis_common::EntityKind::Player,
                x,
                y,
                z,
            })
        }
        // args: [entity_id]
        "entity_disappear" => {
            let entity_id = args.first().and_then(LuaScalar::as_i32).unwrap_or(0) as u64;
            Ok(ServerMessage::EntityDisappear { entity_id })
        }
        // args: [ui_id, xml]
        "ui_show" => {
            let id = args
                .first()
                .and_then(LuaScalar::as_str)
                .ok_or("ui_show: missing ui_id")?
                .to_string();
            let xml = args
                .get(1)
                .and_then(LuaScalar::as_str)
                .ok_or("ui_show: missing xml")?
                .to_string();
            if xml.len() > exsolis_common::UI_MAX_XML_BYTES {
                return Err(format!(
                    "ui_show: xml too large ({} > {})",
                    xml.len(),
                    exsolis_common::UI_MAX_XML_BYTES
                ));
            }
            Ok(ServerMessage::UiShow { id, xml })
        }
        // args: [ui_id]
        "ui_hide" => {
            let id = args
                .first()
                .and_then(LuaScalar::as_str)
                .ok_or("ui_hide: missing ui_id")?
                .to_string();
            Ok(ServerMessage::UiHide { id })
        }
        other => Err(format!("unknown message kind '{other}'")),
    }
}

/// Same mutate-then-notify-loaded-viewers logic as
/// `interaction.rs::apply_and_broadcast`, reachable from scripting too so
/// `api.set_block` behaves identically to a player-driven placement from
/// the client's point of view. Kept as its own copy rather than importing
/// `interaction`'s private helper — the two call sites have different
/// enough inputs (no inventory, no reach check: a mod is server-side
/// trusted code, not a client claim) that sharing the fn would mean
/// threading unused params through one side or the other.
fn set_block_and_broadcast(
    pos: BlockPos,
    block: BlockType,
    world: &mut ChunkWorld,
    outgoing: &mut MessageWriter<OutgoingPacket>,
    loaded_by: &Query<(&Player, &crate::chunks::ChunkRenderState)>,
) {
    let (coord, idx) = world_to_chunk_index(pos);

    let Some(existing) = world.ready.get(&coord) else {
        warn!("scripting: set_block: chunk {coord:?} not loaded");
        return;
    };
    let Some(updated) = existing.set_block(idx, block) else {
        warn!("scripting: set_block: bad index {idx} for chunk {coord:?}");
        return;
    };
    world.ready.insert(coord, updated);

    for (player, state) in loaded_by {
        if state.loaded.contains(&coord) {
            outgoing.write(OutgoingPacket::to(
                player.0,
                ServerMessage::BlockUpdate { pos, kind: block },
            ));
        }
    }
}
