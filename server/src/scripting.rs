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
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{error, info, warn};
use mlua::{Lua, MultiValue, Table, Value};

use exsolis_common::{BlockPos, BlockType, ClientMessage, KeybindState, PlayerId, ServerMessage};
use fastpack;

use crate::Player;
use crate::chunks::{ChunkWorld, PlayerPosition};
use crate::events::{PlayerJoined, PlayerLeaved};
use crate::interaction::{BreakBlockRequested, PlaceBlockRequested, SelectHotbarRequested};
use crate::net::OutgoingPacket;
use crate::ui::{UiActionRequested, UiDragRequested};

use std::fs;
use std::time::SystemTime;

// ---------------------------------------------------------------------
// Public plugin
// ---------------------------------------------------------------------

pub struct ScriptingPlugin;

impl Plugin for ScriptingPlugin {
    fn build(&self, app: &mut App) {
        let runtime = ScriptRuntime::new().expect("failed to initialize Lua runtime");

        app.insert_non_send(runtime)
            .add_systems(
                Update,
                (
                    reload_mods_system,
                    forward_place_block,
                    forward_break_block,
                    forward_select_hotbar,
                    forward_ui_action,
                    forward_player_join,
                    forward_player_leave,
                    forward_ui_drag,
                    forward_keybind_action,
                )
                    // `ui::dispatch_ui_packets` is what actually produces
                    // `UiActionRequested`/`UiDragRequested`, not
                    // `events::dispatch_incoming` directly — see
                    // `ui.rs::UiPlugin` for why UI validation is its own
                    // system rather than another match arm in
                    // `dispatch_incoming`. Ordering after the latter is
                    // still correct (both run in `GameplaySet`, and
                    // `dispatch_ui_packets` is also `in_set(GameplaySet)`,
                    // just not guaranteed to run strictly before this
                    // tuple by that alone) — pin directly to it instead.
                    .after(crate::events::dispatch_incoming)
                    .after(crate::ui::dispatch_ui_packets)
                    .before(run_on_tick),
            )
            .add_systems(
                Update,
                run_on_tick
                    .after(forward_select_hotbar)
                    .after(forward_ui_action)
                    .after(forward_ui_drag),
            )
            .add_message::<KeybindActionRequested>()
            // FIX: `drain_script_requests` writes `OutgoingPacket` (mod
            // `api.send`/`broadcast`/`UiShow`/`UiHide` all go through it),
            // but was only ordered `.after(run_on_tick)` — never anchored
            // to `net::GameplaySet`/`net::NetSet::Send`. `configure_sets`
            // in `net.rs` only chains `Receive -> GameplaySet -> Send`;
            // a system that's merely transitively downstream of something
            // *in* `GameplaySet` isn't guaranteed by that chain to run
            // before `NetSet::Send`'s `flush_outgoing`. In practice this
            // usually worked because Bevy tends to schedule transitively-
            // ordered systems together, but "usually" isn't a scheduling
            // guarantee — `.before(net::flush_outgoing)` makes it one.
            .add_systems(
                Update,
                drain_script_requests
                    .after(run_on_tick)
                    .before(crate::net::flush_outgoing),
            );
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
    mod_timestamps: RefCell<HashMap<PathBuf, SystemTime>>,
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
    BroadcastAction {
        id: String,
        payload_json: String,
        unreliable: bool,
    },
    SetBlock {
        pos: BlockPos,
        block: BlockType,
    },
    RegisterKeybind {
        id: String,
        default_key: String,
        description: String,
        category: String,
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
        let runtime = Self {
            lua,
            outbox,
            mod_timestamps: Default::default(),
        };
        runtime.reload_mods()?;
        Ok(runtime)
    }

    pub fn check_and_reload_mods(&self) {
        if self.should_reload() {
            info!("scripting: detected mod changes or new mods, hot-reloading...");
            if let Err(e) = self.reload_mods() {
                error!("scripting: error hot-reloading mods: {e}");
            }
        }
    }

    fn should_reload(&self) -> bool {
        let discovered = match discover_lua_files(&crate::assets::mods_directory()) {
            Ok(paths) => paths,
            Err(error) => {
                warn!("scripting: could not scan mod files: {error}");
                return false;
            }
        };
        let current: HashMap<_, _> = discovered.into_iter().collect();
        let previous = self.mod_timestamps.borrow();
        current.len() != previous.len()
            || current
                .iter()
                .any(|(path, mtime)| previous.get(path) != Some(mtime))
    }

    pub fn reload_mods(&self) -> mlua::Result<()> {
        let dir = crate::assets::mods_directory();
        if !dir.is_dir() {
            warn!(
                "scripting: no '{}' directory found, no mods loaded",
                dir.display()
            );
            self.mod_timestamps.borrow_mut().clear();
            return Ok(());
        }

        // Очищаємо обробники подій перед повторним виконанням модів
        let globals = self.lua.globals();
        globals.set("__handlers", self.lua.create_table()?)?;
        globals.set("__tick_handlers", self.lua.create_table()?)?;

        let paths = discover_lua_files(&dir).map_err(|e| {
            mlua::Error::RuntimeError(format!("could not scan {}: {e}", dir.display()))
        })?;
        let new_timestamps: HashMap<_, _> = paths.iter().cloned().collect();

        for (path, _) in paths {
            let src = match fs::read_to_string(&path) {
                Ok(src) => src,
                Err(e) => {
                    error!("scripting: failed to read {}: {e}", path.display());
                    continue;
                }
            };
            let name = path.display().to_string();
            if let Err(e) = self.lua.load(&src).set_name(&name).exec() {
                error!("scripting: failed to load {name}: {e}");
            } else {
                info!("[mod] loaded {}", name);
            }
        }

        *self.mod_timestamps.borrow_mut() = new_timestamps;
        Ok(())
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

    {
        let outbox = outbox.clone();
        api.set(
            "register_keybind",
            lua.create_function(move |_, config: mlua::Table| {
                let id: String = config.get("id")?;
                let default_key: String = config.get("key")?;
                let description: String = config.get("description").unwrap_or_default();
                let category: String = config
                    .get("category")
                    .unwrap_or_else(|_| "Mods".to_string());

                outbox.borrow_mut().push(ScriptRequest::RegisterKeybind {
                    id,
                    default_key,
                    description,
                    category,
                });
                Ok(())
            })?,
        )?;
    }

    api.set(
        "load_xml",
        lua.create_function(|_, filename: String| {
            let rel = std::path::Path::new(&filename);
            if rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(mlua::Error::RuntimeError(
                    "load_xml: parent directory navigation ('..') is not allowed".to_string(),
                ));
            }

            let mut path = PathBuf::from("assets/xml").join(rel);
            if path.extension().is_none() {
                path.set_extension("xml");
            }

            std::fs::read_to_string(&path).map_err(|e| {
                mlua::Error::RuntimeError(format!(
                    "load_xml: failed to read '{}': {e}",
                    path.display()
                ))
            })
        })?,
    )?;

    // --- api.render_xml(filename|template, data?) → expanded XML string --
    // Loads `assets/xml/{filename}.xml` (same path rules as load_xml) and
    // expands `{{path}}` / `v-if` / `v-for` against `data` (a plain table).
    // Result is client-ready XML for `api.send(..., "ui_show", id, xml)`.
    // If the first argument starts with `<`, it is treated as an inline
    // template string instead of a file name.
    api.set(
        "render_xml",
        lua.create_function(|_, (src, data): (String, Option<mlua::Value>)| {
            let data = match data {
                Some(v) => crate::ui_template::data_from_lua(&v)?,
                None => crate::ui_template::DataValue::Nil,
            };
            let xml = if src.trim_start().starts_with('<') {
                crate::ui_template::render(&src, &data)
            } else {
                crate::ui_template::render_file(&src, &data)
            }
            .map_err(|e| mlua::Error::RuntimeError(format!("render_xml: {e}")))?;
            Ok(xml)
        })?,
    )?;

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

    // Reliable actions suit one-shot events. The unreliable variant suits
    // frequent visual updates from `api.on_tick` that can be superseded.
    for (name, unreliable) in [
        ("broadcast_action", false),
        ("broadcast_action_unreliable", true),
    ] {
        let outbox = outbox.clone();
        api.set(
            name,
            lua.create_function(move |_, (id, payload): (String, Value)| {
                if !exsolis_common::is_valid_mod_action_id(&id) {
                    return Err(mlua::Error::RuntimeError(format!(
                        "{name}: invalid namespaced action id '{id}'"
                    )));
                }
                let mut nodes = 0;
                let payload = lua_value_to_json(&payload, 0, &mut nodes)
                    .map_err(mlua::Error::RuntimeError)?;
                if !payload.is_object() {
                    return Err(mlua::Error::RuntimeError(format!(
                        "{name}: payload must be an object table"
                    )));
                }
                let payload_json = serde_json::to_string(&payload)
                    .map_err(|e| mlua::Error::RuntimeError(e.to_string()))?;
                if payload_json.len() > exsolis_common::MOD_ACTION_MAX_PAYLOAD_BYTES {
                    return Err(mlua::Error::RuntimeError(format!(
                        "{name}: payload too large ({} > {})",
                        payload_json.len(),
                        exsolis_common::MOD_ACTION_MAX_PAYLOAD_BYTES
                    )));
                }
                outbox.borrow_mut().push(ScriptRequest::BroadcastAction {
                    id,
                    payload_json,
                    unreliable,
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

/// Discover server Lua scripts recursively. Symlinks are ignored so a mod
/// cannot make the server traverse files outside its configured mods tree.
fn discover_lua_files(root: &Path) -> io::Result<Vec<(PathBuf, SystemTime)>> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, SystemTime)>) -> io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let kind = entry.file_type()?;
            if kind.is_dir() {
                walk(&path, out)?;
            } else if kind.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("lua"))
            {
                let modified = entry
                    .metadata()?
                    .modified()
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                out.push((path, modified));
            }
        }
        Ok(())
    }

    let mut paths = Vec::new();
    if root.is_dir() {
        walk(root, &mut paths)?;
    }
    paths.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(paths)
}

fn forward_player_join(mut events: MessageReader<PlayerJoined>, runtime: NonSend<ScriptRuntime>) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.0));
        runtime.dispatch("player_join", t);
    }
}

fn forward_player_leave(mut events: MessageReader<PlayerLeaved>, runtime: NonSend<ScriptRuntime>) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.0));
        runtime.dispatch("player_leave", t);
    }
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

/// Forwards a validated `UiActionRequested` (see `ui.rs` for what
/// "validated" means here — length/screen-open checks only, nothing
/// about what the action *means*) to `on("ui_action", fn(ev) ... end)`.
/// `ev.data` is `nil` when the widget had no `data` attribute — Lua's
/// natural way to spell "absent", so no separate `has_data` flag is
/// needed the way a fixed-shape wire struct might want one.
fn forward_ui_action(
    mut events: MessageReader<UiActionRequested>,
    runtime: NonSend<ScriptRuntime>,
) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("ui_id", ev.ui_id.as_str());
        let _ = t.set("element_id", ev.element_id.as_str());
        let _ = t.set("action", ev.action.as_str());
        let _ = t.set("data", ev.data.as_deref());
        runtime.dispatch("ui_action", t);
    }
}

/// Forwards a validated `UiDragRequested` to `on("ui_drag", fn(ev) ...
/// end)`. Field names deliberately match the client XML vocabulary
/// (`source_id`/`target_id`, not e.g. `from`/`to`) so a mod author
/// reading both the UI markup and the Lua handler isn't translating
/// between two naming schemes for the same concept.
fn forward_ui_drag(mut events: MessageReader<UiDragRequested>, runtime: NonSend<ScriptRuntime>) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("ui_id", ev.ui_id.as_str());
        let _ = t.set("source_id", ev.source_id.as_str());
        let _ = t.set("target_id", ev.target_id.as_str());
        let _ = t.set("data", ev.data.as_deref());
        runtime.dispatch("ui_drag", t);
    }
}

#[derive(Message)]
pub struct KeybindActionRequested {
    pub player_id: PlayerId,
    pub id: String,
    pub state: KeybindState,
}

fn forward_keybind_action(
    mut events: MessageReader<KeybindActionRequested>,
    positions: Query<(&Player, &PlayerPosition)>,
    runtime: NonSend<ScriptRuntime>,
) {
    for ev in events.read() {
        let Ok(t) = runtime.lua.create_table() else {
            continue;
        };
        let _ = t.set("player_id", player_id_to_lua(ev.player_id));
        let _ = t.set("id", ev.id.as_str());
        let _ = t.set("pressed", matches!(ev.state, KeybindState::Pressed));
        if let Some((_, position)) = positions
            .iter()
            .find(|(player, _)| player.0 == ev.player_id)
            && let Ok(position_table) = runtime.lua.create_table()
        {
            let _ = position_table.set("x", position.0.x);
            let _ = position_table.set("y", position.0.y);
            let _ = position_table.set("z", position.0.z);
            let _ = t.set("position", position_table);
        }
        runtime.dispatch("keybind", t);
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
// `pub(crate)` (was private) so `ui.rs::track_open_screens` can order
// itself `.after(drain_script_requests)` explicitly — see the comment on
// that ordering in `ui.rs::UiPlugin::build` for why: this system's own
// placement in the schedule is only transitively `.after(run_on_tick)`,
// not anchored to `GameplaySet`/`NetSet::Send`, so anything downstream
// that needs to run strictly after it has to reference it by name.
pub(crate) fn drain_script_requests(
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
            ScriptRequest::BroadcastAction {
                id,
                payload_json,
                unreliable,
            } => {
                let msg = ServerMessage::ModAction { id, payload_json };
                outgoing.write(if unreliable {
                    OutgoingPacket::broadcast_unreliable(msg)
                } else {
                    OutgoingPacket::broadcast(msg)
                });
            }
            ScriptRequest::RegisterKeybind {
                id,
                default_key,
                description,
                category,
            } => {
                outgoing.write(OutgoingPacket::broadcast(ServerMessage::RegisterKeybind {
                    id,
                    default_key,
                    description,
                    category,
                }));
            }
            ScriptRequest::SetBlock { pos, block } => {
                set_block_and_broadcast(pos, block, &mut world, &mut outgoing, &loaded_by);
            }
        }
    }
}

const MOD_ACTION_MAX_DEPTH: usize = 8;
const MOD_ACTION_MAX_NODES: usize = 256;

fn lua_value_to_json(
    value: &Value,
    depth: usize,
    nodes: &mut usize,
) -> Result<serde_json::Value, String> {
    use serde_json::{Map, Number, Value as JsonValue};

    *nodes += 1;
    if *nodes > MOD_ACTION_MAX_NODES {
        return Err(format!("payload exceeds {MOD_ACTION_MAX_NODES} values"));
    }
    if depth > MOD_ACTION_MAX_DEPTH {
        return Err(format!("payload nesting exceeds {MOD_ACTION_MAX_DEPTH}"));
    }

    Ok(match value {
        Value::Nil => JsonValue::Null,
        Value::Boolean(value) => JsonValue::Bool(*value),
        Value::Integer(value) => JsonValue::Number((*value).into()),
        Value::Number(value) => Number::from_f64(*value)
            .map(JsonValue::Number)
            .ok_or_else(|| "payload numbers must be finite".to_string())?,
        Value::String(value) => {
            JsonValue::String(value.to_str().map_err(|e| e.to_string())?.to_string())
        }
        Value::Table(table) => {
            let mut pairs = Vec::new();
            for pair in table.pairs::<Value, Value>() {
                pairs.push(pair.map_err(|e| e.to_string())?);
                if pairs.len() > MOD_ACTION_MAX_NODES {
                    return Err(format!("table exceeds {MOD_ACTION_MAX_NODES} entries"));
                }
            }
            let is_array = !pairs.is_empty()
                && pairs
                    .iter()
                    .all(|(key, _)| matches!(key, Value::Integer(i) if *i > 0))
                && pairs.len()
                    == pairs
                        .iter()
                        .filter_map(|(key, _)| match key {
                            Value::Integer(i) => Some(*i as usize),
                            _ => None,
                        })
                        .max()
                        .unwrap_or(0);
            if is_array {
                let mut values = vec![JsonValue::Null; pairs.len()];
                for (key, value) in pairs {
                    let Value::Integer(index) = key else {
                        unreachable!()
                    };
                    values[index as usize - 1] = lua_value_to_json(&value, depth + 1, nodes)?;
                }
                JsonValue::Array(values)
            } else {
                let mut values = Map::new();
                for (key, value) in pairs {
                    let Value::String(key) = key else {
                        return Err("object payload tables require string keys".to_string());
                    };
                    let key = key.to_str().map_err(|e| e.to_string())?.to_string();
                    values.insert(key, lua_value_to_json(&value, depth + 1, nodes)?);
                }
                JsonValue::Object(values)
            }
        }
        _ => {
            return Err(
                "payload supports only nil, booleans, numbers, strings, arrays, and objects".into(),
            );
        }
    })
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
        // TEMPORARY test API. args: [entity_id, path, x, y, z, scale?]
        "spawn_model" => {
            let entity_id = args.first().and_then(LuaScalar::as_i32).unwrap_or(0) as u64;
            let path = args
                .get(1)
                .and_then(LuaScalar::as_str)
                .ok_or("spawn_model: missing path")?
                .to_string();
            if path.len() > exsolis_common::MODEL_PATH_MAX_LEN {
                return Err(format!(
                    "spawn_model: path too long ({} > {})",
                    path.len(),
                    exsolis_common::MODEL_PATH_MAX_LEN
                ));
            }
            if fastpack::normalize_path(&path).is_err() {
                return Err(format!("spawn_model: invalid path {path:?}"));
            }
            let x = args.get(2).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            let y = args.get(3).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            let z = args.get(4).and_then(LuaScalar::as_f32).unwrap_or(0.0);
            let scale = args.get(5).and_then(LuaScalar::as_f32).unwrap_or(1.0);
            if !scale.is_finite() || scale <= 0.0 || scale > 1000.0 {
                return Err("spawn_model: scale must be in (0, 1000]".to_string());
            }
            Ok(ServerMessage::SpawnModel {
                entity_id,
                path,
                x,
                y,
                z,
                scale,
            })
        }
        // TEMPORARY test API. args: [entity_id]
        "despawn_model" => {
            let entity_id = args.first().and_then(LuaScalar::as_i32).unwrap_or(0) as u64;
            Ok(ServerMessage::DespawnModel { entity_id })
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

fn reload_mods_system(runtime: NonSend<ScriptRuntime>) {
    runtime.check_and_reload_mods();
}
