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
use bevy_log::{debug, error, info, warn};
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
    /// `api.spawn_entity(kind, { x, y, z, model, scale })`. Ids are
    /// allocated by the server; the request carries no id because a mod
    /// cannot be trusted to pick a unique one.
    SpawnEntity {
        kind: String,
        x: f32,
        y: f32,
        z: f32,
        yaw: f32,
        model: Option<String>,
        scale: f32,
    },
    /// `api.despawn_entity(entity_id)`.
    DespawnEntity {
        entity_id: u64,
    },
    /// `api.set_entity_state(entity_id, { x, y, z, yaw })`.
    SetEntityState {
        entity_id: u64,
        x: f32,
        y: f32,
        z: f32,
        yaw: f32,
    },
    /// `api.get_block_entity(x, y, z)` -> the entity id, or nil.
    QueryBlockEntity {
        pos: BlockPos,
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

    // --- api.spawn_entity(kind, opts) : queue an entity spawn -------------
    // The server allocates the id, so a mod cannot pick a colliding one.
    // Returns nothing; read the id back with `api.get_block_entity` or
    // keep the id from `api.spawn_block_entity` below.
    {
        let outbox = outbox.clone();
        api.set(
            "spawn_entity",
            lua.create_function(move |_, (kind, opts): (String, mlua::Table)| {
                let x: f32 = opts.get("x").unwrap_or(0.0);
                let y: f32 = opts.get("y").unwrap_or(0.0);
                let z: f32 = opts.get("z").unwrap_or(0.0);
                let yaw: f32 = opts.get("yaw").unwrap_or(0.0);
                let scale: f32 = opts.get("scale").unwrap_or(1.0);
                let model: Option<String> = opts.get("model").unwrap_or(None);
                if let Some(ref m) = model
                    && m.len() > exsolis_common::MODEL_PATH_MAX_LEN
                {
                    return Err(mlua::Error::RuntimeError(format!(
                        "spawn_entity: model path too long ({})",
                        exsolis_common::MODEL_PATH_MAX_LEN
                    )));
                }
                if !scale.is_finite() || scale <= 0.0 || scale > 1000.0 {
                    return Err(mlua::Error::RuntimeError(
                        "spawn_entity: scale must be in (0, 1000]".to_string(),
                    ));
                }
                outbox.borrow_mut().push(ScriptRequest::SpawnEntity {
                    kind,
                    x,
                    y,
                    z,
                    yaw,
                    model,
                    scale,
                });
                Ok(())
            })?,
        )?;
    }

    // --- api.despawn_entity(entity_id) ---------------------------------
    {
        let outbox = outbox.clone();
        api.set(
            "despawn_entity",
            lua.create_function(move |_, entity_id: Value| {
                let id = pop_entity_id(&mut std::iter::once(entity_id))
                    .map_err(mlua::Error::RuntimeError)?;
                outbox
                    .borrow_mut()
                    .push(ScriptRequest::DespawnEntity { entity_id: id });
                Ok(())
            })?,
        )?;
    }

    // --- api.set_entity_state(entity_id, opts) -------------------------
    {
        let outbox = outbox.clone();
        api.set(
            "set_entity_state",
            lua.create_function(move |_, (id, opts): (Value, mlua::Table)| {
                let entity_id =
                    pop_entity_id(&mut std::iter::once(id)).map_err(mlua::Error::RuntimeError)?;
                let x: f32 = opts.get("x").unwrap_or(f32::NAN);
                let y: f32 = opts.get("y").unwrap_or(f32::NAN);
                let z: f32 = opts.get("z").unwrap_or(f32::NAN);
                let yaw: f32 = opts.get("yaw").unwrap_or(0.0);
                if !x.is_finite() || !y.is_finite() || !z.is_finite() || !yaw.is_finite() {
                    return Err(mlua::Error::RuntimeError(
                        "set_entity_state: x, y, z and yaw are required and must be finite"
                            .to_string(),
                    ));
                }
                outbox.borrow_mut().push(ScriptRequest::SetEntityState {
                    entity_id,
                    x,
                    y,
                    z,
                    yaw,
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

/// Parse a mod-supplied entity id.
///
/// Same `u64`-through-Lua caveat as [`pop_player_id`]: ids go to and from
/// Lua as decimal strings, because a Lua number cannot represent every
/// `u64`. Small integers are tolerated for convenience.
fn pop_entity_id(it: &mut impl Iterator<Item = Value>) -> Result<u64, String> {
    match it.next() {
        Some(Value::String(s)) => s
            .to_str()
            .map_err(|e| e.to_string())?
            .parse::<u64>()
            .map_err(|_| format!("'{s:?}' is not a valid entity_id")),
        Some(Value::Integer(i)) if i >= 0 => Ok(i as u64),
        Some(Value::Number(n)) if n >= 0.0 && n.fract() == 0.0 => Ok(n as u64),
        other => Err(format!("expected entity_id (string), got {other:?}")),
    }
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
    registry: Res<crate::entities::EntityRegistry>,
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

        // The entity id of the block entity at a position, or nil. Same
        // string-id caveat as every other id crossing into Lua.
        let registry = &registry;
        let get_block_entity = scope.create_function(|lua, (x, y, z): (i32, i32, i32)| {
            match registry.at_block(BlockPos::new(x, y, z)) {
                Some(id) => Ok(Value::String(lua.create_string(id.to_string())?)),
                None => Ok(Value::Nil),
            }
        })?;
        api.set("get_block_entity", get_block_entity)?;

        // Every live entity, as { id = "...", kind = "...", x, y, z }.
        let list_entities = scope.create_function(|lua, ()| {
            let t = lua.create_table()?;
            for (i, record) in registry.records().iter().enumerate() {
                let e = lua.create_table()?;
                e.set("id", record.id.to_string())?;
                e.set("kind", format!("{:?}", record.kind).to_lowercase())?;
                e.set("x", record.state.x)?;
                e.set("y", record.state.y)?;
                e.set("z", record.state.z)?;
                t.raw_set(i + 1, e)?;
            }
            Ok(t)
        })?;
        api.set("list_entities", list_entities)?;

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
    mut commands: Commands,
    mut world: ResMut<ChunkWorld>,
    registry: Res<crate::entities::EntityRegistry>,
    mut spawns: MessageWriter<crate::entities::SpawnEntityRequest>,
    mut despawns: MessageWriter<crate::entities::DespawnEntityRequest>,
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
            ScriptRequest::SpawnEntity {
                kind,
                x,
                y,
                z,
                yaw,
                model,
                scale,
            } => {
                // An unknown kind is a warning, never a panic: the string
                // came from a mod.
                let Ok(kind) = parse_entity_kind(&kind) else {
                    warn!("scripting: api.spawn_entity: unknown kind {kind:?}");
                    continue;
                };
                if let Some(block_pos) = block_pos_from_state(kind, x, y, z) {
                    if registry.at_block(block_pos).is_some() {
                        warn!(
                            "scripting: api.spawn_entity: a block entity already exists at {block_pos:?}"
                        );
                        continue;
                    }
                }
                let mut extra = crate::entities::SpawnExtra::None;
                let mut block_pos = None;
                if kind == exsolis_common::EntityKind::Block {
                    let block = block_pos_from_state(kind, x, y, z)
                        .map(|p| block_at(&world, p))
                        .unwrap_or(BlockType::Chest);
                    if !crate::entities::is_block_entity(block) {
                        warn!(
                            "scripting: api.spawn_entity: {:?} is not a block entity",
                            block
                        );
                        continue;
                    }
                    extra = crate::entities::SpawnExtra::BlockEntity(block);
                    block_pos = block_pos_from_state(kind, x, y, z);
                }
                spawns.write(crate::entities::SpawnEntityRequest {
                    kind,
                    state: exsolis_common::EntityState {
                        x,
                        y,
                        z,
                        yaw,
                        ..Default::default()
                    },
                    model,
                    scale,
                    block_pos,
                    extra,
                    player_token: None,
                });
            }
            ScriptRequest::DespawnEntity { entity_id } => {
                despawns.write(crate::entities::DespawnEntityRequest(entity_id));
            }
            ScriptRequest::SetEntityState {
                entity_id,
                x,
                y,
                z,
                yaw,
            } => {
                let Some(record) = registry.get(entity_id) else {
                    warn!("scripting: api.set_entity_state: unknown entity {entity_id}");
                    continue;
                };
                if !x.is_finite() || !y.is_finite() || !z.is_finite() || !yaw.is_finite() {
                    warn!("scripting: api.set_entity_state: non-finite state for {entity_id}");
                    continue;
                }
                commands
                    .entity(record.entity)
                    .insert(crate::entities::EntityTransform(
                        exsolis_common::EntityState {
                            x,
                            y,
                            z,
                            yaw,
                            ..record.state
                        },
                    ));
                // Push it out immediately rather than waiting for the next
                // sync: a mod that moves an entity expects clients to see it
                // move now.
                outgoing.write(OutgoingPacket::broadcast_unreliable(
                    ServerMessage::EntityStateUpdate {
                        entity_id,
                        state: exsolis_common::EntityState {
                            x,
                            y,
                            z,
                            yaw,
                            ..record.state
                        },
                    },
                ));
            }
            ScriptRequest::QueryBlockEntity { pos } => {
                // Mods get the id synchronously from `get_block_entity`, so
                // this variant only exists to make the lookup requestable
                // from a context without world access; the answer is logged
                // rather than pushed back into Lua.
                debug!(
                    "scripting: block entity at {pos:?} is {:?}",
                    registry.at_block(pos)
                );
            }
        }
    }
}

/// Parse a mod-supplied entity kind name.
fn parse_entity_kind(name: &str) -> Result<exsolis_common::EntityKind, ()> {
    use exsolis_common::EntityKind;
    match name {
        "player" => Ok(EntityKind::Player),
        "block" => Ok(EntityKind::Block),
        "item" => Ok(EntityKind::Item),
        "prop" => Ok(EntityKind::Prop),
        _ => Err(()),
    }
}

/// The block position an entity state refers to, for block-kind entities.
fn block_pos_from_state(
    kind: exsolis_common::EntityKind,
    x: f32,
    y: f32,
    z: f32,
) -> Option<BlockPos> {
    (kind == exsolis_common::EntityKind::Block)
        .then(|| BlockPos::new(x.floor() as i32, y.floor() as i32, z.floor() as i32))
}

/// The block currently at `pos`, or `Air`.
fn block_at(world: &ChunkWorld, pos: BlockPos) -> BlockType {
    let (coord, idx) = world_to_chunk_index(pos);
    world
        .ready
        .get(&coord)
        .and_then(|c| c.unpack())
        .map(|blocks| blocks[idx])
        .unwrap_or(BlockType::Air)
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

fn reload_mods_system(runtime: NonSend<ScriptRuntime>) {
    runtime.check_and_reload_mods();
}

#[cfg(test)]
mod entity_api_tests {
    use super::*;

    /// Ids cross into Lua as decimal strings because a Lua number cannot
    /// represent every `u64`. Getting this wrong silently truncates ids
    /// above 2^53, so the string path is the one that matters.
    #[test]
    fn entity_ids_parse_from_strings_and_small_numbers() {
        let lua = mlua::Lua::new();
        let big = lua.create_string("18446744073709551615").unwrap();
        assert_eq!(
            pop_entity_id(&mut std::iter::once(Value::String(big))).unwrap(),
            u64::MAX,
            "ids above 2^53 must survive the round trip, which is why they go as strings"
        );
        assert_eq!(
            pop_entity_id(&mut std::iter::once(Value::Integer(42))).unwrap(),
            42
        );
    }

    #[test]
    fn entity_ids_reject_junk() {
        for bad in [
            pop_entity_id(&mut std::iter::once(Value::Nil)),
            pop_entity_id(&mut std::iter::once(Value::Integer(-1))),
            pop_entity_id(&mut std::iter::once(Value::Number(-2.0))),
            pop_entity_id(&mut std::iter::once(Value::Number(1.5))),
        ] {
            assert!(bad.is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn entity_kinds_are_a_closed_set() {
        for name in ["player", "block", "item", "prop"] {
            assert!(parse_entity_kind(name).is_ok(), "{name} should parse");
        }
        for name in ["", "PLAYER", "npc", "players"] {
            assert!(
                parse_entity_kind(name).is_err(),
                "{name:?} should not parse"
            );
        }
    }
}
