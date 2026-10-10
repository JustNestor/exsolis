//! Persistent, generic client asset cache and asset handlers.
//!
//! Control traffic uses Renet. Pak bodies are fetched with HTTP(S) on worker
//! threads, verified with fastpak checksums, merged into a persistent cache,
//! and extracted for existing path-based loaders.

use std::{
    any::Any,
    collections::HashMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    thread,
};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use exsolis_common::{AssetHash, ClientMessage, ServerMessage};
use fastpack::{Conflict, MergeOptions, OpenOptions, Pak, PakBuilder};

use crate::net::{self, IncomingPacket, OutgoingPacket};

const ASSET_SYNC_UI_ID: &str = "clientui:asset_sync";
const ASSET_SYNC_UI_XML: &str = r##"<div anchor="center" modal="true" layout="column" align="center" justify="center" width="100%" height="100%" bg="#101418ee"><label font-size="28">Synchronizing assets...</label></div>"##;

pub type LoadedAsset = Arc<dyn Any + Send + Sync>;
type Loaded = LoadedAsset;
pub type AssetLoader = fn(&[u8]) -> Result<LoadedAsset, String>;

#[derive(Resource, Default)]
pub struct AssetHandlerRegistry {
    loaders: HashMap<String, AssetLoader>,
    resident: HashMap<String, Loaded>,
    pending_load: std::collections::HashSet<String>,
}

impl AssetHandlerRegistry {
    /// Register a decoder/loader for an extension such as `png`, `wgsl`, or `ogg`.
    /// Its returned value is retained until application shutdown, providing the
    /// initial resident-asset policy while GPU unload behavior remains backend-specific.
    pub fn register(&mut self, extension: impl Into<String>, loader: AssetLoader) {
        self.loaders.insert(
            extension
                .into()
                .trim_start_matches('.')
                .to_ascii_lowercase(),
            loader,
        );
    }

    pub fn contains(&self, path: &str) -> bool {
        self.resident.contains_key(path)
    }

    pub fn get<T: Any + Send + Sync>(&self, path: &str) -> Option<&T> {
        self.resident.get(path)?.as_ref().downcast_ref::<T>()
    }

    pub fn iter_loaded<T: Any + Send + Sync>(&self) -> impl Iterator<Item = (&str, &T)> {
        self.resident.iter().filter_map(|(path, asset)| {
            asset
                .as_ref()
                .downcast_ref::<T>()
                .map(|value| (path.as_str(), value))
        })
    }
}

#[derive(Resource)]
struct WorkerQueue {
    jobs: mpsc::Sender<AssetJob>,
    receiver: Mutex<mpsc::Receiver<WorkerEvent>>,
}

enum AssetJob {
    Manifest(Vec<AssetHash>),
    Download {
        url: String,
        assets: Vec<AssetHash>,
        update: bool,
    },
}

enum WorkerEvent {
    NeedAssets(Vec<AssetHash>),
    SyncReady(Vec<AssetHash>),
    Updated(Vec<AssetHash>),
    Failed(String),
}

#[derive(Resource, Clone)]
pub struct AssetCachePaths {
    pub pak: PathBuf,
    pub extracted: PathBuf,
}

impl Default for AssetCachePaths {
    fn default() -> Self {
        let root = std::env::var_os("EXSOLIS_ASSET_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_asset_cache_dir);
        Self {
            pak: root.join("assets.pak"),
            extracted: root.join("assets"),
        }
    }
}

fn default_asset_cache_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .map(|home| home.join(".exso").join("downloaded"))
        .unwrap_or_else(|| PathBuf::from("downloaded"))
}

/// Directory used by texture/UI loaders for assets downloaded from the server.
pub fn asset_directory() -> PathBuf {
    AssetCachePaths::default().extracted
}

pub struct AssetClientPlugin;

impl Plugin for AssetClientPlugin {
    fn build(&self, app: &mut App) {
        let (event_sender, receiver) = mpsc::channel();
        let (job_sender, job_receiver) = mpsc::channel();
        let paths = AssetCachePaths::default();
        thread::Builder::new()
            .name("exsolis-asset-worker".into())
            .spawn(move || asset_worker(job_receiver, event_sender, paths))
            .expect("failed to start client asset worker");
        let mut registry = AssetHandlerRegistry::default();
        registry.register("png", decode_png);
        registry.register("wgsl", decode_wgsl);
        registry.register("fxm", crate::render::effects::decode_manifest);
        // Register GLTF loaders
        crate::gltf::register_gltf_loaders(&mut registry);
        app.insert_resource(WorkerQueue {
            jobs: job_sender,
            receiver: Mutex::new(receiver),
        })
        .init_resource::<AssetCachePaths>()
        .insert_resource(registry)
        .add_message::<AssetCacheChanged>()
        .add_systems(
            Update,
            (handle_server_asset_messages, poll_workers)
                .chain()
                .in_set(net::GameplaySet),
        );
    }
}

/// Validated WGSL source retained as a CPU-side asset. Pipeline entry points
/// and SPIR-V generation are selected later from the effect manifest.
#[derive(Debug)]
pub struct WgslAsset {
    pub source: String,
    pub entry_points: Vec<WgslEntryPoint>,
}

#[derive(Debug)]
pub struct WgslEntryPoint {
    pub name: String,
    pub stage: naga::ShaderStage,
    /// SPIR-V words generated for this exact stage/entry-point pair.
    pub spirv: Vec<u32>,
}

fn decode_wgsl(bytes: &[u8]) -> Result<LoadedAsset, String> {
    let source = expand_engine_imports(
        std::str::from_utf8(bytes).map_err(|e| format!("WGSL source is not UTF-8: {e}"))?,
    )?;
    let module = naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("WGSL validation failed: {e}"))?;

    let mut spv_options = naga::back::spv::Options::default();
    // Effect WGSL declares the Vulkano descriptor-set ABI directly with
    // @group/@binding, so preserve those values when Naga resolves resources.
    spv_options.fake_missing_bindings = true;
    // Effects index the engine's bindless set (group 0) with unsized
    // `binding_array`s; see `declare_runtime_descriptor_arrays`.
    let uses_unsized_binding_arrays = module.global_variables.iter().any(|(_, global)| {
        matches!(
            module.types[global.ty].inner,
            naga::TypeInner::BindingArray {
                size: naga::ArraySize::Dynamic,
                ..
            }
        )
    });
    let mut entry_points = Vec::with_capacity(module.entry_points.len());
    for entry in &module.entry_points {
        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: entry.stage,
            entry_point: entry.name.clone(),
        };
        let mut spirv =
            naga::back::spv::write_vec(&module, &info, &spv_options, Some(&pipeline_options))
                .map_err(|e| format!("WGSL SPIR-V generation failed for '{}': {e}", entry.name))?;
        if uses_unsized_binding_arrays {
            declare_runtime_descriptor_arrays(&mut spirv);
        }
        entry_points.push(WgslEntryPoint {
            name: entry.name.clone(),
            stage: entry.stage,
            spirv,
        });
    }

    Ok(Arc::new(WgslAsset {
        source,
        entry_points,
    }))
}

// ---------------------------------------------------------------------------
// Engine WGSL imports
//
// A line consisting of `//!import <name>` is replaced by the engine-owned
// declarations below, so mods don't copy ABI structs that change with the
// engine (the frame layout grows over time). Each import expands once, and
// imports pull in what they depend on. The directive is a plain comment to
// every other tool, and `enable ...;` lines must still come first in the file.
//
//   engine/frame     FrameData + `@group(1) @binding(1) var<storage, read> frame`
//   engine/bindless  `samplers` / `textures` binding arrays (needs
//                    `enable wgpu_binding_array;` at the top of the file)
//   engine/world     block data around the camera (needs engine/frame)
// ---------------------------------------------------------------------------

const ENGINE_FRAME_WGSL: &str = r#"
struct FrameData {
    frame_index: u32,
    time_seconds: f32,
    delta_time: f32,
    _padding: u32,
    viewport: vec2<f32>,
    _padding2: vec2<u32>,
    camera_position: vec3<f32>,
    _padding3: f32,
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    model_view_projection: mat4x4<f32>,
    sampler_nearest: u32,
    sampler_linear: u32,
    texture_count: u32,
    _padding4: u32,
    textures: array<u32, 8>,
    // Seconds since this instance spawned.
    instance_age: f32,
    // Seconds since a `stop` action hit this instance, or -1.0 while running.
    stop_age: f32,
    // Slot of this instance within the effect (stable for its lifetime).
    instance_slot: u32,
    _padding5: u32,
    // xyz: window size in chunks, w: chunk edge in blocks. Zero when absent.
    world_dims: vec4<u32>,
    // xyz: chunk coordinate the window is centred on, w: 1 when usable.
    world_anchor: vec4<i32>,
}
@group(1) @binding(1) var<storage, read> frame: FrameData;
"#;

const ENGINE_BINDLESS_WGSL: &str = r#"
@group(0) @binding(0) var samplers: binding_array<sampler>;
@group(0) @binding(1) var textures: binding_array<texture_2d<f32>>;
"#;

const ENGINE_WORLD_WGSL: &str = r#"
// Read-only block data in a window of chunks around the local camera. Voxels
// are packed as u16: bits 0..8 block id, bits 8..11 flags. A position outside
// the window or in a chunk that hasn't arrived yet reads as WORLD_UNLOADED.
@group(1) @binding(24) var<storage, read> world_cells: array<u32>;

const WORLD_UNLOADED: u32 = 0xFFFFu;
const WORLD_FLAG_SOLID: u32 = 0x100u;
const WORLD_FLAG_TRANSPARENT: u32 = 0x200u;
const WORLD_FLAG_AIR: u32 = 0x400u;

fn world_enabled() -> bool {
    return frame.world_anchor.w != 0;
}

fn world_floor_div(a: i32, b: i32) -> i32 {
    var q = a / b;
    if ((a % b) != 0 && ((a < 0) != (b < 0))) {
        q = q - 1;
    }
    return q;
}

// Packed voxel at an integer block position, or WORLD_UNLOADED.
fn world_block(p: vec3<i32>) -> u32 {
    if (frame.world_anchor.w == 0) {
        return WORLD_UNLOADED;
    }
    let cs = i32(frame.world_dims.w);
    let dims = vec3<i32>(frame.world_dims.xyz);
    let c = vec3<i32>(
        world_floor_div(p.x, cs),
        world_floor_div(p.y, cs),
        world_floor_div(p.z, cs),
    );
    let l = p - c * cs;
    let m = ((c % dims) + dims) % dims;
    let cell = u32((m.z * dims.y + m.y) * dims.x + m.x);
    let tag = cell * 4u;
    if (world_cells[tag + 3u] == 0u
        || bitcast<i32>(world_cells[tag]) != c.x
        || bitcast<i32>(world_cells[tag + 1u]) != c.y
        || bitcast<i32>(world_cells[tag + 2u]) != c.z) {
        return WORLD_UNLOADED;
    }
    let cells = u32(dims.x * dims.y * dims.z);
    let cs_u = u32(cs);
    let voxel = u32((l.y * cs + l.z) * cs + l.x);
    let base = cells * 4u + cell * (cs_u * cs_u * cs_u / 2u);
    let word = world_cells[base + voxel / 2u];
    return (word >> ((voxel & 1u) * 16u)) & 0xFFFFu;
}

fn world_block_at(p: vec3<f32>) -> u32 {
    return world_block(vec3<i32>(floor(p)));
}

fn world_is_loaded(v: u32) -> bool { return v != WORLD_UNLOADED; }
fn world_is_solid(v: u32) -> bool { return v != WORLD_UNLOADED && (v & WORLD_FLAG_SOLID) != 0u; }
fn world_is_air(v: u32) -> bool { return v != WORLD_UNLOADED && (v & WORLD_FLAG_AIR) != 0u; }
fn world_block_id(v: u32) -> u32 { return v & 0xFFu; }

// Collision shorthand: true only for loaded, solid blocks.
fn world_solid_at(p: vec3<f32>) -> bool {
    return world_is_solid(world_block_at(p));
}

// Top surface of the first solid block at or below `p` within `max_down`
// blocks, or `none` if there isn't one in loaded data.
fn world_ground_below(p: vec3<f32>, max_down: i32, none: f32) -> f32 {
    let start = vec3<i32>(floor(p));
    for (var i = 0; i <= max_down; i = i + 1) {
        let v = world_block(vec3<i32>(start.x, start.y - i, start.z));
        if (world_is_solid(v)) {
            return f32(start.y - i + 1);
        }
    }
    return none;
}
"#;

fn engine_import(name: &str) -> Option<(&'static [&'static str], &'static str)> {
    match name {
        "engine/frame" => Some((&[], ENGINE_FRAME_WGSL)),
        "engine/bindless" => Some((&[], ENGINE_BINDLESS_WGSL)),
        "engine/world" => Some((&["engine/frame"], ENGINE_WORLD_WGSL)),
        _ => None,
    }
}

fn expand_engine_import(
    name: &str,
    seen: &mut std::collections::HashSet<String>,
    out: &mut String,
) -> Result<(), String> {
    if !seen.insert(name.to_owned()) {
        return Ok(());
    }
    let Some((dependencies, text)) = engine_import(name) else {
        return Err(format!("unknown engine import '{name}'"));
    };
    for dependency in dependencies {
        expand_engine_import(dependency, seen, out)?;
    }
    out.push_str(text);
    out.push('\n');
    Ok(())
}

fn expand_engine_imports(source: &str) -> Result<String, String> {
    if !source.contains("//!import") {
        return Ok(source.to_owned());
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = String::with_capacity(source.len() + 4096);
    for line in source.lines() {
        match line.trim().strip_prefix("//!import") {
            Some(name) => expand_engine_import(name.trim(), &mut seen, &mut out)?,
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    Ok(out)
}

/// Naga (30.x) lowers an unsized `binding_array<T>` to a UniformConstant
/// `OpTypeRuntimeArray` of opaque handles but never declares the capability
/// Vulkan requires for that (`RuntimeDescriptorArray`), nor the extension that
/// provides it below SPIR-V 1.5 (`SPV_EXT_descriptor_indexing`). `spirv-val
/// --target-env vulkan1.3` rejects the module otherwise
/// (VUID-StandaloneSpirv-OpTypeRuntimeArray-04680). The bindless context
/// already requires the matching device feature, so declaring both is safe.
fn declare_runtime_descriptor_arrays(spirv: &mut Vec<u32>) {
    const HEADER_WORDS: usize = 5;
    const OP_EXTENSION: u32 = 10;
    const OP_CAPABILITY: u32 = 17;
    const CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;

    let mut at = HEADER_WORDS;
    while at < spirv.len() {
        let word = spirv[at];
        let (op, len) = (word & 0xffff, (word >> 16) as usize);
        if op != OP_CAPABILITY || len == 0 || at + len > spirv.len() {
            break;
        }
        if spirv[at + 1] == CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY {
            return;
        }
        at += len;
    }

    let mut name = b"SPV_EXT_descriptor_indexing\0".to_vec();
    while name.len() % 4 != 0 {
        name.push(0);
    }
    let name_words = name
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    let mut inserted = vec![
        (2 << 16) | OP_CAPABILITY,
        CAPABILITY_RUNTIME_DESCRIPTOR_ARRAY,
        (((1 + name.len() / 4) as u32) << 16) | OP_EXTENSION,
    ];
    inserted.extend(name_words);
    spirv.splice(at..at, inserted);
}

#[derive(Message, Debug, Clone)]
pub struct AssetCacheChanged {
    pub assets: Vec<AssetHash>,
}

fn handle_server_asset_messages(
    mut incoming: MessageReader<IncomingPacket>,
    queue: Res<WorkerQueue>,
    paths: Res<AssetCachePaths>,
    mut handlers: ResMut<AssetHandlerRegistry>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut commands: Commands,
) {
    for packet in incoming.read() {
        match &packet.0 {
            ServerMessage::AssetManifest { assets, .. } => {
                commands.write_message(IncomingPacket(ServerMessage::UiShow {
                    id: ASSET_SYNC_UI_ID.to_owned(),
                    xml: ASSET_SYNC_UI_XML.to_owned(),
                }));
                let _ = queue.jobs.send(AssetJob::Manifest(assets.clone()));
            }
            ServerMessage::AssetDownload { url, assets }
            | ServerMessage::AssetUpdate { url, assets } => {
                let is_update = matches!(&packet.0, ServerMessage::AssetUpdate { .. });
                let _ = queue.jobs.send(AssetJob::Download {
                    url: url.clone(),
                    assets: assets.clone(),
                    update: is_update,
                });
            }
            ServerMessage::AssetLoad { paths: asset_paths } => {
                let mut loaded_paths = Vec::new();
                for path in asset_paths {
                    let Ok(path) = fastpack::normalize_path(path) else {
                        warn!("server requested invalid asset path");
                        continue;
                    };
                    let full_path = paths.extracted.join(&path);
                    match fs::read(&full_path) {
                        Ok(bytes) => {
                            let ext = Path::new(&path)
                                .extension()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_ascii_lowercase();
                            let loaded = handlers
                                .loaders
                                .get(&ext)
                                .map(|loader| loader(&bytes))
                                .unwrap_or_else(|| Ok(Arc::new(bytes) as Loaded));
                            match loaded {
                                Ok(value) => {
                                    handlers.resident.insert(path.clone(), value);
                                    loaded_paths.push(path.clone());
                                }
                                Err(e) => warn!("asset handler failed: {e}"),
                            }
                        }
                        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                            handlers.pending_load.insert(path.clone());
                        }
                        Err(e) => warn!(
                            "server requested unavailable asset {}: {e}",
                            full_path.display()
                        ),
                    }
                }
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetLoadComplete {
                    paths: loaded_paths,
                }));
            }
            ServerMessage::AssetRemove { paths } => {
                // GPU allocations are deliberately retained until a backend
                // safe-unload policy exists. The next session manifest will
                // simply stop referencing these paths.
                warn!(
                    "server removed {} assets; retaining resident copies for this session",
                    paths.len()
                );
            }
            ServerMessage::AssetDownloadFailed { reason } => {
                warn!("server asset transfer failed: {reason}");
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetSyncComplete));
                commands.write_message(IncomingPacket(ServerMessage::UiHide {
                    id: ASSET_SYNC_UI_ID.to_owned(),
                }));
            }
            _ => {}
        }
    }
}

fn asset_worker(
    jobs: mpsc::Receiver<AssetJob>,
    events: mpsc::Sender<WorkerEvent>,
    paths: AssetCachePaths,
) {
    // Retain the authoritative full manifest across the two-step request:
    // the server's pak response contains only assets that were transferred,
    // while effect shaders already present in the local cache still need to
    // be decoded and made resident before gameplay starts.
    let mut active_manifest = Vec::new();
    while let Ok(job) = jobs.recv() {
        let event = match job {
            AssetJob::Manifest(manifest) => {
                active_manifest = manifest.clone();
                match prepare_cache(&paths, &manifest) {
                    Ok(missing) if missing.is_empty() => WorkerEvent::SyncReady(manifest),
                    Ok(missing) => WorkerEvent::NeedAssets(missing),
                    Err(e) => WorkerEvent::Failed(e),
                }
            }
            AssetJob::Download {
                url,
                assets,
                update,
            } => match fetch_merge_extract(&url, &assets, &paths) {
                Ok(()) if update => WorkerEvent::Updated(assets),
                Ok(()) => {
                    let assets = if active_manifest.is_empty() {
                        assets
                    } else {
                        active_manifest.clone()
                    };
                    WorkerEvent::SyncReady(assets)
                }
                Err(e) => WorkerEvent::Failed(e),
            },
        };
        if events.send(event).is_err() {
            break;
        }
    }
}

fn poll_workers(
    queue: Res<WorkerQueue>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut changed: MessageWriter<AssetCacheChanged>,
    mut commands: Commands,
    paths: Res<AssetCachePaths>,
    mut handlers: ResMut<AssetHandlerRegistry>,
) {
    let Ok(receiver) = queue.receiver.lock() else {
        return;
    };
    while let Ok(event) = receiver.try_recv() {
        match event {
            WorkerEvent::NeedAssets(assets) => {
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetRequest {
                    assets,
                }));
            }
            WorkerEvent::SyncReady(assets) => {
                let mut loaded_paths = load_pending_assets(&assets, &paths, &mut handlers);
                loaded_paths.extend(load_effect_assets(&assets, &paths, &mut handlers));
                if !assets.is_empty() {
                    changed.write(AssetCacheChanged { assets });
                }
                if !loaded_paths.is_empty() {
                    outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetLoadComplete {
                        paths: std::mem::take(&mut loaded_paths),
                    }));
                }
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetSyncComplete));
                commands.write_message(IncomingPacket(ServerMessage::UiHide {
                    id: ASSET_SYNC_UI_ID.to_owned(),
                }));
            }
            WorkerEvent::Updated(assets) => {
                for asset in &assets {
                    let full_path = paths.extracted.join(&asset.path);
                    match fs::read(&full_path) {
                        Ok(bytes) => {
                            let ext = Path::new(&asset.path)
                                .extension()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_ascii_lowercase();
                            let loaded = handlers
                                .loaders
                                .get(&ext)
                                .map(|loader| loader(&bytes))
                                .unwrap_or_else(|| Ok(Arc::new(bytes) as Loaded));
                            match loaded {
                                Ok(value) => {
                                    handlers.resident.insert(asset.path.clone(), value);
                                }
                                Err(e) => {
                                    warn!("hot-reload handler failed for {}: {e}", asset.path)
                                }
                            }
                        }
                        Err(e) => warn!("hot-reloaded asset {} is unavailable: {e}", asset.path),
                    }
                }
                changed.write(AssetCacheChanged {
                    assets: assets.clone(),
                });
                for asset in &assets {
                    handlers.pending_load.remove(&asset.path);
                }
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetLoadComplete {
                    paths: assets.into_iter().map(|a| a.path).collect(),
                }));
            }
            WorkerEvent::Failed(e) => {
                warn!("asset synchronization failed: {e}; continuing with available assets");
                outgoing.write(OutgoingPacket::reliable(ClientMessage::AssetSyncComplete));
                commands.write_message(IncomingPacket(ServerMessage::UiHide {
                    id: ASSET_SYNC_UI_ID.to_owned(),
                }));
            }
        }
    }
}

fn load_effect_assets(
    assets: &[AssetHash],
    paths: &AssetCachePaths,
    handlers: &mut AssetHandlerRegistry,
) -> Vec<String> {
    let mut loaded = Vec::new();
    for asset in assets {
        let ext = Path::new(&asset.path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(ext.as_str(), "fxm" | "wgsl" | "glb")
            || handlers.resident.contains_key(&asset.path)
        {
            continue;
        }
        let full_path = paths.extracted.join(&asset.path);
        let result = fs::read(&full_path)
            .map_err(|e| e.to_string())
            .and_then(|bytes| {
                let loader = handlers
                    .loaders
                    .get(&ext)
                    .ok_or_else(|| format!("no loader for .{ext}"))?;
                loader(&bytes)
            });
        match result {
            Ok(value) => {
                handlers.resident.insert(asset.path.clone(), value);
                loaded.push(asset.path.clone());
            }
            Err(error) => warn!("effect asset {} failed to load: {error}", asset.path),
        }
    }
    loaded
}

fn load_pending_assets(
    assets: &[AssetHash],
    paths: &AssetCachePaths,
    handlers: &mut AssetHandlerRegistry,
) -> Vec<String> {
    let available: std::collections::HashSet<_> =
        assets.iter().map(|asset| asset.path.as_str()).collect();
    let pending: Vec<_> = handlers
        .pending_load
        .iter()
        .filter(|path| available.contains(path.as_str()))
        .cloned()
        .collect();
    let mut loaded = Vec::new();
    for path in pending {
        let full_path = paths.extracted.join(&path);
        let Ok(bytes) = fs::read(&full_path) else {
            continue;
        };
        let ext = Path::new(&path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let result = handlers
            .loaders
            .get(&ext)
            .map(|loader| loader(&bytes))
            .unwrap_or_else(|| Ok(Arc::new(bytes) as Loaded));
        match result {
            Ok(value) => {
                handlers.resident.insert(path.clone(), value);
                handlers.pending_load.remove(&path);
                loaded.push(path);
            }
            Err(e) => warn!("asset handler failed for {path}: {e}"),
        }
    }
    loaded
}

fn prepare_cache(
    paths: &AssetCachePaths,
    manifest: &[AssetHash],
) -> Result<Vec<AssetHash>, String> {
    if let Some(parent) = paths.pak.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let current = match read_cache_manifest(&paths.pak) {
        Ok(current) => current,
        Err(_) => {
            // Keep client assets independent from any local/server source
            // directory. The server manifest will request every required
            // asset that is absent from this empty cache.
            let builder = PakBuilder::new();
            let initial = temp_path(&paths.pak, "initial");
            builder.build(&initial).map_err(|e| e.to_string())?;
            replace_file(&initial, &paths.pak)?;
            read_cache_manifest(&paths.pak)?
        }
    };
    let missing: Vec<_> = manifest
        .iter()
        .filter(|asset| current.get(&asset.path) != Some(&asset.checksum))
        .cloned()
        .collect();
    reconcile_extracted(&paths.pak, &paths.extracted, manifest)?;
    Ok(missing)
}

/// Keep the extracted view aligned with the authoritative manifest. A valid
/// pak can outlive a partially deleted extraction, so restore matching files
/// from it before asking the server for anything. Historical pak entries are
/// deliberately ignored and stale extracted files are removed.
fn reconcile_extracted(
    pak_path: &Path,
    output: &Path,
    manifest: &[AssetHash],
) -> Result<(), String> {
    let pak = Pak::open_with(
        pak_path,
        OpenOptions {
            verify_reads: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    pak.verify(1).map_err(|e| e.to_string())?;
    let required: HashMap<_, _> = manifest
        .iter()
        .map(|asset| (asset.path.as_str(), asset.checksum))
        .collect();
    fs::create_dir_all(output).map_err(|e| e.to_string())?;

    for entry in pak.iter() {
        if required.get(entry.path).copied() != Some(entry.checksum) {
            continue;
        }
        let relative = fastpack::normalize_path(entry.path).map_err(|e| e.to_string())?;
        let destination = output.join(relative);
        if !destination.is_file() {
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let data = pak.get(entry.path).map_err(|e| e.to_string())?;
            fs::write(destination, data.as_ref()).map_err(|e| e.to_string())?;
        }
    }

    prune_extracted(output, output, &required)
}

fn prune_extracted(
    root: &Path,
    directory: &Path,
    required: &HashMap<&str, u64>,
) -> Result<(), String> {
    for item in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let item = item.map_err(|e| e.to_string())?;
        let path = item.path();
        let kind = item.file_type().map_err(|e| e.to_string())?;
        if kind.is_dir() {
            prune_extracted(root, &path, required)?;
            if fs::read_dir(&path)
                .map_err(|e| e.to_string())?
                .next()
                .is_none()
            {
                fs::remove_dir(&path).map_err(|e| e.to_string())?;
            }
        } else if kind.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            if !required.contains_key(relative.as_str()) {
                fs::remove_file(path).map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

fn read_cache_manifest(path: &Path) -> Result<HashMap<String, u64>, String> {
    if !path.exists() {
        return Err("asset cache does not exist".into());
    }
    let pak = Pak::open_with(
        path,
        OpenOptions {
            verify_reads: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    pak.verify(1).map_err(|e| e.to_string())?;
    Ok(pak
        .iter()
        .map(|entry| (entry.path.to_owned(), entry.checksum))
        .collect())
}

fn fetch_merge_extract(
    url: &str,
    expected: &[AssetHash],
    paths: &AssetCachePaths,
) -> Result<(), String> {
    let mut last_error = String::new();
    for attempt in 0..5 {
        match fetch_merge_extract_once(url, expected, paths) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last_error = e;
                thread::sleep(std::time::Duration::from_millis(250 * (attempt + 1)));
            }
        }
    }
    Err(format!("download failed after retries: {last_error}"))
}

fn fetch_merge_extract_once(
    url: &str,
    expected: &[AssetHash],
    paths: &AssetCachePaths,
) -> Result<(), String> {
    let response = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?
        .get(url)
        .send()
        .map_err(|e| e.to_string())?
        .error_for_status()
        .map_err(|e| e.to_string())?;
    const MAX_PAK_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    if response
        .content_length()
        .is_some_and(|len| len > MAX_PAK_BYTES)
    {
        return Err("downloaded pak exceeds 2 GiB limit".into());
    }
    let mut bytes = Vec::new();
    response
        .take(MAX_PAK_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_PAK_BYTES {
        return Err("downloaded pak exceeds 2 GiB limit".into());
    }
    let delta_path = temp_path(&paths.pak, "download");
    if let Some(parent) = delta_path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(&delta_path, &bytes).map_err(|e| e.to_string())?;

    let result = (|| {
        let delta = Pak::open_with(
            &delta_path,
            OpenOptions {
                verify_reads: true,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;
        delta.verify(1).map_err(|e| e.to_string())?;
        for item in expected {
            let Some(id) = delta.find(&item.path) else {
                return Err(format!("delta pak is missing {}", item.path));
            };
            if delta.info(id).map(|i| i.checksum) != Some(item.checksum) {
                return Err(format!("checksum mismatch for {}", item.path));
            }
        }
        if let Some(parent) = paths.pak.parent() {
            fs::copy(&delta_path, parent.join("last_delta.pak")).map_err(|e| e.to_string())?;
        }
        // Write only this delta to the extracted overlay. This preserves the
        // previous pak's unchanged files without resurrecting deleted files.
        extract_pak(&delta_path, &paths.extracted)?;
        let merged_path = temp_path(&paths.pak, "merged");
        {
            let current = Pak::open_with(
                &paths.pak,
                OpenOptions {
                    verify_reads: true,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
            fastpack::merge(
                &[&current, &delta],
                &merged_path,
                &MergeOptions {
                    on_conflict: Conflict::LastWins,
                    ..Default::default()
                },
            )
            .map_err(|e| e.to_string())?;
        }
        drop(delta);
        replace_file(&merged_path, &paths.pak)?;
        Ok(())
    })();
    let _ = fs::remove_file(delta_path);
    result
}

fn replace_file(source: &Path, destination: &Path) -> Result<(), String> {
    if !destination.exists() {
        return fs::rename(source, destination).map_err(|e| e.to_string());
    }
    let backup = temp_path(destination, "backup");
    fs::rename(destination, &backup).map_err(|e| e.to_string())?;
    match fs::rename(source, destination) {
        Ok(()) => {
            let _ = fs::remove_file(backup);
            Ok(())
        }
        Err(e) => {
            let _ = fs::rename(&backup, destination);
            Err(e.to_string())
        }
    }
}

fn extract_pak(pak_path: &Path, output: &Path) -> Result<(), String> {
    let pak = Pak::open_with(
        pak_path,
        OpenOptions {
            verify_reads: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    pak.verify(1).map_err(|e| e.to_string())?;
    for entry in pak.iter() {
        let normalized = fastpack::normalize_path(entry.path).map_err(|e| e.to_string())?;
        let destination = output.join(normalized);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let data = pak.get(entry.path).map_err(|e| e.to_string())?;
        fs::write(destination, data.as_ref()).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn temp_path(base: &Path, suffix: &str) -> PathBuf {
    let parent = base.parent().unwrap_or_else(|| Path::new("."));
    parent.join(format!(
        ".asset-{}-{suffix}-{}.pak",
        std::process::id(),
        rand::random::<u64>()
    ))
}

// A useful default decoder for common image assets; specialized render/audio
// plugins can register more expensive GPU/device-specific handlers.
pub fn decode_png(bytes: &[u8]) -> Result<LoadedAsset, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| e.to_string())?
        .into_rgba8();
    Ok(Arc::new(image))
}
