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
    let source = std::str::from_utf8(bytes)
        .map_err(|e| format!("WGSL source is not UTF-8: {e}"))?
        .to_owned();
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
    let mut entry_points = Vec::with_capacity(module.entry_points.len());
    for entry in &module.entry_points {
        let pipeline_options = naga::back::spv::PipelineOptions {
            shader_stage: entry.stage,
            entry_point: entry.name.clone(),
        };
        let spirv =
            naga::back::spv::write_vec(&module, &info, &spv_options, Some(&pipeline_options))
                .map_err(|e| format!("WGSL SPIR-V generation failed for '{}': {e}", entry.name))?;
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
        if !matches!(ext.as_str(), "fxm" | "wgsl") || handlers.resident.contains_key(&asset.path) {
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
