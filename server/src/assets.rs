//! Asset catalog and on-demand HTTP(S) pak delivery.
//!
//! Renet carries only control messages. Asset bytes are sent by this Axum
//! service as fastpak archives, and the listener is stopped after 30 idle
//! seconds with no connected Renet clients.

use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    Router,
    body::Body,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    routing::get,
};
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::{error, info, warn};
use bevy_renet::RenetServer;
use exsolis_common::{AssetHash, ClientMessage, PlayerId, ServerMessage};
use fastpack::{Intent, Pak, PakBuilder};
use tokio::sync::oneshot;

use crate::{
    PlayerMap,
    net::{self, IncomingPacket, OutgoingPacket},
};

const IDLE_SHUTDOWN: Duration = Duration::from_secs(30);

#[derive(Resource)]
pub struct AssetCatalog {
    assets: HashMap<String, AssetHash>,
    files: HashMap<String, PathBuf>,
    disk_state: HashMap<String, (u64, Option<SystemTime>)>,
    last_scan: Instant,
}

impl AssetCatalog {
    fn new() -> Self {
        let assets_root = asset_root();
        let mods_root = mods_directory();
        let (assets, files, disk_state) =
            scan_catalog(&assets_root, &mods_root).unwrap_or_else(|e| {
                warn!("could not build asset catalog: {e}");
                (HashMap::new(), HashMap::new(), HashMap::new())
            });
        Self {
            assets,
            files,
            disk_state,
            last_scan: Instant::now(),
        }
    }

    pub fn manifest(&self) -> Vec<AssetHash> {
        let mut assets: Vec<_> = self.assets.values().cloned().collect();
        assets.sort_by(|a, b| a.path.cmp(&b.path));
        assets
    }

    fn make_delta(&self, requested: &[AssetHash]) -> Result<(Vec<u8>, Vec<AssetHash>), String> {
        let mut builder = PakBuilder::new();
        builder.intent(Intent::Fast);
        let mut included = Vec::new();
        for item in requested {
            let Some(actual) = self.assets.get(&item.path) else {
                continue;
            };
            let Some(path) = self.files.get(&item.path) else {
                continue;
            };
            builder
                .add_file(&item.path, path)
                .map_err(|e| e.to_string())?;
            included.push(actual.clone());
        }
        let output = temp_pak_path("delta");
        builder.build(&output).map_err(|e| e.to_string())?;
        let bytes = fs::read(&output).map_err(|e| e.to_string());
        let _ = fs::remove_file(output);
        bytes.map(|bytes| (bytes, included))
    }

    fn refresh_if_changed(&mut self) -> Option<(Vec<AssetHash>, Vec<String>)> {
        if self.last_scan.elapsed() < Duration::from_millis(750) {
            return None;
        }
        self.last_scan = Instant::now();
        let assets_root = asset_root();
        let mods_root = mods_directory();
        let current_state = disk_snapshot(&assets_root, &mods_root).unwrap_or_default();
        if current_state == self.disk_state {
            return None;
        }
        self.disk_state = current_state;
        match scan_catalog(&assets_root, &mods_root) {
            Ok((assets, files, state)) => {
                self.disk_state = state;
                let changed: Vec<_> = assets
                    .iter()
                    .filter_map(|(path, hash)| {
                        (self.assets.get(path).map(|old| old.checksum) != Some(hash.checksum))
                            .then_some(hash.clone())
                    })
                    .collect();
                let removed = self
                    .assets
                    .keys()
                    .filter(|path| !assets.contains_key(*path))
                    .cloned()
                    .collect();
                self.assets = assets;
                self.files = files;
                Some((changed, removed))
            }
            Err(e) => {
                warn!("asset catalog refresh failed: {e}");
                None
            }
        }
    }
}

#[derive(Clone, Default)]
struct HttpState(Arc<RwLock<HashMap<String, (Instant, bytes::Bytes)>>>);

async fn download(
    State(state): State<HttpState>,
    AxumPath(ticket): AxumPath<String>,
) -> Result<Body, StatusCode> {
    let mut downloads = state
        .0
        .write()
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    downloads.retain(|_, (created, _)| created.elapsed() < Duration::from_secs(600));
    downloads
        .get(&ticket)
        .map(|(_, bytes)| Body::from(bytes.clone()))
        .ok_or(StatusCode::NOT_FOUND)
}

struct RunningHttp {
    shutdown: oneshot::Sender<()>,
    thread: JoinHandle<()>,
}

#[derive(Resource)]
pub struct AssetHttpService {
    state: HttpState,
    running: Option<RunningHttp>,
    empty_since: Option<Instant>,
    bind_addr: SocketAddr,
    public_base: String,
}

impl AssetHttpService {
    fn from_env() -> Self {
        let bind_addr = std::env::var("EXSOLIS_ASSET_BIND_ADDR")
            .unwrap_or_else(|_| "0.0.0.0:5001".into())
            .parse()
            .expect("bad EXSOLIS_ASSET_BIND_ADDR");
        let public_base = std::env::var("EXSOLIS_ASSET_BASE_URL").unwrap_or_else(|_| {
            let addr =
                std::env::var("EXSOLIS_PUBLIC_ADDR").unwrap_or_else(|_| "127.0.0.1:5000".into());
            let host = addr
                .rsplit_once(':')
                .map(|(host, _)| host)
                .unwrap_or("127.0.0.1");
            format!("http://{host}:5001")
        });
        Self {
            state: HttpState::default(),
            running: None,
            empty_since: None,
            bind_addr,
            public_base,
        }
    }

    fn ensure_running(&mut self) -> Result<(), String> {
        if self.running.is_some() {
            return Ok(());
        }
        let listener = std::net::TcpListener::bind(self.bind_addr).map_err(|e| e.to_string())?;
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let (shutdown, shutdown_rx) = oneshot::channel();
        let state = self.state.clone();
        let thread = thread::Builder::new()
            .name("exsolis-asset-http".into())
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(e) => {
                        eprintln!("asset HTTP runtime failed: {e}");
                        return;
                    }
                };
                runtime.block_on(async move {
                    let listener = match tokio::net::TcpListener::from_std(listener) {
                        Ok(listener) => listener,
                        Err(e) => {
                            eprintln!("asset HTTP listener failed: {e}");
                            return;
                        }
                    };
                    let app = Router::new()
                        .route("/assets/{ticket}", get(download))
                        .with_state(state);
                    if let Err(e) = axum::serve(listener, app)
                        .with_graceful_shutdown(async {
                            let _ = shutdown_rx.await;
                        })
                        .await
                    {
                        eprintln!("asset HTTP server stopped: {e}");
                    }
                });
            })
            .map_err(|e| e.to_string())?;
        self.running = Some(RunningHttp { shutdown, thread });
        info!("asset HTTP server listening on {}", self.bind_addr);
        Ok(())
    }

    fn publish(&self, bytes: Vec<u8>) -> Result<String, String> {
        let ticket = format!(
            "{:016x}{:016x}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            rand::random::<u64>()
        );
        self.state
            .0
            .write()
            .map_err(|_| "asset HTTP state poisoned".to_owned())?
            .insert(ticket.clone(), (Instant::now(), bytes::Bytes::from(bytes)));
        Ok(format!(
            "{}/assets/{ticket}",
            self.public_base.trim_end_matches('/')
        ))
    }

    fn stop(&mut self) {
        if let Some(running) = self.running.take() {
            let _ = running.shutdown.send(());
            let _ = running.thread.join();
        }
        if let Ok(mut downloads) = self.state.0.write() {
            downloads.clear();
        }
        self.empty_since = None;
    }
}

#[derive(Resource, Default)]
pub struct PendingAssetJoins(pub std::collections::HashSet<PlayerId>);

pub struct AssetServerPlugin;

impl Plugin for AssetServerPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AssetCatalog::new())
            .insert_resource(AssetHttpService::from_env())
            .init_resource::<PendingAssetJoins>()
            .add_systems(
                Update,
                (
                    handle_asset_packets,
                    refresh_and_push_updates,
                    maintain_http,
                )
                    .chain()
                    .in_set(net::GameplaySet),
            );
    }
}

pub fn handle_connected(
    client_id: PlayerId,
    catalog: &AssetCatalog,
    http: &mut AssetHttpService,
    pending: &mut PendingAssetJoins,
    outgoing: &mut MessageWriter<OutgoingPacket>,
) {
    pending.0.insert(client_id);
    if let Err(e) = http.ensure_running() {
        error!("could not start asset HTTP server: {e}");
    }
    outgoing.write(OutgoingPacket::to(
        client_id,
        ServerMessage::AssetManifest {
            assets: catalog.manifest(),
            base_url: http.public_base.clone(),
        },
    ));
}

pub fn handle_disconnected(client_id: PlayerId, pending: &mut PendingAssetJoins) {
    pending.0.remove(&client_id);
}

/// Push named assets to one connected client as a pak delta. Paths are
/// resolved through the server catalog, so caller-provided strings never
/// become filesystem paths.
pub fn push_assets(
    player_id: PlayerId,
    paths: &[String],
    catalog: &AssetCatalog,
    http: &mut AssetHttpService,
    outgoing: &mut MessageWriter<OutgoingPacket>,
) -> Result<(), String> {
    let selected: Vec<_> = paths
        .iter()
        .filter_map(|path| catalog.assets.get(path).cloned())
        .collect();
    let (bytes, selected) = catalog.make_delta(&selected)?;
    http.ensure_running()?;
    let url = http.publish(bytes)?;
    outgoing.write(OutgoingPacket::to(
        player_id,
        ServerMessage::AssetUpdate {
            url,
            assets: selected,
        },
    ));
    Ok(())
}

/// Instruct a client to make assets resident using its registered handlers.
pub fn request_load(
    outgoing: &mut MessageWriter<OutgoingPacket>,
    player_id: PlayerId,
    paths: Vec<String>,
) {
    outgoing.write(OutgoingPacket::to(
        player_id,
        ServerMessage::AssetLoad { paths },
    ));
}

fn handle_asset_packets(
    mut incoming: MessageReader<IncomingPacket>,
    catalog: Res<AssetCatalog>,
    mut http: ResMut<AssetHttpService>,
    mut pending: ResMut<PendingAssetJoins>,
    mut outgoing: MessageWriter<OutgoingPacket>,
    mut commands: Commands,
    mut player_map: ResMut<PlayerMap>,
    positions: Query<(&crate::chunks::PlayerPosition, &crate::movement::PlayerLook)>,
) {
    for packet in incoming.read() {
        match &packet.msg {
            ClientMessage::AssetRequest { assets } if pending.0.contains(&packet.player_id) => {
                match http
                    .ensure_running()
                    .and_then(|_| catalog.make_delta(assets))
                    .and_then(|(bytes, included)| http.publish(bytes).map(|url| (url, included)))
                {
                    Ok((url, included)) => {
                        outgoing.write(OutgoingPacket::to(
                            packet.player_id,
                            ServerMessage::AssetDownload {
                                url,
                                assets: included,
                            },
                        ));
                    }
                    Err(e) => {
                        warn!("could not build asset delta for {}: {e}", packet.player_id);
                        outgoing.write(OutgoingPacket::to(
                            packet.player_id,
                            ServerMessage::AssetDownloadFailed { reason: e },
                        ));
                    }
                }
            }
            ClientMessage::AssetSyncComplete if pending.0.remove(&packet.player_id) => {
                crate::events::spawn_joined_player(
                    packet.player_id,
                    &mut commands,
                    &mut player_map,
                    &positions,
                    &mut outgoing,
                );
            }
            ClientMessage::AssetLoadComplete { .. } => {}
            _ => {}
        }
    }
}

fn refresh_and_push_updates(
    mut catalog: ResMut<AssetCatalog>,
    mut http: ResMut<AssetHttpService>,
    player_map: Res<PlayerMap>,
    mut outgoing: MessageWriter<OutgoingPacket>,
) {
    let Some((changed, removed)) = catalog.refresh_if_changed() else {
        return;
    };
    if player_map.playerid_to_entity.is_empty() {
        return;
    }
    if changed.is_empty() && removed.is_empty() {
        return;
    }
    if !changed.is_empty() {
        let (bytes, changed) = match catalog.make_delta(&changed) {
            Ok(delta) => delta,
            Err(e) => {
                warn!("could not build hot-reload pak: {e}");
                return;
            }
        };
        if let Err(e) = http.ensure_running() {
            warn!("could not start asset HTTP server: {e}");
            return;
        }
        let url = match http.publish(bytes) {
            Ok(url) => url,
            Err(e) => {
                warn!("could not publish hot-reload pak: {e}");
                return;
            }
        };
        for player in player_map.playerid_to_entity.keys() {
            outgoing.write(OutgoingPacket::to(
                *player,
                ServerMessage::AssetUpdate {
                    url: url.clone(),
                    assets: changed.clone(),
                },
            ));
        }
    }
    if !removed.is_empty() {
        for player in player_map.playerid_to_entity.keys() {
            outgoing.write(OutgoingPacket::to(
                *player,
                ServerMessage::AssetRemove {
                    paths: removed.clone(),
                },
            ));
        }
    }
}

fn maintain_http(mut http: ResMut<AssetHttpService>, server: Res<RenetServer>) {
    if !server.clients_id().is_empty() {
        http.empty_since = None;
    } else if http.running.is_some() {
        let since = http.empty_since.get_or_insert_with(Instant::now);
        if since.elapsed() >= IDLE_SHUTDOWN {
            http.stop();
        }
    }
}

fn asset_root() -> PathBuf {
    if let Some(path) = std::env::var_os("EXSOLIS_ASSET_DIR") {
        return PathBuf::from(path);
    }
    let cwd = PathBuf::from("assets");
    if cwd.is_dir() {
        return cwd;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets")
}

/// Directory containing server-side mod folders. Each mod may have Lua
/// sources anywhere below its folder and client assets under `assets/`.
pub(crate) fn mods_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("EXSOLIS_MODS_DIR") {
        return PathBuf::from(path);
    }
    let cwd = PathBuf::from("mods");
    if cwd.is_dir() {
        return cwd;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../mods")
}

fn scan_catalog(
    assets_root: &Path,
    mods_root: &Path,
) -> Result<
    (
        HashMap<String, AssetHash>,
        HashMap<String, PathBuf>,
        HashMap<String, (u64, Option<SystemTime>)>,
    ),
    String,
> {
    let files = discover_client_assets(assets_root, mods_root)?;
    let state = snapshot_files(&files).map_err(|e| e.to_string())?;
    let mut builder = PakBuilder::new();
    builder.intent(Intent::Fast);
    for (logical_path, disk_path) in &files {
        builder
            .add_file(logical_path, disk_path)
            .map_err(|e| format!("could not add asset '{logical_path}': {e}"))?;
    }
    let pak_path = temp_pak_path("catalog");
    builder.build(&pak_path).map_err(|e| e.to_string())?;
    let pak = Pak::open(&pak_path).map_err(|e| e.to_string())?;
    let assets = pak
        .iter()
        .map(|entry| {
            (
                entry.path.to_owned(),
                AssetHash {
                    path: entry.path.to_owned(),
                    checksum: entry.checksum,
                },
            )
        })
        .collect();
    drop(pak);
    let _ = fs::remove_file(pak_path);
    Ok((assets, files, state))
}

fn discover_client_assets(
    assets_root: &Path,
    mods_root: &Path,
) -> Result<HashMap<String, PathBuf>, String> {
    fn walk_assets(
        dir: &Path,
        root: &Path,
        logical_prefix: &str,
        filter_extensions: bool,
        out: &mut HashMap<String, PathBuf>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let ty = entry.file_type()?;
            if ty.is_dir() {
                walk_assets(&path, root, logical_prefix, filter_extensions, out)?;
            } else if ty.is_file() {
                if filter_extensions && !is_client_asset_path(&path) {
                    continue;
                }
                let relative = path.strip_prefix(root).unwrap_or(&path);
                let relative = relative.to_string_lossy().replace('\\', "/");
                let logical = if logical_prefix.is_empty() {
                    relative
                } else {
                    format!("{logical_prefix}/{relative}")
                };
                out.insert(logical, path);
            }
        }
        Ok(())
    }

    fn walk_mod_directories(
        dir: &Path,
        mods_root: &Path,
        out: &mut HashMap<String, PathBuf>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name();
            if name.to_string_lossy().starts_with('.') {
                continue;
            }
            if name == "assets" {
                let Some(mod_dir) = path.parent() else {
                    continue;
                };
                let relative_mod = mod_dir.strip_prefix(mods_root).unwrap_or(mod_dir);
                if relative_mod.as_os_str().is_empty() {
                    continue;
                }
                let mod_name = relative_mod.to_string_lossy().replace('\\', "/");
                let prefix = format!("mods/{mod_name}");
                walk_assets(&path, &path, &prefix, true, out)?;
            } else {
                walk_mod_directories(&path, mods_root, out)?;
            }
        }
        Ok(())
    }

    let mut files = HashMap::new();
    if assets_root.is_dir() {
        walk_assets(assets_root, assets_root, "", false, &mut files)
            .map_err(|e| format!("{}: {e}", assets_root.display()))?;
    }

    if mods_root.is_dir() {
        walk_mod_directories(mods_root, mods_root, &mut files)
            .map_err(|e| format!("{}: {e}", mods_root.display()))?;
    }
    Ok(files)
}

fn is_client_asset_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "fxm" | "wgsl" | "png" | "xml" | "json"
            )
        })
}

fn snapshot_files(
    files: &HashMap<String, PathBuf>,
) -> std::io::Result<HashMap<String, (u64, Option<SystemTime>)>> {
    let mut out = HashMap::with_capacity(files.len());
    for (logical_path, path) in files {
        let metadata = fs::metadata(path)?;
        out.insert(
            logical_path.clone(),
            (metadata.len(), metadata.modified().ok()),
        );
    }
    Ok(out)
}

fn disk_snapshot(
    assets_root: &Path,
    mods_root: &Path,
) -> std::io::Result<HashMap<String, (u64, Option<SystemTime>)>> {
    let files = discover_client_assets(assets_root, mods_root).map_err(std::io::Error::other)?;
    snapshot_files(&files)
}

fn temp_pak_path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "exsolis-{label}-{}-{}.pak",
        std::process::id(),
        rand::random::<u64>()
    ))
}
