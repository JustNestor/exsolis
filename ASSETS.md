# Asset delivery

The server scans its own `assets/` source directory (override with
`EXSOLIS_ASSET_DIR`) and builds a
manifest of normalized paths and fastpak xxh3-64 content checksums. Renet
carries the manifest, missing-asset requests, download tickets, preload
instructions, and hot-reload notifications. Pak data is transferred only over
the Axum HTTP service.

The HTTP listener starts when the first Renet client connects and shuts down
after 30 seconds with no connected clients. Defaults are `0.0.0.0:5001` for
the listener and `http://<EXSOLIS_PUBLIC_ADDR host>:5001` for client URLs.
Configure `EXSOLIS_ASSET_BIND_ADDR` to change the listener address. Set
`EXSOLIS_ASSET_BASE_URL` to the public origin clients can reach. For HTTPS,
terminate TLS at a reverse proxy and point this variable at that HTTPS origin;
forward `/assets/*` to the configured listener.

Clients do not read the server's `assets/` source directory. By default they
keep their cache outside the repository at `~/.exso/downloaded/`: the merged
pak is `assets.pak`, the most recent verified server transfer is `last_delta.pak`,
and extracted assets are under `assets/`. Set `EXSOLIS_ASSET_CACHE_DIR` to move
the cache; keep it outside the server's `EXSOLIS_ASSET_DIR`. On join, clients compare the server manifest with their pak and
request only missing or changed entries. The merged pak remains on disk for the
next session. The server does not create the player until synchronization
completes. Failed transfers are logged and the client proceeds with the assets
already in its downloaded cache.

The client `AssetHandlerRegistry` dispatches resident-load requests by file
extension. The default PNG handler decodes image bytes into memory; unknown
extensions are retained as bytes. Register additional handlers for shaders,
models, audio, or backend-specific GPU uploads. Resident values currently stay
alive for the process lifetime.

The server polls asset file metadata during development. Added or changed files
are delivered as small fastpak deltas and reloaded by the client. Removed files
are no longer referenced by future manifests; resident copies are retained for
the current process because the GPU backend does not yet support safe unloading.
xxh3 checksums detect accidental corruption, not malicious modification.
