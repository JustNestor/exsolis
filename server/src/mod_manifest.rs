//! Per-mod manifests: `mods/ts/<id>.toml`, containing **only dependencies**.
//!
//! A manifest's existence is optional — no file means "no dependencies". The
//! `<id>` is the file stem of the matching `<id>.ts` source. Everything else
//! (component ownership, type layout, version) belongs to the hand-written
//! `mods/types/<id>.d.ts` and the runtime registration, not to this file.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct ManifestFile {
    #[serde(default)]
    dependencies: Vec<String>,
}

/// A validated manifest for one mod.
#[derive(Debug, Clone, Default)]
pub(crate) struct ModManifest {
    pub(crate) id: String,
    pub(crate) dependencies: Vec<String>,
}

impl ModManifest {
    /// Read `root/<id>.toml`. Missing file → empty manifest (legal).
    pub(crate) fn load(root: &Path, id: &str) -> Result<Self, String> {
        let path = root.join(format!("{id}.toml"));
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self { id: id.to_owned(), dependencies: Vec::new() });
            }
            Err(e) => return Err(format!("read {}: {e}", path.display())),
        };
        let parsed: ManifestFile = toml::from_str(&text)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        Ok(Self { id: id.to_owned(), dependencies: parsed.dependencies })
    }
}

/// Load every manifest in `root`, keyed by mod id (file stem).
pub(crate) fn load_all(root: &Path) -> std::collections::HashMap<String, ModManifest> {
    let mut out = std::collections::HashMap::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Some(id) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        match ModManifest::load(root, id) {
            Ok(m) => { out.insert(id.to_owned(), m); }
            Err(e) => bevy_log::warn!("[ts mod] manifest error: {e}"),
        }
    }
    out
}

/// Check that every dependency names a mod that is actually present. Returns
/// a list of readable error strings (empty when all dependencies resolve).
pub(crate) fn check_dependencies(
    manifests: &std::collections::HashMap<String, ModManifest>,
) -> Vec<String> {
    let mut errors = Vec::new();
    for (id, manifest) in manifests {
        for dep in &manifest.dependencies {
            if !manifests.contains_key(dep) {
                errors.push(format!(
                    "mod `{id}` depends on `{dep}`, which is not loaded"
                ));
            }
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_manifest_is_empty() {
        let dir = std::env::temp_dir().join(format!("exsolis-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let m = ModManifest::load(&dir, "absent").unwrap();
        assert!(m.dependencies.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_dependencies_only() {
        let dir = std::env::temp_dir().join(format!("exsolis-manifest-parse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("mymod.toml"),
            "dependencies = [\"core\", \"util\"]\n",
        ).unwrap();
        let m = ModManifest::load(&dir, "mymod").unwrap();
        assert_eq!(m.dependencies, vec!["core", "util"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_dependency_is_reported() {
        let mut manifests = std::collections::HashMap::new();
        manifests.insert("a".to_owned(), ModManifest {
            id: "a".to_owned(),
            dependencies: vec!["ghost".to_owned()],
        });
        let errors = check_dependencies(&manifests);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("ghost"));
    }
}
