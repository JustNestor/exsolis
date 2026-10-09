//! Client-side dispatch boundary for server-authored mod visual actions.
//!
//! This module intentionally runs no client scripts. A package manifest maps
//! network actions to validated, typed inputs for its declared GPU passes.

use std::{collections::HashSet, path::Path};

use bevy_ecs::prelude::*;
use bevy_log::warn;
use exsolis_common::{
    MOD_ACTION_MAX_ID_BYTES, MOD_ACTION_MAX_PAYLOAD_BYTES, ServerMessage, is_valid_mod_action_id,
};
use serde::Deserialize;

use crate::{
    assets::{AssetHandlerRegistry, LoadedAsset, WgslAsset},
    net::IncomingPacket,
};

#[derive(Message, Debug, Clone)]
pub struct ModActionEvent {
    pub effect_id: String,
    pub id: String,
    pub compute_pass: String,
    /// Payload fields encoded with WGSL-compatible alignment and padding.
    pub payload: Vec<u8>,
    pub model_player_id: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectManifest {
    pub format_version: u32,
    pub id: String,
    /// Number of independent in-flight action instances this effect can hold.
    /// When full, the oldest active instance is reused.
    #[serde(default = "default_max_instances")]
    pub max_instances: u32,
    /// Instances become reusable after this many seconds.
    #[serde(default = "default_instance_lifetime_seconds")]
    pub instance_lifetime_seconds: f32,
    #[serde(default)]
    pub frame_inputs: Vec<FrameInput>,
    #[serde(default)]
    pub actions: Vec<ActionRoute>,
    #[serde(default)]
    pub passes: Vec<EffectPass>,
    #[serde(default)]
    pub buffers: Vec<BufferDecl>,
    #[serde(default)]
    pub images: Vec<ImageDecl>,
    /// PNG assets sampled by this effect's shaders. Order matters: shaders
    /// read `frame.textures[i]` for the i-th entry (see `MAX_EFFECT_TEXTURES`).
    #[serde(default)]
    pub textures: Vec<TextureDecl>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureDecl {
    /// Human-readable name; unique within the effect.
    pub id: String,
    /// Asset path of a `.png`, e.g. `mods/fly/effects/spark.png`.
    pub path: String,
}

fn default_max_instances() -> u32 {
    8
}

fn default_instance_lifetime_seconds() -> f32 {
    3.0
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FrameInput {
    FrameIndex,
    Time,
    DeltaTime,
    Viewport,
    CameraPosition,
    ViewProjection,
    Model,
    ModelViewProjection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionRoute {
    pub id: String,
    pub compute_pass: String,
    #[serde(default)]
    pub fields: Vec<PayloadField>,
    /// Payload path holding a decimal PlayerId whose client transform is
    /// exposed through the engine model-matrix binding.
    pub model_player_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadField {
    pub name: String,
    pub source: String,
    #[serde(rename = "type")]
    pub ty: PayloadType,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadType {
    I32,
    U32,
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectPass {
    pub id: String,
    pub kind: PassKind,
    pub shader: Option<String>,
    pub entry_point: Option<String>,
    pub vertex: Option<StageRef>,
    pub fragment: Option<StageRef>,
    pub phase: Option<RenderPhase>,
    pub blend: Option<BlendMode>,
    pub depth: Option<DepthMode>,
    #[serde(default)]
    pub run_each_frame: bool,
    #[serde(default = "default_vertex_count")]
    pub vertex_count: u32,
    #[serde(default = "default_instance_count")]
    pub instance_count: u32,
    #[serde(default = "default_workgroups")]
    pub workgroups: [u32; 3],
    #[serde(default)]
    pub reads: Vec<String>,
    #[serde(default)]
    pub writes: Vec<String>,
    #[serde(default)]
    pub after: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageRef {
    pub shader: String,
    pub entry_point: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PassKind {
    Compute,
    Graphics,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RenderPhase {
    WorldTransparent,
    WorldOverlay,
    UiOverlay,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    Opaque,
    Alpha,
    Additive,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthMode {
    Test,
    Disabled,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BufferDecl {
    pub id: String,
    pub bytes: u32,
    pub persistent: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageDecl {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
    pub usage: Vec<ImageUsage>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Rgba8Unorm,
    Rgba16Float,
    Rgba32Float,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageUsage {
    Sampled,
    Storage,
    RenderTarget,
}

const MAX_MANIFEST_BYTES: usize = 32 * 1024;
const MAX_PASSES: usize = 16;
const MAX_RESOURCES: usize = 16;
/// Capacity of the texture table in the frame buffer; the WGSL-side
/// `textures: array<u32, 8>` must match.
pub(crate) const MAX_EFFECT_TEXTURES: usize = 8;
const MAX_EFFECT_BUFFER_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EFFECT_IMAGE_PIXELS: u64 = 4096 * 4096;
const MAX_EFFECT_IMAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PAYLOAD_LAYOUT_BYTES: usize = 1024;
const MAX_WORKGROUPS_PER_PASS: u64 = 65_535;

fn default_workgroups() -> [u32; 3] {
    [1, 1, 1]
}

fn default_vertex_count() -> u32 {
    6
}

fn default_instance_count() -> u32 {
    1
}

pub(crate) fn decode_manifest(bytes: &[u8]) -> Result<LoadedAsset, String> {
    if bytes.len() > MAX_MANIFEST_BYTES {
        return Err(format!("effect manifest too large ({} bytes)", bytes.len()));
    }
    let manifest: EffectManifest =
        serde_json::from_slice(bytes).map_err(|e| format!("effect manifest parse failed: {e}"))?;
    validate_manifest(&manifest)?;
    Ok(std::sync::Arc::new(manifest))
}

fn validate_manifest(manifest: &EffectManifest) -> Result<(), String> {
    if manifest.format_version != 1 {
        return Err(format!(
            "unsupported effect manifest version {}",
            manifest.format_version
        ));
    }
    if !is_valid_mod_action_id(&manifest.id) {
        return Err(format!("invalid effect id '{}'", manifest.id));
    }
    if !(1..=64).contains(&manifest.max_instances) {
        return Err("effect max_instances must be between 1 and 64".into());
    }
    if !manifest.instance_lifetime_seconds.is_finite()
        || !(0.05..=120.0).contains(&manifest.instance_lifetime_seconds)
    {
        return Err("effect instance_lifetime_seconds must be between 0.05 and 120".into());
    }
    if manifest.frame_inputs.len() > 16
        || manifest.actions.len() > 64
        || manifest.passes.len() > MAX_PASSES
        || manifest.buffers.len() > MAX_RESOURCES
        || manifest.images.len() > MAX_RESOURCES
        || manifest.textures.len() > MAX_EFFECT_TEXTURES
    {
        return Err("effect manifest exceeds a declaration limit".into());
    }

    let mut frame_inputs = HashSet::new();
    if manifest
        .frame_inputs
        .iter()
        .any(|input| !frame_inputs.insert(*input))
    {
        return Err("duplicate frame input".into());
    }

    let mut pass_ids = HashSet::new();
    for pass in &manifest.passes {
        if !is_identifier(&pass.id) || !pass_ids.insert(pass.id.as_str()) {
            return Err(format!("invalid or duplicate pass id '{}'", pass.id));
        }
        match pass.kind {
            PassKind::Compute => {
                let shader = pass
                    .shader
                    .as_deref()
                    .ok_or_else(|| format!("compute pass '{}' has no shader", pass.id))?;
                let entry_point = pass
                    .entry_point
                    .as_deref()
                    .ok_or_else(|| format!("compute pass '{}' has no entry point", pass.id))?;
                if pass.vertex.is_some()
                    || pass.fragment.is_some()
                    || pass.phase.is_some()
                    || pass.blend.is_some()
                    || pass.depth.is_some()
                {
                    return Err(format!(
                        "compute pass '{}' has graphics-only fields",
                        pass.id
                    ));
                }
                validate_wgsl_path(shader)?;
                if !is_identifier(entry_point) {
                    return Err(format!("invalid WGSL entry point '{entry_point}'"));
                }
                let groups = pass.workgroups.iter().try_fold(1u64, |total, count| {
                    if *count == 0 {
                        None
                    } else {
                        total.checked_mul(u64::from(*count))
                    }
                });
                if groups.is_none_or(|groups| groups > MAX_WORKGROUPS_PER_PASS) {
                    return Err(format!(
                        "compute pass '{}' has invalid workgroup count",
                        pass.id
                    ));
                }
            }
            PassKind::Graphics => {
                if pass.shader.is_some() || pass.entry_point.is_some() {
                    return Err(format!(
                        "graphics pass '{}' uses compute shader fields",
                        pass.id
                    ));
                }
                let vertex = pass
                    .vertex
                    .as_ref()
                    .ok_or_else(|| format!("graphics pass '{}' has no vertex stage", pass.id))?;
                let fragment = pass
                    .fragment
                    .as_ref()
                    .ok_or_else(|| format!("graphics pass '{}' has no fragment stage", pass.id))?;
                validate_stage(vertex, &pass.id)?;
                validate_stage(fragment, &pass.id)?;
                if pass.phase.is_none() || pass.blend.is_none() || pass.depth.is_none() {
                    return Err(format!(
                        "graphics pass '{}' must declare phase, blend, and depth",
                        pass.id
                    ));
                }
                if pass.run_each_frame {
                    return Err(format!(
                        "graphics pass '{}' is rendered every frame automatically",
                        pass.id
                    ));
                }
                if pass.vertex_count == 0
                    || pass.vertex_count > 1_000_000
                    || pass.instance_count == 0
                    || pass.instance_count > 65_535
                {
                    return Err(format!(
                        "graphics pass '{}' has invalid draw counts",
                        pass.id
                    ));
                }
            }
        }
    }
    for pass in &manifest.passes {
        for dependency in &pass.after {
            if !pass_ids.contains(dependency.as_str()) || dependency == &pass.id {
                return Err(format!(
                    "pass '{}' has invalid dependency '{dependency}'",
                    pass.id
                ));
            }
        }
    }
    // Dependencies describe execution order, so reject cycles while the
    // package is still being loaded rather than failing during graph build.
    fn visit_pass(
        id: &str,
        passes: &[EffectPass],
        marks: &mut std::collections::HashMap<String, u8>,
    ) -> bool {
        match marks.get(id) {
            Some(1) => return false,
            Some(2) => return true,
            _ => {}
        }
        marks.insert(id.to_owned(), 1);
        let pass = passes
            .iter()
            .find(|pass| pass.id == id)
            .expect("known pass");
        for dependency in &pass.after {
            if !visit_pass(dependency, passes, marks) {
                return false;
            }
        }
        marks.insert(id.to_owned(), 2);
        true
    }
    let mut dependency_marks = std::collections::HashMap::new();
    for pass in &manifest.passes {
        if !visit_pass(&pass.id, &manifest.passes, &mut dependency_marks) {
            return Err(format!(
                "pass dependencies contain a cycle at '{}'",
                pass.id
            ));
        }
    }

    let mut buffer_ids = HashSet::new();
    let mut total_buffer_bytes = 0u64;
    for buffer in &manifest.buffers {
        if !is_identifier(&buffer.id) || !buffer_ids.insert(buffer.id.as_str()) || buffer.bytes == 0
        {
            return Err(format!("invalid or duplicate buffer '{}'", buffer.id));
        }
        total_buffer_bytes += u64::from(buffer.bytes);
    }
    if total_buffer_bytes > MAX_EFFECT_BUFFER_BYTES {
        return Err("effect buffers exceed the 4 MiB package limit".into());
    }

    let mut texture_ids = HashSet::new();
    for texture in &manifest.textures {
        if !is_identifier(&texture.id) || !texture_ids.insert(texture.id.as_str()) {
            return Err(format!("invalid or duplicate texture '{}'", texture.id));
        }
        // `.png` exactly: texture.rs only discovers lowercase-extension files.
        if fastpack::normalize_path(&texture.path).is_err() || !texture.path.ends_with(".png") {
            return Err(format!("invalid texture asset path '{}'", texture.path));
        }
    }

    let mut image_ids = HashSet::new();
    let mut total_image_bytes = 0u64;
    for image in &manifest.images {
        let pixels = u64::from(image.width) * u64::from(image.height);
        if !is_identifier(&image.id)
            || !image_ids.insert(image.id.as_str())
            || buffer_ids.contains(image.id.as_str())
            || image.width == 0
            || image.height == 0
            || pixels > MAX_EFFECT_IMAGE_PIXELS
        {
            return Err(format!("invalid or duplicate image '{}'", image.id));
        }
        let bytes_per_pixel = match image.format {
            ImageFormat::Rgba8Unorm => 4,
            ImageFormat::Rgba16Float => 8,
            ImageFormat::Rgba32Float => 16,
        };
        total_image_bytes = total_image_bytes
            .checked_add(pixels * bytes_per_pixel)
            .ok_or_else(|| "effect image allocation size overflow".to_string())?;
        if total_image_bytes > MAX_EFFECT_IMAGE_BYTES {
            return Err("effect images exceed the 64 MiB package limit".into());
        }
        if image.usage.is_empty() {
            return Err(format!("image '{}' has no declared usage", image.id));
        }
    }

    let resource_ids: HashSet<&str> = buffer_ids.union(&image_ids).copied().collect();
    for pass in &manifest.passes {
        let mut read_ids = HashSet::new();
        let mut write_ids = HashSet::new();
        for id in &pass.reads {
            if !resource_ids.contains(id.as_str()) || !read_ids.insert(id.as_str()) {
                return Err(format!(
                    "pass '{}' has invalid read resource '{id}'",
                    pass.id
                ));
            }
        }
        for id in &pass.writes {
            if !resource_ids.contains(id.as_str()) || !write_ids.insert(id.as_str()) {
                return Err(format!(
                    "pass '{}' has invalid write resource '{id}'",
                    pass.id
                ));
            }
        }
        for id in &pass.reads {
            if let Some(image) = manifest.images.iter().find(|image| image.id == *id) {
                if !image.usage.contains(&ImageUsage::Sampled) {
                    return Err(format!(
                        "pass '{}' samples image '{}' without sampled usage",
                        pass.id, id
                    ));
                }
            }
        }
        for id in &pass.writes {
            if let Some(image) = manifest.images.iter().find(|image| image.id == *id) {
                let required_usage = match pass.kind {
                    PassKind::Compute => ImageUsage::Storage,
                    PassKind::Graphics => ImageUsage::RenderTarget,
                };
                if !image.usage.contains(&required_usage) {
                    return Err(format!(
                        "pass '{}' writes image '{}' without {:?} usage",
                        pass.id, id, required_usage
                    ));
                }
            }
        }
    }

    let compute_pass_ids: HashSet<&str> = manifest
        .passes
        .iter()
        .filter(|pass| pass.kind == PassKind::Compute)
        .map(|pass| pass.id.as_str())
        .collect();
    let mut action_ids = HashSet::new();
    for action in &manifest.actions {
        if !is_valid_mod_action_id(&action.id)
            || !action_ids.insert(action.id.as_str())
            || !compute_pass_ids.contains(action.compute_pass.as_str())
        {
            return Err(format!("invalid or duplicate action route '{}'", action.id));
        }
        let mut field_names = HashSet::new();
        let mut source_names = HashSet::new();
        let mut size = 0usize;
        for field in &action.fields {
            if !is_identifier(&field.name)
                || field.source.is_empty()
                || !field.source.split('.').all(is_identifier)
                || !field_names.insert(field.name.as_str())
                || !source_names.insert(field.source.as_str())
            {
                return Err(format!(
                    "invalid or duplicate field in action '{}'",
                    action.id
                ));
            }
            let (alignment, field_size) = field.ty.layout();
            size = align_up(size, alignment) + field_size;
        }
        if size > MAX_PAYLOAD_LAYOUT_BYTES {
            return Err(format!(
                "action '{}' payload layout exceeds 1 KiB",
                action.id
            ));
        }
        if let Some(path) = &action.model_player_id {
            if path.is_empty() || !path.split('.').all(is_identifier) {
                return Err(format!("invalid model player-id path for '{}'", action.id));
            }
        }
    }
    Ok(())
}

fn validate_wgsl_path(path: &str) -> Result<(), String> {
    if fastpack::normalize_path(path).is_err()
        || !Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("wgsl"))
    {
        return Err(format!("invalid WGSL asset path '{path}'"));
    }
    Ok(())
}

fn validate_stage(stage: &StageRef, pass_id: &str) -> Result<(), String> {
    validate_wgsl_path(&stage.shader)?;
    if !is_identifier(&stage.entry_point) {
        return Err(format!(
            "graphics pass '{pass_id}' has invalid entry point '{}'",
            stage.entry_point
        ));
    }
    Ok(())
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

impl PayloadType {
    fn layout(self) -> (usize, usize) {
        match self {
            Self::I32 | Self::U32 | Self::F32 => (4, 4),
            Self::Vec2 => (8, 8),
            Self::Vec3 => (16, 12),
            Self::Vec4 => (16, 16),
            Self::Mat4 => (16, 64),
        }
    }
}

pub(super) fn route_mod_actions(
    mut incoming: MessageReader<IncomingPacket>,
    mut actions: MessageWriter<ModActionEvent>,
    assets: Res<AssetHandlerRegistry>,
) {
    for packet in incoming.read() {
        let ServerMessage::ModAction { id, payload_json } = &packet.0 else {
            continue;
        };
        if id.len() > MOD_ACTION_MAX_ID_BYTES
            || !is_valid_mod_action_id(id)
            || payload_json.len() > MOD_ACTION_MAX_PAYLOAD_BYTES
        {
            warn!("dropping malformed or oversized mod action '{id}'");
            continue;
        }
        let payload: serde_json::Value =
            match serde_json::from_str::<serde_json::Value>(payload_json) {
                Ok(value) if value.is_object() => value,
                _ => {
                    warn!("dropping mod action '{id}' with invalid object payload");
                    continue;
                }
            };
        for (_, manifest) in assets.iter_loaded::<EffectManifest>() {
            for route in manifest.actions.iter().filter(|route| route.id == *id) {
                let pass = manifest
                    .passes
                    .iter()
                    .find(|pass| pass.id == route.compute_pass)
                    .expect("validated action references a compute pass");
                let shader_path = pass.shader.as_deref().expect("compute shader validated");
                let entry_point = pass
                    .entry_point
                    .as_deref()
                    .expect("compute entry point validated");
                let Some(shader) = assets.get::<WgslAsset>(shader_path) else {
                    warn!(
                        "effect '{}' action '{}' references unloaded shader '{}'",
                        manifest.id, id, shader_path
                    );
                    continue;
                };
                if !shader.entry_points.iter().any(|entry| {
                    entry.name == entry_point && entry.stage == naga::ShaderStage::Compute
                }) {
                    warn!(
                        "effect '{}' pass '{}' has no compiled compute entry point '{}'",
                        manifest.id, pass.id, entry_point
                    );
                    continue;
                }
                match pack_payload(route, &payload) {
                    Ok((bytes, model_player_id)) => {
                        actions.write(ModActionEvent {
                            effect_id: manifest.id.clone(),
                            id: id.clone(),
                            compute_pass: route.compute_pass.clone(),
                            payload: bytes,
                            model_player_id,
                        });
                    }
                    Err(error) => {
                        warn!("effect '{}' rejected action '{}': {error}", manifest.id, id)
                    }
                }
            }
        }
    }
}

fn pack_payload(
    route: &ActionRoute,
    payload: &serde_json::Value,
) -> Result<(Vec<u8>, Option<u64>), String> {
    let mut bytes = Vec::new();
    for field in &route.fields {
        let (alignment, size) = field.ty.layout();
        let aligned = align_up(bytes.len(), alignment);
        bytes.resize(aligned + size, 0);
        let value = json_path(payload, &field.source)
            .ok_or_else(|| format!("payload field '{}' is missing", field.source))?;
        write_payload_field(&mut bytes[aligned..aligned + size], value, field.ty)
            .map_err(|error| format!("field '{}': {error}", field.source))?;
    }

    let model_player_id = route
        .model_player_id
        .as_deref()
        .map(|path| {
            json_path(payload, path)
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("model player id '{path}' must be a decimal string"))?
                .parse::<u64>()
                .map_err(|_| format!("model player id '{path}' is not a u64"))
        })
        .transpose()?;
    Ok((bytes, model_player_id))
}

fn json_path<'a>(value: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    path.split('.')
        .try_fold(value, |current, part| current.get(part))
}

fn write_payload_field(
    output: &mut [u8],
    value: &serde_json::Value,
    ty: PayloadType,
) -> Result<(), String> {
    match ty {
        PayloadType::I32 => write_number(output, value, |n| {
            n.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .map(i32::to_le_bytes)
        }),
        PayloadType::U32 => write_number(output, value, |n| {
            n.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(u32::to_le_bytes)
        }),
        PayloadType::F32 => write_number(output, value, |n| {
            n.as_f64()
                .filter(|n| n.is_finite() && n.abs() <= f32::MAX as f64)
                .map(|n| (n as f32).to_le_bytes())
        }),
        PayloadType::Vec2 => write_vector(output, value, 2),
        PayloadType::Vec3 => write_vector(output, value, 3),
        PayloadType::Vec4 => write_vector(output, value, 4),
        PayloadType::Mat4 => write_vector(output, value, 16),
    }
}

fn write_number<const N: usize>(
    output: &mut [u8],
    value: &serde_json::Value,
    convert: impl FnOnce(&serde_json::Number) -> Option<[u8; N]>,
) -> Result<(), String> {
    let number = value
        .as_number()
        .and_then(convert)
        .ok_or_else(|| "expected a number in range".to_string())?;
    output[..N].copy_from_slice(&number);
    Ok(())
}

fn write_vector(
    output: &mut [u8],
    value: &serde_json::Value,
    components: usize,
) -> Result<(), String> {
    let components = if let Some(array) = value.as_array() {
        if array.len() != components {
            return Err(format!("expected an array of {components} numbers"));
        }
        array.iter().collect::<Vec<_>>()
    } else {
        let keys = ["x", "y", "z", "w"];
        if components > 4 {
            return Err("mat4 payloads must be flat arrays".into());
        }
        keys[..components]
            .iter()
            .map(|key| value.get(*key))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("expected an array or an object with {components} components"))?
    };
    for (i, value) in components.iter().enumerate() {
        let number = value
            .as_f64()
            .filter(|number| number.is_finite() && number.abs() <= f32::MAX as f64)
            .ok_or_else(|| format!("component {i} is not a finite f32"))?
            as f32;
        output[i * 4..i * 4 + 4].copy_from_slice(&number.to_le_bytes());
    }
    Ok(())
}
