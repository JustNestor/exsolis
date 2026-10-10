//!import engine/frame

// A continuous effect: started once, kept alive until stopped, and attached to
// a player's model matrix (the route sets `follow_model`).

struct AuraAction {
    intensity: f32,
}

struct AuraState {
    intensity: f32,
}

@group(1) @binding(0) var<storage, read> action: AuraAction;
@group(1) @binding(2) var<storage, read_write> state: AuraState;

const PARTICLES: u32 = 24u;
const FADE_OUT_SECONDS: f32 = 0.6;
const PARTICLE_SIZE: f32 = 0.05;

// Spawn pass: runs once when the aura starts.
@compute @workgroup_size(1)
fn init() {
    state.intensity = action.intensity;
}

// Update pass: runs once per `update` action; the payload is this action's.
@compute @workgroup_size(1)
fn set_intensity() {
    state.intensity = action.intensity;
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) fade: f32,
    @location(1) local_corner: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0)
    );
    let corner = corners[vertex_index % 6u];
    let t = frame.instance_age;
    let n = f32(instance_index);

    // Orbit in the player's local space. `frame.model` tracks the player every
    // frame because the route opted in with follow_model.
    let angle = n / f32(PARTICLES) * 6.2831853 + t * 1.2;
    let ring = 0.55 + 0.08 * sin(t * 2.0 + n);
    let height = 0.15 + fract(n * 0.37 + t * 0.25) * 1.6;
    let local_position = vec3<f32>(cos(angle) * ring, height, sin(angle) * ring);
    let centre = (frame.model * vec4<f32>(local_position, 1.0)).xyz;

    var fade = clamp(t * 3.0, 0.0, 1.0);
    if (frame.stop_age >= 0.0) {
        fade = fade * clamp(1.0 - frame.stop_age / FADE_OUT_SECONDS, 0.0, 1.0);
    }

    let vp = frame.view_projection;
    let right = normalize(vec3<f32>(vp[0][0], vp[1][0], vp[2][0]));
    let forward = normalize(vec3<f32>(vp[0][3], vp[1][3], vp[2][3]));
    let up = cross(right, forward);

    var out: VertexOutput;
    out.clip_position = vp * vec4<f32>(centre + (right * corner.x + up * corner.y) * PARTICLE_SIZE, 1.0);
    out.fade = fade * state.intensity;
    out.local_corner = corner;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let glow = 1.0 - smoothstep(0.1, 1.0, length(in.local_corner));
    return vec4<f32>(vec3<f32>(0.2, 0.9, 0.8), glow * in.fade);
}
