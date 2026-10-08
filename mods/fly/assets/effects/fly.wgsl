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
}

struct FlyAction {
    strength: f32,
    position: vec3<f32>,
}

struct Particle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
}

@group(1) @binding(0) var<storage, read> action: FlyAction;
@group(1) @binding(1) var<storage, read> frame: FrameData;
@group(1) @binding(2) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    let angle = f32(index) * 2.39996323;
    let ring = 0.18 + f32(index % 11u) * 0.035;
    let origin = action.position + vec3<f32>(0.0, 0.06, 0.0);
    var particle = particles[index];
    particle.position_age = vec4<f32>(
        origin + vec3<f32>(cos(angle) * ring, 0.0, sin(angle) * ring),
        0.0
    );
    particle.velocity_lifetime = vec4<f32>(
        cos(angle) * (0.6 + f32(index % 7u) * 0.09),
        0.45 + f32(index % 7u) * 0.08,
        sin(angle) * (0.6 + f32(index % 7u) * 0.09),
        0.75 + (f32(index % 9u) * 0.04) / max(action.strength, 0.1)
    );
    particles[index] = particle;
}

@compute @workgroup_size(64)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= arrayLength(&particles)) {
        return;
    }

    var particle = particles[index];
    if (particle.position_age.w < 0.0) {
        return;
    }
    let dt = min(frame.delta_time, 0.05);
    let age = particle.position_age.w + dt;
    if (age >= particle.velocity_lifetime.w) {
        particle.position_age = vec4<f32>(particle.position_age.xyz, -1.0);
    } else {
        particle.position_age = vec4<f32>(
            particle.position_age.xyz + particle.velocity_lifetime.xyz * dt,
            age
        );
        particle.velocity_lifetime = vec4<f32>(
            particle.velocity_lifetime.x,
            particle.velocity_lifetime.y - 0.9 * dt,
            particle.velocity_lifetime.z,
            particle.velocity_lifetime.w
        );
    }
    particles[index] = particle;
}

const PARTICLE_SIZE: f32 = 0.035;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) life: f32,
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
    let particle = particles[instance_index];
    let life = particle.position_age.w;
    let corner = corners[vertex_index % 6u];
    var out: VertexOutput;
    if (life < 0.0) {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.life = 0.0;
        out.local_corner = vec2<f32>(2.0, 2.0);
    } else {
        // Camera-facing billboard. Recover the camera basis from the rows of
        // view_projection (column-major: vp[col][row]):
        //   row 0 = P00 * right      (clip.x depends only on view x)
        //   row 3 = forward          (clip.w = distance along forward)
        // up = right x forward  (right-handed). Rows 0 and 3 are unaffected
        // by the NDC Y convention, so this holds whether or not Y is flipped.
        let vp = frame.view_projection;
        let right = normalize(vec3<f32>(vp[0][0], vp[1][0], vp[2][0]));
        let forward = normalize(vec3<f32>(vp[0][3], vp[1][3], vp[2][3]));
        let up = cross(right, forward);
        let world_position = particle.position_age.xyz
            + (right * corner.x + up * corner.y) * PARTICLE_SIZE;
        out.clip_position = frame.view_projection * vec4<f32>(world_position, 1.0);
        out.life = clamp(1.0 - life / particle.velocity_lifetime.w, 0.0, 1.0);
        out.local_corner = corner;
    }
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let edge = 1.0 - smoothstep(0.45, 1.0, length(in.local_corner));
    let alpha = in.life * edge;
    return vec4<f32>(1.0, 0.42, 0.08, alpha);
}
