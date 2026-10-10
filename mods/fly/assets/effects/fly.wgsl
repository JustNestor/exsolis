enable wgpu_binding_array;

// Engine-owned declarations (see `ENGINE_*_WGSL` in assets.rs):
//   engine/frame    FrameData + `frame` (group 1, binding 1)
//   engine/bindless `samplers` / `textures` binding arrays (group 0)
//   engine/world    block data around the camera (group 1, binding 24)
//!import engine/frame
//!import engine/bindless
//!import engine/world

struct FlyAction {
    strength: f32,
    position: vec3<f32>,
}

struct Particle {
    position_age: vec4<f32>,
    velocity_lifetime: vec4<f32>,
}

// Layout of a DrawIndirectCommand. Atomics let every alive particle append
// itself without a separate counting pass.
struct DrawArgs {
    vertex_count: atomic<u32>,
    instance_count: atomic<u32>,
    first_vertex: atomic<u32>,
    first_instance: atomic<u32>,
}

const TEX_SPARK: u32 = 0u;
const VERTICES_PER_PARTICLE: u32 = 6u;
const PARTICLE_SIZE: f32 = 0.035;

@group(1) @binding(0) var<storage, read> action: FlyAction;
@group(1) @binding(2) var<storage, read_write> particles: array<Particle>;
// Non-persistent buffers are zeroed every frame, so `args` starts at
// instance_count = 0 and `alive` starts empty without any reset pass.
@group(1) @binding(3) var<storage, read_write> alive: array<u32>;
@group(1) @binding(4) var<storage, read_write> args: DrawArgs;

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
        particles[index] = particle;
        return;
    }

    var position = particle.position_age.xyz;
    var velocity = particle.velocity_lifetime.xyz;
    velocity.y = velocity.y - 0.9 * dt;
    let next = position + velocity * dt;

    // Sweep one axis at a time against the world: sparks slide along walls and
    // bounce off the ground. Blocks that aren't loaded never count as solid.
    if (!world_solid_at(vec3<f32>(next.x, position.y, position.z))) {
        position.x = next.x;
    } else {
        velocity.x = -velocity.x * 0.4;
    }
    if (!world_solid_at(vec3<f32>(position.x, next.y, position.z))) {
        position.y = next.y;
    } else {
        velocity.y = -velocity.y * 0.35;
        velocity.x = velocity.x * 0.8;
        velocity.z = velocity.z * 0.8;
    }
    if (!world_solid_at(vec3<f32>(position.x, position.y, next.z))) {
        position.z = next.z;
    } else {
        velocity.z = -velocity.z * 0.4;
    }

    particle.position_age = vec4<f32>(position, age);
    particle.velocity_lifetime = vec4<f32>(velocity, particle.velocity_lifetime.w);
    particles[index] = particle;

    // Append to the alive list; the draw reads its size from `args`.
    let slot = atomicAdd(&args.instance_count, 1u);
    alive[slot] = index;
    atomicStore(&args.vertex_count, VERTICES_PER_PARTICLE);
}

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
    // `instance_index` counts only live particles; `alive` maps it back.
    let particle = particles[alive[instance_index]];
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
    // corner is [-1, 1] with +y up; texture v grows downward.
    let uv = vec2<f32>(in.local_corner.x, -in.local_corner.y) * 0.5 + vec2<f32>(0.5, 0.5);
    let spark = textureSample(
        textures[frame.textures[TEX_SPARK]],
        samplers[frame.sampler_linear],
        uv,
    );
    // Additive blend is SrcAlpha/One, so alpha scales the colour.
    return vec4<f32>(spark.rgb * vec3<f32>(1.0, 0.42, 0.08), spark.a * in.life);
}
