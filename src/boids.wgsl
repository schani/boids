const FLAG_PREDATOR: u32 = 1u;
const FLAG_ALIVE: u32 = 2u;

const WORKGROUP_SIZE: u32 = 256u;

struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
  life: f32,
  species: u32,
  flags: u32,
  _pad: u32,
};

struct Params {
  world_size: vec2<f32>,
  cell_size: f32,
  radius: f32,
  start_life: f32,
  separation_coeff: f32,
  wall_distance: f32,
  wall_factor: f32,
  predator_distance: f32,
  predator_food_gain: f32,
  prey_avoidance_bonus: f32,
  max_velocity: f32,
  predator_speed_bonus: f32,
  predator_life_mult: f32,
  prey_to_predator_mutation_denom: u32,
  _pad_after_mutation: u32,
  grid_size: vec2<u32>,
  capacity: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read> boid_in: array<Boid>;
@group(0) @binding(2) var<storage, read_write> boid_out: array<Boid>;
@group(0) @binding(3) var<storage, read_write> grid_counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> grid_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> grid_offsets_write: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> grid_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> dead_free: array<u32>;
@group(0) @binding(8) var<storage, read_write> dead_free_count: atomic<i32>;
@group(0) @binding(9) var<storage, read_write> dead_new: array<u32>;
@group(0) @binding(10) var<storage, read_write> dead_new_count: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> spawn_list: array<Boid>;
@group(0) @binding(12) var<storage, read_write> spawn_count: atomic<u32>;
@group(0) @binding(13) var<storage, read_write> species_counts: array<atomic<u32>>;

fn normalize_or_zero(v: vec2<f32>) -> vec2<f32> {
  let len = length(v);
  if (len > 0.0001) {
    return v / len;
  }
  return vec2(0.0, 0.0);
}

fn clamp_cell(coord: i32, max_val: i32) -> u32 {
  if (coord < 0) { return 0u; }
  if (coord > max_val) { return u32(max_val); }
  return u32(coord);
}

fn hash_u32(x: u32) -> u32 {
  var h = x;
  h = (h ^ 61u) ^ (h >> 16u);
  h = h * 9u;
  h = h ^ (h >> 4u);
  h = h * 0x27d4eb2du;
  h = h ^ (h >> 15u);
  return h;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_dead_new(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u) {
    atomicStore(&dead_new_count, 0u);
    atomicStore(&spawn_count, 0u);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn clear_grid_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let cell_count = params.grid_size.x * params.grid_size.y;
  if (idx >= cell_count) { return; }
  atomicStore(&grid_counts[idx], 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn build_grid_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_in[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) { return; }

  let cx = clamp_cell(i32(b.pos.x / params.cell_size), i32(params.grid_size.x) - 1);
  let cy = clamp_cell(i32(b.pos.y / params.cell_size), i32(params.grid_size.y) - 1);
  let cell = cx + cy * params.grid_size.x;
  atomicAdd(&grid_counts[cell], 1u);
}

@compute @workgroup_size(1)
fn scan_grid_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  let cell_count = params.grid_size.x * params.grid_size.y;
  var sum = 0u;
  for (var i = 0u; i < cell_count; i = i + 1u) {
    let c = atomicLoad(&grid_counts[i]);
    grid_offsets[i] = sum;
    sum = sum + c;
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn copy_grid_offsets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let cell_count = params.grid_size.x * params.grid_size.y;
  if (idx >= cell_count) { return; }
  atomicStore(&grid_offsets_write[idx], grid_offsets[idx]);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn scatter_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_in[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) { return; }

  let cx = clamp_cell(i32(b.pos.x / params.cell_size), i32(params.grid_size.x) - 1);
  let cy = clamp_cell(i32(b.pos.y / params.cell_size), i32(params.grid_size.y) - 1);
  let cell = cx + cy * params.grid_size.x;
  let offset = atomicAdd(&grid_offsets_write[cell], 1u);
  grid_indices[offset] = idx;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn update_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_in[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) {
    boid_out[idx] = b;
    return;
  }

  let predator = (b.flags & FLAG_PREDATOR) != 0u;

  let life_for_motion = select(b.life, min(b.life, params.start_life), predator);
  var target_velocity = params.max_velocity;
  if (predator) {
    target_velocity = target_velocity * params.predator_speed_bonus;
  }
  target_velocity = target_velocity
    * (1.0 + (life_for_motion - params.start_life) / params.start_life / 2.0);

  var next_vel = b.vel;
  let vlen = length(b.vel);
  next_vel = next_vel + normalize_or_zero(b.vel) * (target_velocity - vlen) * 0.2;

  var align_steer = vec2(0.0, 0.0);
  var cohesion_steer = vec2(0.0, 0.0);
  var separation_steer = vec2(0.0, 0.0);
  var num_friends = 0u;
  var num_prey = 0u;
  var num_prey_eaten = 0u;
  var was_eaten = false;

  let cx = i32(b.pos.x / params.cell_size);
  let cy = i32(b.pos.y / params.cell_size);

  for (var oy = -1i; oy <= 1i; oy = oy + 1i) {
    for (var ox = -1i; ox <= 1i; ox = ox + 1i) {
      let nx = cx + ox;
      let ny = cy + oy;
      if (nx < 0 || ny < 0) { continue; }
      if (nx >= i32(params.grid_size.x) || ny >= i32(params.grid_size.y)) { continue; }
      let cell = u32(nx) + u32(ny) * params.grid_size.x;
      let start = grid_offsets[cell];
      let count = atomicLoad(&grid_counts[cell]);
      for (var i = 0u; i < count; i = i + 1u) {
        let other_idx = grid_indices[start + i];
        if (other_idx == idx) { continue; }
        let other = boid_in[other_idx];
        if ((other.flags & FLAG_ALIVE) == 0u) { continue; }

        let diff = b.pos - other.pos;
        let dist = length(diff);
        if (dist >= params.radius || dist <= 0.0001) { continue; }

        if (predator) {
          if ((other.flags & FLAG_PREDATOR) == 0u) {
            if (dist < params.predator_distance) {
              num_prey_eaten = num_prey_eaten + 1u;
            }
            num_prey = num_prey + 1u;
            cohesion_steer = cohesion_steer + other.pos;
          }
        } else {
          let inv = 1.0 / dist;
          var steer = diff * (params.separation_coeff * inv * inv);
          if ((other.flags & FLAG_PREDATOR) != 0u) {
            if (dist < params.predator_distance) {
              was_eaten = true;
            }
            steer = steer * params.prey_avoidance_bonus;
          }
          separation_steer = separation_steer + steer;

          if ((other.flags & FLAG_PREDATOR) == 0u && other.species == b.species) {
            align_steer = align_steer + other.vel;
            cohesion_steer = cohesion_steer + other.pos;
            num_friends = num_friends + 1u;
          }
        }
      }
    }
  }

  next_vel = next_vel + separation_steer;

  if (!predator && num_friends > 0u) {
    let inv = 1.0 / f32(num_friends);
    align_steer = align_steer * inv;
    let align = align_steer - b.vel;
    next_vel = next_vel + normalize_or_zero(align) / 4.0;

    cohesion_steer = cohesion_steer * inv;
    let cohesion = cohesion_steer - b.pos;
    next_vel = next_vel + normalize_or_zero(cohesion) / 4.0;
  }

  if (predator && num_prey > 0u) {
    let inv = 1.0 / f32(num_prey);
    cohesion_steer = cohesion_steer * inv;
    let cohesion = cohesion_steer - b.pos;
    next_vel = next_vel + normalize_or_zero(cohesion) / 2.0;
  }

  if (b.pos.x < params.wall_distance) {
    next_vel.x = next_vel.x + (params.wall_factor / max(b.pos.x, 0.001));
  }
  if (b.pos.y < params.wall_distance) {
    next_vel.y = next_vel.y + (params.wall_factor / max(b.pos.y, 0.001));
  }
  if (b.pos.x > params.world_size.x - params.wall_distance) {
    next_vel.x = next_vel.x + (params.wall_factor / (b.pos.x - params.world_size.x));
  }
  if (b.pos.y > params.world_size.y - params.wall_distance) {
    next_vel.y = next_vel.y + (params.wall_factor / (b.pos.y - params.world_size.y));
  }

  if (b.pos.x < 0.0) {
    next_vel.x = abs(next_vel.x);
  } else if (b.pos.x > params.world_size.x) {
    next_vel.x = -abs(next_vel.x);
  }
  if (b.pos.y < 0.0) {
    next_vel.y = abs(next_vel.y);
  } else if (b.pos.y > params.world_size.y) {
    next_vel.y = -abs(next_vel.y);
  }

  var next_life = b.life;
  var alive_out = true;
  if (predator) {
    next_life = next_life - 1.0 + f32(num_prey_eaten) * params.predator_food_gain;
  } else {
    if (was_eaten) {
      alive_out = false;
    } else if (num_friends < 5u || num_friends > 19u) {
      next_life = next_life - 1.0;
    } else {
      next_life = next_life + 1.0;
    }
  }

  if (next_life <= 0.0) {
    alive_out = false;
  }

  var out_flags = b.flags;
  if (!alive_out) {
    out_flags = out_flags & ~FLAG_ALIVE;
    let slot = atomicAdd(&dead_new_count, 1u);
    dead_new[slot] = idx;
  }

  let next_pos = b.pos + next_vel;
  boid_out[idx] = Boid(next_pos, next_vel, next_life, b.species, out_flags, 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reproduce_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_out[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) { return; }
  let threshold = select(
    params.start_life * 2.0,
    params.start_life * 2.0 * params.predator_life_mult,
    (b.flags & FLAG_PREDATOR) != 0u
  );
  if (b.life <= threshold) { return; }

  let half_life = b.life * 0.5;
  boid_out[idx] = Boid(b.pos, b.vel, half_life, b.species, b.flags, 0u);

  let slot = atomicAdd(&spawn_count, 1u);
  if (slot < params.capacity) {
    var child_species = b.species;
    var child_flags = b.flags;
    if ((b.flags & FLAG_PREDATOR) == 0u) {
      let seed =
        idx ^ slot ^
        bitcast<u32>(b.pos.x) ^ bitcast<u32>(b.pos.y) ^
        bitcast<u32>(b.vel.x) ^ bitcast<u32>(b.vel.y) ^
        bitcast<u32>(half_life);
      let denom = max(1u, params.prey_to_predator_mutation_denom);
      if ((hash_u32(seed) % denom) == 0u) {
        child_flags = (child_flags | FLAG_PREDATOR);
        child_species = 0u;
      }
    }
    spawn_list[slot] = Boid(
      b.pos + vec2(3.0, 3.0),
      b.vel,
      half_life,
      child_species,
      child_flags,
      0u
    );
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_spawns(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = min(atomicLoad(&spawn_count), params.capacity);
  if (idx >= count) { return; }
  let spawn = spawn_list[idx];

  let old = atomicAdd(&dead_free_count, -1);
  if (old > 0) {
    let spawn_index = dead_free[u32(old - 1)];
    boid_out[spawn_index] = spawn;
  } else {
    atomicAdd(&dead_free_count, 1);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn merge_dead(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = atomicLoad(&dead_new_count);
  if (idx >= count) { return; }
  let slot = atomicAdd(&dead_free_count, 1);
  if (slot >= 0) {
    dead_free[u32(slot)] = dead_new[idx];
  }
}

@compute @workgroup_size(32)
fn clear_species_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < 2u) {
    atomicStore(&species_counts[gid.x], 0u);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn count_species(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_out[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) { return; }

  if ((b.flags & FLAG_PREDATOR) != 0u) {
    atomicAdd(&species_counts[1u], 1u);
  } else {
    atomicAdd(&species_counts[0u], 1u);
  }
}

struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) alive: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VsOut {
  let b = boid_in[iid];
  let alive = (b.flags & FLAG_ALIVE) != 0u;
  if (!alive) {
    return VsOut(vec4(2.0, 2.0, 0.0, 1.0), vec3(0.0, 0.0, 0.0), 0.0);
  }

  var local: vec2<f32>;
  if (vid == 0u) {
    local = vec2(20.0, 0.0);
  } else if (vid == 1u) {
    local = vec2(-7.0, 7.0);
  } else {
    local = vec2(-7.0, -7.0);
  }

  let v = b.vel;
  let vlen = length(v);
  let dir = select(vec2(1.0, 0.0), v / vlen, vlen > 0.0001);
  let sinv = dir.y;
  let cosv = dir.x;

  let life_for_size = select(b.life, min(b.life, params.start_life), (b.flags & FLAG_PREDATOR) != 0u);
  var scale = 0.5 + 0.5 * (life_for_size / 600.0);
  if ((b.flags & FLAG_PREDATOR) != 0u) {
    scale = scale * 2.0;
  }

  let rotated = vec2(
    local.x * cosv - local.y * sinv,
    local.x * sinv + local.y * cosv
  ) * scale;
  let world = b.pos + rotated;
  let ndc = vec2(
    world.x / params.world_size.x * 2.0 - 1.0,
    1.0 - world.y / params.world_size.y * 2.0
  );

  var color = vec3(0.639, 0.659, 0.310);
  let s = b.species % 5u;
  if (s == 1u) {
    color = vec3(1.0, 0.443, 0.553);
  } else if (s == 2u) {
    color = vec3(0.161, 0.804, 1.0);
  } else if (s == 3u) {
    color = vec3(0.259, 0.898, 0.878);
  } else if (s == 4u) {
    color = vec3(0.478, 0.475, 1.0);
  }
  if ((b.flags & FLAG_PREDATOR) != 0u) {
    color = vec3(1.0, 0.267, 0.267);
  }

  return VsOut(vec4(ndc, 0.0, 1.0), color, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  if (in.alive < 0.5) {
    discard;
  }
  return vec4(in.color, 1.0);
}

struct GraphIn {
  @location(0) pos: vec2<f32>,
  @location(1) color: vec3<f32>,
};

struct GraphOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_graph(in: GraphIn) -> GraphOut {
  return GraphOut(vec4(in.pos, 0.0, 1.0), in.color);
}

@fragment
fn fs_graph(in: GraphOut) -> @location(0) vec4<f32> {
  return vec4(in.color, 1.0);
}
