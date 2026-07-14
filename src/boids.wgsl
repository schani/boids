const FLAG_PREDATOR: u32 = 1u;
const FLAG_ALIVE: u32 = 2u;
const FLAG_WARDEN_CHARGING: u32 = 4u;

const KIND_STANDARD: u32 = 0u;
const KIND_WEAVER: u32 = 1u;
const KIND_COURIER: u32 = 2u;
const KIND_WARDEN: u32 = 3u;
const PREY_KIND_COUNT: u32 = 4u;

const PANIC_SHIFT: u32 = 8u;
const PANIC_MASK: u32 = 0x0000ff00u;
const PANIC_MAX: u32 = 12u;
const DEPLETION_REASON_MASK: u32 = 0x000000ffu;
const WARDEN_CHARGE_FORCE: f32 = 2.2;
const WARDEN_REPEL_FORCE: f32 = 45.0;
const WARDEN_CHARGE_LIFE_COST: f32 = 2.0;
const DIVERSITY_MIN_NEIGHBORS: u32 = 6u;
const DIVERSITY_DOMINANCE_COST: f32 = 1.25;
const DIVERSITY_RARITY_BONUS: f32 = 0.35;
const KIND_SHIFT_DENOM: u32 = 2048u;
const FIELD_GRID_SIZE: u32 = 200u;
const FIELD_CELL_COUNT: u32 = FIELD_GRID_SIZE * FIELD_GRID_SIZE;
const FIELD_DEPOSIT_SCALE: f32 = 1024.0;
const TRAIL_FOLLOW_FORCE: f32 = 0.42;

const WORKGROUP_SIZE: u32 = 256u;
const CAUSE_REPRODUCTION_SPLIT: u32 = 0u;
const CAUSE_PREY_TO_PREDATOR_MUTATION: u32 = 1u;
const CAUSE_BIRTH_DROPPED: u32 = 2u;
const CAUSE_PREY_EATEN: u32 = 3u;
const CAUSE_PREY_LONELINESS: u32 = 4u;
const CAUSE_PREY_OVERCROWDING: u32 = 5u;
const CAUSE_PREDATOR_STARVATION: u32 = 6u;
const CAUSE_OLD_AGE: u32 = 7u;
const CAUSE_PREY_COLOR_SHIFT: u32 = 8u;
const DEATH_NONE: u32 = 0u;
const DEATH_PREY_EATEN: u32 = 1u;
const DEATH_PREY_LONELINESS: u32 = 2u;
const DEATH_PREY_OVERCROWDING: u32 = 3u;
const DEATH_PREDATOR_STARVATION: u32 = 4u;
const DEATH_OLD_AGE: u32 = 5u;
const MUTATION_FLAG: u32 = 0x80000000u;
const COLOR_SHIFT_FLAG: u32 = 0x40000000u;
const PREY_LIFETIME_MIN: u32 = 1000u;
const PREY_LIFETIME_RANGE: u32 = 1001u;
const PREDATOR_LIFETIME_MIN: u32 = 500u;
const PREDATOR_LIFETIME_RANGE: u32 = 501u;
const PREY_SPECIES_COUNT: u32 = 5u;
const PREDATOR_POPULATION_INDEX: u32 = PREY_SPECIES_COUNT;
const KIND_POPULATION_START: u32 = PREDATOR_POPULATION_INDEX + 1u;
const PANICKED_POPULATION_INDEX: u32 = KIND_POPULATION_START + PREY_KIND_COUNT;
const CHARGING_WARDEN_POPULATION_INDEX: u32 = PANICKED_POPULATION_INDEX + 1u;
const POPULATION_SERIES: u32 = CHARGING_WARDEN_POPULATION_INDEX + 1u;

struct Boid {
  pos: vec2<f32>,
  vel: vec2<f32>,
  life: f32,
  lifetime: u32,
  species: u32,
  flags: u32,
  _pad: u32,
  kind: u32,
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
  prey_to_predator_mutation_denom: u32,
  loneliness_enabled: u32,
  overcrowding_enabled: u32,
  old_age_enabled: u32,
  prey_gain_min_neighbors: u32,
  prey_gain_max_neighbors: u32,
  prey_color_shift_denom: u32,
  grid_size: vec2<u32>,
  capacity: u32,
  _pad: u32,
};

struct FieldCell {
  flow: vec2<f32>,
  strength: f32,
  _pad: f32,
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
@group(0) @binding(14) var<storage, read_write> cause_counts: array<atomic<u32>>;
@group(0) @binding(15) var<storage, read> field_state: array<FieldCell>;
@group(0) @binding(16) var<storage, read_write> field_out: array<FieldCell>;
@group(0) @binding(17) var<storage, read_write> field_deposits: array<atomic<i32>>;

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

fn field_index(x: u32, y: u32) -> u32 {
  return min(x, FIELD_GRID_SIZE - 1u) + min(y, FIELD_GRID_SIZE - 1u) * FIELD_GRID_SIZE;
}

fn field_index_for_pos(pos: vec2<f32>) -> u32 {
  let normalized = clamp(pos / params.world_size, vec2(0.0), vec2(0.999999));
  let cell = vec2<u32>(normalized * f32(FIELD_GRID_SIZE));
  return field_index(cell.x, cell.y);
}

fn trail_at(pos: vec2<f32>) -> FieldCell {
  return field_state[field_index_for_pos(pos)];
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

fn random_lifetime(seed: u32, predator: bool) -> u32 {
  let h = hash_u32(seed);
  let prey_lifetime = PREY_LIFETIME_MIN + (h % PREY_LIFETIME_RANGE);
  let predator_lifetime = PREDATOR_LIFETIME_MIN + (h % PREDATOR_LIFETIME_RANGE);
  return select(prey_lifetime, predator_lifetime, predator);
}

fn panic_level(packed: u32) -> u32 {
  return (packed & PANIC_MASK) >> PANIC_SHIFT;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reset_dead_new(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u) {
    atomicStore(&dead_new_count, 0u);
    atomicStore(&spawn_count, 0u);
    atomicStore(&cause_counts[CAUSE_REPRODUCTION_SPLIT], 0u);
    atomicStore(&cause_counts[CAUSE_PREY_TO_PREDATOR_MUTATION], 0u);
    atomicStore(&cause_counts[CAUSE_BIRTH_DROPPED], 0u);
    atomicStore(&cause_counts[CAUSE_PREY_EATEN], 0u);
    atomicStore(&cause_counts[CAUSE_PREY_LONELINESS], 0u);
    atomicStore(&cause_counts[CAUSE_PREY_OVERCROWDING], 0u);
    atomicStore(&cause_counts[CAUSE_PREDATOR_STARVATION], 0u);
    atomicStore(&cause_counts[CAUSE_OLD_AGE], 0u);
    atomicStore(&cause_counts[CAUSE_PREY_COLOR_SHIFT], 0u);
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
  let predator = (b.flags & FLAG_PREDATOR) != 0u;
  let field_cell = field_index_for_pos(b.pos) * 4u;
  if (!predator && (b.kind % PREY_KIND_COUNT) == KIND_WEAVER) {
    let heading = normalize_or_zero(b.vel);
    atomicAdd(&field_deposits[field_cell], i32(heading.x * FIELD_DEPOSIT_SCALE));
    atomicAdd(&field_deposits[field_cell + 1u], i32(heading.y * FIELD_DEPOSIT_SCALE));
    atomicAdd(&field_deposits[field_cell + 2u], i32(FIELD_DEPOSIT_SCALE));
  } else if (predator) {
    // Predators punch temporary holes in routes. A flock visibly streams around
    // those breaks instead of blindly following a stale trail into danger.
    atomicAdd(&field_deposits[field_cell + 3u], i32(FIELD_DEPOSIT_SCALE));
  }
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
fn clear_trail_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= FIELD_CELL_COUNT) { return; }
  let deposit = idx * 4u;
  atomicStore(&field_deposits[deposit], 0i);
  atomicStore(&field_deposits[deposit + 1u], 0i);
  atomicStore(&field_deposits[deposit + 2u], 0i);
  atomicStore(&field_deposits[deposit + 3u], 0i);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn update_trail_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= FIELD_CELL_COUNT) { return; }
  let x = idx % FIELD_GRID_SIZE;
  let y = idx / FIELD_GRID_SIZE;
  let center = field_state[idx];
  let left = field_state[field_index(x - min(x, 1u), y)].flow;
  let right = field_state[field_index(min(x + 1u, FIELD_GRID_SIZE - 1u), y)].flow;
  let up = field_state[field_index(x, y - min(y, 1u))].flow;
  let down = field_state[field_index(x, min(y + 1u, FIELD_GRID_SIZE - 1u))].flow;
  let neighbor_flow = (left + right + up + down) * 0.25;

  let deposit = idx * 4u;
  let deposited_flow = vec2(
    f32(atomicLoad(&field_deposits[deposit])),
    f32(atomicLoad(&field_deposits[deposit + 1u]))
  ) / FIELD_DEPOSIT_SCALE;
  let weaver_presence = clamp(
    f32(atomicLoad(&field_deposits[deposit + 2u])) / FIELD_DEPOSIT_SCALE,
    0.0,
    4.0
  );
  let predator_break = clamp(
    f32(atomicLoad(&field_deposits[deposit + 3u])) / FIELD_DEPOSIT_SCALE,
    0.0,
    3.0
  );

  // Directional wakes diffuse just enough to connect consecutive positions,
  // then fade quickly enough that every visible route is recent boid history.
  var flow = mix(center.flow, neighbor_flow, 0.075) * 0.975;
  flow = flow + deposited_flow * (0.32 + weaver_presence * 0.06);
  let route_before_break = length(flow);
  flow = flow * (1.0 - 0.58 * min(predator_break, 1.0));
  let strength = min(length(flow), 4.0);
  let visible_break = predator_break * smoothstep(0.04, 0.22, route_before_break);
  field_out[idx] = FieldCell(flow, strength, visible_break);
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
  let kind = select(b.kind % PREY_KIND_COUNT, KIND_STANDARD, predator);
  let panic_in = select(panic_level(b._pad), 0u, predator);

  let life_for_motion = select(b.life, min(b.life, params.start_life), predator);
  var target_velocity = params.max_velocity;
  if (predator) {
    target_velocity = target_velocity * params.predator_speed_bonus;
  } else if (panic_in > 0u) {
    target_velocity = target_velocity * 1.35;
  }
  target_velocity = target_velocity
    * (1.0 + (life_for_motion - params.start_life) / params.start_life / 2.0);

  var next_vel = b.vel;
  let vlen = length(b.vel);
  next_vel = next_vel + normalize_or_zero(b.vel) * (target_velocity - vlen) * 0.2;
  let trail = trail_at(b.pos);
  if (!predator && trail.strength > 0.04) {
    let follow = normalize_or_zero(trail.flow);
    let alignment = max(0.2, dot(normalize_or_zero(b.vel), follow));
    let kind_response = select(1.0, 0.45, kind == KIND_WEAVER);
    next_vel = next_vel + follow * min(trail.strength, 1.5)
      * TRAIL_FOLLOW_FORCE * alignment * kind_response;
  }

  var align_steer = vec2(0.0, 0.0);
  var cohesion_steer = vec2(0.0, 0.0);
  var separation_steer = vec2(0.0, 0.0);
  var warden_repulsion = vec2(0.0, 0.0);
  var predator_center = vec2(0.0, 0.0);
  var num_friends = 0u;
  var num_prey = 0u;
  var num_predators = 0u;
  var num_wardens = 0u;
  var num_prey_neighbors = 0u;
  var num_same_kind = 0u;
  var num_prey_eaten = 0u;
  var next_panic = 0u;
  if (panic_in > 0u) {
    next_panic = panic_in - 1u;
  }
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

        let other_predator = (other.flags & FLAG_PREDATOR) != 0u;
        if (predator) {
          if (!other_predator) {
            let charging_warden = (other.flags & FLAG_WARDEN_CHARGING) != 0u;
            if (dist < params.predator_distance && !charging_warden) {
              num_prey_eaten = num_prey_eaten + 1u;
            }
            num_prey = num_prey + 1u;
            cohesion_steer = cohesion_steer + other.pos;
            if (charging_warden) {
              let inv = 1.0 / dist;
              warden_repulsion = warden_repulsion
                + diff * (WARDEN_REPEL_FORCE * inv * inv);
            }
          }
        } else {
          let inv = 1.0 / dist;
          var steer = diff * (params.separation_coeff * inv * inv);
          if (other_predator) {
            if (dist < params.predator_distance) {
              was_eaten = true;
            }
            steer = steer * params.prey_avoidance_bonus;
            predator_center = predator_center + other.pos;
            num_predators = num_predators + 1u;
          } else {
            num_prey_neighbors = num_prey_neighbors + 1u;
            if (other.kind == kind) {
              num_same_kind = num_same_kind + 1u;
            }
            let other_panic = panic_level(other._pad);
            if (other_panic > 1u) {
              next_panic = max(next_panic, other_panic - 1u);
            }
            if (other.kind == KIND_WARDEN) {
              num_wardens = num_wardens + 1u;
            }
          }
          separation_steer = separation_steer + steer;

          if (!other_predator && other.species == b.species) {
            align_steer = align_steer + other.vel;
            cohesion_steer = cohesion_steer + other.pos;
            num_friends = num_friends + 1u;
          }
        }
      }
    }
  }

  next_vel = next_vel + separation_steer;

  var warden_charging = false;
  if (!predator && kind == KIND_COURIER && num_predators > 0u) {
    next_panic = PANIC_MAX;
  }
  if (!predator && kind == KIND_WARDEN && num_predators > 0u && num_wardens > 0u) {
    predator_center = predator_center / f32(num_predators);
    next_vel = next_vel
      + normalize_or_zero(predator_center - b.pos) * WARDEN_CHARGE_FORCE;
    warden_charging = true;
    was_eaten = false;
  }

  if (!predator && num_friends > 0u) {
    let inv = 1.0 / f32(num_friends);
    align_steer = align_steer * inv;
    let align = align_steer - b.vel;
    let flock_divisor = select(4.0, 10.0, panic_in > 0u);
    next_vel = next_vel + normalize_or_zero(align) / flock_divisor;

    cohesion_steer = cohesion_steer * inv;
    let cohesion = cohesion_steer - b.pos;
    next_vel = next_vel + normalize_or_zero(cohesion) / flock_divisor;
  }

  if (predator && num_prey > 0u) {
    let inv = 1.0 / f32(num_prey);
    cohesion_steer = cohesion_steer * inv;
    let cohesion = cohesion_steer - b.pos;
    next_vel = next_vel + normalize_or_zero(cohesion) / 2.0 + warden_repulsion;
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
  let old_age_active = params.old_age_enabled != 0u;
  var next_lifetime = b.lifetime;
  if (old_age_active && b.lifetime > 0u) {
    next_lifetime = b.lifetime - 1u;
  }
  var alive_out = true;
  var death_reason = DEATH_NONE;
  var last_depletion_reason = b._pad & DEPLETION_REASON_MASK;
  let loneliness_active = params.loneliness_enabled != 0u;
  let overcrowding_active = params.overcrowding_enabled != 0u;
  let prey_gain_min_neighbors = min(params.prey_gain_min_neighbors, params.prey_gain_max_neighbors);
  let prey_gain_max_neighbors = max(params.prey_gain_min_neighbors, params.prey_gain_max_neighbors);
  if (predator) {
    next_life = next_life - 1.0 + f32(num_prey_eaten) * params.predator_food_gain;
    last_depletion_reason = DEATH_PREDATOR_STARVATION;
  } else {
    if (was_eaten) {
      alive_out = false;
      death_reason = DEATH_PREY_EATEN;
    } else if (num_friends < prey_gain_min_neighbors && loneliness_active) {
      next_life = next_life - 1.0;
      last_depletion_reason = DEATH_PREY_LONELINESS;
    } else if (num_friends > prey_gain_max_neighbors && overcrowding_active) {
      next_life = next_life - 1.0;
      last_depletion_reason = DEATH_PREY_OVERCROWDING;
    } else if (num_friends >= prey_gain_min_neighbors && num_friends <= prey_gain_max_neighbors) {
      next_life = next_life + 1.0;
    }

    // Negative frequency dependence keeps a locally dominant kind from turning its
    // temporary advantage into a permanent monoculture. Rare kinds get a smaller
    // recovery bonus, so population shares continue to move instead of locking.
    if (num_prey_neighbors >= DIVERSITY_MIN_NEIGHBORS) {
      if (num_same_kind * 3u > num_prey_neighbors * 2u) {
        next_life = next_life - DIVERSITY_DOMINANCE_COST;
      } else if (num_same_kind * 4u < num_prey_neighbors) {
        next_life = next_life + DIVERSITY_RARITY_BONUS;
      }
    }

    // Charging is powerful but metabolically expensive. Wardens win during a
    // predator surge, then yield ground once prolonged defense drains them.
    if (warden_charging) {
      next_life = next_life - WARDEN_CHARGE_LIFE_COST;
    }
  }

  if (alive_out && old_age_active && next_lifetime == 0u) {
    alive_out = false;
    death_reason = DEATH_OLD_AGE;
  }

  if (alive_out && next_life <= 0.0) {
    alive_out = false;
    if (death_reason == DEATH_NONE) {
      death_reason = select(last_depletion_reason, DEATH_PREDATOR_STARVATION, predator);
      if (death_reason == DEATH_NONE) {
        death_reason = DEATH_PREY_LONELINESS;
      }
    }
  }

  var out_flags = b.flags & ~FLAG_WARDEN_CHARGING;
  if (warden_charging) {
    out_flags = out_flags | FLAG_WARDEN_CHARGING;
  }
  if (!alive_out) {
    out_flags = out_flags & ~FLAG_ALIVE;
    let slot = atomicAdd(&dead_new_count, 1u);
    dead_new[slot] = (idx & 0x00ffffffu) | (death_reason << 24u);
  }

  let next_pos = b.pos + next_vel;
  let packed_state = last_depletion_reason
    | (next_panic << PANIC_SHIFT);
  let out_pad = select(0u, packed_state, alive_out);
  boid_out[idx] = Boid(next_pos, next_vel, next_life, next_lifetime, b.species, out_flags, out_pad, kind);
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn reproduce_boids(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.capacity) { return; }
  let b = boid_out[idx];
  if ((b.flags & FLAG_ALIVE) == 0u) { return; }
  let threshold = params.start_life * 2.0;
  if (b.life <= threshold) { return; }

  let half_life = b.life * 0.5;
  boid_out[idx] = Boid(b.pos, b.vel, half_life, b.lifetime, b.species, b.flags, b._pad, b.kind);

  let slot = atomicAdd(&spawn_count, 1u);
  if (slot < params.capacity) {
    var child_species = b.species;
    var child_flags = b.flags & ~FLAG_WARDEN_CHARGING;
    var child_pad = b._pad;
    var child_kind = b.kind;
    let seed =
      idx ^ slot ^
      bitcast<u32>(b.pos.x) ^ bitcast<u32>(b.pos.y) ^
      bitcast<u32>(b.vel.x) ^ bitcast<u32>(b.vel.y) ^
      bitcast<u32>(half_life);
    if ((b.flags & FLAG_PREDATOR) == 0u) {
      let denom = max(1u, params.prey_to_predator_mutation_denom);
      if ((hash_u32(seed) % denom) == 0u) {
        child_flags = (child_flags | FLAG_PREDATOR);
        child_species = 0u;
        child_kind = KIND_STANDARD;
        child_pad = child_pad | MUTATION_FLAG;
      } else {
        let color_roll = hash_u32(seed ^ 0xa511e9b3u);
        let color_denom = max(1u, params.prey_color_shift_denom);
        if ((color_roll % color_denom) == 0u) {
          let step = 1u + (hash_u32(color_roll ^ 0x9e3779b9u) % 4u);
          child_species = (child_species + step) % 5u;
          child_pad = child_pad | COLOR_SHIFT_FLAG;
        }
        let kind_roll = hash_u32(seed ^ 0x6c8e9cf5u);
        if ((kind_roll % KIND_SHIFT_DENOM) == 0u) {
          let kind_step = 1u + (hash_u32(kind_roll ^ 0x85ebca6bu) % (PREY_KIND_COUNT - 1u));
          child_kind = (child_kind + kind_step) % PREY_KIND_COUNT;
          child_pad = child_pad & DEPLETION_REASON_MASK;
        }
      }
    }
    let child_predator = (child_flags & FLAG_PREDATOR) != 0u;
    let child_lifetime = random_lifetime(seed ^ child_flags ^ child_species, child_predator);
    spawn_list[slot] = Boid(
      b.pos + vec2(3.0, 3.0),
      b.vel,
      half_life,
      child_lifetime,
      child_species,
      child_flags,
      child_pad,
      child_kind
    );
  } else {
    atomicAdd(&cause_counts[CAUSE_BIRTH_DROPPED], 1u);
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
    boid_out[spawn_index] = Boid(
      spawn.pos,
      spawn.vel,
      spawn.life,
      spawn.lifetime,
      spawn.species,
      spawn.flags,
      spawn._pad & ~(MUTATION_FLAG | COLOR_SHIFT_FLAG | PANIC_MASK),
      spawn.kind
    );
    atomicAdd(&cause_counts[CAUSE_REPRODUCTION_SPLIT], 1u);
    if ((spawn._pad & MUTATION_FLAG) != 0u) {
      atomicAdd(&cause_counts[CAUSE_PREY_TO_PREDATOR_MUTATION], 1u);
    }
    if ((spawn._pad & COLOR_SHIFT_FLAG) != 0u) {
      atomicAdd(&cause_counts[CAUSE_PREY_COLOR_SHIFT], 1u);
    }
  } else {
    atomicAdd(&dead_free_count, 1);
    atomicAdd(&cause_counts[CAUSE_BIRTH_DROPPED], 1u);
  }
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn merge_dead(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let count = atomicLoad(&dead_new_count);
  if (idx >= count) { return; }
  let packed = dead_new[idx];
  let death_reason = packed >> 24u;
  let dead_idx = packed & 0x00ffffffu;

  let slot = atomicAdd(&dead_free_count, 1);
  if (slot >= 0) {
    dead_free[u32(slot)] = dead_idx;
  }

  switch death_reason {
    case DEATH_PREY_EATEN: {
      atomicAdd(&cause_counts[CAUSE_PREY_EATEN], 1u);
    }
    case DEATH_PREY_LONELINESS: {
      atomicAdd(&cause_counts[CAUSE_PREY_LONELINESS], 1u);
    }
    case DEATH_PREY_OVERCROWDING: {
      atomicAdd(&cause_counts[CAUSE_PREY_OVERCROWDING], 1u);
    }
    case DEATH_PREDATOR_STARVATION: {
      atomicAdd(&cause_counts[CAUSE_PREDATOR_STARVATION], 1u);
    }
    case DEATH_OLD_AGE: {
      atomicAdd(&cause_counts[CAUSE_OLD_AGE], 1u);
    }
    default: {
    }
  }
}

@compute @workgroup_size(32)
fn clear_species_counts(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < POPULATION_SERIES) {
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
    atomicAdd(&species_counts[PREDATOR_POPULATION_INDEX], 1u);
  } else {
    atomicAdd(&species_counts[b.species % PREY_SPECIES_COUNT], 1u);
    atomicAdd(&species_counts[KIND_POPULATION_START + (b.kind % PREY_KIND_COUNT)], 1u);
    if (panic_level(b._pad) > 0u) {
      atomicAdd(&species_counts[PANICKED_POPULATION_INDEX], 1u);
    }
    if ((b.flags & FLAG_WARDEN_CHARGING) != 0u) {
      atomicAdd(&species_counts[CHARGING_WARDEN_POPULATION_INDEX], 1u);
    }
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

  let predator = (b.flags & FLAG_PREDATOR) != 0u;
  let kind = select(b.kind % PREY_KIND_COUNT, KIND_STANDARD, predator);
  var local: vec2<f32>;
  if (vid == 0u) {
    local = vec2(20.0, 0.0);
  } else if (vid == 1u) {
    local = vec2(-7.0, 7.0);
  } else {
    local = vec2(-7.0, -7.0);
  }
  if (kind == KIND_COURIER) {
    local.x = select(-9.0, 30.0, vid == 0u);
    local.y = select(select(-4.0, 4.0, vid == 1u), 0.0, vid == 0u);
  } else if (kind == KIND_WARDEN) {
    local.x = select(-8.0, 19.0, vid == 0u);
    local.y = select(select(-12.0, 12.0, vid == 1u), 0.0, vid == 0u);
  } else if (kind == KIND_WEAVER) {
    local.x = select(-14.0, 24.0, vid == 0u);
    local.y = select(select(-7.0, 7.0, vid == 1u), 0.0, vid == 0u);
  }

  let v = b.vel;
  let vlen = length(v);
  let dir = select(vec2(1.0, 0.0), v / vlen, vlen > 0.0001);
  let sinv = dir.y;
  let cosv = dir.x;

  let life_for_size = select(b.life, min(b.life, params.start_life), predator);
  var scale = 0.5 + 0.5 * (life_for_size / 600.0);
  if (predator) {
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
  if (kind == KIND_WEAVER) {
    color = vec3(0.08, 0.92, 1.0);
  } else if (kind == KIND_COURIER) {
    color = vec3(1.0, 0.12, 0.72);
  } else if (kind == KIND_WARDEN) {
    color = select(
      vec3(0.08, 0.72, 0.24),
      vec3(0.15, 1.0, 0.35),
      (b.flags & FLAG_WARDEN_CHARGING) != 0u
    );
  }
  let panic = panic_level(b._pad);
  if (!predator && panic > 0u) {
    let panic_strength = f32(panic) / f32(PANIC_MAX);
    color = mix(color, vec3(1.0, 0.03, 0.72), 0.22 + panic_strength * 0.35);
    if (kind == KIND_COURIER && panic == PANIC_MAX) {
      color = mix(color, vec3(1.0, 0.92, 1.0), 0.72);
    }
  }
  if (predator) {
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

struct EffectOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) phase: f32,
  @location(2) @interpolate(flat) effect: u32,
};

@vertex
fn vs_effect(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> EffectOut {
  let b = boid_in[iid];
  if ((b.flags & FLAG_ALIVE) == 0u || (b.flags & FLAG_PREDATOR) != 0u) {
    return EffectOut(vec4(2.0, 2.0, 0.0, 1.0), vec2(0.0), 0.0, 0u);
  }

  let kind = b.kind % PREY_KIND_COUNT;
  let panic = panic_level(b._pad);
  var effect = 0u;
  var extent = 1.0;
  var phase = 0.0;
  if (kind == KIND_COURIER && panic == PANIC_MAX && (hash_u32(iid) % 8u) == 0u) {
    effect = 2u;
    extent = 180.0;
    phase = f32(b.lifetime % 72u) / 72.0;
  } else if (panic > 0u && (hash_u32(iid ^ 0x9e3779b9u) % 128u) == 0u) {
    effect = 3u;
    extent = 95.0;
    phase = f32(panic) / f32(PANIC_MAX);
  }
  if (effect == 0u) {
    return EffectOut(vec4(2.0, 2.0, 0.0, 1.0), vec2(0.0), 0.0, 0u);
  }

  var local = vec2(-1.0, -1.0);
  if (vid == 1u || vid == 4u) {
    local = vec2(1.0, -1.0);
  } else if (vid == 2u || vid == 3u) {
    local = vec2(-1.0, 1.0);
  } else if (vid == 5u) {
    local = vec2(1.0, 1.0);
  }
  let world = b.pos + local * extent;
  let ndc = vec2(
    world.x / params.world_size.x * 2.0 - 1.0,
    1.0 - world.y / params.world_size.y * 2.0
  );
  return EffectOut(vec4(ndc, 0.0, 1.0), local, phase, effect);
}

@fragment
fn fs_effect(in: EffectOut) -> @location(0) vec4<f32> {
  let dist = length(in.local);
  if (dist > 1.0 || in.effect == 0u) { discard; }

  if (in.effect == 2u) {
    let ring_radius = 0.18 + (1.0 - in.phase) * 0.72;
    let ring = 1.0 - smoothstep(0.025, 0.075, abs(dist - ring_radius));
    let second_ring = 1.0 - smoothstep(0.02, 0.065, abs(dist - ring_radius * 0.55));
    return vec4(1.0, 0.05, 0.78, ring * 0.58 + second_ring * 0.22);
  }

  let ring_radius = 0.38 + (1.0 - in.phase) * 0.38;
  let ring = 1.0 - smoothstep(0.035, 0.12, abs(dist - ring_radius));
  let halo = (1.0 - smoothstep(0.0, 1.0, dist)) * in.phase * 0.09;
  return vec4(1.0, 0.02, 0.72, ring * in.phase * 0.26 + halo);
}

struct FieldOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_field(@builtin(vertex_index) vid: u32) -> FieldOut {
  var pos = vec2(-1.0, -1.0);
  if (vid == 1u || vid == 4u) {
    pos = vec2(1.0, -1.0);
  } else if (vid == 2u || vid == 3u) {
    pos = vec2(-1.0, 1.0);
  } else if (vid == 5u) {
    pos = vec2(1.0, 1.0);
  }
  let uv = vec2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
  return FieldOut(vec4(pos, 0.0, 1.0), uv);
}

fn sample_trail_field(uv: vec2<f32>) -> FieldCell {
  let grid_pos = clamp(uv, vec2(0.0), vec2(1.0)) * f32(FIELD_GRID_SIZE - 1u);
  let x0 = u32(floor(grid_pos.x));
  let y0 = u32(floor(grid_pos.y));
  let x1 = min(x0 + 1u, FIELD_GRID_SIZE - 1u);
  let y1 = min(y0 + 1u, FIELD_GRID_SIZE - 1u);
  let blend = fract(grid_pos);
  let a = field_state[field_index(x0, y0)];
  let b = field_state[field_index(x1, y0)];
  let c = field_state[field_index(x0, y1)];
  let d = field_state[field_index(x1, y1)];
  let top_flow = mix(a.flow, b.flow, blend.x);
  let bottom_flow = mix(c.flow, d.flow, blend.x);
  let strength = mix(mix(a.strength, b.strength, blend.x), mix(c.strength, d.strength, blend.x), blend.y);
  let disruption = mix(mix(a._pad, b._pad, blend.x), mix(c._pad, d._pad, blend.x), blend.y);
  return FieldCell(mix(top_flow, bottom_flow, blend.y), strength, disruption);
}

@fragment
fn fs_field(in: FieldOut) -> @location(0) vec4<f32> {
  let field = sample_trail_field(in.uv);
  let trail_strength = 1.0 - exp(-field.strength * 1.15);
  let force_dir = normalize_or_zero(field.flow);
  let base = vec3(0.008, 0.013, 0.035);
  let direction_color = mix(
    vec3(0.18, 0.32, 1.0),
    vec3(0.05, 0.95, 1.0),
    force_dir.x * 0.5 + 0.5
  );
  var color = mix(base, direction_color, trail_strength * 0.34);

  // Predator-cleared cells flash red at the actual break point, never as an
  // unrelated global overlay.
  let break_grid = fract(in.uv * vec2<f32>(FIELD_GRID_SIZE));
  let slash = 1.0 - smoothstep(0.035, 0.12, abs(break_grid.x + break_grid.y - 1.0));
  color = mix(color, vec3(1.0, 0.035, 0.08), slash * smoothstep(0.1, 0.8, field._pad) * 0.72);
  return vec4(color, 1.0);
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
