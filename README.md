# Boids Simulation

A browser-based predator-prey boids simulation implementing Craig Reynolds' flocking algorithm with extensions for multi-species interaction and life cycles.

## Rust + wgpu (GPU-only)

This repo now includes a native Rust + wgpu implementation that runs **both simulation and rendering on the GPU** (no mouse input, no CPU boid math).

Run it with:

```bash
cargo run --release
```

### Headless population runs

The native simulator can run the same GPU compute passes without creating a window or audio
device. It reports the five prey species and predators at frame 0, every sampling interval, and
the final frame:

```bash
cargo run --release -- --headless --frames 5000 --sample-every 100 --seed 1
```

CSV is written to standard output, while run metadata and timing are written to standard error,
so results can be redirected directly into analysis tools:

```bash
cargo run --release -- --headless --frames 20000 --sample-every 250 --seed 42 > populations.csv
```

Use `--format jsonl` for streaming JSON Lines output. `--initial-count N` supports smaller smoke
tests or density experiments (up to the simulation capacity of 75,000). Run
`cargo run --release -- --headless --help` for the complete option list. A fixed seed reproduces
the initial population; exact trajectories can still vary across GPU models because parallel
floating-point and atomic operation ordering is hardware-dependent.

Headless output also includes each behavioral kind (`standard`, `pulse`, `courier`, and `warden`)
plus the instantaneous numbers of panicked prey and charging wardens. Initial mixes can be isolated
or combined with `--pulse-ratio`, `--courier-ratio`, and `--warden-ratio`. For example, this runs a
Courier-only experiment with 0.2% of prey acting as Couriers:

```bash
cargo run --release -- --headless --frames 5000 --sample-every 100 \
  --pulse-ratio 0 --courier-ratio 0.002 --warden-ratio 0
```

### Emergent prey kinds

Three uncommon prey kinds add large-scale motion without adding a simulation pass or spatial data
structure. They reuse the neighbor checks already performed for flocking:

- **Pulse** (gold, breathing triangles) continuously alternates between attracting and repelling
  nearby boids, producing expanding and contracting pockets in a flock.
- **Panic Courier** (long magenta triangles) emits a short-lived panic signal after spotting a
  predator. Signal strength falls by one at each hop, so alarm fronts travel through nearby prey
  and then dissipate instead of becoming a permanent global state.
- **Warden** (wide green triangles) charges a nearby predator when another Warden is present. A
  charging Warden glows brighter, cannot be eaten while the pair holds, and repels predators,
  producing local pursuit reversals and defensive fronts.

The default initial prey mix is 1% Pulse, 0.2% Courier, and 8% Warden. Offspring inherit their
parent’s kind; prey that mutate into predators become standard predators.

#### Measured cost and population behavior

Optimized Metal runs on an Apple M2 Max used 32,000 initial boids, 300 frames, and three seeds.
In the candidate-isolation sweep, the no-kind control averaged 699.7 frames/s. Pulse-only averaged
696.5 frames/s (-0.46%), Courier-only 705.2 frames/s (+0.79%, within run-to-run noise), and
Warden-only 689.2 frames/s (-1.50%). A final matched run after all tuning averaged 705.9 frames/s
for the control and 704.7 frames/s for the combined mix, a 0.17% reduction. All kinds share the
existing neighborhood pass; the only extra population cost is one atomic kind counter per prey.

In a 2,000-frame seed-42 comparison, the control peaked at 52,429 total boids and ended at 9,028
with 925 predators. The combined mix peaked earlier at 44,293 and ended at 7,311 with 40
predators. Wardens grew from 8% of initial prey to 51% of surviving prey as paired defense became
an evolutionary advantage. Sampled panic activity peaked at 27,013 during high predator density
and fell to zero after the predator collapse, showing that Courier alarms dissipate when their
source disappears.

## Overview

This simulation models a self-regulating ecosystem with two types of agents:

- **Prey** (5 colored species) — Follow classic boids flocking rules
- **Predators** (sharks) — Hunt prey using modified cohesion behavior

The simulation runs on a 1000×1000 pixel canvas representing a 5000×5000 unit world, starting with approximately 4975 prey and 25 predators.

## Files

| File | Description |
|------|-------------|
| `boid.js` | Boid class with behavior rules and life cycle |
| `script.js` | Simulation engine, spatial partitioning, animation loop |
| `vector.js` | 2D vector math utilities |
| `graph.js` | Population graph visualization |
| `index.html` | Canvas setup and script loading |

## The Boids Model

### Constants

```
startLife            = 300    // Initial life for all boids
separationCoefficient = 3     // Separation force multiplier
wallDistance         = 150    // Wall avoidance activation zone
wallFactor           = 30     // Wall repulsion strength
predatorDistance     = 10     // Kill/eat range
predatorFoodGain     = 30     // Life gained per prey consumed
preyAvoidanceBonus   = 10     // Multiplier for predator avoidance
MAX_VELOCITY         = 5      // Base target speed
predatorSpeedBonus   = 1.9    // Predator speed multiplier
radius               = 100    // Neighborhood detection radius
```

### Neighborhood Detection

Boids only interact with neighbors within a radius of 100 units. Spatial partitioning divides the world into grid cells, and each boid checks only its own cell plus the 8 adjacent cells (3×3 neighborhood).

---

## Prey Behavior

Each frame, prey boids compute steering forces from several rules, then update velocity and position.

### 1. Speed Regulation

Target velocity scales with life (healthier boids move faster):

```
v_target = v_base × (1 + (life - 300) / 600)
```

Which simplifies to:

```
v_target = v_base × (0.5 + life / 600)
```

The velocity is smoothly adjusted toward the target:

```
Δv = v_target - |v|
v_next = v + normalize(v) × Δv × 0.2
```

### 2. Separation

Steer away from all nearby boids with force inversely proportional to distance:

```
For each neighbor i within radius:
    d = p_self - p_i                              // vector pointing away from neighbor
    s = normalize(d) × (separationCoefficient / |d|)

    If neighbor is predator:
        s = s × preyAvoidanceBonus                // 10× stronger avoidance

    separationSteer += s

v_next += separationSteer
```

### 3. Alignment

Match velocity with same-colored neighbors:

```
avgVelocity = Σ(v_i) / n                          // average velocity of same-color neighbors
alignDelta = avgVelocity - v_self
v_next += normalize(alignDelta) / 4
```

### 4. Cohesion

Move toward center of mass of same-colored neighbors:

```
centerOfMass = Σ(p_i) / n                         // center of same-color neighbors
cohesionDelta = centerOfMass - p_self
v_next += normalize(cohesionDelta) / 4
```

### 5. Wall Avoidance

When within `wallDistance` (150 units) of a boundary, apply inverse-distance repulsion:

```
If x < wallDistance:
    v_next.x += wallFactor / x

If y < wallDistance:
    v_next.y += wallFactor / y

If x > width - wallDistance:
    v_next.x += wallFactor / (x - width)          // negative, pushes left

If y > height - wallDistance:
    v_next.y += wallFactor / (y - height)         // negative, pushes up
```

### 6. Boundary Reflection

Hard boundary enforcement if boid escapes:

```
If x < 0:       v_next.x = |v_next.x|
If x > width:   v_next.x = -|v_next.x|
If y < 0:       v_next.y = |v_next.y|
If y > height:  v_next.y = -|v_next.y|
```

### 7. Life Cycle

**Life changes per frame:**
```
If 5 ≤ numSameColorNeighbors ≤ 19:
    life += 1                                     // thriving in stable group
Else:
    life -= 1                                     // starving (isolated or overcrowded)
```

**Death:**
- Eaten if within `predatorDistance` (10 units) of a predator
- Starvation if `life ≤ 0`

**Reproduction:**
```
If life > startLife × 2 (600):
    Split into two boids, each with life / 2
    Offspring spawns at position offset by (3, 3)
```

---

## Predator Behavior

Predators use a simplified rule set focused on hunting.

### 1. Speed

Predators move 1.9× faster than prey:

```
v_target = MAX_VELOCITY × predatorSpeedBonus = 5 × 1.9 = 9.5
```

(Also scales with life like prey)

### 2. Prey Seeking (Cohesion)

Move toward center of mass of all nearby prey (stronger than prey cohesion):

```
preyCenter = Σ(p_prey) / numPrey
huntDelta = preyCenter - p_self
v_next += normalize(huntDelta) / 2               // /2 vs /4 for prey = 2× stronger
```

### 3. Separation from Other Predators

Predators do not apply separation to prey (they want to get close), but the separation calculation runs — effectively predators separate from each other.

### 4. Life Cycle

**Life changes per frame:**
```
life += -1 + (numPreyEaten × predatorFoodGain)
life += -1 + (numPreyEaten × 30)
```

A predator loses 1 life per frame and must eat to survive. Each prey consumed within `predatorDistance` (10 units) grants 30 life.

**Death:** `life ≤ 0`

**Reproduction:** Same as prey — split when `life > 600`

---

## Position Update

After all steering forces are computed:

```
v = v_next
p = p + v
```

---

## Rendering

Boids are drawn as triangles pointing in the direction of velocity:

```
scale = (0.5 + life / 600)                        // size scales with health
If predator: scale × 2

Triangle vertices (rotated by velocity angle):
    Front:  (20 × scale, 0)
    Back:   (-7 × scale, ±7 × scale)
```

Colors:
- Prey: `#A3A84F` (olive), `#FF718D` (pink), `#29CDFF` (cyan), `#42E5E0` (turquoise), `#7A79FF` (purple)
- Predators: `#FF4444` (red)

---

## Performance Optimizations

1. **Spatial partitioning** — O(n) neighbor lookup instead of O(n²)
2. **Batch rendering** — All boids of same color drawn in single path
3. **Adaptive frame rate** — If frame time exceeds 20ms, simulation runs 2× steps per frame
4. **Canvas alpha disabled** — `{ alpha: false }` for faster compositing

---

## Emergent Behavior

The simulation produces classic predator-prey population dynamics:

1. Prey reproduce when in stable flocks (5-19 neighbors)
2. Predator population grows when prey are abundant
3. Overhunting causes prey scarcity
4. Predators starve, population crashes
5. Prey recover, cycle repeats

The population graph displays these oscillations in real-time.

---

## References

- Reynolds, C. W. (1987). "Flocks, herds and schools: A distributed behavioral model." *SIGGRAPH '87*.
