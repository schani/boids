const CAUSE_COUNT: usize = 9;
const IDX_REPRODUCTION_SPLIT: usize = 0;
const IDX_PREY_TO_PREDATOR_MUTATION: usize = 1;
const IDX_PREY_EATEN: usize = 3;
const IDX_PREY_COLOR_SHIFT: usize = 8;

#[derive(Clone, Copy, Default)]
pub struct EventControls {
    pub prey_level: f32,
    pub predator_level: f32,
    pub birth_rate: f32,
    pub eaten_rate: f32,
}

pub struct AudioControlMapper {
    pending_births: u64,
    pending_eaten: u64,
    pending_dt: f32,
    window_sec: f32,
    prey_count: u32,
    predator_count: u32,
    prey_capacity_hint: f32,
    predator_capacity_hint: f32,
    smoothed: EventControls,
}

impl AudioControlMapper {
    pub fn new(window_sec: f32, prey_capacity_hint: u32, predator_capacity_hint: u32) -> Self {
        Self {
            pending_births: 0,
            pending_eaten: 0,
            pending_dt: 0.0,
            window_sec: window_sec.max(0.01),
            prey_count: 0,
            predator_count: 0,
            prey_capacity_hint: prey_capacity_hint.max(1) as f32,
            predator_capacity_hint: predator_capacity_hint.max(1) as f32,
            smoothed: EventControls::default(),
        }
    }

    pub fn update_populations(&mut self, prey: u32, predators: u32) {
        self.prey_count = prey;
        self.predator_count = predators;
    }

    pub fn ingest_events(
        &mut self,
        counts: [u32; CAUSE_COUNT],
        dt_sec: f32,
    ) -> Option<EventControls> {
        let dt = dt_sec.clamp(1.0 / 500.0, 0.5);
        let births = counts[IDX_REPRODUCTION_SPLIT]
            + counts[IDX_PREY_TO_PREDATOR_MUTATION]
            + counts[IDX_PREY_COLOR_SHIFT];
        let eaten = counts[IDX_PREY_EATEN];
        self.pending_births += births as u64;
        self.pending_eaten += eaten as u64;
        self.pending_dt += dt;

        if self.pending_dt < self.window_sec {
            return None;
        }

        let window_dt = self.pending_dt.max(1e-4);
        let birth_rate_hz = self.pending_births as f32 / window_dt;
        let eaten_rate_hz = self.pending_eaten as f32 / window_dt;
        self.pending_births = 0;
        self.pending_eaten = 0;
        self.pending_dt = 0.0;

        let prey_raw = normalize_population(self.prey_count as f32, self.prey_capacity_hint);
        let pred_raw =
            normalize_population(self.predator_count as f32, self.predator_capacity_hint);
        let birth_raw = compand_rate(birth_rate_hz, 5.0, 1500.0);
        let eaten_raw = compand_rate(eaten_rate_hz, 3.0, 1200.0);

        self.smoothed.prey_level = slew(self.smoothed.prey_level, prey_raw, window_dt, 0.12, 0.25);
        self.smoothed.predator_level = slew(
            self.smoothed.predator_level,
            pred_raw,
            window_dt,
            0.12,
            0.25,
        );
        self.smoothed.birth_rate = slew(self.smoothed.birth_rate, birth_raw, window_dt, 0.08, 0.20);
        self.smoothed.eaten_rate = slew(self.smoothed.eaten_rate, eaten_raw, window_dt, 0.08, 0.20);

        Some(self.smoothed)
    }
}

fn normalize_population(value: f32, max_hint: f32) -> f32 {
    ((1.0 + value.max(0.0)).ln() / (1.0 + max_hint.max(1.0)).ln()).clamp(0.0, 1.0)
}

fn compand_rate(rate_hz: f32, knee: f32, max_rate: f32) -> f32 {
    let k = knee.max(1e-4);
    let m = max_rate.max(k + 1e-4);
    ((1.0 + rate_hz.max(0.0) / k).ln() / (1.0 + m / k).ln()).clamp(0.0, 1.0)
}

fn slew(current: f32, target: f32, dt: f32, attack_sec: f32, release_sec: f32) -> f32 {
    let tau = if target > current {
        attack_sec.max(1e-4)
    } else {
        release_sec.max(1e-4)
    };
    let alpha = 1.0 - (-dt / tau).exp();
    current + (target - current) * alpha
}
