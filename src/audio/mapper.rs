const CAUSE_COUNT: usize = 9;

const IDX_REPRODUCTION_SPLIT: usize = 0;
const IDX_PREY_TO_PREDATOR_MUTATION: usize = 1;
const IDX_BIRTH_DROPPED: usize = 2;
const IDX_PREY_EATEN: usize = 3;
const IDX_PREY_LONELINESS: usize = 4;
const IDX_PREY_OVERCROWDING: usize = 5;
const IDX_PREDATOR_STARVATION: usize = 6;
const IDX_OLD_AGE: usize = 7;
const IDX_PREY_COLOR_SHIFT: usize = 8;

#[derive(Clone, Copy, Default)]
pub struct EventControls {
    pub birth: f32,
    pub death: f32,
    pub stress: f32,
    pub imbalance: f32,
    pub s_repro_split: f32,
    pub s_mutation: f32,
    pub s_color_shift: f32,
    pub s_birth_dropped: f32,
    pub s_prey_eaten: f32,
    pub s_prey_lonely: f32,
    pub s_prey_crowd: f32,
    pub s_pred_starve: f32,
    pub s_old_age: f32,
}

pub struct AudioControlMapper {
    pending_counts: [u64; CAUSE_COUNT],
    pending_dt: f32,
    smoothed: [f32; CAUSE_COUNT],
    window_sec: f32,
}

impl AudioControlMapper {
    pub fn new(window_sec: f32) -> Self {
        Self {
            pending_counts: [0; CAUSE_COUNT],
            pending_dt: 0.0,
            smoothed: [0.0; CAUSE_COUNT],
            window_sec: window_sec.max(0.01),
        }
    }

    pub fn ingest_frame(
        &mut self,
        counts: [u32; CAUSE_COUNT],
        dt_sec: f32,
    ) -> Option<EventControls> {
        let dt = dt_sec.clamp(1.0 / 500.0, 0.5);
        for (dst, src) in self.pending_counts.iter_mut().zip(counts.iter()) {
            *dst += *src as u64;
        }
        self.pending_dt += dt;

        if self.pending_dt < self.window_sec {
            return None;
        }

        let window_dt = self.pending_dt.max(1e-4);
        let mut rates = [0.0f32; CAUSE_COUNT];
        for i in 0..CAUSE_COUNT {
            rates[i] = self.pending_counts[i] as f32 / window_dt;
            let companded = compand_rate(rates[i], 8.0, 1500.0);
            self.smoothed[i] = slew(self.smoothed[i], companded, window_dt, 0.08, 0.35);
        }

        self.pending_counts = [0; CAUSE_COUNT];
        self.pending_dt = 0.0;

        let birth_rate = rates[IDX_REPRODUCTION_SPLIT]
            + rates[IDX_PREY_TO_PREDATOR_MUTATION]
            + rates[IDX_PREY_COLOR_SHIFT];
        let death_rate = rates[IDX_PREY_EATEN]
            + rates[IDX_PREY_LONELINESS]
            + rates[IDX_PREY_OVERCROWDING]
            + rates[IDX_PREDATOR_STARVATION]
            + rates[IDX_OLD_AGE];
        let birth = (self.smoothed[IDX_REPRODUCTION_SPLIT]
            + self.smoothed[IDX_PREY_TO_PREDATOR_MUTATION]
            + self.smoothed[IDX_PREY_COLOR_SHIFT])
            / 3.0;
        let death = (self.smoothed[IDX_PREY_EATEN]
            + self.smoothed[IDX_PREY_LONELINESS]
            + self.smoothed[IDX_PREY_OVERCROWDING]
            + self.smoothed[IDX_PREDATOR_STARVATION]
            + self.smoothed[IDX_OLD_AGE])
            / 5.0;
        let imbalance = (birth_rate - death_rate) / (birth_rate + death_rate + 1.0);
        let stress = (rates[IDX_PREY_LONELINESS]
            + rates[IDX_PREY_OVERCROWDING]
            + rates[IDX_PREDATOR_STARVATION])
            / (death_rate + 1.0);

        Some(EventControls {
            birth,
            death,
            stress,
            imbalance,
            s_repro_split: self.smoothed[IDX_REPRODUCTION_SPLIT],
            s_mutation: self.smoothed[IDX_PREY_TO_PREDATOR_MUTATION],
            s_color_shift: self.smoothed[IDX_PREY_COLOR_SHIFT],
            s_birth_dropped: self.smoothed[IDX_BIRTH_DROPPED],
            s_prey_eaten: self.smoothed[IDX_PREY_EATEN],
            s_prey_lonely: self.smoothed[IDX_PREY_LONELINESS],
            s_prey_crowd: self.smoothed[IDX_PREY_OVERCROWDING],
            s_pred_starve: self.smoothed[IDX_PREDATOR_STARVATION],
            s_old_age: self.smoothed[IDX_OLD_AGE],
        })
    }
}

fn compand_rate(rate: f32, knee: f32, max_rate: f32) -> f32 {
    let k = knee.max(1e-4);
    let m = max_rate.max(k + 1e-4);
    ((1.0 + rate.max(0.0) / k).ln() / (1.0 + m / k).ln()).clamp(0.0, 1.0)
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
