use crate::audio::EventControls;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct AudioEngine {
    _stream: cpal::Stream,
    shared: Arc<SharedControls>,
}

impl AudioEngine {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "no default output audio device found".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("failed to query default output config: {e}"))?;
        let config = supported.config();
        let sample_rate = config.sample_rate.0 as f32;
        let channels = config.channels as usize;

        let shared = Arc::new(SharedControls::default());
        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => {
                build_stream_f32(&device, &config, sample_rate, channels, Arc::clone(&shared))
            }
            cpal::SampleFormat::I16 => {
                build_stream_i16(&device, &config, sample_rate, channels, Arc::clone(&shared))
            }
            cpal::SampleFormat::U16 => {
                build_stream_u16(&device, &config, sample_rate, channels, Arc::clone(&shared))
            }
            other => {
                return Err(format!("unsupported audio sample format: {other:?}"));
            }
        }
        .map_err(|e| format!("failed to build audio output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("failed to start audio output stream: {e}"))?;

        Ok(Self {
            _stream: stream,
            shared,
        })
    }

    pub fn update_controls(&self, controls: EventControls) {
        self.shared.store(controls);
    }
}

fn build_stream_f32(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_rate: f32,
    channels: usize,
    shared: Arc<SharedControls>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let mut renderer = RenderState::new(sample_rate);
    device.build_output_stream(
        config,
        move |data: &mut [f32], _| {
            write_data(data, channels, &shared, &mut renderer, |s| s);
        },
        move |err| {
            eprintln!("audio stream error: {err}");
        },
        None,
    )
}

fn build_stream_i16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_rate: f32,
    channels: usize,
    shared: Arc<SharedControls>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let mut renderer = RenderState::new(sample_rate);
    device.build_output_stream(
        config,
        move |data: &mut [i16], _| {
            write_data(data, channels, &shared, &mut renderer, |s| {
                (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
            });
        },
        move |err| {
            eprintln!("audio stream error: {err}");
        },
        None,
    )
}

fn build_stream_u16(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_rate: f32,
    channels: usize,
    shared: Arc<SharedControls>,
) -> Result<cpal::Stream, cpal::BuildStreamError> {
    let mut renderer = RenderState::new(sample_rate);
    device.build_output_stream(
        config,
        move |data: &mut [u16], _| {
            write_data(data, channels, &shared, &mut renderer, |s| {
                ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16
            });
        },
        move |err| {
            eprintln!("audio stream error: {err}");
        },
        None,
    )
}

fn write_data<T, F>(
    output: &mut [T],
    channels: usize,
    shared: &SharedControls,
    renderer: &mut RenderState,
    mut convert: F,
) where
    T: Copy,
    F: FnMut(f32) -> T,
{
    let controls = shared.load();
    for frame in output.chunks_mut(channels.max(1)) {
        let sample = renderer.next_sample(controls);
        let out = convert(sample);
        for chan in frame {
            *chan = out;
        }
    }
}

#[derive(Default)]
struct SharedControls {
    birth: AtomicU32,
    death: AtomicU32,
    stress: AtomicU32,
    imbalance: AtomicU32,
    s_repro_split: AtomicU32,
    s_mutation: AtomicU32,
    s_color_shift: AtomicU32,
    s_birth_dropped: AtomicU32,
    s_prey_eaten: AtomicU32,
    s_prey_lonely: AtomicU32,
    s_prey_crowd: AtomicU32,
    s_pred_starve: AtomicU32,
    s_old_age: AtomicU32,
}

impl SharedControls {
    fn store(&self, c: EventControls) {
        store_f32(&self.birth, c.birth);
        store_f32(&self.death, c.death);
        store_f32(&self.stress, c.stress);
        store_f32(&self.imbalance, c.imbalance);
        store_f32(&self.s_repro_split, c.s_repro_split);
        store_f32(&self.s_mutation, c.s_mutation);
        store_f32(&self.s_color_shift, c.s_color_shift);
        store_f32(&self.s_birth_dropped, c.s_birth_dropped);
        store_f32(&self.s_prey_eaten, c.s_prey_eaten);
        store_f32(&self.s_prey_lonely, c.s_prey_lonely);
        store_f32(&self.s_prey_crowd, c.s_prey_crowd);
        store_f32(&self.s_pred_starve, c.s_pred_starve);
        store_f32(&self.s_old_age, c.s_old_age);
    }

    fn load(&self) -> EventControls {
        EventControls {
            birth: load_f32(&self.birth),
            death: load_f32(&self.death),
            stress: load_f32(&self.stress),
            imbalance: load_f32(&self.imbalance),
            s_repro_split: load_f32(&self.s_repro_split),
            s_mutation: load_f32(&self.s_mutation),
            s_color_shift: load_f32(&self.s_color_shift),
            s_birth_dropped: load_f32(&self.s_birth_dropped),
            s_prey_eaten: load_f32(&self.s_prey_eaten),
            s_prey_lonely: load_f32(&self.s_prey_lonely),
            s_prey_crowd: load_f32(&self.s_prey_crowd),
            s_pred_starve: load_f32(&self.s_pred_starve),
            s_old_age: load_f32(&self.s_old_age),
        }
    }
}

fn store_f32(dst: &AtomicU32, value: f32) {
    dst.store(value.to_bits(), Ordering::Relaxed);
}

fn load_f32(src: &AtomicU32) -> f32 {
    f32::from_bits(src.load(Ordering::Relaxed))
}

struct RenderState {
    sample_rate: f32,
    birth_lp: OnePoleLowpass,
    death_lp: OnePoleLowpass,
    noise_state: u32,
    drop_cooldown_sec: f32,
    smooth: EventControls,
    phase_birth_a: f32,
    phase_birth_b: f32,
    phase_rumble: f32,
    phase_mut_carrier: f32,
    phase_mut_mod: f32,
    phase_eaten: f32,
    phase_lonely: f32,
    phase_crowd: f32,
    phase_old_age: f32,
    phase_drop: f32,
    env_eaten: f32,
    env_lonely: f32,
    env_crowd: f32,
    env_old_age: f32,
    env_drop: f32,
}

impl RenderState {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            birth_lp: OnePoleLowpass::default(),
            death_lp: OnePoleLowpass::default(),
            noise_state: 0x9E37_79B9,
            drop_cooldown_sec: 0.0,
            smooth: EventControls::default(),
            phase_birth_a: 0.0,
            phase_birth_b: 0.0,
            phase_rumble: 0.0,
            phase_mut_carrier: 0.0,
            phase_mut_mod: 0.0,
            phase_eaten: 0.0,
            phase_lonely: 0.0,
            phase_crowd: 0.0,
            phase_old_age: 0.0,
            phase_drop: 0.0,
            env_eaten: 0.0,
            env_lonely: 0.0,
            env_crowd: 0.0,
            env_old_age: 0.0,
            env_drop: 0.0,
        }
    }

    fn next_sample(&mut self, controls: EventControls) -> f32 {
        self.smooth.birth = smooth(self.smooth.birth, controls.birth, 0.0015);
        self.smooth.death = smooth(self.smooth.death, controls.death, 0.0015);
        self.smooth.stress = smooth(self.smooth.stress, controls.stress, 0.0015);
        self.smooth.imbalance = smooth(self.smooth.imbalance, controls.imbalance, 0.0015);
        self.smooth.s_repro_split =
            smooth(self.smooth.s_repro_split, controls.s_repro_split, 0.0015);
        self.smooth.s_mutation = smooth(self.smooth.s_mutation, controls.s_mutation, 0.0015);
        self.smooth.s_color_shift =
            smooth(self.smooth.s_color_shift, controls.s_color_shift, 0.0015);
        self.smooth.s_birth_dropped = smooth(
            self.smooth.s_birth_dropped,
            controls.s_birth_dropped,
            0.0015,
        );
        self.smooth.s_prey_eaten = smooth(self.smooth.s_prey_eaten, controls.s_prey_eaten, 0.0015);
        self.smooth.s_prey_lonely =
            smooth(self.smooth.s_prey_lonely, controls.s_prey_lonely, 0.0015);
        self.smooth.s_prey_crowd = smooth(self.smooth.s_prey_crowd, controls.s_prey_crowd, 0.0015);
        self.smooth.s_pred_starve =
            smooth(self.smooth.s_pred_starve, controls.s_pred_starve, 0.0015);
        self.smooth.s_old_age = smooth(self.smooth.s_old_age, controls.s_old_age, 0.0015);

        let pitch_ratio = 2.0f32.powf((7.0 * self.smooth.imbalance) / 12.0);
        let birth_freq_a = 110.0 * pitch_ratio;
        let birth_freq_b = 220.0 * pitch_ratio * (1.004 + 0.01 * self.smooth.s_color_shift);
        self.phase_birth_a = advance_phase(self.phase_birth_a, birth_freq_a, self.sample_rate);
        self.phase_birth_b = advance_phase(self.phase_birth_b, birth_freq_b, self.sample_rate);
        let birth_raw = self.phase_birth_a.sin() * 0.72 + self.phase_birth_b.sin() * 0.28;
        let birth_cutoff = 1200.0 + 5000.0 * self.smooth.birth;
        let birth_amp = 0.05 + 0.30 * self.smooth.birth;
        let birth_voice = self
            .birth_lp
            .process(birth_raw, birth_cutoff, self.sample_rate)
            * birth_amp;

        let noise = self.next_noise();
        let death_cutoff = 1800.0 - 1400.0 * self.smooth.death;
        let death_noise = self.death_lp.process(noise, death_cutoff, self.sample_rate);
        let rumble_hz = 35.0 + 120.0 * self.smooth.s_pred_starve;
        self.phase_rumble = advance_phase(self.phase_rumble, rumble_hz, self.sample_rate);
        let death_rumble = self.phase_rumble.sin();
        let death_amp = 0.03 + 0.40 * self.smooth.death;
        let death_voice = (0.74 * death_noise + 0.26 * death_rumble) * death_amp;

        let eaten_rand = self.next_rand();
        let eaten_layer = event_layer(
            self.sample_rate,
            eaten_rand,
            self.smooth.s_prey_eaten,
            4.0,
            80.0,
            1300.0,
            &mut self.phase_eaten,
            &mut self.env_eaten,
            0.06,
            0.16,
        );
        let lonely_rand = self.next_rand();
        let lonely_layer = event_layer(
            self.sample_rate,
            lonely_rand,
            self.smooth.s_prey_lonely,
            1.0,
            24.0,
            280.0,
            &mut self.phase_lonely,
            &mut self.env_lonely,
            0.11,
            0.10,
        );
        let crowd_rand = self.next_rand();
        let crowd_layer = event_layer(
            self.sample_rate,
            crowd_rand,
            self.smooth.s_prey_crowd,
            2.0,
            40.0,
            760.0,
            &mut self.phase_crowd,
            &mut self.env_crowd,
            0.08,
            0.12,
        );
        let old_age_rand = self.next_rand();
        let old_age_layer = event_layer(
            self.sample_rate,
            old_age_rand,
            self.smooth.s_old_age,
            0.5,
            10.0,
            180.0,
            &mut self.phase_old_age,
            &mut self.env_old_age,
            0.20,
            0.14,
        );
        let texture = eaten_layer + lonely_layer + crowd_layer + old_age_layer;

        let mod_hz = 80.0 + 40.0 * self.smooth.s_mutation;
        let carrier_hz = 250.0 + 40.0 * self.smooth.s_color_shift;
        self.phase_mut_mod = advance_phase(self.phase_mut_mod, mod_hz, self.sample_rate);
        self.phase_mut_carrier =
            advance_phase(self.phase_mut_carrier, carrier_hz, self.sample_rate);
        let fm_index = 0.2 + 6.0 * self.smooth.s_mutation;
        let mut_voice = (self.phase_mut_carrier + self.phase_mut_mod.sin() * fm_index).sin()
            * (0.02 + 0.12 * (self.smooth.s_mutation + self.smooth.s_color_shift) * 0.5);

        self.drop_cooldown_sec = (self.drop_cooldown_sec - 1.0 / self.sample_rate).max(0.0);
        if self.smooth.s_birth_dropped > 0.15 && self.drop_cooldown_sec <= 0.0 {
            self.env_drop = 1.0;
            self.drop_cooldown_sec = 0.25;
        }
        self.phase_drop = advance_phase(
            self.phase_drop,
            1800.0 + 600.0 * self.smooth.s_birth_dropped,
            self.sample_rate,
        );
        self.env_drop *= 0.992;
        let drop_voice = (self.phase_drop.sin() * 0.45 + self.next_noise() * 0.55)
            * self.env_drop
            * (0.15 + 0.35 * self.smooth.s_birth_dropped);

        let mixed = 0.35 * birth_voice
            + 0.45 * death_voice
            + 0.25 * texture
            + 0.2 * mut_voice
            + 0.35 * drop_voice;
        (mixed * 1.4).tanh() * 0.7
    }

    fn next_rand(&mut self) -> f32 {
        self.noise_state ^= self.noise_state << 13;
        self.noise_state ^= self.noise_state >> 17;
        self.noise_state ^= self.noise_state << 5;
        (self.noise_state as f32) / (u32::MAX as f32)
    }

    fn next_noise(&mut self) -> f32 {
        self.next_rand() * 2.0 - 1.0
    }
}

#[derive(Default)]
struct OnePoleLowpass {
    z: f32,
}

impl OnePoleLowpass {
    fn process(&mut self, input: f32, cutoff_hz: f32, sample_rate: f32) -> f32 {
        let c = cutoff_hz.clamp(20.0, sample_rate * 0.45);
        let a = (-2.0 * PI * c / sample_rate).exp();
        self.z = (1.0 - a) * input + a * self.z;
        self.z
    }
}

fn advance_phase(phase: f32, hz: f32, sr: f32) -> f32 {
    let mut next = phase + (2.0 * PI * hz.max(0.0) / sr.max(1.0));
    if next > 2.0 * PI {
        next -= 2.0 * PI;
    }
    next
}

fn smooth(current: f32, target: f32, amount: f32) -> f32 {
    current + (target - current) * amount.clamp(0.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn event_layer(
    sample_rate: f32,
    rand_unit: f32,
    activity: f32,
    base_density: f32,
    density_span: f32,
    tone_hz: f32,
    phase: &mut f32,
    env: &mut f32,
    env_decay: f32,
    gain: f32,
) -> f32 {
    let density = base_density + density_span * activity;
    let trigger_p = (density / sample_rate).clamp(0.0, 0.5);
    if rand_unit < trigger_p {
        *env = 1.0;
    }
    let decay = (-1.0 / (env_decay.max(0.01) * sample_rate)).exp();
    *env *= decay;
    *phase = advance_phase(*phase, tone_hz, sample_rate);
    (*phase).sin() * *env * gain
}
