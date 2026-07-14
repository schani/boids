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
            write_data(data, channels, &shared, &mut renderer, |x| x);
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
            write_data(data, channels, &shared, &mut renderer, |x| {
                (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
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
            write_data(data, channels, &shared, &mut renderer, |x| {
                ((x.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16
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
    prey_level: AtomicU32,
    predator_level: AtomicU32,
    birth_rate: AtomicU32,
    eaten_rate: AtomicU32,
}

impl SharedControls {
    fn store(&self, c: EventControls) {
        store_f32(&self.prey_level, c.prey_level);
        store_f32(&self.predator_level, c.predator_level);
        store_f32(&self.birth_rate, c.birth_rate);
        store_f32(&self.eaten_rate, c.eaten_rate);
    }

    fn load(&self) -> EventControls {
        EventControls {
            prey_level: load_f32(&self.prey_level),
            predator_level: load_f32(&self.predator_level),
            birth_rate: load_f32(&self.birth_rate),
            eaten_rate: load_f32(&self.eaten_rate),
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
    noise_state: u32,
    smooth: EventControls,
    prey_phase: f32,
    prey_octave_phase: f32,
    predator_phase: f32,
    predator_fifth_phase: f32,
    predator_lfo_phase: f32,
    birth_phase: f32,
    birth_overtone_phase: f32,
    birth_freq: f32,
    eaten_phase: f32,
    eaten_fifth_phase: f32,
    eaten_freq: f32,
    eaten_sweep: f32,
    birth_env: f32,
    eaten_env: f32,
}

impl RenderState {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate: sample_rate.max(8_000.0),
            noise_state: 0xA511_E9B3,
            smooth: EventControls::default(),
            prey_phase: 0.0,
            prey_octave_phase: 0.0,
            predator_phase: 0.0,
            predator_fifth_phase: 0.0,
            predator_lfo_phase: 0.0,
            birth_phase: 0.0,
            birth_overtone_phase: 0.0,
            birth_freq: 880.0,
            eaten_phase: 0.0,
            eaten_fifth_phase: 0.0,
            eaten_freq: 196.0,
            eaten_sweep: 1.0,
            birth_env: 0.0,
            eaten_env: 0.0,
        }
    }

    fn next_sample(&mut self, controls: EventControls) -> f32 {
        self.smooth.prey_level = smooth(self.smooth.prey_level, controls.prey_level, 0.0015);
        self.smooth.predator_level =
            smooth(self.smooth.predator_level, controls.predator_level, 0.0015);
        self.smooth.birth_rate = smooth(self.smooth.birth_rate, controls.birth_rate, 0.0025);
        self.smooth.eaten_rate = smooth(self.smooth.eaten_rate, controls.eaten_rate, 0.0025);

        let prey_freq = pick_scale_note(self.smooth.prey_level, 220.0, 2);
        let prey_amp = 0.02 + 0.16 * self.smooth.prey_level;
        self.prey_phase = advance_phase(self.prey_phase, prey_freq, self.sample_rate);
        self.prey_octave_phase =
            advance_phase(self.prey_octave_phase, prey_freq * 2.0, self.sample_rate);
        let prey_voice =
            (0.82 * self.prey_phase.sin() + 0.18 * self.prey_octave_phase.sin()) * prey_amp;

        let predator_freq = pick_scale_note(self.smooth.predator_level, 110.0, 1);
        let predator_amp = 0.01 + 0.16 * self.smooth.predator_level;
        let tremolo_hz = lerp(1.2, 3.8, self.smooth.predator_level);
        self.predator_phase = advance_phase(self.predator_phase, predator_freq, self.sample_rate);
        self.predator_fifth_phase = advance_phase(
            self.predator_fifth_phase,
            predator_freq * 1.5,
            self.sample_rate,
        );
        self.predator_lfo_phase =
            advance_phase(self.predator_lfo_phase, tremolo_hz, self.sample_rate);
        let predator_tremolo = 0.75 + 0.25 * self.predator_lfo_phase.sin().max(0.0);
        let predator_voice = (0.7 * triangle(self.predator_phase)
            + 0.3 * self.predator_fifth_phase.sin())
            * predator_amp
            * predator_tremolo;

        let birth_density_hz = 0.8 + 80.0 * self.smooth.birth_rate;
        if self.next_rand() < (birth_density_hz / self.sample_rate).clamp(0.0, 0.5) {
            self.birth_env = 1.0;
            self.birth_freq = pick_scale_note(self.next_rand(), 440.0, 3);
        }
        self.birth_env *= 0.996;
        self.birth_phase = advance_phase(self.birth_phase, self.birth_freq, self.sample_rate);
        self.birth_overtone_phase = advance_phase(
            self.birth_overtone_phase,
            self.birth_freq * 2.0,
            self.sample_rate,
        );
        let birth_voice = (0.75 * self.birth_phase.sin() + 0.25 * self.birth_overtone_phase.sin())
            * self.birth_env
            * (0.03 + 0.10 * self.smooth.birth_rate);

        let eaten_density_hz = 0.6 + 65.0 * self.smooth.eaten_rate;
        if self.next_rand() < (eaten_density_hz / self.sample_rate).clamp(0.0, 0.5) {
            self.eaten_env = 1.0;
            self.eaten_freq = pick_scale_note(self.next_rand(), 130.81, 1);
            self.eaten_sweep = 1.0;
        }
        self.eaten_env *= 0.994;
        self.eaten_sweep *= 0.9992;
        let eaten_freq_now = self.eaten_freq * (0.75 + 0.25 * self.eaten_sweep);
        self.eaten_phase = advance_phase(self.eaten_phase, eaten_freq_now, self.sample_rate);
        self.eaten_fifth_phase = advance_phase(
            self.eaten_fifth_phase,
            eaten_freq_now * 1.5,
            self.sample_rate,
        );
        let eaten_voice = (0.8 * triangle(self.eaten_phase) + 0.2 * self.eaten_fifth_phase.sin())
            * self.eaten_env
            * (0.02 + 0.09 * self.smooth.eaten_rate);

        let mix = 0.9 * prey_voice + 0.7 * predator_voice + 0.8 * birth_voice + 0.7 * eaten_voice;
        (mix * 1.5).tanh() * 0.55
    }

    fn next_rand(&mut self) -> f32 {
        self.noise_state ^= self.noise_state << 13;
        self.noise_state ^= self.noise_state >> 17;
        self.noise_state ^= self.noise_state << 5;
        (self.noise_state as f32) / (u32::MAX as f32)
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

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn triangle(phase: f32) -> f32 {
    (2.0 / PI) * phase.sin().asin()
}

fn pick_scale_note(level: f32, base_hz: f32, octaves: usize) -> f32 {
    // Major pentatonic intervals for a cheerful/consonant palette.
    const SCALE: [f32; 5] = [1.0, 9.0 / 8.0, 5.0 / 4.0, 3.0 / 2.0, 5.0 / 3.0];
    let octave_count = octaves.max(1);
    let steps = SCALE.len() * octave_count;
    let idx = (level.clamp(0.0, 1.0) * (steps.saturating_sub(1) as f32)).round() as usize;
    let octave = idx / SCALE.len();
    let scale_idx = idx % SCALE.len();
    base_hz * SCALE[scale_idx] * 2.0f32.powi(octave as i32)
}
