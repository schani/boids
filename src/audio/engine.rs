use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

pub struct AudioEngine {
    _stream: cpal::Stream,
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

        let stream = match supported.sample_format() {
            cpal::SampleFormat::F32 => build_stream(&device, &config, 0.0f32),
            cpal::SampleFormat::I16 => build_stream(&device, &config, 0i16),
            cpal::SampleFormat::U16 => build_stream(&device, &config, u16::MAX / 2),
            other => {
                return Err(format!("unsupported audio sample format: {other:?}"));
            }
        }
        .map_err(|e| format!("failed to build audio output stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("failed to start audio output stream: {e}"))?;

        Ok(Self { _stream: stream })
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    silence: T,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + Copy + Send + 'static,
{
    device.build_output_stream(
        config,
        move |output: &mut [T], _| output.fill(silence),
        move |err| eprintln!("audio stream error: {err}"),
        None,
    )
}
