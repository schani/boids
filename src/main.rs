mod audio;

use audio::{AudioControlMapper, AudioEngine};
use bytemuck::{Pod, Zeroable};
use egui_wgpu::ScreenDescriptor;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{collections::VecDeque, env, process::ExitCode, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WORLD_SIZE: [f32; 2] = [8000.0, 8000.0];
const BOID_CAPACITY: u32 = 75_000;
const INITIAL_COUNT: u32 = 32_000;
const PREDATOR_RATIO: f32 = 0.005; // 0.5%

const START_LIFE: f32 = 300.0;
const SEPARATION_COEFF: f32 = 3.0;
const WALL_DISTANCE: f32 = 150.0;
const WALL_FACTOR: f32 = 30.0;
const PREDATOR_DISTANCE: f32 = 10.0;
const PREDATOR_FOOD_GAIN: f32 = 30.0;
const PREY_AVOIDANCE_BONUS: f32 = 10.0;
const MAX_VELOCITY: f32 = 5.0;
const PREDATOR_SPEED_BONUS: f32 = 1.9;
const RADIUS: f32 = 100.0;
const PREY_GAIN_MIN_NEIGHBORS: u32 = 5;
const PREY_GAIN_MAX_NEIGHBORS: u32 = 20;
const PREY_COLOR_SHIFT_DENOM: u32 = 10;
const PREY_LIFETIME_MIN: u32 = 1000;
const PREY_LIFETIME_MAX: u32 = 2000;
const PREDATOR_LIFETIME_MIN: u32 = 500;
const PREDATOR_LIFETIME_MAX: u32 = 1000;

const GRAPH_SERIES: usize = 2; // combined prey + predators
const PREY_SPECIES_COUNT: usize = 5;
const GRAPH_READBACK_INTERVAL: u32 = 4;
const GRAPH_HEIGHT_PX: u32 = 200;
const GRAPH_PADDING_PX: f32 = 16.0;
const GRAPH_MAX_SCALE: u32 = 320;
const GRAPH_MAX_POINTS: usize = 8192;
const DEFAULT_MUTATION_DENOM: u32 = 1000;
const CAUSE_COUNT: usize = 9;
const CAUSE_WINDOW_FRAMES: usize = 60;
const CAUSE_REPRODUCTION_SPLIT: usize = 0;
const CAUSE_PREY_TO_PREDATOR_MUTATION: usize = 1;
const CAUSE_BIRTH_DROPPED: usize = 2;
const CAUSE_PREY_EATEN: usize = 3;
const CAUSE_PREY_LONELINESS: usize = 4;
const CAUSE_PREY_OVERCROWDING: usize = 5;
const CAUSE_PREDATOR_STARVATION: usize = 6;
const CAUSE_OLD_AGE: usize = 7;
const CAUSE_PREY_COLOR_SHIFT: usize = 8;
const AUDIO_CONTROL_WINDOW_SEC: f32 = 0.10;

const FLAG_PREDATOR: u32 = 1 << 0;
const FLAG_ALIVE: u32 = 1 << 1;

const KIND_STANDARD: u32 = 0;
const KIND_PULSE: u32 = 1;
const KIND_COURIER: u32 = 2;
const KIND_WARDEN: u32 = 3;
const PREY_KIND_COUNT: usize = 4;
const PULSE_STATE_SHIFT: u32 = 16;
const PULSE_PERIOD: u32 = 240;

const PREDATOR_POPULATION_INDEX: usize = PREY_SPECIES_COUNT;
const KIND_POPULATION_START: usize = PREDATOR_POPULATION_INDEX + 1;
const PANICKED_POPULATION_INDEX: usize = KIND_POPULATION_START + PREY_KIND_COUNT;
const CHARGING_WARDEN_POPULATION_INDEX: usize = PANICKED_POPULATION_INDEX + 1;
const POPULATION_SERIES: usize = CHARGING_WARDEN_POPULATION_INDEX + 1;

const WORKGROUP_SIZE: u32 = 256;
const HEADLESS_MAX_BATCH_FRAMES: u64 = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Boid {
    pos: [f32; 2],
    vel: [f32; 2],
    life: f32,
    lifetime: u32,
    species: u32,
    flags: u32,
    _pad: u32,
    kind: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    world_size: [f32; 2],
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
    grid_size: [u32; 2],
    capacity: u32,
    _pad: u32,
}

struct Pipelines {
    clear_grid: wgpu::ComputePipeline,
    reset_dead: wgpu::ComputePipeline,
    build_counts: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    copy_offsets: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    reproduce: wgpu::ComputePipeline,
    apply_spawns: wgpu::ComputePipeline,
    merge_dead: wgpu::ComputePipeline,
    clear_species: wgpu::ComputePipeline,
    count_species: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
    graph: wgpu::RenderPipeline,
}

struct Buffers {
    params: wgpu::Buffer,
    boids: [wgpu::Buffer; 2],
    grid_counts: wgpu::Buffer,
    grid_offsets: wgpu::Buffer,
    grid_offsets_write: wgpu::Buffer,
    grid_indices: wgpu::Buffer,
    dead_free: wgpu::Buffer,
    dead_free_count: wgpu::Buffer,
    dead_new: wgpu::Buffer,
    dead_new_count: wgpu::Buffer,
    spawn_list: wgpu::Buffer,
    spawn_count: wgpu::Buffer,
    species_counts: wgpu::Buffer,
    species_counts_read: wgpu::Buffer,
    cause_counts: wgpu::Buffer,
    cause_counts_read: wgpu::Buffer,
    graph_vertices: wgpu::Buffer,
}

struct BindGroups {
    reset_dead: wgpu::BindGroup,
    clear_grid: wgpu::BindGroup,
    scan: wgpu::BindGroup,
    copy_offsets: wgpu::BindGroup,
    merge_dead: wgpu::BindGroup,
    clear_species: wgpu::BindGroup,
    count_species: [wgpu::BindGroup; 2],
    build_counts: [wgpu::BindGroup; 2],
    scatter: [wgpu::BindGroup; 2],
    update: [wgpu::BindGroup; 2],
    reproduce: [wgpu::BindGroup; 2],
    apply_spawns: [wgpu::BindGroup; 2],
    render: [wgpu::BindGroup; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GraphVertex {
    pos: [f32; 2],
    color: [f32; 3],
    _pad: f32,
}

struct GraphState {
    history: [Vec<f32>; GRAPH_SERIES],
    scale: [u32; GRAPH_SERIES],
    acc: [f32; GRAPH_SERIES],
    acc_count: [u32; GRAPH_SERIES],
    frame: u32,
    series_max: [f32; GRAPH_SERIES],
    vertex_counts: [u32; GRAPH_SERIES],
}

struct CauseState {
    history: VecDeque<[u32; CAUSE_COUNT]>,
    sums: [u64; CAUSE_COUNT],
    all_time_max: u64,
}

impl CauseState {
    fn push_frame(&mut self, frame: [u32; CAUSE_COUNT]) {
        if self.history.len() == CAUSE_WINDOW_FRAMES {
            if let Some(oldest) = self.history.pop_front() {
                for i in 0..CAUSE_COUNT {
                    self.sums[i] = self.sums[i].saturating_sub(oldest[i] as u64);
                }
            }
        }

        self.history.push_back(frame);
        for i in 0..CAUSE_COUNT {
            self.sums[i] += frame[i] as u64;
        }
        let frame_peak = self.sums.iter().copied().max().unwrap_or(0);
        self.all_time_max = self.all_time_max.max(frame_peak);
    }
}

impl Default for CauseState {
    fn default() -> Self {
        Self {
            history: VecDeque::with_capacity(CAUSE_WINDOW_FRAMES),
            sums: [0; CAUSE_COUNT],
            all_time_max: 1,
        }
    }
}

#[derive(Copy, Clone)]
struct GraphHover {
    prey: Option<f32>,
    predators: Option<f32>,
    x_ui: f32,
    y_ui: f32,
}

#[derive(Clone)]
struct FrameBreakdown {
    compute_ms: f32,
    sim_render_ms: f32,
    graph_render_ms: f32,
    ui_ms: f32,
    readback_ms: f32,
    submit_present_ms: f32,
    total_ms: f32,
}

impl Default for FrameBreakdown {
    fn default() -> Self {
        Self {
            compute_ms: 0.0,
            sim_render_ms: 0.0,
            graph_render_ms: 0.0,
            ui_ms: 0.0,
            readback_ms: 0.0,
            submit_present_ms: 0.0,
            total_ms: 0.0,
        }
    }
}

struct UiState {
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    fps_smoothed: f32,
    timings: FrameBreakdown,
}

struct State {
    surface: Option<wgpu::Surface<'static>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: Option<wgpu::SurfaceConfiguration>,
    size: PhysicalSize<u32>,
    grid_cells: u32,
    current: usize,
    pipelines: Pipelines,
    _buffers: Buffers,
    bind_groups: BindGroups,
    graph: GraphState,
    causes: CauseState,
    audio: Option<AudioEngine>,
    audio_mapper: AudioControlMapper,
    audio_last_tick: Instant,
    params_cpu: Params,
    ui: Option<UiState>,
    cursor_pos_px: Option<[f32; 2]>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum HeadlessFormat {
    Csv,
    Jsonl,
}

#[derive(Clone, Debug, PartialEq)]
struct HeadlessOptions {
    frames: u64,
    sample_every: u64,
    seed: u64,
    initial_count: u32,
    mix: InitialMix,
    format: HeadlessFormat,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct InitialMix {
    pulse_ratio: f32,
    courier_ratio: f32,
    warden_ratio: f32,
}

impl Default for InitialMix {
    fn default() -> Self {
        Self {
            pulse_ratio: 0.01,
            courier_ratio: 0.002,
            warden_ratio: 0.08,
        }
    }
}

impl Default for HeadlessOptions {
    fn default() -> Self {
        Self {
            frames: 5_000,
            sample_every: 100,
            seed: 1,
            initial_count: INITIAL_COUNT,
            mix: InitialMix::default(),
            format: HeadlessFormat::Csv,
        }
    }
}

enum RunMode {
    Gui,
    Headless(HeadlessOptions),
    Help,
}

impl State {
    async fn new(
        window: Option<&winit::window::Window>,
        initial_count: u32,
        seed: u64,
        mix: InitialMix,
    ) -> Self {
        assert!(initial_count <= BOID_CAPACITY);
        let size = window
            .map(winit::window::Window::inner_size)
            .unwrap_or_else(|| PhysicalSize::new(1, 1));
        let instance = wgpu::Instance::default();
        let surface = window.map(|window| {
            let surface = instance.create_surface(window).expect("surface");
            unsafe { std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface) }
        });
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
        {
            Some(adapter) => adapter,
            None => {
                let fallback = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        compatible_surface: None,
                        force_fallback_adapter: window.is_none(),
                    })
                    .await;

                match fallback {
                    Some(adapter) if window.is_none() => adapter,
                    Some(adapter) => {
                        let info = adapter.get_info();
                        panic!(
                            "No adapter compatible with the window surface. Found adapter: {:?}. \
                             This often means the process has no WindowServer/Metal access (e.g. \
                             headless or sandboxed). Try running in a normal desktop session.",
                            info
                        );
                    }
                    None => {
                        panic!(
                            "No GPU adapters found at all. This usually means Metal is unavailable \
                             or the process is running without graphics access."
                        );
                    }
                }
            }
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("device");

        let (surface_format, config) = if let Some(surface) = &surface {
            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps.formats[0];
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width: size.width,
                height: size.height,
                present_mode: surface_caps.present_modes[0],
                desired_maximum_frame_latency: 2,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![surface_format],
            };
            surface.configure(&device, &config);
            (surface_format, Some(config))
        } else {
            (wgpu::TextureFormat::Rgba8UnormSrgb, None)
        };
        if window.is_none() {
            let info = adapter.get_info();
            eprintln!("Headless GPU adapter: {} ({:?})", info.name, info.backend);
        }

        let cell_size = RADIUS;
        let grid_x = (WORLD_SIZE[0] / cell_size).ceil() as u32;
        let grid_y = (WORLD_SIZE[1] / cell_size).ceil() as u32;
        let grid_cells = grid_x * grid_y;
        // Single-thread scan in WGSL handles any grid size.

        let params = Params {
            world_size: WORLD_SIZE,
            cell_size,
            radius: RADIUS,
            start_life: START_LIFE,
            separation_coeff: SEPARATION_COEFF,
            wall_distance: WALL_DISTANCE,
            wall_factor: WALL_FACTOR,
            predator_distance: PREDATOR_DISTANCE,
            predator_food_gain: PREDATOR_FOOD_GAIN,
            prey_avoidance_bonus: PREY_AVOIDANCE_BONUS,
            max_velocity: MAX_VELOCITY,
            predator_speed_bonus: PREDATOR_SPEED_BONUS,
            prey_to_predator_mutation_denom: DEFAULT_MUTATION_DENOM,
            loneliness_enabled: 0,
            overcrowding_enabled: 0,
            old_age_enabled: 1,
            prey_gain_min_neighbors: PREY_GAIN_MIN_NEIGHBORS,
            prey_gain_max_neighbors: PREY_GAIN_MAX_NEIGHBORS,
            prey_color_shift_denom: PREY_COLOR_SHIFT_DENOM,
            grid_size: [grid_x, grid_y],
            capacity: BOID_CAPACITY,
            _pad: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let initial_boids = create_initial_boids(initial_count, seed, mix);
        let boid_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_a"),
            size: (std::mem::size_of::<Boid>() as u64) * BOID_CAPACITY as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&boid_a, 0, bytemuck::cast_slice(&initial_boids));

        let boid_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boids_b"),
            size: (std::mem::size_of::<Boid>() as u64) * BOID_CAPACITY as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_counts"),
            size: (grid_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_offsets = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_offsets"),
            size: (grid_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_offsets_write = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_offsets_write"),
            size: (grid_cells as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let grid_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_indices"),
            size: (BOID_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dead_free = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dead_free"),
            size: (BOID_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dead_new = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dead_new"),
            size: (BOID_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spawn_list = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spawn_list"),
            size: (std::mem::size_of::<Boid>() as u64) * BOID_CAPACITY as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dead_free_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dead_free_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dead_new_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dead_new_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spawn_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spawn_count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let species_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("species_counts"),
            size: (POPULATION_SERIES as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let species_counts_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("species_counts_read"),
            size: (POPULATION_SERIES as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cause_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cause_counts"),
            size: (CAUSE_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cause_counts_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cause_counts_read"),
            size: (CAUSE_COUNT as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let graph_vertex_capacity =
            (GRAPH_SERIES as u64) * ((GRAPH_MAX_POINTS as u64 - 1) * 2) as u64;
        let graph_vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("graph_vertices"),
            size: graph_vertex_capacity * std::mem::size_of::<GraphVertex>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (dead_free_list, dead_free_count_value) = create_dead_free_list(initial_count);
        queue.write_buffer(&dead_free, 0, bytemuck::cast_slice(&dead_free_list));
        queue.write_buffer(
            &dead_free_count,
            0,
            bytemuck::bytes_of(&(dead_free_count_value as i32)),
        );
        queue.write_buffer(&dead_new_count, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&spawn_count, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&cause_counts, 0, bytemuck::cast_slice(&[0u32; CAUSE_COUNT]));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("boids.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("boids.wgsl").into()),
        });

        let reset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reset_bgl"),
            entries: &[
                storage_entry(10, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(12, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(14, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let clear_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("build_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scan_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(5, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(5, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(6, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("update_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(6, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(9, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(10, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let reproduce_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reproduce_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(11, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(12, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(14, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("apply_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(7, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(8, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(11, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(12, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(14, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let merge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge_bgl"),
            entries: &[
                storage_entry(9, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(10, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(7, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(8, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(14, false, wgpu::ShaderStages::COMPUTE),
            ],
        });

        let clear_species_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clear_species_bgl"),
            entries: &[storage_entry(13, false, wgpu::ShaderStages::COMPUTE)],
        });
        let count_species_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("count_species_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(13, false, wgpu::ShaderStages::COMPUTE),
            ],
        });

        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_bgl"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::VERTEX),
                storage_entry(1, true, wgpu::ShaderStages::VERTEX),
            ],
        });

        let reset_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reset_layout"),
            bind_group_layouts: &[&reset_bgl],
            push_constant_ranges: &[],
        });
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear_layout"),
            bind_group_layouts: &[&clear_bgl],
            push_constant_ranges: &[],
        });
        let build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("build_layout"),
            bind_group_layouts: &[&build_bgl],
            push_constant_ranges: &[],
        });
        let scan_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scan_layout"),
            bind_group_layouts: &[&scan_bgl],
            push_constant_ranges: &[],
        });
        let copy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("copy_layout"),
            bind_group_layouts: &[&copy_bgl],
            push_constant_ranges: &[],
        });
        let scatter_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_layout"),
            bind_group_layouts: &[&scatter_bgl],
            push_constant_ranges: &[],
        });
        let update_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("update_layout"),
            bind_group_layouts: &[&update_bgl],
            push_constant_ranges: &[],
        });
        let reproduce_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reproduce_layout"),
            bind_group_layouts: &[&reproduce_bgl],
            push_constant_ranges: &[],
        });
        let apply_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("apply_layout"),
            bind_group_layouts: &[&apply_bgl],
            push_constant_ranges: &[],
        });
        let merge_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("merge_layout"),
            bind_group_layouts: &[&merge_bgl],
            push_constant_ranges: &[],
        });
        let clear_species_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear_species_layout"),
            bind_group_layouts: &[&clear_species_bgl],
            push_constant_ranges: &[],
        });
        let count_species_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("count_species_layout"),
            bind_group_layouts: &[&count_species_bgl],
            push_constant_ranges: &[],
        });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_layout"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });

        let pipelines = Pipelines {
            clear_grid: compute_pipeline(&device, &shader, &clear_layout, "clear_grid_counts"),
            reset_dead: compute_pipeline(&device, &shader, &reset_layout, "reset_dead_new"),
            build_counts: compute_pipeline(&device, &shader, &build_layout, "build_grid_counts"),
            scan: compute_pipeline(&device, &shader, &scan_layout, "scan_grid_counts"),
            copy_offsets: compute_pipeline(&device, &shader, &copy_layout, "copy_grid_offsets"),
            scatter: compute_pipeline(&device, &shader, &scatter_layout, "scatter_indices"),
            update: compute_pipeline(&device, &shader, &update_layout, "update_boids"),
            reproduce: compute_pipeline(&device, &shader, &reproduce_layout, "reproduce_boids"),
            apply_spawns: compute_pipeline(&device, &shader, &apply_layout, "apply_spawns"),
            merge_dead: compute_pipeline(&device, &shader, &merge_layout, "merge_dead"),
            clear_species: compute_pipeline(
                &device,
                &shader,
                &clear_species_layout,
                "clear_species_counts",
            ),
            count_species: compute_pipeline(
                &device,
                &shader,
                &count_species_layout,
                "count_species",
            ),
            render: render_pipeline(&device, &shader, &render_layout, surface_format),
            graph: graph_pipeline(&device, &shader, surface_format),
        };

        let boids = [boid_a, boid_b];
        let buffers = Buffers {
            params: params_buffer,
            boids,
            grid_counts,
            grid_offsets,
            grid_offsets_write,
            grid_indices,
            dead_free,
            dead_free_count,
            dead_new,
            dead_new_count,
            spawn_list,
            spawn_count,
            species_counts,
            species_counts_read,
            cause_counts,
            cause_counts_read,
            graph_vertices,
        };

        let reset_dead = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reset_dead_bg"),
            layout: &reset_bgl,
            entries: &[
                bind_entry(10, &buffers.dead_new_count),
                bind_entry(12, &buffers.spawn_count),
                bind_entry(14, &buffers.cause_counts),
            ],
        });
        let clear_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_grid_bg"),
            layout: &clear_bgl,
            entries: &[
                bind_entry(0, &buffers.params),
                bind_entry(3, &buffers.grid_counts),
            ],
        });
        let scan = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scan_bg"),
            layout: &scan_bgl,
            entries: &[
                bind_entry(0, &buffers.params),
                bind_entry(3, &buffers.grid_counts),
                bind_entry(4, &buffers.grid_offsets),
            ],
        });
        let copy_offsets = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_offsets_bg"),
            layout: &copy_bgl,
            entries: &[
                bind_entry(0, &buffers.params),
                bind_entry(4, &buffers.grid_offsets),
                bind_entry(5, &buffers.grid_offsets_write),
            ],
        });
        let merge_dead = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("merge_dead_bg"),
            layout: &merge_bgl,
            entries: &[
                bind_entry(9, &buffers.dead_new),
                bind_entry(10, &buffers.dead_new_count),
                bind_entry(7, &buffers.dead_free),
                bind_entry(8, &buffers.dead_free_count),
                bind_entry(14, &buffers.cause_counts),
            ],
        });
        let clear_species = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_species_bg"),
            layout: &clear_species_bgl,
            entries: &[bind_entry(13, &buffers.species_counts)],
        });

        let build_counts = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("build_counts_bg_0"),
                layout: &build_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[0]),
                    bind_entry(3, &buffers.grid_counts),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("build_counts_bg_1"),
                layout: &build_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[1]),
                    bind_entry(3, &buffers.grid_counts),
                ],
            }),
        ];
        let scatter = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_bg_0"),
                layout: &scatter_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[0]),
                    bind_entry(5, &buffers.grid_offsets_write),
                    bind_entry(6, &buffers.grid_indices),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scatter_bg_1"),
                layout: &scatter_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[1]),
                    bind_entry(5, &buffers.grid_offsets_write),
                    bind_entry(6, &buffers.grid_indices),
                ],
            }),
        ];
        let update = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("update_bg_0"),
                layout: &update_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[0]),
                    bind_entry(2, &buffers.boids[1]),
                    bind_entry(3, &buffers.grid_counts),
                    bind_entry(4, &buffers.grid_offsets),
                    bind_entry(6, &buffers.grid_indices),
                    bind_entry(9, &buffers.dead_new),
                    bind_entry(10, &buffers.dead_new_count),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("update_bg_1"),
                layout: &update_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(1, &buffers.boids[1]),
                    bind_entry(2, &buffers.boids[0]),
                    bind_entry(3, &buffers.grid_counts),
                    bind_entry(4, &buffers.grid_offsets),
                    bind_entry(6, &buffers.grid_indices),
                    bind_entry(9, &buffers.dead_new),
                    bind_entry(10, &buffers.dead_new_count),
                ],
            }),
        ];
        let reproduce = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reproduce_bg_0"),
                layout: &reproduce_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[0]),
                    bind_entry(11, &buffers.spawn_list),
                    bind_entry(12, &buffers.spawn_count),
                    bind_entry(14, &buffers.cause_counts),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reproduce_bg_1"),
                layout: &reproduce_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[1]),
                    bind_entry(11, &buffers.spawn_list),
                    bind_entry(12, &buffers.spawn_count),
                    bind_entry(14, &buffers.cause_counts),
                ],
            }),
        ];
        let apply_spawns = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apply_spawns_bg_0"),
                layout: &apply_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[0]),
                    bind_entry(7, &buffers.dead_free),
                    bind_entry(8, &buffers.dead_free_count),
                    bind_entry(11, &buffers.spawn_list),
                    bind_entry(12, &buffers.spawn_count),
                    bind_entry(14, &buffers.cause_counts),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("apply_spawns_bg_1"),
                layout: &apply_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[1]),
                    bind_entry(7, &buffers.dead_free),
                    bind_entry(8, &buffers.dead_free_count),
                    bind_entry(11, &buffers.spawn_list),
                    bind_entry(12, &buffers.spawn_count),
                    bind_entry(14, &buffers.cause_counts),
                ],
            }),
        ];

        let count_species = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("count_species_bg_0"),
                layout: &count_species_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[0]),
                    bind_entry(13, &buffers.species_counts),
                ],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("count_species_bg_1"),
                layout: &count_species_bgl,
                entries: &[
                    bind_entry(0, &buffers.params),
                    bind_entry(2, &buffers.boids[1]),
                    bind_entry(13, &buffers.species_counts),
                ],
            }),
        ];

        let render = [
            create_render_bind_group(&device, &render_bgl, &buffers, 0),
            create_render_bind_group(&device, &render_bgl, &buffers, 1),
        ];

        let bind_groups = BindGroups {
            reset_dead,
            clear_grid,
            scan,
            copy_offsets,
            merge_dead,
            clear_species,
            count_species,
            build_counts,
            scatter,
            update,
            reproduce,
            apply_spawns,
            render,
        };

        let ui = window.map(|window| {
            let egui_ctx = egui::Context::default();
            let egui_winit = egui_winit::State::new(
                egui_ctx.clone(),
                egui::ViewportId::ROOT,
                window,
                Some(window.scale_factor() as f32),
                None,
            );
            let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);
            UiState {
                egui_ctx,
                egui_winit,
                egui_renderer,
                fps_smoothed: 0.0,
                timings: FrameBreakdown::default(),
            }
        });
        let audio = if window.is_some() {
            match AudioEngine::new() {
                Ok(engine) => Some(engine),
                Err(err) => {
                    eprintln!("Audio disabled: {err}");
                    None
                }
            }
        } else {
            None
        };

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            size,
            grid_cells,
            current: 0,
            pipelines,
            _buffers: buffers,
            bind_groups,
            graph: GraphState {
                history: std::array::from_fn(|_| Vec::new()),
                scale: [1u32; GRAPH_SERIES],
                acc: [0.0; GRAPH_SERIES],
                acc_count: [0u32; GRAPH_SERIES],
                frame: 0,
                series_max: [1.0; GRAPH_SERIES],
                vertex_counts: [0u32; GRAPH_SERIES],
            },
            causes: CauseState::default(),
            audio,
            audio_mapper: AudioControlMapper::new(
                AUDIO_CONTROL_WINDOW_SEC,
                BOID_CAPACITY,
                ((BOID_CAPACITY as f32) * 0.05).round() as u32,
            ),
            audio_last_tick: Instant::now(),
            params_cpu: params,
            ui,
            cursor_pos_px: None,
        };
        state.update_graph_vertices();
        state
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            if let (Some(surface), Some(config)) = (&self.surface, &mut self.config) {
                config.width = new_size.width;
                config.height = new_size.height;
                surface.configure(&self.device, config);
            }
        }
    }

    fn handle_window_event(&mut self, window: &winit::window::Window, event: &WindowEvent) -> bool {
        self.ui
            .as_mut()
            .expect("GUI state")
            .egui_winit
            .on_window_event(window, event)
            .consumed
    }

    fn render(&mut self, window: &winit::window::Window) -> Result<(), wgpu::SurfaceError> {
        let frame_start = Instant::now();
        let now = Instant::now();
        let audio_dt_sec = (now - self.audio_last_tick)
            .as_secs_f32()
            .clamp(1.0 / 500.0, 0.5);
        self.audio_last_tick = now;
        let output = self
            .surface
            .as_ref()
            .expect("GUI surface")
            .get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let ui_start = Instant::now();

        let mut ui = self.ui.take().expect("GUI state");
        let raw_input = ui.egui_winit.take_egui_input(window);
        let timings = ui.timings.clone();
        let fps = ui.fps_smoothed;
        let audio_enabled = self.audio.is_some();
        let hover_info = self.graph_hover(window.scale_factor() as f32);
        let cause_frame_count = self.causes.history.len();
        let cause_sums = self.causes.sums;
        let cause_all_time_max = self.causes.all_time_max.max(1) as f32;
        let cause_bars = [
            (
                "Births: Reproduction Split",
                cause_sums[CAUSE_REPRODUCTION_SPLIT],
            ),
            (
                "Births: Prey->Predator Mutation",
                cause_sums[CAUSE_PREY_TO_PREDATOR_MUTATION],
            ),
            (
                "Births: Prey Color Shift",
                cause_sums[CAUSE_PREY_COLOR_SHIFT],
            ),
            ("Births: Dropped (No Slot)", cause_sums[CAUSE_BIRTH_DROPPED]),
            ("Deaths: Prey Eaten", cause_sums[CAUSE_PREY_EATEN]),
            ("Deaths: Prey Loneliness", cause_sums[CAUSE_PREY_LONELINESS]),
            (
                "Deaths: Prey Overcrowding",
                cause_sums[CAUSE_PREY_OVERCROWDING],
            ),
            (
                "Deaths: Predator Starvation",
                cause_sums[CAUSE_PREDATOR_STARVATION],
            ),
            ("Deaths: Old Age", cause_sums[CAUSE_OLD_AGE]),
        ];
        let overlay_margin = 12.0f32;
        let overlay_spacing = 8.0f32;
        let collapsed_height = 28.0f32;
        let viewport_height = self.size.height.max(1) as f32;
        let causes_default_y =
            (viewport_height - overlay_margin - collapsed_height).max(overlay_margin);
        let sim_default_y =
            (causes_default_y - overlay_spacing - collapsed_height).max(overlay_margin);
        let mut local_params = self.params_cpu;
        let full_output = ui.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Sim Controls")
                .default_pos([overlay_margin, sim_default_y])
                .default_open(false)
                .show(ctx, |ui| {
                    ui.label(format!("FPS: {:.1}", fps));
                    ui.label(format!(
                        "Audio: {}",
                        if audio_enabled {
                            "On"
                        } else {
                            "Off (no output device)"
                        }
                    ));
                    ui.separator();
                    ui.label("Simulation Parameters");
                    ui.add(
                        egui::Slider::new(&mut local_params.separation_coeff, 0.0..=10.0)
                            .text("Separation"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.wall_factor, 0.0..=80.0)
                            .text("Wall Force"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.predator_distance, 1.0..=40.0)
                            .text("Predator Distance"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.predator_food_gain, 1.0..=120.0)
                            .text("Predator Food Gain"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.prey_avoidance_bonus, 1.0..=30.0)
                            .text("Prey Avoidance"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut local_params.prey_to_predator_mutation_denom,
                            100..=20_000,
                        )
                        .text("Mutation 1/N"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.prey_gain_min_neighbors, 0..=64)
                            .text("Prey Gain Min Neighbors"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.prey_gain_max_neighbors, 0..=64)
                            .text("Prey Gain Max Neighbors"),
                    );
                    ui.add(
                        egui::Slider::new(&mut local_params.prey_color_shift_denom, 1..=100)
                            .text("Prey Color Shift 1/N"),
                    );
                    let mut loneliness_enabled = local_params.loneliness_enabled != 0;
                    ui.checkbox(&mut loneliness_enabled, "Loneliness Enabled");
                    local_params.loneliness_enabled = u32::from(loneliness_enabled);
                    let mut overcrowding_enabled = local_params.overcrowding_enabled != 0;
                    ui.checkbox(&mut overcrowding_enabled, "Overcrowding Enabled");
                    local_params.overcrowding_enabled = u32::from(overcrowding_enabled);
                    let mut old_age_enabled = local_params.old_age_enabled != 0;
                    ui.checkbox(&mut old_age_enabled, "Old Age Enabled");
                    local_params.old_age_enabled = u32::from(old_age_enabled);

                    ui.separator();
                    ui.label("Frame Time Breakdown (ms)");
                    let bars = [
                        ("Compute", timings.compute_ms),
                        ("Boid Render", timings.sim_render_ms),
                        ("Graph Render", timings.graph_render_ms),
                        ("UI", timings.ui_ms),
                        ("Readback+Graph", timings.readback_ms),
                        ("Submit+Present", timings.submit_present_ms),
                    ];
                    let max_bar = bars
                        .iter()
                        .fold(1.0f32, |acc, (_, v)| acc.max(*v))
                        .max(timings.total_ms);
                    for (label, value) in bars {
                        let frac = (value / max_bar).clamp(0.0, 1.0);
                        ui.horizontal(|ui| {
                            ui.label(format!("{label:>14}"));
                            ui.add(
                                egui::widgets::ProgressBar::new(frac)
                                    .desired_width(210.0)
                                    .text(format!("{value:.2} ms")),
                            );
                        });
                    }
                    ui.label(format!("Total: {:.2} ms", timings.total_ms));
                });

            egui::Window::new("Birth/Death Causes")
                .default_pos([overlay_margin, causes_default_y])
                .default_open(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "Rolling sum over last {} frames (target 60 ~ 1s), scaled to all-time peak",
                        cause_frame_count
                    ));
                    ui.separator();
                    let label_width = 270.0f32;
                    let row_spacing = ui.spacing().item_spacing.x;
                    let bar_width = (ui.available_width() - label_width - row_spacing).max(140.0);
                    for (label, value) in cause_bars {
                        let frac = (value as f32 / cause_all_time_max).clamp(0.0, 1.0);
                        ui.horizontal(|ui| {
                            ui.add_sized([label_width, 0.0], egui::Label::new(label).wrap(false));
                            let bar_response = ui.add(
                                egui::widgets::ProgressBar::new(frac)
                                    .desired_width(bar_width)
                                    .text(""),
                            );
                            let text_pos = egui::pos2(
                                bar_response.rect.right() - 8.0,
                                bar_response.rect.center().y,
                            );
                            ui.painter().text(
                                text_pos,
                                egui::Align2::RIGHT_CENTER,
                                value.to_string(),
                                egui::TextStyle::Body.resolve(ui.style()),
                                egui::Color32::WHITE,
                            );
                        });
                    }
                });

            if let Some(hover) = hover_info {
                let screen = ctx.input(|i| i.screen_rect());
                let overlay_size = egui::vec2(150.0, 66.0);
                let margin = 8.0;
                let edge_offset = 14.0;

                let mut overlay_pos = egui::pos2(hover.x_ui + edge_offset, hover.y_ui - 52.0);
                if overlay_pos.x + overlay_size.x > screen.right() - margin {
                    overlay_pos.x = hover.x_ui - overlay_size.x - edge_offset;
                }
                if overlay_pos.y < screen.top() + margin {
                    overlay_pos.y = hover.y_ui + edge_offset;
                }
                overlay_pos.x = overlay_pos.x.clamp(
                    screen.left() + margin,
                    screen.right() - overlay_size.x - margin,
                );
                overlay_pos.y = overlay_pos.y.clamp(
                    screen.top() + margin,
                    screen.bottom() - overlay_size.y - margin,
                );

                egui::Area::new(egui::Id::new("graph_hover_overlay"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(overlay_pos)
                    .show(ctx, |ui| {
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
                            ui.set_min_width(140.0);
                            let prey_text = hover
                                .prey
                                .map(|v| format!("{}", v.round() as u32))
                                .unwrap_or_else(|| "-".to_string());
                            let pred_text = hover
                                .predators
                                .map(|v| format!("{}", v.round() as u32))
                                .unwrap_or_else(|| "-".to_string());
                            ui.add(egui::Label::new(format!("Prey: {prey_text}")).wrap(false));
                            ui.add(egui::Label::new(format!("Predators: {pred_text}")).wrap(false));
                        });
                    });
            }
        });

        self.params_cpu = local_params;
        self.queue.write_buffer(
            &self._buffers.params,
            0,
            bytemuck::bytes_of(&self.params_cpu),
        );

        ui.egui_winit
            .handle_platform_output(window, full_output.platform_output);
        let paint_jobs = ui
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.size.width, self.size.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, image_delta) in &full_output.textures_delta.set {
            ui.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        ui.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );
        let ui_ms = ui_start.elapsed().as_secs_f32() * 1000.0;

        let compute_start = Instant::now();
        self.run_compute_passes(&mut encoder);
        let compute_ms = compute_start.elapsed().as_secs_f32() * 1000.0;
        encoder.copy_buffer_to_buffer(
            &self._buffers.cause_counts,
            0,
            &self._buffers.cause_counts_read,
            0,
            (CAUSE_COUNT as u64) * 4,
        );
        let readback = self.graph.frame % GRAPH_READBACK_INTERVAL == 0;
        if readback {
            encoder.copy_buffer_to_buffer(
                &self._buffers.species_counts,
                0,
                &self._buffers.species_counts_read,
                0,
                (POPULATION_SERIES as u64) * 4,
            );
        }
        self.current = 1 - self.current;

        let sim_render_ms: f32;
        let mut graph_render_ms = 0.0f32;
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let graph_height = GRAPH_HEIGHT_PX.min(self.size.height.saturating_sub(1));
            let boid_height = self.size.height.saturating_sub(graph_height).max(1);

            render_pass.set_viewport(
                0.0,
                0.0,
                self.size.width as f32,
                boid_height as f32,
                0.0,
                1.0,
            );
            render_pass.set_scissor_rect(0, 0, self.size.width, boid_height);
            let sim_start = Instant::now();
            render_pass.set_pipeline(&self.pipelines.render);
            render_pass.set_bind_group(0, &self.bind_groups.render[self.current], &[]);
            render_pass.draw(0..3, 0..BOID_CAPACITY);
            sim_render_ms = sim_start.elapsed().as_secs_f32() * 1000.0;

            if graph_height > 0 {
                render_pass.set_viewport(
                    0.0,
                    boid_height as f32,
                    self.size.width as f32,
                    graph_height as f32,
                    0.0,
                    1.0,
                );
                render_pass.set_scissor_rect(0, boid_height, self.size.width, graph_height);
                let graph_start = Instant::now();
                render_pass.set_pipeline(&self.pipelines.graph);
                render_pass.set_vertex_buffer(0, self._buffers.graph_vertices.slice(..));
                let mut start = 0u32;
                for s in 0..GRAPH_SERIES {
                    let count = self.graph.vertex_counts[s];
                    if count > 0 {
                        render_pass.draw(start..start + count, 0..1);
                    }
                    start += count;
                }
                graph_render_ms = graph_start.elapsed().as_secs_f32() * 1000.0;
            }
        }

        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            ui.egui_renderer
                .render(&mut ui_pass, &paint_jobs, &screen_desc);
        }

        for id in &full_output.textures_delta.free {
            ui.egui_renderer.free_texture(id);
        }

        let submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));
        output.present();
        let submit_present_ms = submit_start.elapsed().as_secs_f32() * 1000.0;

        let readback_ms = {
            let readback_start = Instant::now();
            if readback {
                if let Some(counts) = self.read_population_counts() {
                    let prey_count = counts[..PREY_SPECIES_COUNT].iter().sum();
                    let predator_count = counts[PREDATOR_POPULATION_INDEX];
                    self.audio_mapper
                        .update_populations(prey_count, predator_count);
                    let width_limit = self.graph_width_limit();
                    self.add_graph_point(0, prey_count as f32, width_limit);
                    self.add_graph_point(1, predator_count as f32, width_limit);
                }
                self.update_graph_vertices();
            }
            let cause_counts = self.read_cause_counts();
            if let Some(counts) = cause_counts {
                if let Some(controls) = self.audio_mapper.ingest_events(counts, audio_dt_sec) {
                    if let Some(audio) = &self.audio {
                        audio.update_controls(controls);
                    }
                }
            }
            readback_start.elapsed().as_secs_f32() * 1000.0
        };
        self.graph.frame += 1;
        let total_ms = frame_start.elapsed().as_secs_f32() * 1000.0;
        ui.timings = FrameBreakdown {
            compute_ms,
            sim_render_ms,
            graph_render_ms,
            ui_ms,
            readback_ms,
            submit_present_ms,
            total_ms,
        };
        if total_ms > 0.0 {
            let instant_fps = 1000.0 / total_ms;
            ui.fps_smoothed = if ui.fps_smoothed <= 0.0 {
                instant_fps
            } else {
                ui.fps_smoothed * 0.9 + instant_fps * 0.1
            };
        }
        self.ui = Some(ui);
        Ok(())
    }

    fn run_compute_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let grid_dispatch = dispatch_count(self.grid_cells, WORKGROUP_SIZE);
        let boid_dispatch = dispatch_count(BOID_CAPACITY, WORKGROUP_SIZE);
        let in_idx = self.current;
        let out_idx = 1 - self.current;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reset_dead"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reset_dead);
            pass.set_bind_group(0, &self.bind_groups.reset_dead, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_grid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.clear_grid);
            pass.set_bind_group(0, &self.bind_groups.clear_grid, &[]);
            pass.dispatch_workgroups(grid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_counts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.build_counts);
            pass.set_bind_group(0, &self.bind_groups.build_counts[in_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scan"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.scan);
            pass.set_bind_group(0, &self.bind_groups.scan, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("copy_offsets"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.copy_offsets);
            pass.set_bind_group(0, &self.bind_groups.copy_offsets, &[]);
            pass.dispatch_workgroups(grid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.scatter);
            pass.set_bind_group(0, &self.bind_groups.scatter[in_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.update);
            pass.set_bind_group(0, &self.bind_groups.update[in_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reproduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.reproduce);
            pass.set_bind_group(0, &self.bind_groups.reproduce[out_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply_spawns"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.apply_spawns);
            pass.set_bind_group(0, &self.bind_groups.apply_spawns[out_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_species"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.clear_species);
            pass.set_bind_group(0, &self.bind_groups.clear_species, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count_species"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.count_species);
            pass.set_bind_group(0, &self.bind_groups.count_species[out_idx], &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("merge_dead"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.merge_dead);
            pass.set_bind_group(0, &self.bind_groups.merge_dead, &[]);
            pass.dispatch_workgroups(boid_dispatch, 1, 1);
        }
    }

    fn read_current_population(&mut self) -> Result<[u32; POPULATION_SERIES], String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless_initial_population"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("clear_population"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.clear_species);
            pass.set_bind_group(0, &self.bind_groups.clear_species, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("count_population"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.count_species);
            pass.set_bind_group(0, &self.bind_groups.count_species[self.current], &[]);
            pass.dispatch_workgroups(dispatch_count(BOID_CAPACITY, WORKGROUP_SIZE), 1, 1);
        }
        self.copy_population_to_readback(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        self.read_population_counts()
            .ok_or_else(|| "failed to read initial population from GPU".to_string())
    }

    fn run_headless_batch(
        &mut self,
        frame_count: u64,
        read_population: bool,
    ) -> Result<Option<[u32; POPULATION_SERIES]>, String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("headless_frames"),
            });
        for _ in 0..frame_count {
            self.run_compute_passes(&mut encoder);
            self.current = 1 - self.current;
        }
        if read_population {
            self.copy_population_to_readback(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        if read_population {
            self.read_population_counts()
                .map(Some)
                .ok_or_else(|| "failed to read population from GPU".to_string())
        } else {
            Ok(None)
        }
    }

    fn copy_population_to_readback(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self._buffers.species_counts,
            0,
            &self._buffers.species_counts_read,
            0,
            (POPULATION_SERIES as u64) * 4,
        );
    }

    fn read_cause_counts(&mut self) -> Option<[u32; CAUSE_COUNT]> {
        let slice = self._buffers.cause_counts_read.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv() {
            let data = slice.get_mapped_range();
            let counts_src: &[u32] = bytemuck::cast_slice(&data);
            let mut counts = [0u32; CAUSE_COUNT];
            if counts_src.len() >= CAUSE_COUNT {
                counts.copy_from_slice(&counts_src[..CAUSE_COUNT]);
            }
            drop(data);
            self._buffers.cause_counts_read.unmap();
            self.causes.push_frame(counts);
            return Some(counts);
        }
        None
    }

    fn read_population_counts(&mut self) -> Option<[u32; POPULATION_SERIES]> {
        let slice = self._buffers.species_counts_read.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv() {
            let data = slice.get_mapped_range();
            let counts_src: &[u32] = bytemuck::cast_slice(&data);
            let mut counts = [0u32; POPULATION_SERIES];
            if counts_src.len() >= POPULATION_SERIES {
                counts.copy_from_slice(&counts_src[..POPULATION_SERIES]);
            }
            drop(data);
            self._buffers.species_counts_read.unmap();
            return Some(counts);
        }
        None
    }

    fn update_graph_vertices(&mut self) {
        let width = self.size.width.max(1) as f32;
        let graph_height = GRAPH_HEIGHT_PX.min(self.size.height.saturating_sub(1)) as f32;
        if graph_height < 4.0 {
            return;
        }
        let origin = [GRAPH_PADDING_PX, GRAPH_PADDING_PX];
        let size = [
            (width - 2.0 * GRAPH_PADDING_PX).max(1.0),
            (graph_height - 2.0 * GRAPH_PADDING_PX).max(1.0),
        ];

        // Match the original graph behavior: one prey line and one predator line.
        let colors: [[f32; 3]; GRAPH_SERIES] = [
            [0.0, 0.0, 1.0], // prey
            [1.0, 0.0, 0.0], // predators
        ];

        let mut vertices = Vec::new();
        self.graph.vertex_counts = [0u32; GRAPH_SERIES];
        for s in 0..GRAPH_SERIES {
            let history = &self.graph.history[s];
            if history.len() < 2 {
                continue;
            }

            let color = colors[s];
            let denom = self.graph.series_max[s].max(1.0);
            for i in 0..(history.len() - 1) {
                // Align to the right edge, one x-unit per sample, like the original canvas graph.
                let x0 = origin[0] + (size[0] - history.len() as f32 + i as f32);
                let x1 = x0 + 1.0;
                let y0 = origin[1] + size[1] * (1.0 - history[i] / denom);
                let y1 = origin[1] + size[1] * (1.0 - history[i + 1] / denom);

                let ndc0 = [x0 / width * 2.0 - 1.0, 1.0 - y0 / graph_height * 2.0];
                let ndc1 = [x1 / width * 2.0 - 1.0, 1.0 - y1 / graph_height * 2.0];
                vertices.push(GraphVertex {
                    pos: ndc0,
                    color,
                    _pad: 0.0,
                });
                vertices.push(GraphVertex {
                    pos: ndc1,
                    color,
                    _pad: 0.0,
                });
            }
            self.graph.vertex_counts[s] = ((history.len() - 1) * 2) as u32;
        }

        self.queue.write_buffer(
            &self._buffers.graph_vertices,
            0,
            bytemuck::cast_slice(&vertices),
        );
    }

    fn graph_width_limit(&self) -> usize {
        let graph_width = (self.size.width as f32 - 2.0 * GRAPH_PADDING_PX).max(2.0) as usize;
        graph_width.min(GRAPH_MAX_POINTS)
    }

    fn add_graph_point(&mut self, series: usize, value: f32, width_limit: usize) {
        self.graph.acc[series] += value;
        self.graph.acc_count[series] += 1;
        if self.graph.acc_count[series] < self.graph.scale[series] {
            return;
        }

        let point = self.graph.acc[series] / self.graph.acc_count[series] as f32;
        self.graph.acc[series] = 0.0;
        self.graph.acc_count[series] = 0;
        self.graph.history[series].push(point);
        self.graph.series_max[series] = self.graph.series_max[series].max(point);

        if self.graph.history[series].len() <= width_limit {
            return;
        }

        if self.graph.scale[series] < GRAPH_MAX_SCALE {
            let old = std::mem::take(&mut self.graph.history[series]);
            let mut downsampled = Vec::with_capacity(old.len() / 2);
            let mut i = 0usize;
            while i + 1 < old.len() {
                downsampled.push((old[i] + old[i + 1]) * 0.5);
                i += 2;
            }
            self.graph.history[series] = downsampled;
            self.graph.scale[series] *= 2;
            return;
        }

        let overflow = self.graph.history[series].len() - width_limit;
        self.graph.history[series].drain(0..overflow);
    }

    fn graph_hover(&self, pixels_per_point: f32) -> Option<GraphHover> {
        let cursor = self.cursor_pos_px?;
        let graph_height = GRAPH_HEIGHT_PX.min(self.size.height.saturating_sub(1)) as f32;
        if graph_height <= 0.0 {
            return None;
        }
        let boid_height = self.size.height as f32 - graph_height;
        if cursor[1] < boid_height || cursor[1] > boid_height + graph_height {
            return None;
        }
        if cursor[0] < 0.0 || cursor[0] > self.size.width as f32 {
            return None;
        }

        let inner_x = GRAPH_PADDING_PX;
        let inner_w = (self.size.width as f32 - 2.0 * GRAPH_PADDING_PX).max(1.0);
        let local_x = cursor[0];

        let prey = Self::sample_history_at_x(&self.graph.history[0], local_x, inner_x, inner_w);
        let predators =
            Self::sample_history_at_x(&self.graph.history[1], local_x, inner_x, inner_w);
        if prey.is_none() && predators.is_none() {
            return None;
        }

        Some(GraphHover {
            prey,
            predators,
            x_ui: cursor[0] / pixels_per_point,
            y_ui: cursor[1] / pixels_per_point,
        })
    }

    fn sample_history_at_x(
        history: &[f32],
        local_x: f32,
        inner_x: f32,
        inner_w: f32,
    ) -> Option<f32> {
        if history.is_empty() {
            return None;
        }
        let len = history.len() as f32;
        let first_x = inner_x + (inner_w - len).max(0.0);
        let last_x = first_x + (len - 1.0).max(0.0);
        if local_x < first_x || local_x > last_x {
            return None;
        }

        let idx = (local_x - first_x).round() as usize;
        history.get(idx.min(history.len() - 1)).copied()
    }
}

fn dispatch_count(total: u32, group: u32) -> u32 {
    (total + group - 1) / group
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module: shader,
        entry_point: entry,
    })
}

fn render_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

fn graph_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("graph_layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("graph_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_graph",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<GraphVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 8,
                        shader_location: 1,
                    },
                ],
            }],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_graph",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    })
}

fn create_render_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &Buffers,
    index: usize,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("render_bind_group"),
        layout,
        entries: &[
            bind_entry(0, &buffers.params),
            bind_entry(1, &buffers.boids[index]),
        ],
    })
}

fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn create_initial_boids(initial_count: u32, seed: u64, mix: InitialMix) -> Vec<Boid> {
    let predator_count = (initial_count as f32 * PREDATOR_RATIO).round() as u32;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut boids = Vec::with_capacity(BOID_CAPACITY as usize);
    for i in 0..initial_count {
        let predator = i < predator_count;
        let species = if predator {
            0
        } else {
            rng.gen_range(0..PREY_SPECIES_COUNT as u32)
        };
        let kind = if predator {
            KIND_STANDARD
        } else {
            let roll = rng.gen_range(0.0..1.0);
            if roll < mix.pulse_ratio {
                KIND_PULSE
            } else if roll < mix.pulse_ratio + mix.courier_ratio {
                KIND_COURIER
            } else if roll < mix.pulse_ratio + mix.courier_ratio + mix.warden_ratio {
                KIND_WARDEN
            } else {
                KIND_STANDARD
            }
        };
        let flags = FLAG_ALIVE | if predator { FLAG_PREDATOR } else { 0 };
        let pulse_state = if kind == KIND_PULSE {
            ((i.wrapping_mul(73) ^ seed as u32) % PULSE_PERIOD) << PULSE_STATE_SHIFT
        } else {
            0
        };
        let life = START_LIFE;
        let lifetime = if predator {
            rng.gen_range(PREDATOR_LIFETIME_MIN..=PREDATOR_LIFETIME_MAX)
        } else {
            rng.gen_range(PREY_LIFETIME_MIN..=PREY_LIFETIME_MAX)
        };
        boids.push(Boid {
            pos: [
                rng.gen_range(0.0..WORLD_SIZE[0]),
                rng.gen_range(0.0..WORLD_SIZE[1]),
            ],
            vel: [
                (rng.gen_range(-0.5..0.5)) * 10.0,
                (rng.gen_range(-0.5..0.5)) * 10.0,
            ],
            life,
            lifetime,
            species,
            flags,
            _pad: pulse_state,
            kind,
        });
    }
    for _ in initial_count..BOID_CAPACITY {
        boids.push(Boid {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            life: 0.0,
            lifetime: 0,
            species: 0,
            flags: 0,
            _pad: 0,
            kind: KIND_STANDARD,
        });
    }
    boids
}

fn create_dead_free_list(initial_count: u32) -> (Vec<u32>, u32) {
    let mut list = vec![0u32; BOID_CAPACITY as usize];
    let mut count = 0u32;
    for i in initial_count..BOID_CAPACITY {
        list[count as usize] = i;
        count += 1;
    }
    (list, count)
}

fn parse_run_mode(args: impl IntoIterator<Item = String>) -> Result<RunMode, String> {
    let args: Vec<String> = args.into_iter().collect();
    if args.is_empty() {
        return Ok(RunMode::Gui);
    }
    if args[0] == "--help" || args[0] == "-h" {
        return Ok(RunMode::Help);
    }
    if args[0] != "--headless" {
        return Err(format!("unknown argument: {}", args[0]));
    }

    let mut options = HeadlessOptions::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--help" || args[i] == "-h" {
            return Ok(RunMode::Help);
        }
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("{} requires a value", args[i]))?;
        match args[i].as_str() {
            "--frames" => {
                options.frames = value
                    .parse()
                    .map_err(|_| format!("invalid frame count: {value}"))?;
            }
            "--sample-every" => {
                options.sample_every = value
                    .parse()
                    .map_err(|_| format!("invalid sampling interval: {value}"))?;
            }
            "--seed" => {
                options.seed = value
                    .parse()
                    .map_err(|_| format!("invalid seed: {value}"))?;
            }
            "--initial-count" => {
                options.initial_count = value
                    .parse()
                    .map_err(|_| format!("invalid initial count: {value}"))?;
            }
            "--pulse-ratio" => {
                options.mix.pulse_ratio = parse_ratio("pulse", value)?;
            }
            "--courier-ratio" => {
                options.mix.courier_ratio = parse_ratio("courier", value)?;
            }
            "--warden-ratio" => {
                options.mix.warden_ratio = parse_ratio("warden", value)?;
            }
            "--format" => {
                options.format = match value.as_str() {
                    "csv" => HeadlessFormat::Csv,
                    "jsonl" => HeadlessFormat::Jsonl,
                    _ => return Err(format!("invalid format: {value} (expected csv or jsonl)")),
                };
            }
            option => return Err(format!("unknown headless option: {option}")),
        }
        i += 2;
    }

    if options.sample_every == 0 {
        return Err("--sample-every must be greater than zero".to_string());
    }
    if options.initial_count > BOID_CAPACITY {
        return Err(format!(
            "--initial-count cannot exceed the capacity of {BOID_CAPACITY}"
        ));
    }
    let special_ratio =
        options.mix.pulse_ratio + options.mix.courier_ratio + options.mix.warden_ratio;
    if special_ratio > 1.0 {
        return Err(format!(
            "pulse, courier, and warden ratios must total at most 1.0 (got {special_ratio})"
        ));
    }
    Ok(RunMode::Headless(options))
}

fn parse_ratio(name: &str, value: &str) -> Result<f32, String> {
    let ratio: f32 = value
        .parse()
        .map_err(|_| format!("invalid {name} ratio: {value}"))?;
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(format!("{name} ratio must be between 0.0 and 1.0"));
    }
    Ok(ratio)
}

fn print_usage() {
    println!(
        "Boids GPU\n\n\
         Usage:\n\
           boids                         Run the interactive simulation\n\
           boids --headless [OPTIONS]    Run without a window or audio\n\n\
         Headless options:\n\
           --frames N          Frames to simulate (default: 5000)\n\
           --sample-every N    Emit population every N frames (default: 100)\n\
           --seed N            Initial population seed (default: 1)\n\
           --initial-count N   Initial boids, max {BOID_CAPACITY} (default: {INITIAL_COUNT})\n\
           --pulse-ratio R     Initial fraction of prey that pulse (default: 0.01)\n\
           --courier-ratio R   Initial fraction of prey that carry panic (default: 0.002)\n\
           --warden-ratio R    Initial fraction of prey that defend (default: 0.08)\n\
           --format FORMAT     csv or jsonl (default: csv)\n\
           -h, --help          Show this help"
    );
}

fn emit_population(format: HeadlessFormat, frame: u64, counts: &[u32; POPULATION_SERIES]) {
    let total: u32 =
        counts[..PREY_SPECIES_COUNT].iter().sum::<u32>() + counts[PREDATOR_POPULATION_INDEX];
    match format {
        HeadlessFormat::Csv => println!(
            "{frame},{},{},{},{},{},{},{},{},{},{},{},{},{total}",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            counts[PREDATOR_POPULATION_INDEX],
            counts[KIND_POPULATION_START + KIND_STANDARD as usize],
            counts[KIND_POPULATION_START + KIND_PULSE as usize],
            counts[KIND_POPULATION_START + KIND_COURIER as usize],
            counts[KIND_POPULATION_START + KIND_WARDEN as usize],
            counts[PANICKED_POPULATION_INDEX],
            counts[CHARGING_WARDEN_POPULATION_INDEX]
        ),
        HeadlessFormat::Jsonl => println!(
            "{{\"frame\":{frame},\"populations\":{{\"prey_0\":{},\"prey_1\":{},\"prey_2\":{},\"prey_3\":{},\"prey_4\":{},\"predators\":{},\"standard\":{},\"pulse\":{},\"courier\":{},\"warden\":{}}},\"activity\":{{\"panicked\":{},\"charging_wardens\":{}}},\"total\":{total}}}",
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            counts[4],
            counts[PREDATOR_POPULATION_INDEX],
            counts[KIND_POPULATION_START + KIND_STANDARD as usize],
            counts[KIND_POPULATION_START + KIND_PULSE as usize],
            counts[KIND_POPULATION_START + KIND_COURIER as usize],
            counts[KIND_POPULATION_START + KIND_WARDEN as usize],
            counts[PANICKED_POPULATION_INDEX],
            counts[CHARGING_WARDEN_POPULATION_INDEX]
        ),
    }
}

async fn run_headless(options: HeadlessOptions) -> Result<(), String> {
    let started = Instant::now();
    let mut state = State::new(None, options.initial_count, options.seed, options.mix).await;
    let initialization_elapsed = started.elapsed();
    eprintln!(
        "Headless run: frames={}, sample_every={}, seed={}, initial_count={}, pulse_ratio={}, courier_ratio={}, warden_ratio={}, format={:?}",
        options.frames,
        options.sample_every,
        options.seed,
        options.initial_count,
        options.mix.pulse_ratio,
        options.mix.courier_ratio,
        options.mix.warden_ratio,
        options.format
    );
    if options.format == HeadlessFormat::Csv {
        println!(
            "frame,prey_0,prey_1,prey_2,prey_3,prey_4,predators,standard,pulse,courier,warden,panicked,charging_wardens,total"
        );
    }

    let initial = state.read_current_population()?;
    emit_population(options.format, 0, &initial);

    let simulation_started = Instant::now();
    let mut frame = 0;
    while frame < options.frames {
        let sample_frame = (frame + options.sample_every).min(options.frames);
        let mut counts = None;
        while frame < sample_frame {
            let batch = HEADLESS_MAX_BATCH_FRAMES.min(sample_frame - frame);
            let reaches_sample = frame + batch == sample_frame;
            counts = state.run_headless_batch(batch, reaches_sample)?;
            frame += batch;
        }
        let counts = counts.ok_or_else(|| "population sample was not produced".to_string())?;
        emit_population(options.format, frame, &counts);
    }
    let simulation_elapsed = simulation_started.elapsed();
    let frames_per_second = if simulation_elapsed.is_zero() {
        0.0
    } else {
        options.frames as f64 / simulation_elapsed.as_secs_f64()
    };
    eprintln!(
        "Completed: initialization={:.3}s, simulation={:.3}s, throughput={frames_per_second:.1} frames/s, total={:.3}s",
        initialization_elapsed.as_secs_f64(),
        simulation_elapsed.as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn run_gui() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("Boids GPU (wgpu)")
        .with_inner_size(PhysicalSize::new(2000, 2000))
        .build(&event_loop)
        .expect("window");

    let mut state = pollster::block_on(State::new(
        Some(&window),
        INITIAL_COUNT,
        rand::random::<u64>(),
        InitialMix::default(),
    ));

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, .. } => {
                let _consumed = state.handle_window_event(&window, &event);
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(size) => state.resize(size),
                    WindowEvent::CursorMoved { position, .. } => {
                        state.cursor_pos_px = Some([position.x as f32, position.y as f32]);
                    }
                    WindowEvent::CursorLeft { .. } => {
                        state.cursor_pos_px = None;
                    }
                    WindowEvent::ScaleFactorChanged {
                        mut inner_size_writer,
                        ..
                    } => {
                        let size = window.inner_size();
                        let _ = inner_size_writer.request_inner_size(size);
                        state.resize(size);
                    }
                    WindowEvent::RedrawRequested => {
                        if let Err(err) = state.render(&window) {
                            match err {
                                wgpu::SurfaceError::Lost => state.resize(state.size),
                                wgpu::SurfaceError::OutOfMemory => target.exit(),
                                wgpu::SurfaceError::Outdated => {}
                                wgpu::SurfaceError::Timeout => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        })
        .expect("run");
}

fn main() -> ExitCode {
    let mode = match parse_run_mode(env::args().skip(1)) {
        Ok(mode) => mode,
        Err(err) => {
            eprintln!("Error: {err}\n");
            print_usage();
            return ExitCode::from(2);
        }
    };

    match mode {
        RunMode::Gui => run_gui(),
        RunMode::Headless(options) => {
            if let Err(err) = pollster::block_on(run_headless(options)) {
                eprintln!("Headless run failed: {err}");
                return ExitCode::FAILURE;
            }
        }
        RunMode::Help => print_usage(),
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_headless_options() {
        let mode = parse_run_mode(
            [
                "--headless",
                "--frames",
                "250",
                "--sample-every",
                "25",
                "--seed",
                "42",
                "--initial-count",
                "1000",
                "--pulse-ratio",
                "0.03",
                "--courier-ratio",
                "0.004",
                "--warden-ratio",
                "0.12",
                "--format",
                "jsonl",
            ]
            .map(str::to_string),
        )
        .unwrap();
        let RunMode::Headless(options) = mode else {
            panic!("expected headless mode");
        };
        assert_eq!(
            options,
            HeadlessOptions {
                frames: 250,
                sample_every: 25,
                seed: 42,
                initial_count: 1000,
                mix: InitialMix {
                    pulse_ratio: 0.03,
                    courier_ratio: 0.004,
                    warden_ratio: 0.12,
                },
                format: HeadlessFormat::Jsonl,
            }
        );
    }

    #[test]
    fn rejects_zero_sampling_interval() {
        let result = parse_run_mode(["--headless", "--sample-every", "0"].map(str::to_string));
        assert!(result.is_err());
    }

    #[test]
    fn rejects_species_ratios_over_one() {
        let result = parse_run_mode(
            [
                "--headless",
                "--pulse-ratio",
                "0.4",
                "--courier-ratio",
                "0.3",
                "--warden-ratio",
                "0.31",
            ]
            .map(str::to_string),
        );
        assert!(result.is_err());
    }
}
