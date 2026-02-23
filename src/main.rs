use bytemuck::{Pod, Zeroable};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const WORLD_SIZE: [f32; 2] = [8000.0, 8000.0];
const BOID_CAPACITY: u32 = 50_000;
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
const PREDATOR_LIFE_MULT: f32 = 1.5;
const RADIUS: f32 = 100.0;

const GRAPH_SERIES: usize = 2; // combined prey + predators
const GRAPH_READBACK_INTERVAL: u32 = 4;
const GRAPH_HEIGHT_PX: u32 = 200;
const GRAPH_PADDING_PX: f32 = 16.0;
const GRAPH_MAX_SCALE: u32 = 320;
const GRAPH_MAX_POINTS: usize = 8192;

const FLAG_PREDATOR: u32 = 1 << 0;
const FLAG_ALIVE: u32 = 1 << 1;

const WORKGROUP_SIZE: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Boid {
    pos: [f32; 2],
    vel: [f32; 2],
    life: f32,
    species: u32,
    flags: u32,
    _pad: u32,
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
    predator_life_mult: f32,
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

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    grid_cells: u32,
    current: usize,
    pipelines: Pipelines,
    _buffers: Buffers,
    bind_groups: BindGroups,
    graph: GraphState,
}

impl State {
    async fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window).expect("surface");
        let surface = unsafe {
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        };
        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
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
                        force_fallback_adapter: false,
                    })
                    .await;

                match fallback {
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
            predator_life_mult: PREDATOR_LIFE_MULT,
            grid_size: [grid_x, grid_y],
            capacity: BOID_CAPACITY,
            _pad: 0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let initial_boids = create_initial_boids();
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
            size: (GRAPH_SERIES as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let species_counts_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("species_counts_read"),
            size: (GRAPH_SERIES as u64) * 4,
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

        let (dead_free_list, dead_free_count_value) = create_dead_free_list(INITIAL_COUNT);
        queue.write_buffer(&dead_free, 0, bytemuck::cast_slice(&dead_free_list));
        queue.write_buffer(
            &dead_free_count,
            0,
            bytemuck::bytes_of(&(dead_free_count_value as i32)),
        );
        queue.write_buffer(&dead_new_count, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&spawn_count, 0, bytemuck::bytes_of(&0u32));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("boids.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("boids.wgsl").into()),
        });

        let reset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reset_bgl"),
            entries: &[
                storage_entry(10, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(12, false, wgpu::ShaderStages::COMPUTE),
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
            ],
        });
        let merge_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("merge_bgl"),
            entries: &[
                storage_entry(9, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(10, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(7, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(8, false, wgpu::ShaderStages::COMPUTE),
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
            graph_vertices,
        };

        let reset_dead = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reset_dead_bg"),
            layout: &reset_bgl,
            entries: &[
                bind_entry(10, &buffers.dead_new_count),
                bind_entry(12, &buffers.spawn_count),
            ],
        });
        let clear_grid = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_grid_bg"),
            layout: &clear_bgl,
            entries: &[bind_entry(0, &buffers.params), bind_entry(3, &buffers.grid_counts)],
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
        };
        state.update_graph_vertices();
        state
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        self.run_compute_passes(&mut encoder);
        let readback = self.graph.frame % GRAPH_READBACK_INTERVAL == 0;
        if readback {
            encoder.copy_buffer_to_buffer(
                &self._buffers.species_counts,
                0,
                &self._buffers.species_counts_read,
                0,
                (GRAPH_SERIES as u64) * 4,
            );
        }
        self.current = 1 - self.current;

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
            render_pass.set_pipeline(&self.pipelines.render);
            render_pass.set_bind_group(0, &self.bind_groups.render[self.current], &[]);
            render_pass.draw(0..3, 0..BOID_CAPACITY);

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
            }
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        if readback {
            self.read_species_counts();
            self.update_graph_vertices();
        }
        self.graph.frame += 1;
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

    fn read_species_counts(&mut self) {
        let slice = self._buffers.species_counts_read.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = receiver.recv() {
            let data = slice.get_mapped_range();
            let counts_src: &[u32] = bytemuck::cast_slice(&data);
            let mut counts = [0u32; GRAPH_SERIES];
            if counts_src.len() >= GRAPH_SERIES {
                counts.copy_from_slice(&counts_src[..GRAPH_SERIES]);
            }
            drop(data);
            self._buffers.species_counts_read.unmap();
            let width_limit = self.graph_width_limit();
            for s in 0..GRAPH_SERIES {
                self.add_graph_point(s, counts[s] as f32, width_limit);
            }
        }
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

        self.queue
            .write_buffer(&self._buffers.graph_vertices, 0, bytemuck::cast_slice(&vertices));
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

fn create_initial_boids() -> Vec<Boid> {
    let predator_count = (INITIAL_COUNT as f32 * PREDATOR_RATIO).round() as u32;
    let mut rng = rand::thread_rng();
    let mut boids = Vec::with_capacity(BOID_CAPACITY as usize);
    for i in 0..INITIAL_COUNT {
        let predator = i < predator_count;
        let species = if predator {
            0
        } else {
            rng.gen_range(0..5)
        };
        let flags = FLAG_ALIVE | if predator { FLAG_PREDATOR } else { 0 };
        let life = if predator {
            START_LIFE * PREDATOR_LIFE_MULT
        } else {
            START_LIFE
        };
        boids.push(Boid {
            pos: [rng.gen_range(0.0..WORLD_SIZE[0]), rng.gen_range(0.0..WORLD_SIZE[1])],
            vel: [
                (rng.gen_range(-0.5..0.5)) * 10.0,
                (rng.gen_range(-0.5..0.5)) * 10.0,
            ],
            life,
            species,
            flags,
            _pad: 0,
        });
    }
    for _ in INITIAL_COUNT..BOID_CAPACITY {
        boids.push(Boid {
            pos: [0.0, 0.0],
            vel: [0.0, 0.0],
            life: 0.0,
            species: 0,
            flags: 0,
            _pad: 0,
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

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_title("Boids GPU (wgpu)")
        .with_inner_size(PhysicalSize::new(1000, 1000))
        .build(&event_loop)
        .expect("window");

    let mut state = pollster::block_on(State::new(&window));

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, target| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => state.resize(size),
                WindowEvent::ScaleFactorChanged {
                    mut inner_size_writer,
                    ..
                } => {
                    let size = window.inner_size();
                    let _ = inner_size_writer.request_inner_size(size);
                    state.resize(size);
                }
                WindowEvent::RedrawRequested => {
                    if let Err(err) = state.render() {
                        match err {
                            wgpu::SurfaceError::Lost => state.resize(state.size),
                            wgpu::SurfaceError::OutOfMemory => target.exit(),
                            wgpu::SurfaceError::Outdated => {}
                            wgpu::SurfaceError::Timeout => {}
                        }
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        })
        .expect("run");
}
