use crate::{
    bond::EMPTY,
    scene::{Scene, CELL},
    tool::Tool,
    ui::Rect,
};
use voxel_physics_lab::world::{GeneratedCellContent, GeneratedWorld, WORLD_SIDE_M};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::{collections::HashSet, sync::Arc};
use wgpu::util::DeviceExt;
use winit::window::Window;

// wgpu 0.19.4 API checked with cargo check; GPU/window runtime validation remains.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    center: [f32; 3],
    size: [f32; 3],
    color: [f32; 4],
    rotation: [f32; 4],
    surface: f32,
}
fn surface_kind(content: GeneratedCellContent) -> f32 {
    match content { GeneratedCellContent::Terrain => 1.0, GeneratedCellContent::PlantWood => 2.0, _ => 0.0 }
}
impl Instance {
    fn cube(center: Vec3, rotation: Quat, color: [f32; 4]) -> Self {
        Self {
            center: center.to_array(),
            size: [CELL * 0.98; 3],
            color,
            rotation: rotation.to_array(),
            surface: 0.0,
        }
    }
}
fn push_plant_lod(
    instances: &mut Vec<Instance>,
    occupied: &mut HashSet<(i32, i32, i32, u8, u8)>,
    voxel: &voxel_physics_lab::world::TerrainVoxel,
    lod: f32,
    lod_level: u8,
    opacity: f32,
) {
    if opacity <= 0.001 {
        return;
    }
    let material = match voxel.content {
        GeneratedCellContent::PlantWood => 0,
        GeneratedCellContent::PlantFoliage => 1,
        _ => 2,
    };
    if lod_level == 0 {
        let mut color = voxel.color;
        color[3] = opacity;
        instances.push(Instance {
            center: voxel.center.to_array(),
            size: voxel.size.to_array(),
            color,
            rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
        });
        return;
    }
    // A seed is a 1 cm biological state marker. Keep it at native display
    // scale nearby instead of expanding it into a tree-sized LOD billboard.
    if voxel.content == GeneratedCellContent::Seed {
        return;
    }
    // Coarse LOD cells summarize occupied plant voxels on a single world
    // aligned cubic lattice. Extending each branch along a quantized fiber
    // direction made long, thin bars that changed the tree silhouette and
    // opened gaps at diagonal joints. Sampling occupied cells into cubes keeps
    // the branch network connected and preserves the voxelized silhouette.
    let world_origin = Vec3::new(-WORLD_SIDE_M * 0.5, -WORLD_SIDE_M * 0.5, 0.0);
    let local_center = voxel.center - world_origin;
    let lod_cell = (
        (local_center.x / lod).floor() as i32,
        (local_center.y / lod).floor() as i32,
        (local_center.z / lod).floor() as i32,
    );
    let key = (
        lod_cell.0,
        lod_cell.1,
        lod_cell.2,
        lod_level,
        material,
    );
    if !occupied.insert(key) {
        return;
    }
    let center = world_origin
        + Vec3::new(
            (key.0 as f32 + 0.5) * lod,
            (key.1 as f32 + 0.5) * lod,
            (key.2 as f32 + 0.5) * lod,
        );
    let mut color = voxel.color;
    color[3] = opacity;
    instances.push(Instance {
        center: center.to_array(),
        // Adjacent render cells should meet at their shared voxel boundary.
        // Shrinking coarse cells makes visible gaps between branch voxels,
        // especially at the distant LODs where one cell is already an
        // approximation of several source cells.
        size: [lod; 3],
        color,
        rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
    });
}
type WorldCacheKey = (u64, u64, i32, i32, i32);

#[derive(Default)]
struct WorldGeometryCache {
    world_instance_cache: Vec<Instance>,
    world_transition_cache: Vec<Instance>,
    world_terrain_instance_count: usize,
    world_transition_terrain_count: usize,
}

impl WorldGeometryCache {
    fn restrict_to_bounds(&mut self, world: &GeneratedWorld) {
        let Some((min,max)) = world.view_bounds else { return; };
        fn clip(instances: &mut Vec<Instance>, terrain: &mut usize, min: Vec3, max: Vec3) {
            let mut retained_terrain = 0;
            let old_terrain = *terrain;
            let mut index = 0;
            instances.retain_mut(|instance| {
                let was_terrain = index < old_terrain;
                index += 1;
                let center = Vec3::from_array(instance.center);
                let size = Vec3::from_array(instance.size);
                let lower = (center-size*0.5).max(min);
                let upper = (center+size*0.5).min(max);
                if upper.x <= lower.x || upper.y <= lower.y || upper.z <= lower.z { return false; }
                instance.center = ((lower+upper)*0.5).to_array();
                instance.size = (upper-lower).to_array();
                if was_terrain { retained_terrain += 1; }
                true
            });
            *terrain = retained_terrain;
        }
        clip(&mut self.world_instance_cache,&mut self.world_terrain_instance_count,min,max);
        clip(&mut self.world_transition_cache,&mut self.world_transition_terrain_count,min,max);
    }
    fn overview(world: &GeneratedWorld, camera_position: Vec3) -> Self {
        let world_instance_cache = world.voxels.iter()
            .filter(|voxel| {
                let delta = voxel.center - camera_position;
                let radius = 1_000.0 + voxel.size.x.max(voxel.size.y) * 0.5;
                delta.x * delta.x + delta.y * delta.y <= radius * radius
            })
            .map(|voxel| Instance {
                center: voxel.center.to_array(),
                size: voxel.size.to_array(),
                color: voxel.color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
            })
            .collect();
        let mut cache = Self { world_instance_cache, ..Self::default() };
        cache.restrict_to_bounds(world);
        cache
    }

    fn build(world: &GeneratedWorld, camera_position: Vec3, load_center: Vec3) -> Self {
        let mut cache = Self::default();
        cache.world_instance_cache.clear();
        cache.world_transition_cache.clear();
        cache.world_instance_cache.reserve(100_000);
        cache.world_transition_cache.reserve(16_000);
        for voxel in world.terrain_patch(load_center, 1_000.0) {
            // Render only the surface and a short skirt. The sampled
            // REV terrain still extends below this; hidden solid depth
            // does not need tens of metres of rasterized cube faces.
            let skirt = (voxel.size.x * 0.75).max(0.1);
            let instance = Instance {
                center: [voxel.center.x, voxel.center.y, voxel.size.z - skirt * 0.5],
                size: [voxel.size.x, voxel.size.y, skirt],
                color: voxel.color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
            };
            if voxel.color[3] < 0.999 {
                cache.world_transition_cache.push(instance);
            } else {
                cache.world_instance_cache.push(instance);
            }
        }
        cache.world_terrain_instance_count = cache.world_instance_cache.len();
        cache.world_transition_terrain_count = cache.world_transition_cache.len();
        let mut occupied_lod_cells = HashSet::new();
        // Only refine cells actually rendered; the old 18 m query discarded
        // every result beyond 10.8 m after building it.
        let local_wood = world.plant_wood_voxels_near(camera_position, 10.8);
        for voxel in &world.voxels {
            let is_terrain = voxel.content == GeneratedCellContent::Terrain;
            let is_tree_cell = matches!(
                voxel.content,
                GeneratedCellContent::Seed
                    | GeneratedCellContent::PlantWood
                    | GeneratedCellContent::PlantFoliage
            );
            let delta = voxel.center - camera_position;
            let distance_squared = delta.length_squared();
            if is_terrain || distance_squared > 1_000_000.0 {
                continue;
            }
            // Close wood is replaced by its 1 cm local refinement. Keep
            // the coarse level in the blend band so the tree never fades
            // into empty space while crossing the detail boundary.
            if voxel.content == GeneratedCellContent::PlantWood
                && distance_squared <= 5.2 * 5.2
            {
                continue;
            }
            if is_tree_cell {
                const LOD_RADII: [f32; 5] = [8.0, 32.0, 128.0, 512.0, 1_000.0];
                // Keep the coarsest plant cell below the scale of a
                // small tree crown. Multi-metre cubes erase branch
                // shape and turn distant trees into block silhouettes.
                const LOD_SIZES: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.4];
                let mut transition = None;
                for boundary in 0..4 {
                    let radius = LOD_RADII[boundary];
                    let width = radius * 0.35;
                    let start = radius - width;
                    let end = radius + width;
                    if (start * start..=end * end).contains(&distance_squared) {
                        let distance = distance_squared.sqrt();
                        let t = ((distance - start) / (end - start)).clamp(0.0, 1.0);
                        let t = t * t * (3.0 - 2.0 * t);
                        transition = Some((boundary, t));
                        break;
                    }
                }
                if let Some((lower_level, blend)) = transition {
                    if lower_level != 0 || voxel.content != GeneratedCellContent::PlantWood {
                        push_plant_lod(
                            &mut cache.world_instance_cache,
                            &mut occupied_lod_cells,
                            voxel,
                            LOD_SIZES[lower_level],
                            lower_level as u8,
                            1.0 - blend,
                        );
                    }
                    push_plant_lod(
                        &mut cache.world_transition_cache,
                        &mut occupied_lod_cells,
                        voxel,
                        LOD_SIZES[lower_level + 1],
                        (lower_level + 1) as u8,
                        blend,
                    );
                } else {
                    let lod_level = LOD_RADII
                        .iter()
                        .filter(|&&radius| distance_squared >= radius * radius)
                        .count();
                    let lod_level = lod_level.min(4);
                    let fade = if distance_squared > 850.0 * 850.0 {
                        let distance = distance_squared.sqrt();
                        let t = ((distance - 850.0) / 150.0).clamp(0.0, 1.0);
                        let t = t * t * (3.0 - 2.0 * t);
                        1.0 - t
                    } else {
                        1.0
                    };
                    let destination = if fade < 0.999 {
                        &mut cache.world_transition_cache
                    } else {
                        &mut cache.world_instance_cache
                    };
                    push_plant_lod(
                        destination,
                        &mut occupied_lod_cells,
                        voxel,
                        LOD_SIZES[lod_level],
                        lod_level as u8,
                        fade,
                    );
                }
            } else {
                // Non-tree ground plants and fungi are sparse and keep
                // their original size at every LOD.
                cache.world_instance_cache.push(Instance {
                    center: voxel.center.to_array(),
                    size: voxel.size.to_array(),
                    color: voxel.color,
                    rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
                });
            }
        }
        for voxel in local_wood {
            let distance_squared = (voxel.center - camera_position).length_squared();
            if distance_squared > 10.8 * 10.8 {
                continue;
            }
            let opacity = if distance_squared <= 5.2 * 5.2 {
                1.0
            } else {
                let distance = distance_squared.sqrt();
                let t = ((distance - 5.2) / (10.8 - 5.2)).clamp(0.0, 1.0);
                let t = t * t * (3.0 - 2.0 * t);
                1.0 - t
            };
            let mut color = voxel.color;
            color[3] = opacity;
            cache.world_instance_cache.push(Instance {
                center: voxel.center.to_array(),
                size: voxel.size.to_array(),
                color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
            });
        }
        cache.restrict_to_bounds(world);
        cache
    }
}

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: Vec3,
    free_position: Option<Vec3>,
    first_person: bool,
    pub free_speed_mps: f32,
    pub look_sensitivity: f32,
}
impl Camera {
    pub fn new() -> Self {
        Self {
            yaw: -1.0,
            pitch: 0.35,
            radius: 5.0,
            target: Vec3::new(0., 0., 1.5),
            free_position: None,
            first_person: false,
            free_speed_mps: 20.0,
            look_sensitivity: 0.005,
        }
    }
    fn offset_direction(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
    }
    pub fn forward(&self) -> Vec3 {
        -self.offset_direction()
    }
    pub fn is_freecam(&self) -> bool {
        self.free_position.is_some()
    }
    pub fn is_first_person(&self) -> bool {
        self.first_person && self.free_position.is_none()
    }
    pub fn set_first_person(&mut self, eye_position: Vec3) {
        self.target = eye_position;
        self.first_person = true;
        self.free_position = None;
    }
    pub fn set_orbit(&mut self, target: Vec3, radius: f32) {
        self.target = target;
        self.radius = radius;
        self.first_person = false;
        self.free_position = None;
    }
    pub fn enable_freecam(&mut self) {
        if self.free_position.is_none() {
            let position = if self.first_person {
                self.target
            } else {
                self.target + self.offset_direction() * self.radius
            };
            self.free_position = Some(position);
            self.free_speed_mps = (self.radius * 0.06).clamp(10.0, 200.0);
        }
    }
    pub fn disable_freecam(&mut self) {
        if let Some(position) = self.free_position.take() {
            if self.first_person {
                self.target = position;
            } else {
                self.target = position + self.forward() * self.radius;
            }
        }
    }
    pub fn rotate_freecam(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx as f32 * self.look_sensitivity;
        // Mouse-down looks down: positive pitch lowers the view direction in
        // this camera convention because forward is the negative orbit offset.
        self.pitch = (self.pitch + dy as f32 * self.look_sensitivity).clamp(-1.55, 1.55);
    }
    pub fn position(&self) -> Vec3 {
        self.free_position
            .unwrap_or_else(|| {
                if self.first_person {
                    self.target
                } else {
                    self.target + self.offset_direction() * self.radius
                }
            })
    }
    pub fn move_freecam(&mut self, right: f32, forward: f32, up: f32, dt: f32, fast: bool) {
        if self.free_position.is_none() {
            return;
        }
        let view_forward = self.forward();
        let mut view_right = view_forward.cross(Vec3::Z);
        if view_right.length_squared() < 1.0e-6 {
            view_right = Vec3::X;
        } else {
            view_right = view_right.normalize();
        }
        let mut direction = view_right * right + view_forward * forward + Vec3::Z * up;
        if direction.length_squared() > 1.0 {
            direction = direction.normalize();
        }
        let speed_multiplier = if fast { 5.0 } else { 1.0 };
        if let Some(position) = self.free_position.as_mut() {
            *position += direction * self.free_speed_mps * speed_multiplier * dt;
        }
    }
    pub fn adjust_free_speed(&mut self, amount: f32) {
        self.free_speed_mps = (self.free_speed_mps * (amount * 0.2).exp()).clamp(0.5, 5000.0);
    }
    fn matrix(&self, aspect: f32) -> Mat4 {
        let offset = self.offset_direction() * self.radius;
        // The overview camera can sit several kilometres from the map. Keep
        // the near plane proportional to camera distance for depth precision,
        // and include the whole seeded map in the far plane.
        let near = (self.radius * 0.00001).clamp(0.01, 0.5);
        let far = (self.radius + WORLD_SIDE_M * 2.0).max(100.0);
        let (eye, target) = match self.free_position {
            Some(position) => (position, position - self.offset_direction()),
            None if self.first_person => (self.target, self.target + self.forward()),
            None => (self.target + offset, self.target),
        };
        Mat4::perspective_rh(50f32.to_radians(), aspect, near, far)
            * Mat4::look_at_rh(eye, target, Vec3::Z)
    }
}
pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    transition_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    terrain_vertices: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    physics_instances: wgpu::Buffer,
    physics_capacity: usize,
    world_instance_cache: Vec<Instance>,
    world_transition_cache: Vec<Instance>,
    world_terrain_instance_count: usize,
    world_transition_terrain_count: usize,
    cached_world_key: Option<WorldCacheKey>,
    world_cache_job: Option<std::sync::mpsc::Receiver<(WorldCacheKey, WorldGeometryCache)>>,
    world_snapshot: Option<((u64, u64), Arc<GeneratedWorld>)>,
    ui_instances: wgpu::Buffer,
    ui_capacity: usize,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    screen_buffer: wgpu::Buffer,
    screen_group: wgpu::BindGroup,
    depth: wgpu::TextureView,
}
fn depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}
fn instance_buffer(device: &wgpu::Device, count: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: (count.max(1) * stride) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn terrain_surface_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(30);
    let mut quad = |points: [Vec3; 4], normal: Vec3| {
        for index in [0, 1, 2, 0, 2, 3] {
            vertices.push(Vertex {
                position: points[index].to_array(),
                normal: normal.to_array(),
            });
        }
    };
    let half = 0.5;
    quad(
        [
            Vec3::new(-half, -half, half),
            Vec3::new(half, -half, half),
            Vec3::new(half, half, half),
            Vec3::new(-half, half, half),
        ],
        Vec3::Z,
    );
    for (normal, points) in [
        (
            Vec3::X,
            [
                Vec3::new(half, -half, half),
                Vec3::new(half, half, half),
                Vec3::new(half, half, -half),
                Vec3::new(half, -half, -half),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                Vec3::new(-half, half, half),
                Vec3::new(-half, -half, half),
                Vec3::new(-half, -half, -half),
                Vec3::new(-half, half, -half),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(half, half, half),
                Vec3::new(-half, half, half),
                Vec3::new(-half, half, -half),
                Vec3::new(half, half, -half),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                Vec3::new(-half, -half, half),
                Vec3::new(half, -half, half),
                Vec3::new(half, -half, -half),
                Vec3::new(-half, -half, -half),
            ],
        ),
    ] {
        quad(points, normal);
    }
    vertices
}
fn uniform(
    device: &wgpu::Device,
    size: u64,
) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, layout, group)
}
impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No graphics adapter")?;
        let adapter_info = adapter.get_info();
        eprintln!(
            "wgpu renderer adapter: {} ({:?}, {:?}) — {}",
            adapter_info.name, adapter_info.device_type, adapter_info.backend, adapter_info.driver
        );
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("voxel lab"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let (camera_buffer, camera_layout, camera_group) = uniform(&device, 64);
        let (screen_buffer, screen_layout, screen_group) = uniform(&device, 16);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/voxel.wgsl").into()),
        });
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let ui_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&screen_layout],
            push_constant_ranges: &[],
        });
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let instance_attrs = wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32x3, 4 => Float32x4, 5 => Float32x4, 6 => Float32];
        let ui_attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let target = Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instanced cubes"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[target.clone()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });
        let transition_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LOD transition blend"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[target.clone()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI rectangles"),
            layout: Some(&ui_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: "vs",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Rect>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &ui_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: "fs",
                targets: &[target],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });
        let mut cube = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut normal = Vec3::ZERO;
                normal[axis] = sign;
                let u = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 1) % 3];
                let v = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 2) % 3];
                for [a, b] in [
                    [-1., -1.],
                    [1., -1.],
                    [-1., 1.],
                    [-1., 1.],
                    [1., -1.],
                    [1., 1.],
                ] {
                    cube.push(Vertex {
                        position: ((normal + a * u + b * v) * 0.5).to_array(),
                        normal: normal.to_array(),
                    });
                }
            }
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube"),
            contents: bytemuck::cast_slice(&cube),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let terrain_mesh = terrain_surface_vertices();
        let terrain_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain surface and skirts"),
            contents: bytemuck::cast_slice(&terrain_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instances = instance_buffer(&device, 100_000, std::mem::size_of::<Instance>());
        let physics_instances = instance_buffer(&device, 1024, std::mem::size_of::<Instance>());
        let ui_instances = instance_buffer(&device, 10_000, std::mem::size_of::<Rect>());
        let depth = depth(&device, &config);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            transition_pipeline,
            ui_pipeline,
            vertices,
            terrain_vertices,
            instances,
            instance_capacity: 100_000,
            physics_instances,
            physics_capacity: 1024,
            world_instance_cache: Vec::new(),
            world_transition_cache: Vec::new(),
            world_terrain_instance_count: 0,
            world_transition_terrain_count: 0,
            cached_world_key: None,
            world_cache_job: None,
            world_snapshot: None,
            ui_instances,
            ui_capacity: 10_000,
            camera_buffer,
            camera_group,
            screen_buffer,
            screen_group,
            depth,
        })
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = depth(&self.device, &self.config);
    }
    pub fn draw(
        &mut self,
        scene: &Scene,
        tool: &Option<Tool>,
        generated_world: Option<&GeneratedWorld>,
        physics: Option<&voxel_physics_lab::world_physics::PhysicsFrame>,
        camera: &Camera,
        ui: &[Rect],
        show_bonds: bool,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut world_cache_changed = false;
        let physical_instances: Vec<_> = physics.into_iter().flat_map(|frame| &frame.poses)
            .map(|&(p,q)| Instance { center:p.to_array(), size:[CELL;3],
                color:[0.62,0.35,0.13,1.0], rotation:q.to_array(), surface:2.0 }).collect();
        if physical_instances.len() > self.physics_capacity {
            self.physics_capacity = physical_instances.len().next_power_of_two();
            self.physics_instances = instance_buffer(&self.device,self.physics_capacity,
                std::mem::size_of::<Instance>());
        }
        if !physical_instances.is_empty() {
            self.queue.write_buffer(&self.physics_instances,0,bytemuck::cast_slice(&physical_instances));
        }
        if let Some(world) = generated_world {
            let camera_position = camera.position();
            // Keep a seeded 1 km radius around the camera. The cache follows
            // the camera in 8 m increments to avoid rebuilding every frame.
            let camera_cell = (
                (camera_position.x / 8.0).floor() as i32,
                (camera_position.y / 8.0).floor() as i32,
                (camera_position.z / 8.0).floor() as i32,
            );
            let load_center = Vec3::new(
                (camera_cell.0 as f32 + 0.5) * 8.0,
                (camera_cell.1 as f32 + 0.5) * 8.0,
                (camera_cell.2 as f32 + 0.5) * 8.0,
            );
            let (seed_revision, geometry_revision) = world.render_cache_key();
            let cache_key = (
                seed_revision,
                geometry_revision,
                camera_cell.0,
                camera_cell.1,
                camera_cell.2,
            );
            let source_key = (seed_revision, geometry_revision);
            let reset_world = self.world_snapshot.as_ref().map_or(true, |(key, _)| {
                key.0 != seed_revision || key.1 > geometry_revision
            });
            if reset_world {
                // Show the existing overview immediately while local geometry
                // is prepared. No synchronous centimetre rasterization here.
                let overview = WorldGeometryCache::overview(world, camera_position);
                self.world_instance_cache = overview.world_instance_cache;
                self.world_transition_cache = overview.world_transition_cache;
                self.world_terrain_instance_count = overview.world_terrain_instance_count;
                self.world_transition_terrain_count = overview.world_transition_terrain_count;
                self.cached_world_key = None;
                self.world_snapshot = Some((source_key, Arc::new(world.clone())));
                world_cache_changed = true;
            }
            if let Some(receiver) = &self.world_cache_job {
                match receiver.try_recv() {
                    Ok((built_key, geometry)) => {
                        self.world_cache_job = None;
                        // Accept a completed earlier growth snapshot while
                        // requesting the latest one, so accelerated growth
                        // cannot starve all completed geometry updates.
                        if built_key.0 == seed_revision && built_key.1 <= geometry_revision {
                            self.world_instance_cache = geometry.world_instance_cache;
                            self.world_transition_cache = geometry.world_transition_cache;
                            self.world_terrain_instance_count = geometry.world_terrain_instance_count;
                            self.world_transition_terrain_count = geometry.world_transition_terrain_count;
                            self.cached_world_key = Some(built_key);
                            world_cache_changed = true;
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.world_cache_job = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
            }
            if self.cached_world_key != Some(cache_key) && self.world_cache_job.is_none() {
                if self.world_snapshot.as_ref().map(|(key, _)| *key) != Some(source_key) {
                    self.world_snapshot = Some((source_key, Arc::new(world.clone())));
                }
                let snapshot = Arc::clone(&self.world_snapshot.as_ref().unwrap().1);
                let (sender, receiver) = std::sync::mpsc::channel();
                match std::thread::Builder::new().name("world-render-geometry".into()).spawn(move || {
                    let start = std::time::Instant::now();
                    let geometry = WorldGeometryCache::build(&snapshot, camera_position, load_center);
                    if std::env::var_os("VOXEL_PROFILE_WORLD_GEOMETRY").is_some() {
                        eprintln!("world geometry: build_ms={:.2} instances={} transition_instances={}",
                            start.elapsed().as_secs_f64() * 1_000.0,
                            geometry.world_instance_cache.len(), geometry.world_transition_cache.len());
                    }
                    let _ = sender.send((cache_key, geometry));
                }) {
                    Ok(_) => self.world_cache_job = Some(receiver),
                    Err(error) => eprintln!("world geometry worker could not start: {error}"),
                }
            }
        } else {
            self.cached_world_key = None;
            self.world_snapshot = None;
            self.world_terrain_instance_count = 0;
        }
        let mut instances: Vec<_> = if generated_world.is_some() {
            Vec::new()
        } else {
            let damage = scene.incident_bond_damage();
            scene
                .positions
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let color = if scene.voxels[i].kinematic {
                        [0.24, 0.28, 0.3, 1.]
                    } else if scene.axis_targets[i].iter().any(Option::is_some) {
                        [0.18, 0.55, 0.85, 1.]
                    } else if damage[i].1 >= 2 {
                        [0.83, 0.28, 0.24, 1.]
                    } else if damage[i].1 == 1 {
                        [0.92, 0.52, 0.24, 1.]
                    } else {
                        let d = damage[i].0;
                        [0.55 + 0.39 * d, 0.32 + 0.43 * d, 0.12 + 0.18 * d, 1.]
                    };
                    Instance::cube(p.as_vec3(), scene.rotations[i], color)
                })
                .collect()
        };
        if generated_world.is_none() {
        if let Some(t) = tool {
            if let Some(m) = t.mount {
                // External guide visualization, not an additional collision body.
                let direction = if m.direction == Vec3::ZERO {
                    Vec3::X
                } else {
                    m.direction
                };
                let offset = Vec3::Y * 0.045;
                instances.push(Instance {
                    center: (m.world_base + offset.as_dvec3()).as_vec3().to_array(),
                    size: [0.8, 0.008, 0.008],
                    color: [0.18, 0.55, 0.85, 1.0],
                    rotation: Quat::from_rotation_arc(Vec3::X, direction).to_array(),
                    surface: 0.0,
                });
                let base = t.position + (t.rotation * m.local_base).as_dvec3();
                instances.push(Instance::cube(base.as_vec3(), m.rotation, [0.18, 0.55, 0.85, 1.0]));
            }
            instances.extend(
                t.world_cells()
                    .map(|p| Instance::cube(p.as_vec3(), t.rotation, t.color)),
            );
        }
        if show_bonds {
            for axis in 0..3 {
                for (i, b) in scene.bonds[axis].iter().enumerate() {
                    if b.neighbor == EMPTY || scene.voxels[i].kinematic {
                        continue;
                    }
                    let a = scene.positions[i];
                    let d = scene.positions[b.neighbor as usize] - a;
                    if d.length_squared() < 1e-12 {
                        continue;
                    }
                    instances.push(Instance {
                        center: (a + d * 0.5).as_vec3().to_array(),
                        size: [CELL * 0.08, CELL * 0.08, d.length() as f32],
                        rotation: Quat::from_rotation_arc(Vec3::Z, d.normalize().as_vec3()).to_array(),
                        surface: 0.0,
                        color: [b.damage, 1. - b.damage, 0.2, 1.],
                    });
                }
            }
            // Shrink cubes to expose the lattice; absent bonds are never drawn.
            for i in 0..scene.voxels.len() {
                if !scene.voxels[i].kinematic {
                    instances[i].size = [CELL * 0.35; 3];
                }
            }
        }
        }
        let static_instance_count = if generated_world.is_some() {
            self.world_instance_cache.len() + self.world_transition_cache.len()
        } else {
            instances.len()
        };
        let instance_count = static_instance_count;
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            self.instances = instance_buffer(
                &self.device,
                self.instance_capacity,
                std::mem::size_of::<Instance>(),
            );
        }
        if ui.len() > self.ui_capacity {
            self.ui_capacity = ui.len().next_power_of_two();
            self.ui_instances =
                instance_buffer(&self.device, self.ui_capacity, std::mem::size_of::<Rect>());
        }
        if generated_world.is_some() {
            if world_cache_changed {
                self.queue.write_buffer(
                    &self.instances,
                    0,
                    bytemuck::cast_slice(&self.world_instance_cache),
                );
                if !self.world_transition_cache.is_empty() {
                    let transition_offset = self.world_instance_cache.len()
                        * std::mem::size_of::<Instance>();
                    self.queue.write_buffer(
                        &self.instances,
                        transition_offset as u64,
                        bytemuck::cast_slice(&self.world_transition_cache),
                    );
                }
            }
        } else {
            self.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }
        self.queue
            .write_buffer(&self.ui_instances, 0, bytemuck::cast_slice(ui));
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(
                &camera
                    .matrix(self.config.width as f32 / self.config.height as f32)
                    .to_cols_array(),
            ),
        );
        self.queue.write_buffer(
            &self.screen_buffer,
            0,
            bytemuck::cast_slice(&[self.config.width as f32, self.config.height as f32, 0., 0.]),
        );
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.035,
                            b: 0.05,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_group, &[]);
            if generated_world.is_some() {
                if self.world_terrain_instance_count > 0 {
                    let terrain_bytes = self.world_terrain_instance_count
                        * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.terrain_vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(..terrain_bytes as u64));
                    pass.draw(0..30, 0..self.world_terrain_instance_count as u32);
                }
                // Transition instances have their own pass below. Including
                // them here draws them twice and treats terrain skirts as
                // full cubes in the opaque object pass.
                let object_count = self.world_instance_cache.len()
                    .saturating_sub(self.world_terrain_instance_count);
                if object_count > 0 {
                    let object_offset = self.world_terrain_instance_count
                        * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(object_offset as u64..));
                    pass.draw(0..36, 0..object_count as u32);
                }
            } else {
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_vertex_buffer(1, self.instances.slice(..));
                pass.draw(0..36, 0..instance_count as u32);
            }
            if !physical_instances.is_empty() {
                pass.set_vertex_buffer(0,self.vertices.slice(..));
                pass.set_vertex_buffer(1,self.physics_instances.slice(..));
                pass.draw(0..36,0..physical_instances.len() as u32);
            }
            if generated_world.is_some() && !self.world_transition_cache.is_empty() {
                pass.set_pipeline(&self.transition_pipeline);
                pass.set_bind_group(0, &self.camera_group, &[]);
                let transition_start = self.world_instance_cache.len();
                if self.world_transition_terrain_count > 0 {
                    let start_bytes = transition_start * std::mem::size_of::<Instance>();
                    let end_bytes = (transition_start + self.world_transition_terrain_count)
                        * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.terrain_vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(start_bytes as u64..end_bytes as u64));
                    pass.draw(0..30, 0..self.world_transition_terrain_count as u32);
                }
                let transition_object_start = transition_start + self.world_transition_terrain_count;
                let transition_object_count = self.world_transition_cache.len()
                    .saturating_sub(self.world_transition_terrain_count);
                if transition_object_count > 0 {
                    let start_bytes = transition_object_start * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(start_bytes as u64..));
                    pass.draw(0..36, 0..transition_object_count as u32);
                }
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI"),
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
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.screen_group, &[]);
            pass.set_vertex_buffer(0, self.ui_instances.slice(..));
            pass.draw(0..6, 0..ui.len() as u32);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
