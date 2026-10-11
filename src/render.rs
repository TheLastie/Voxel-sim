use crate::{
    bond::EMPTY,
    scene::{Scene, CELL},
    tool::Tool,
    ui::Rect,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;
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
}
impl Instance {
    fn cube(center: Vec3, rotation: Quat, color: [f32; 4]) -> Self {
        Self {
            center: center.to_array(),
            size: [CELL * 0.98; 3],
            color,
            rotation: rotation.to_array(),
        }
    }
}
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: Vec3,
}
impl Camera {
    pub fn new() -> Self {
        Self {
            yaw: -1.0,
            pitch: 0.35,
            radius: 5.0,
            target: Vec3::new(0., 0., 1.5),
        }
    }
    fn matrix(&self, aspect: f32) -> Mat4 {
        let offset = Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        ) * self.radius;
        Mat4::perspective_rh(50f32.to_radians(), aspect, 0.01, 100.0)
            * Mat4::look_at_rh(self.target + offset, self.target, Vec3::Z)
    }
}
pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
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
        let instance_attrs = wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32x3, 4 => Float32x4, 5 => Float32x4];
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
        let instances = instance_buffer(&device, 100_000, std::mem::size_of::<Instance>());
        let ui_instances = instance_buffer(&device, 10_000, std::mem::size_of::<Rect>());
        let depth = depth(&device, &config);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            ui_pipeline,
            vertices,
            instances,
            instance_capacity: 100_000,
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
        camera: &Camera,
        ui: &[Rect],
        show_bonds: bool,
    ) -> Result<(), wgpu::SurfaceError> {
        let damage = scene.incident_bond_damage();
        let mut instances: Vec<_> = scene
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let color = if scene.voxels[i].mass.is_infinite() {
                    [0.24, 0.28, 0.3, 1.]
                } else if damage[i].1 >= 2 {
                    [0.83, 0.28, 0.24, 1.]
                } else if damage[i].1 == 1 {
                    [0.92, 0.52, 0.24, 1.]
                } else {
                    let d = damage[i].0;
                    [0.55 + 0.39*d, 0.32 + 0.43*d, 0.12 + 0.18*d, 1.]
                };
                Instance::cube(p, scene.rotations[i], color)
            })
            .collect();
        if let Some(t) = tool {
            instances.extend(
                t.world_cells()
                    .map(|p| Instance::cube(p, t.rotation, t.color)),
            );
        }
        if show_bonds {
            for axis in 0..3 {
                for (i, b) in scene.bonds[axis].iter().enumerate() {
                    if b.neighbor == EMPTY || scene.voxels[i].mass.is_infinite() {
                        continue;
                    }
                    let a = scene.positions[i];
                    let d = scene.positions[b.neighbor as usize] - a;
                    if d.length_squared() < 1e-12 {
                        continue;
                    }
                    instances.push(Instance {
                        center: (a + d * 0.5).to_array(),
                        size: [CELL * 0.08, CELL * 0.08, d.length()],
                        rotation: Quat::from_rotation_arc(Vec3::Z, d.normalize()).to_array(),
                        color: [b.damage, 1. - b.damage, 0.2, 1.],
                    });
                }
            }
            // Shrink cubes to expose the lattice; absent bonds are never drawn.
            for i in 0..scene.voxels.len() {
                if scene.voxels[i].mass.is_finite() {
                    instances[i].size = [CELL * 0.35; 3];
                }
            }
        }
        if instances.len() > self.instance_capacity {
            self.instance_capacity = instances.len().next_power_of_two();
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
        self.queue
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
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
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_vertex_buffer(1, self.instances.slice(..));
            pass.draw(0..36, 0..instances.len() as u32);
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
