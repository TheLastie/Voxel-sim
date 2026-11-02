//! Native, read-only renderer for snapshots emitted by `live_tests::observe_scene`.
#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use serde::Deserialize;
use std::{
    error::Error,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use voxel_physics_lab::live_tests::LiveTestFrame;
use voxel_physics_lab::ui_text::{draw_text, Rect};
use voxel_physics_lab::voxel_surface::exterior_face_masks;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    face_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    position: [f32; 3],
    color: [f32; 3],
    face_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
    cell_size: f32,
    padding: [f32; 3],
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default)]
struct RunStatus {
    label: String,
    status: String,
    passed: usize,
    failed: usize,
    ignored: usize,
    detail: String,
}

impl Default for RunStatus {
    fn default() -> Self {
        Self {
            label: "test run".to_owned(),
            status: "starting".to_owned(),
            passed: 0,
            failed: 0,
            ignored: 0,
            detail: String::new(),
        }
    }
}

struct LiveFeed {
    frame: Arc<Mutex<Option<FrameSnapshot>>>,
    status: Arc<Mutex<Option<RunStatus>>>,
}

struct FrameSnapshot {
    frame: LiveTestFrame,
    process_epoch: u64,
    surface_masks: Vec<u8>,
}

struct Viewer {
    window: std::sync::Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    hud_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    instance_count: u32,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    screen_buffer: wgpu::Buffer,
    screen_group: wgpu::BindGroup,
    hud_buffer: wgpu::Buffer,
    hud_capacity: usize,
    hud_count: u32,
    hud_rects: Vec<Rect>,
    hud_dirty: bool,
    last_hud_age_second: u64,
    depth: wgpu::TextureView,
    frame: Option<LiveTestFrame>,
    scene_identity: Option<(String, u64)>,
    anchor_origin: Option<[f64; 3]>,
    last_frame_at: Option<Instant>,
    run_status: Option<RunStatus>,
    target: Vec3,
    yaw: f32,
    pitch: f32,
    radius: f32,
    cell_size: f32,
    dragging: Option<MouseButton>,
    last_cursor: Option<(f64, f64)>,
}

impl Viewer {
    async fn new(window: std::sync::Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No graphics adapter found")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("live test viewer"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("live test camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera layout"),
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
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("live cube shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("live test pipeline layout"),
            bind_group_layouts: &[&camera_layout],
            push_constant_ranges: &[],
        });
        let vertex_attributes =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 4 => Uint32];
        let instance_attributes =
            wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32x3, 5 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("live test REV cubes"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attributes,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attributes,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });
        let screen_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HUD screen dimensions"),
            contents: bytemuck::cast_slice(&[
                size.width.max(1) as f32,
                size.height.max(1) as f32,
                0.0_f32,
                0.0_f32,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let screen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HUD screen layout"),
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
        let screen_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HUD screen group"),
            layout: &screen_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: screen_buffer.as_entire_binding(),
            }],
        });
        let hud_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("project UI text shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/ui.wgsl").into()),
        });
        let hud_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HUD pipeline layout"),
            bind_group_layouts: &[&screen_layout],
            push_constant_ranges: &[],
        });
        let hud_attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let hud_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("live test HUD"),
            layout: Some(&hud_layout),
            vertex: wgpu::VertexState {
                module: &hud_shader,
                entry_point: "vs",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Rect>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &hud_attributes,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &hud_shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });
        let vertices = cube_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube mesh"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("REV instances"),
            size: std::mem::size_of::<Instance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hud_capacity = 1;
        let hud_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("live test HUD rectangles"),
            size: (hud_capacity * std::mem::size_of::<Rect>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let depth = depth_view(&device, &config);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            hud_pipeline,
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            instance_buffer,
            instance_capacity: 1,
            instance_count: 0,
            camera_buffer,
            camera_group,
            screen_buffer,
            screen_group,
            hud_buffer,
            hud_capacity,
            hud_count: 0,
            hud_rects: Vec::new(),
            hud_dirty: true,
            last_hud_age_second: u64::MAX,
            depth,
            frame: None,
            scene_identity: None,
            anchor_origin: None,
            last_frame_at: None,
            run_status: None,
            target: Vec3::ZERO,
            yaw: 0.65,
            pitch: 0.45,
            radius: 2.0,
            cell_size: 0.01,
            dragging: None,
            last_cursor: None,
        })
    }

    fn accept_frame(&mut self, snapshot: FrameSnapshot) {
        let frame = snapshot.frame;
        let identity = (frame.test_name.clone(), snapshot.process_epoch);
        let new_scene = self.scene_identity.as_ref() != Some(&identity);
        if new_scene {
            self.anchor_origin = Some(frame.origin);
            self.target = bounds_center(&frame);
            self.radius = bounds_radius(&frame).max(frame.cell_m * 3.0);
            self.scene_identity = Some(identity);
        }
        let anchor_origin = self.anchor_origin.unwrap_or(frame.origin);
        let origin_offset = Vec3::new(
            (frame.origin[0] - anchor_origin[0]) as f32,
            (frame.origin[1] - anchor_origin[1]) as f32,
            (frame.origin[2] - anchor_origin[2]) as f32,
        );
        self.cell_size = frame.cell_m.max(1e-6);
        let instances: Vec<Instance> = frame
            .positions
            .iter()
            .zip(&frame.colors)
            .zip(&snapshot.surface_masks)
            .filter_map(|((position, color), mask)| {
                (*mask != 0).then_some(Instance {
                    position: (Vec3::from_array(*position) + origin_offset).to_array(),
                    color: *color,
                    face_mask: u32::from(*mask),
                })
            })
            .collect();
        if instances.len() > self.instance_capacity {
            self.instance_capacity = instances.len().next_power_of_two();
            self.instance_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("REV instances"),
                size: (self.instance_capacity * std::mem::size_of::<Instance>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !instances.is_empty() {
            self.queue
                .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
        }
        self.instance_count = instances.len() as u32;
        self.frame = Some(frame);
        self.last_frame_at = Some(Instant::now());
        self.hud_dirty = true;
        self.update_title(false);
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.depth = depth_view(&self.device, &self.config);
        self.queue.write_buffer(
            &self.screen_buffer,
            0,
            bytemuck::cast_slice(&[size.width as f32, size.height as f32, 0.0_f32, 0.0_f32]),
        );
        self.hud_dirty = true;
    }

    fn update_title(&self, stale: bool) {
        let run = self
            .run_status
            .as_ref()
            .map(|status| {
                format!(
                    " | {}: {} | passed {} failed {}{}",
                    status.label,
                    status.status,
                    status.passed,
                    status.failed,
                    if status.detail.is_empty() {
                        String::new()
                    } else {
                        format!(" | {}", status.detail)
                    }
                )
            })
            .unwrap_or_default();
        let scene = match (&self.frame, self.last_frame_at) {
            (Some(frame), Some(at)) => {
                let state = if stale {
                    format!(
                        "LAST SCENE {} · stale {:.1}s",
                        frame.test_name,
                        at.elapsed().as_secs_f32()
                    )
                } else {
                    format!("scene {}", frame.test_name)
                };
                format!(
                    " | {state} | t={:.4}s | {} REVs | step={:.2}ms | frame #{}",
                    frame.simulated_seconds, frame.voxel_count, frame.wall_ms, frame.sequence,
                )
            }
            _ => " | waiting for a scene snapshot".to_owned(),
        };
        let title = format!("Live physics tests{run}{scene}");
        self.window.set_title(&title);
    }

    fn refresh_hud(&mut self) {
        let age = self
            .last_frame_at
            .map(|at| at.elapsed())
            .unwrap_or_default();
        let age_second = age.as_secs();
        if !self.hud_dirty && age_second == self.last_hud_age_second {
            return;
        }
        self.last_hud_age_second = age_second;
        self.hud_dirty = false;
        self.hud_rects.clear();
        let width = (self.config.width as f32 - 24.0).clamp(280.0, 700.0);
        let max_chars = ((width - 28.0) / 7.5).max(20.0) as usize;
        let lines = [
            "LIVE PHYSICS TESTS".to_owned(),
            self.run_status.as_ref().map_or_else(
                || "RUN waiting for runner status".to_owned(),
                |status| format!("RUN {}", status.label),
            ),
            self.run_status.as_ref().map_or_else(
                || "RESULT waiting".to_owned(),
                |status| {
                    format!(
                        "RESULT {} | passed {} failed {} ignored {}",
                        status.status, status.passed, status.failed, status.ignored
                    )
                },
            ),
            self.run_status
                .as_ref()
                .filter(|status| !status.detail.is_empty())
                .map_or_else(
                    || "DETAIL -".to_owned(),
                    |status| format!("DETAIL {}", status.detail),
                ),
            self.frame.as_ref().map_or_else(
                || "SCENE waiting for a physical snapshot".to_owned(),
                |frame| {
                    if age > Duration::from_millis(500) {
                        format!(
                            "LAST SCENE {} | stale {:.1}s",
                            frame.test_name,
                            age.as_secs_f32(),
                        )
                    } else {
                        format!("SCENE {}", frame.test_name,)
                    }
                },
            ),
            self.frame.as_ref().map_or_else(
                || "PHYSICS waiting".to_owned(),
                |frame| {
                    format!(
                        "t={:.4}s | {} REVs | {} surface | step {:.2}ms | frame #{}",
                        frame.simulated_seconds,
                        frame.voxel_count,
                        self.instance_count,
                        frame.wall_ms,
                        frame.sequence
                    )
                },
            ),
            "CAMERA: left drag orbit | right drag pan | wheel zoom".to_owned(),
        ];

        self.hud_rects.push(Rect {
            rect: [12.0, 12.0, width, 188.0],
            color: [0.025, 0.04, 0.065, 0.91],
        });
        for (row, line) in lines.iter().enumerate() {
            let text = fit_line(line, max_chars);
            draw_text(
                &mut self.hud_rects,
                24.0,
                20.0 + row as f32 * 23.0,
                &text,
                if row == 0 { 1.12 } else { 1.0 },
            );
        }
        if self.hud_rects.len() > self.hud_capacity {
            self.hud_capacity = self.hud_rects.len().next_power_of_two();
            self.hud_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("live test HUD rectangles"),
                size: (self.hud_capacity * std::mem::size_of::<Rect>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !self.hud_rects.is_empty() {
            self.queue
                .write_buffer(&self.hud_buffer, 0, bytemuck::cast_slice(&self.hud_rects));
        }
        self.hud_count = self.hud_rects.len() as u32;
    }

    fn draw(&mut self) {
        self.refresh_hud();
        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let direction = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
        );
        let eye = self.target + direction * self.radius * 2.3;
        let near = (self.radius * 0.0001).max(0.00001);
        let view_projection =
            Mat4::perspective_infinite_reverse_rh(45_f32.to_radians(), aspect, near)
                * Mat4::look_at_rh(eye, self.target, Vec3::Z);
        let uniform = CameraUniform {
            view_projection: view_projection.to_cols_array_2d(),
            cell_size: self.cell_size,
            padding: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        let output = match self.surface.get_current_texture() {
            Ok(output) => output,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::Timeout) => return,
            Err(wgpu::SurfaceError::OutOfMemory) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("live test frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("REV cubes"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.035,
                            g: 0.045,
                            b: 0.065,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..self.instance_count);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("live test HUD"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hud_pipeline);
            pass.set_bind_group(0, &self.screen_group, &[]);
            pass.set_vertex_buffer(0, self.hud_buffer.slice(..));
            pass.draw(0..6, 0..self.hud_count);
        }
        self.queue.submit(Some(encoder.finish()));
        output.present();
        if self
            .last_frame_at
            .is_some_and(|at| at.elapsed() > Duration::from_millis(500))
        {
            self.update_title(true);
        }
    }
}

fn depth_view(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer depth"),
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
        .create_view(&wgpu::TextureViewDescriptor::default())
}

fn cube_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(36);
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            let mut normal = Vec3::ZERO;
            normal[axis] = sign;
            let u = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 1) % 3];
            let v = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 2) % 3] * sign;
            for [a, b] in [
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ] {
                vertices.push(Vertex {
                    position: ((normal + a * u + b * v) * 0.5).to_array(),
                    normal: normal.to_array(),
                    face_index: (axis as u32 * 2 + u32::from(sign > 0.0)),
                });
            }
        }
    }
    vertices
}

fn bounds_center(frame: &LiveTestFrame) -> Vec3 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in &frame.positions {
        let point = Vec3::from_array(*position);
        min = min.min(point);
        max = max.max(point);
    }
    if !min.is_finite() || !max.is_finite() {
        Vec3::ZERO
    } else {
        (min + max) * 0.5
    }
}

fn fit_line(text: &str, max_chars: usize) -> String {
    let max_chars = max_chars.max(4);
    let mut chars = text.chars();
    let visible: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!(
            "{}...",
            visible
                .chars()
                .take(max_chars - 3)
                .collect::<String>()
                .trim_end()
        )
    } else {
        visible
    }
}

fn bounds_radius(frame: &LiveTestFrame) -> f32 {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for position in &frame.positions {
        let point = Vec3::from_array(*position);
        min = min.min(point);
        max = max.max(point);
    }
    if !min.is_finite() || !max.is_finite() {
        1.0
    } else {
        (max - min).length().max(frame.cell_m)
    }
}

fn snapshot_directory() -> PathBuf {
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--dir" {
            if let Some(directory) = args.next() {
                return PathBuf::from(directory);
            }
        }
    }
    std::env::var_os("VOXEL_LIVE_TESTS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "E:/Projects/gpt/voxel-sim/Voxel-sim-guided-knife-validation/validation/live-tests",
            )
        })
}

fn is_newer_frame(
    candidate_modified: std::time::SystemTime,
    candidate_sequence: u64,
    latest_modified: Option<std::time::SystemTime>,
    latest_sequence: u64,
) -> bool {
    match latest_modified {
        Some(latest) if candidate_modified < latest => false,
        Some(latest) if candidate_modified == latest => candidate_sequence > latest_sequence,
        _ => true,
    }
}

fn is_new_process_snapshot(
    candidate_sequence: u64,
    latest_sequence: u64,
    has_latest: bool,
) -> bool {
    has_latest && candidate_sequence <= latest_sequence
}

fn start_reader(directory: PathBuf) -> LiveFeed {
    let latest = Arc::new(Mutex::new(None));
    let status_latest = Arc::new(Mutex::new(None));
    let reader_latest = latest.clone();
    let reader_status = status_latest.clone();
    thread::Builder::new()
        .name("voxel-live-test-reader".into())
        .spawn(move || {
            let mut latest_sequence = 0;
            let mut process_epoch = 0_u64;
            let mut latest_modified = None;
            let mut observed: [Option<(std::time::SystemTime, u64)>; 2] = [None, None];
            let status_path = directory.join("run-status.json");
            let mut observed_status = None;
            loop {
                for slot in 0..2 {
                    let path = directory.join(format!("frame-{slot}.bin"));
                    let Ok(metadata) = fs::metadata(&path) else {
                        continue;
                    };
                    let Ok(modified) = metadata.modified() else {
                        continue;
                    };
                    let signature = (modified, metadata.len());
                    if observed[slot] == Some(signature) {
                        continue;
                    }
                    let Ok(bytes) = fs::read(path) else { continue };
                    let Ok(frame) = bincode::deserialize::<LiveTestFrame>(&bytes) else {
                        continue;
                    };
                    observed[slot] = Some(signature);
                    if frame.version == 1
                        && is_newer_frame(
                            modified,
                            frame.sequence,
                            latest_modified,
                            latest_sequence,
                        )
                    {
                        let new_process = is_new_process_snapshot(
                            frame.sequence,
                            latest_sequence,
                            latest_modified.is_some(),
                        );
                        if new_process {
                            process_epoch = process_epoch.wrapping_add(1);
                        }
                        latest_sequence = frame.sequence;
                        latest_modified = Some(modified);
                        let surface_masks = exterior_face_masks(&frame.positions, frame.cell_m);
                        if let Ok(mut latest) = reader_latest.lock() {
                            *latest = Some(FrameSnapshot {
                                frame,
                                process_epoch,
                                surface_masks,
                            });
                        }
                    }
                }
                if let Ok(bytes) = fs::read(&status_path) {
                    if let Ok(status) = serde_json::from_slice::<RunStatus>(&bytes) {
                        if observed_status.as_ref() != Some(&status) {
                            observed_status = Some(status.clone());
                            if let Ok(mut latest) = reader_status.lock() {
                                *latest = Some(status);
                            }
                        }
                    }
                }
                thread::sleep(Duration::from_millis(40));
            }
        })
        .expect("start snapshot reader thread");
    LiveFeed {
        frame: latest,
        status: status_latest,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let window = std::sync::Arc::new(
        WindowBuilder::new()
            .with_title("Live physics tests | waiting for a test snapshot")
            .with_inner_size(PhysicalSize::new(1280, 800))
            .build(&event_loop)?,
    );
    let mut viewer = pollster::block_on(Viewer::new(window))?;
    let feed = start_reader(snapshot_directory());
    let mut last_title_refresh = Instant::now();

    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { window_id, event } if window_id == viewer.window.id() => {
                match event {
                    WindowEvent::CloseRequested => target.exit(),
                    WindowEvent::Resized(size) => viewer.resize(size),
                    WindowEvent::ScaleFactorChanged { .. } => {
                        viewer.resize(viewer.window.inner_size());
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        viewer.dragging = (state == ElementState::Pressed).then_some(button);
                        if state == ElementState::Released {
                            viewer.last_cursor = None;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if let Some(button) = viewer.dragging {
                            if let Some((last_x, last_y)) = viewer.last_cursor {
                                let dx = (position.x - last_x) as f32;
                                let dy = (position.y - last_y) as f32;
                                if button == MouseButton::Left {
                                    viewer.yaw -= dx * 0.008;
                                    viewer.pitch = (viewer.pitch + dy * 0.008).clamp(-1.48, 1.48);
                                } else {
                                    let right = Vec3::new(-viewer.yaw.sin(), viewer.yaw.cos(), 0.0);
                                    let up = Vec3::Z;
                                    viewer.target +=
                                        (right * -dx + up * dy) * viewer.radius * 0.0015;
                                }
                            }
                        }
                        viewer.last_cursor = Some((position.x, position.y));
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let amount = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(position) => position.y as f32 / 60.0,
                        };
                        viewer.radius = (viewer.radius * (1.0 - amount * 0.08)).max(0.0001);
                    }
                    WindowEvent::RedrawRequested => viewer.draw(),
                    _ => {}
                }
            }
            Event::AboutToWait => {
                if let Ok(mut latest) = feed.frame.lock() {
                    if let Some(frame) = latest.take() {
                        viewer.accept_frame(frame);
                    }
                }
                if let Ok(mut latest) = feed.status.lock() {
                    if let Some(status) = latest.take() {
                        viewer.run_status = Some(status);
                        viewer.hud_dirty = true;
                        viewer.update_title(
                            viewer
                                .last_frame_at
                                .is_some_and(|at| at.elapsed() > Duration::from_millis(500)),
                        );
                    }
                }
                if last_title_refresh.elapsed() >= Duration::from_millis(250) {
                    if viewer
                        .last_frame_at
                        .is_some_and(|at| at.elapsed() > Duration::from_millis(500))
                    {
                        viewer.update_title(true);
                    }
                    last_title_refresh = Instant::now();
                }
                viewer.window.request_redraw();
            }
            _ => {}
        }
    })?;
    Ok(())
}

const SHADER: &str = r#"
struct Camera {
    view_projection: mat4x4<f32>,
    cell_size: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) instance_position: vec3<f32>,
    @location(3) color: vec3<f32>,
    @location(4) face_index: u32,
    @location(5) face_mask: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};
@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    if ((input.face_mask & (1u << input.face_index)) == 0u) {
        output.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    } else {
        let world = input.instance_position + input.position * camera.cell_size;
        output.position = camera.view_projection * vec4<f32>(world, 1.0);
    }
    output.normal = input.normal;
    output.color = input.color;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(-0.45, -0.55, 0.72));
    let diffuse = max(dot(normalize(input.normal), light), 0.0);
    let illumination = 0.42 + 0.58 * diffuse;
    return vec4<f32>(input.color * illumination, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::{cube_vertices, fit_line, is_new_process_snapshot, is_newer_frame};
    use glam::{Mat4, Vec3, Vec4};
    use std::time::{Duration, SystemTime};

    #[test]
    fn newer_process_snapshot_wins_even_when_sequence_restarts() {
        let previous = SystemTime::UNIX_EPOCH + Duration::from_secs(10);
        let current = previous + Duration::from_millis(1);
        assert!(is_newer_frame(current, 1, Some(previous), 900));
        assert!(!is_newer_frame(previous, 901, Some(current), 1));
        assert!(is_newer_frame(current, 2, Some(current), 1));
        assert!(is_new_process_snapshot(1, 1, true));
        assert!(is_new_process_snapshot(1, 900, true));
        assert!(!is_new_process_snapshot(2, 1, true));
        assert!(!is_new_process_snapshot(1, 0, false));
    }

    #[test]
    fn hud_detail_is_truncated_to_available_line_width() {
        assert_eq!(fit_line("short", 8), "short");
        assert_eq!(fit_line("long status detail", 8), "long...");
    }

    #[test]
    fn every_cube_triangle_winds_outward() {
        let vertices = cube_vertices();
        for (face_index, face) in vertices.chunks_exact(6).enumerate() {
            assert!(face
                .iter()
                .all(|vertex| vertex.face_index == face_index as u32));
        }
        for triangle in vertices.chunks_exact(3) {
            let a = Vec3::from_array(triangle[0].position);
            let b = Vec3::from_array(triangle[1].position);
            let c = Vec3::from_array(triangle[2].position);
            let normal = Vec3::from_array(triangle[0].normal);
            assert!((b - a).cross(c - a).dot(normal) > 0.0);
        }
    }

    #[test]
    fn reverse_z_orders_near_points_ahead_of_far_points() {
        let projection = Mat4::perspective_infinite_reverse_rh(45_f32.to_radians(), 1.0, 0.01);
        let projected_depth = |distance: f32| {
            let clip = projection * Vec4::new(0.0, 0.0, -distance, 1.0);
            clip.z / clip.w
        };
        let near_depth = projected_depth(0.01);
        let far_depth = projected_depth(100.0);
        assert!(near_depth > far_depth);
        assert!(far_depth >= 0.0);
    }
}
