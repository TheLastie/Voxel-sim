mod render;
mod ui;
use voxel_physics_lab::{bond, scene, solver, tool};

use glam::Vec3;
use render::{Camera, Renderer};
use scene::{Scene, SceneConfig};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tool::{Shape, Tool};
use ui::Ui;
use winit::{
    event::{ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

fn launch(shape: Shape, scene: &Scene, ui: &Ui) -> Tool {
    let mut tool = Tool::new(shape, scene, ui.start(), ui.direction(), ui.speed);
    if matches!(shape, Shape::Wedge) {
        let rear = tool.cells.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        tool.mount_base(Vec3::new(rear, 0.0, 0.0), ui.direction()).expect("finite guide");
    }
    tool
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let path = args
        .windows(2)
        .find(|p| p[0] == "--scene")
        .map(|p| p[1].as_str())
        .unwrap_or("scenes/mounted_axe.ron");
    let config = SceneConfig::load(path)?;
    let mut scene = Scene::new(config.clone());
    let mut ui = Ui::new(config.initial_speed, Vec3::from_array(config.tool_start));
    if let Some(index) = args.iter().position(|s| s == "--headless") {
        let steps: usize = args
            .get(index + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        let shape = match args
            .windows(2)
            .find(|p| p[0] == "--tool")
            .map(|p| p[1].as_str())
        {
            Some("knife") => Shape::Blade,
            Some("rod") => Shape::Rod,
            Some("stone") => Shape::Rounded,
            _ => Shape::Wedge,
        };
        let mut tool = Some(launch(shape, &scene, &ui));
        let start = Instant::now();
        for _ in 0..steps {
            solver::step(&mut scene, &mut tool);
        }
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "voxels={} steps={} elapsed_s={:.3} steps_per_s={:.2} broken={} clusters={} linear_residual={:.2e}",
            scene.voxels.len(),
            steps,
            seconds,
            steps as f64 / seconds,
            scene.broken,
            scene.cluster_count,
            scene.solver_residual
        );
        if scene.positions.iter().any(|p| !p.is_finite()) {
            return Err("Non-finite solver output".into());
        }
        return Ok(());
    }
    // API checked with cargo check; opening a real window still needs display validation.
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Voxel mechanics laboratory")
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
            .build(&event_loop)?,
    );
    let mut renderer = pollster::block_on(Renderer::new(window.clone()))?;
    let mut camera = Camera::new();
    let height = config.dimensions[2] as f32 * scene::CELL;
    camera.target = Vec3::Z * (height * 0.5);
    camera.radius = (height * 1.5).max(config.ground_dimensions[0].max(config.ground_dimensions[1]) as f32 * scene::CELL * 1.6);
    let mut tool = None;
    let mut pointer = (0.0f32, 0.0f32);
    let mut orbit = false;
    let mut dragging = false;
    let mut last = Instant::now();
    let mut accumulator = 0.0f32;
    let mut sim_ms = 0.0;
    let mut fps = 0.0;
    let mut frames = 0;
    let mut fps_time = Instant::now();
    event_loop.run(move |event, target| {
        target.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
                WindowEvent::CursorMoved { position, .. } => {
                    let next = (position.x as f32, position.y as f32);
                    if orbit {
                        camera.yaw -= (next.0 - pointer.0) * 0.005;
                        camera.pitch =
                            (camera.pitch + (next.1 - pointer.1) * 0.005).clamp(-1.4, 1.4);
                    }
                    if dragging && next.1 >= 185.0 {
                        ui.pointer(next.0, next.1);
                    }
                    pointer = next;
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Right,
                    state,
                    ..
                } => orbit = state == ElementState::Pressed,
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    dragging =
                        state == ElementState::Pressed && pointer.0 < 295.0 && pointer.1 >= 185.0;
                    if state == ElementState::Pressed {
                        if let Some(shape) = ui.pointer(pointer.0, pointer.1) {
                            tool = Some(launch(shape, &scene, &ui));
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                    };
                    camera.radius = (camera.radius * (-amount * 0.1).exp()).clamp(0.3, 30.0);
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat =>
                {
                    let shape = match event.physical_key {
                        PhysicalKey::Code(KeyCode::Digit1) => Some(Shape::Wedge),
                        PhysicalKey::Code(KeyCode::Digit2) => Some(Shape::Blade),
                        PhysicalKey::Code(KeyCode::Digit3) => Some(Shape::Rod),
                        PhysicalKey::Code(KeyCode::Digit4) => Some(Shape::Rounded),
                        PhysicalKey::Code(KeyCode::Space) => {
                            ui.paused = !ui.paused;
                            accumulator = 0.0;
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            ui.show_bonds = !ui.show_bonds;
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            scene = Scene::new(config.clone());
                            tool = None;
                            accumulator = 0.0;
                            None
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            target.exit();
                            None
                        }
                        _ => None,
                    };
                    if let Some(shape) = shape {
                        ui.selected = shape;
                        tool = Some(launch(shape, &scene, &ui));
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last).as_secs_f32();
                    last = now;
                    if !ui.paused {
                        accumulator = (accumulator + elapsed).min(config.fixed_dt * 2.0);
                        // Bounded catch-up: an overloaded simulation slows down, rather than spiralling.
                        let start = Instant::now();
                        let mut steps = 0;
                        while accumulator >= config.fixed_dt && steps < 2 {
                            solver::step(&mut scene, &mut tool);
                            accumulator -= config.fixed_dt;
                            steps += 1;
                        }
                        if steps > 0 {
                            sim_ms = start.elapsed().as_secs_f32() * 1000.0 / steps as f32;
                        }
                        if scene.positions.iter().any(|p| !p.is_finite()) {
                            eprintln!("Solver became non-finite; paused.");
                            ui.paused = true;
                        }
                    }
                    let rects = ui.draw(scene.broken, scene.cluster_count, fps, sim_ms);
                    match renderer.draw(&scene, &tool, &camera, &rects, ui.show_bonds) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let s = window.inner_size();
                            renderer.resize(s.width, s.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(wgpu::SurfaceError::Timeout) => {}
                    }
                    frames += 1;
                    if fps_time.elapsed() >= Duration::from_secs(1) {
                        fps = frames as f32 / fps_time.elapsed().as_secs_f32();
                        frames = 0;
                        fps_time = Instant::now();
                        window.set_title(&format!(
                            "Voxel lab | {:.1} FPS | {:.1} ms/step | broken {}",
                            fps, sim_ms, scene.broken
                        ));
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                if !window.is_minimized().unwrap_or(false) {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    })?;
    Ok(())
}
