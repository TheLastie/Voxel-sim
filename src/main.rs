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

const RECORDING_DURATION_SECONDS: f64 = 17.6;
const PLAYBACK_FRAME_INTERVAL_SECONDS: f64 = 0.1;
const MAX_SIMULATED_SECONDS_PER_REDRAW: f64 = 0.040;
const MAX_SOLVER_WALL_TIME_PER_REDRAW: Duration = Duration::from_millis(40);

#[derive(Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Interactive,
    Recording,
    Playback,
}

struct PlaybackFrame {
    time_s: f64,
    positions: Vec<Vec3>,
    rotations: Vec<glam::Quat>,
    bond_neighbors: Vec<u32>,
    bond_damage: Vec<f32>,
    broken: usize,
    clusters: usize,
    tool_pose: Option<(Vec3, glam::Quat)>,
}

fn capture_frame(scene: &Scene, tool: &Option<Tool>, time_s: f64) -> PlaybackFrame {
    PlaybackFrame {
        time_s,
        positions: scene.positions.clone(),
        rotations: scene.rotations.clone(),
        bond_neighbors: scene
            .bonds
            .iter()
            .flat_map(|axis| axis.iter().map(|bond| bond.neighbor))
            .collect(),
        bond_damage: scene
            .bonds
            .iter()
            .flat_map(|axis| axis.iter().map(|bond| bond.damage))
            .collect(),
        broken: scene.broken,
        clusters: scene.cluster_count,
        tool_pose: tool.as_ref().map(|tool| (tool.position, tool.rotation)),
    }
}

fn apply_frame(scene: &mut Scene, tool: &mut Option<Tool>, frame: &PlaybackFrame) {
    scene.positions.clone_from(&frame.positions);
    scene.rotations.clone_from(&frame.rotations);
    for ((bond, &neighbor), &damage) in scene
        .bonds
        .iter_mut()
        .flatten()
        .zip(&frame.bond_neighbors)
        .zip(&frame.bond_damage)
    {
        bond.neighbor = neighbor;
        bond.damage = damage;
    }
    scene.broken = frame.broken;
    scene.cluster_count = frame.clusters;
    if let (Some(tool), Some((position, rotation))) = (tool.as_mut(), frame.tool_pose) {
        tool.position = position;
        tool.rotation = rotation;
    }
}

fn launch(shape: Shape, scene: &Scene, ui: &Ui) -> Tool {
    let mut tool = Tool::new(shape, scene, ui.start(), ui.direction(), ui.speed);
    if matches!(shape, Shape::Wedge) {
        let rear = tool.cells.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        tool.mount_base(Vec3::new(rear, 0.0, 0.0), ui.direction())
            .expect("finite guide");
    }
    tool
}

fn fix_wood_base(scene: &mut Scene) {
    let wood_ids: Vec<_> = ["wood", "oak"]
        .into_iter()
        .filter_map(voxel_physics_lab::material::Material::component_id)
        .collect();
    if wood_ids.is_empty() {
        return;
    }
    let wood: Vec<_> = scene
        .voxels
        .iter()
        .enumerate()
        .filter(|(_, voxel)| {
            voxel
                .composition
                .components
                .iter()
                .any(|&(component, fraction)| {
                    wood_ids.contains(&component) && fraction > 0.0
                })
        })
        .map(|(index, _)| index)
        .collect();
    let Some(lowest_z) = wood
        .iter()
        .map(|&index| scene.positions[index].z)
        .min_by(f32::total_cmp)
    else {
        return;
    };
    let base: Vec<_> = wood
        .into_iter()
        .filter(|&index| (scene.positions[index].z - lowest_z).abs() <= scene::CELL * 0.1)
        .collect();
    scene
        .fix_voxels(&base)
        .expect("wood base indices come from this scene");
    eprintln!("fixed wood base: {} REV at Z={lowest_z:.4} m", base.len());
}

fn selected_shape(args: &[String]) -> Option<Shape> {
    args.windows(2)
        .find(|pair| pair[0] == "--tool")
        .map(|pair| match pair[1].as_str() {
            "knife" => Shape::Blade,
            "rod" => Shape::Rod,
            "stone" => Shape::Rounded,
            _ => Shape::Wedge,
        })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let pause_on_fracture = args.iter().any(|arg| arg == "--pause-on-fracture");
    let path = args
        .windows(2)
        .find(|p| p[0] == "--scene")
        .map(|p| p[1].as_str())
        .unwrap_or("scenes/mounted_axe.ron");
    let config = SceneConfig::load(path)?;
    let mut scene = Scene::new(config.clone());
    fix_wood_base(&mut scene);
    let mut ui = Ui::new(
        config.initial_speed,
        Vec3::from_array(config.tool_start),
        Vec3::from_array(config.tool_direction),
    );
    if let Some(index) = args.iter().position(|s| s == "--headless") {
        let steps: usize = args
            .get(index + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        let shape = selected_shape(&args).unwrap_or(Shape::Wedge);
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
    camera.radius = (height * 1.5).max(
        config.ground_dimensions[0].max(config.ground_dimensions[1]) as f32 * scene::CELL * 1.6,
    );
    let initial_shape = selected_shape(&args);
    if let Some(shape) = initial_shape {
        ui.selected = shape;
    }
    let mut tool = initial_shape.map(|shape| launch(shape, &scene, &ui));
    let mut pointer = (0.0f32, 0.0f32);
    let mut orbit = false;
    let mut dragging = false;
    let mut last = Instant::now();
    let mut accumulator = 0.0f64;
    let mut mode = AppMode::Interactive;
    let mut playback_frames = Vec::new();
    let mut playback_time_s = 0.0;
    let mut live_simulated_s = 0.0;
    let mut live_wall_s = 0.0;
    let mut next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
    let mut sim_ms = 0.0;
    let mut fracture_pause_triggered = false;
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
                            if mode == AppMode::Interactive {
                                tool = Some(launch(shape, &scene, &ui));
                            }
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
                            if mode != AppMode::Playback {
                                accumulator = 0.0;
                            }
                            last = Instant::now();
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyB) => {
                            ui.show_bonds = !ui.show_bonds;
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            mode = AppMode::Interactive;
                            scene = Scene::new(config.clone());
                            fix_wood_base(&mut scene);
                            tool = None;
                            playback_frames.clear();
                            playback_time_s = 0.0;
                            live_simulated_s = 0.0;
                            live_wall_s = 0.0;
                            next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
                            accumulator = 0.0;
                            ui.paused = false;
                            fracture_pause_triggered = false;
                            last = Instant::now();
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyP) => {
                            match mode {
                                AppMode::Interactive => {
                                    playback_frames.clear();
                                    playback_frames.push(capture_frame(&scene, &tool, 0.0));
                                    playback_time_s = 0.0;
                                    live_simulated_s = 0.0;
                                    live_wall_s = 0.0;
                                    next_record_frame_s = PLAYBACK_FRAME_INTERVAL_SECONDS;
                                    accumulator = 0.0;
                                    ui.paused = false;
                                    fracture_pause_triggered = false;
                                    mode = AppMode::Recording;
                                    last = Instant::now();
                                }
                                AppMode::Recording => {
                                    if let Some(frame) = playback_frames.last() {
                                        if live_simulated_s > frame.time_s + 1e-9 {
                                            playback_frames.push(capture_frame(
                                                &scene,
                                                &tool,
                                                live_simulated_s,
                                            ));
                                        }
                                    }
                                    mode = AppMode::Playback;
                                    playback_time_s = 0.0;
                                    accumulator = 0.0;
                                    ui.paused = false;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    last = Instant::now();
                                }
                                AppMode::Playback => {
                                    playback_time_s = 0.0;
                                    ui.paused = false;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    last = Instant::now();
                                }
                            }
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
                        if mode == AppMode::Interactive {
                            tool = Some(launch(shape, &scene, &ui));
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    let now = Instant::now();
                    let elapsed = now.duration_since(last).as_secs_f32();
                    last = now;

                    if matches!(mode, AppMode::Interactive | AppMode::Recording) && !ui.paused {
                        live_wall_s += f64::from(elapsed);
                        accumulator += f64::from(elapsed);
                        // Preserve elapsed time. Limit simulation catch-up to 40 ms per redraw
                        // and retain any backlog instead of dropping simulated time.
                        let start = Instant::now();
                        let mut steps = 0;
                        let fixed_dt = f64::from(config.fixed_dt);
                        let max_steps =
                            (MAX_SIMULATED_SECONDS_PER_REDRAW / fixed_dt).floor() as usize;
                        while accumulator >= fixed_dt
                            && steps < max_steps.max(1)
                            && (steps == 0 || start.elapsed() < MAX_SOLVER_WALL_TIME_PER_REDRAW)
                        {
                            if let Some(active_tool) = tool.as_mut() {
                                active_tool
                                    .set_external_force(ui.direction() * ui.force_newtons)
                                    .expect("UI force is finite");
                            }
                            solver::step(&mut scene, &mut tool);
                            accumulator -= fixed_dt;
                            live_simulated_s += fixed_dt;
                            steps += 1;
                            if mode == AppMode::Recording {
                                while live_simulated_s + fixed_dt * 0.5
                                    >= next_record_frame_s
                                {
                                    playback_frames.push(capture_frame(
                                        &scene,
                                        &tool,
                                        live_simulated_s,
                                    ));
                                    next_record_frame_s += PLAYBACK_FRAME_INTERVAL_SECONDS;
                                }
                                if live_simulated_s + fixed_dt * 0.5
                                    >= RECORDING_DURATION_SECONDS
                                {
                                    if playback_frames
                                        .last()
                                        .is_some_and(|frame| {
                                            frame.time_s + fixed_dt * 0.5
                                                < live_simulated_s
                                        })
                                    {
                                        playback_frames.push(capture_frame(
                                            &scene,
                                            &tool,
                                            live_simulated_s,
                                        ));
                                    }
                                    mode = AppMode::Playback;
                                    playback_time_s = 0.0;
                                    accumulator = 0.0;
                                    if let Some(frame) = playback_frames.first() {
                                        apply_frame(&mut scene, &mut tool, frame);
                                    }
                                    break;
                                }
                            }
                        }
                        if pause_on_fracture && !fracture_pause_triggered && scene.broken > 0 {
                            ui.paused = true;
                            ui.show_bonds = true;
                            fracture_pause_triggered = true;
                        }
                        if steps > 0 {
                            sim_ms = start.elapsed().as_secs_f32() * 1000.0 / steps as f32;
                        }
                        if scene.positions.iter().any(|p| !p.is_finite()) {
                            eprintln!("Solver became non-finite; paused.");
                            ui.paused = true;
                        }
                    } else if mode == AppMode::Playback && !ui.paused {
                        let duration = playback_frames
                            .last()
                            .map_or(0.0, |frame| frame.time_s);
                        // Keep the last saved state visible for one sample interval, then
                        // wrap to the beginning. Playback is a view of captured solver states;
                        // it must not stop the physical run or synthesize intermediate states.
                        let loop_duration = duration + PLAYBACK_FRAME_INTERVAL_SECONDS;
                        playback_time_s = (playback_time_s + f64::from(elapsed)) % loop_duration;
                        let sample_time_s = playback_time_s.min(duration);
                        let frame_index = playback_frames
                            .partition_point(|frame| frame.time_s <= sample_time_s)
                            .saturating_sub(1);
                        if let Some(frame) = playback_frames.get(frame_index) {
                            apply_frame(&mut scene, &mut tool, frame);
                        }
                    }
                    let rects = ui.draw(scene.broken, scene.cluster_count, fps, sim_ms);
                    match renderer.draw(&scene, &tool, &camera, &rects, ui.show_bonds) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let s = window.inner_size();
                            renderer.resize(s.width, s.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            target.exit();
                        }
                        Err(wgpu::SurfaceError::Timeout) => {}
                    }
                    frames += 1;
                    if fps_time.elapsed() >= Duration::from_secs(1) {
                        fps = frames as f32 / fps_time.elapsed().as_secs_f32();
                        frames = 0;
                        fps_time = Instant::now();
                        let tool_axis_position = tool
                            .as_ref()
                            .map_or(f32::NAN, |t| t.position.dot(ui.direction()));
                        let tool_speed = tool.as_ref().map_or(0.0, |t| t.velocity.length());
                        let live_rate = if live_wall_s > 0.0 {
                            live_simulated_s / live_wall_s
                        } else {
                            0.0
                        };
                        let mode_title = match mode {
                            AppMode::Interactive => {
                                if ui.paused {
                                    "LIVE PAUSED".to_owned()
                                } else {
                                    format!("LIVE {:.3}x", live_rate)
                                }
                            }
                            AppMode::Recording => format!(
                                "RECORD {} {:.1}/{:.1}s | LIVE {:.3}x",
                                if ui.paused { "PAUSED" } else { "LIVE" },
                                live_simulated_s,
                                RECORDING_DURATION_SECONDS,
                                live_rate
                            ),
                            AppMode::Playback => {
                                let state = if ui.paused { "PAUSED" } else { "1.00x" };
                                let frame_count = playback_frames.len();
                                let loop_duration = playback_frames
                                    .last()
                                    .map_or(PLAYBACK_FRAME_INTERVAL_SECONDS, |frame| {
                                        frame.time_s + PLAYBACK_FRAME_INTERVAL_SECONDS
                                    });
                                format!(
                                    "PLAYBACK {state} {:.1}/{:.1}s | {frame_count} frames",
                                    playback_time_s.min(loop_duration),
                                    loop_duration,
                                )
                            }
                        };
                        window.set_title(&format!(
                            "Voxel lab | {mode_title} | {:.1} FPS | {:.1} ms/step | tool {:.2} m/s axis {:.3} | broken {}",
                            fps, sim_ms, tool_speed, tool_axis_position, scene.broken
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
