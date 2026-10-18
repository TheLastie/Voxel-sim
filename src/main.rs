mod render;
mod ui;
use voxel_physics_lab::{bond, scene, solver, tool};

use glam::{DVec3, Vec3};
use render::{Camera, Renderer};
use scene::{Scene, SceneConfig, VoxelSpawn};
use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};
use tool::{Shape, Tool};
use ui::{Ui, UiAction};
use voxel_physics_lab::world::{GeneratedWorld, GrowthScheduler, WorldCharacter, WORLD_SIDE_M};
use voxel_physics_lab::world_physics::{self, PhysicsWorker};
use winit::{
    event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, WindowBuilder},
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
    positions: Vec<DVec3>,
    rotations: Vec<glam::Quat>,
    bond_neighbors: Vec<u32>,
    bond_damage: Vec<f32>,
    broken: usize,
    clusters: usize,
    tool_pose: Option<(DVec3, glam::Quat)>,
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
        .min_by(f64::total_cmp)
    else {
        return;
    };
    let base: Vec<_> = wood
        .into_iter()
        .filter(|&index| (scene.positions[index].z - lowest_z).abs() <= scene::DCELL * 0.1)
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

fn two_cube_collision_scene(mut config: SceneConfig) -> Result<Scene, String> {
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = 0.001;
    config.substeps = 1;
    config.iterations = 40;
    config.corotation_iterations = 2;
    // The pair coefficient is the sum of oak and concrete surface values,
    // scaled by this scene setting. Tangential loss is accounted for as heat.
    config.fracture.friction = 0.45;
    let oak = voxel_physics_lab::material::Material::component_id("oak")
        .ok_or_else(|| "oak material is unavailable".to_owned())?;
    let concrete = voxel_physics_lab::material::Material::component_id("concrete")
        .ok_or_else(|| "concrete material is unavailable".to_owned())?;
    let mut spawns = Vec::with_capacity(16 + config.ground_dimensions[0] * config.ground_dimensions[1]);
    for (cells, velocity) in [([-3, -2], Vec3::X * 0.025), ([1, 2], Vec3::NEG_X * 0.025)] {
        for z in 120..122 {
            for y in -1..=0 {
                for x in cells {
                    spawns.push(
                        VoxelSpawn::at_cell([x, y, z], oak).with_velocity(velocity),
                    );
                }
            }
        }
    }
    let [floor_x, floor_y, _] = config.ground_dimensions;
    for y in 0..floor_y {
        for x in 0..floor_x {
            let cell = [x as i32 - floor_x as i32 / 2, y as i32 - floor_y as i32 / 2, -1];
            spawns.push(VoxelSpawn::at_cell(cell, concrete).fixed());
        }
    }
    Scene::from_voxels(config, spawns)
}

fn friction_heat_road_scene(mut config: SceneConfig) -> Result<Scene, String> {
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = 0.001;
    config.substeps = 1;
    config.iterations = 48;
    config.corotation_iterations = 2;
    config.fracture.friction = 0.45;
    let oak = voxel_physics_lab::material::Material::component_id("oak")
        .ok_or_else(|| "oak material is unavailable".to_owned())?;
    let concrete = voxel_physics_lab::material::Material::component_id("concrete")
        .ok_or_else(|| "concrete material is unavailable".to_owned())?;
    let mut spawns = Vec::with_capacity(240 * 9 + 27);
    for x in -120..120 {
        for y in -4..=4 {
            spawns.push(VoxelSpawn::at_cell([x, y, -1], concrete).fixed());
        }
    }
    let center = DVec3::new(-0.555, 0.005, 0.015);
    let omega = Vec3::new(0.0, 18.0, 0.0);
    for z in 0..3 {
        for y in -1..=1 {
            for x in -57..=-55 {
                let cell = [x, y, z];
                let position = DVec3::new(
                    f64::from(x) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                    f64::from(y) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                    f64::from(z) * scene::CELL as f64 + scene::CELL as f64 * 0.5,
                );
                let velocity = Vec3::new(0.3, 0.0, 0.0)
                    + omega.cross((position - center).as_vec3());
                spawns.push(VoxelSpawn::at_cell(cell, oak).with_velocity(velocity));
            }
        }
    }
    Scene::from_voxels(config, spawns)
}

fn cube_center(scene: &Scene, range: std::ops::Range<usize>) -> DVec3 {
    let sum = range
        .map(|index| scene.positions[index])
        .fold(DVec3::ZERO, |sum, position| sum + position);
    sum / 8.0
}

fn cube_velocity(scene: &Scene, range: std::ops::Range<usize>) -> Vec3 {
    let sum = range
        .map(|index| scene.voxels[index].velocity)
        .fold(Vec3::ZERO, |sum, velocity| sum + velocity);
    sum / 8.0
}

fn total_momentum(scene: &Scene) -> DVec3 {
    scene.voxels.iter().fold(DVec3::ZERO, |sum, voxel| {
        sum + DVec3::new(
            f64::from(voxel.mass * voxel.velocity.x),
            f64::from(voxel.mass * voxel.velocity.y),
            f64::from(voxel.mass * voxel.velocity.z),
        )
    })
}

fn angular_momentum(scene: &Scene, range: std::ops::Range<usize>) -> DVec3 {
    let total_mass: f64 = range.clone().map(|index| f64::from(scene.voxels[index].mass)).sum();
    if total_mass <= 0.0 {
        return DVec3::ZERO;
    }
    let center = range
        .clone()
        .map(|index| scene.positions[index] * f64::from(scene.voxels[index].mass))
        .fold(DVec3::ZERO, |sum, weighted_position| sum + weighted_position)
        / total_mass;
    range.fold(DVec3::ZERO, |sum, index| {
        let voxel = &scene.voxels[index];
        let momentum = DVec3::new(
            f64::from(voxel.mass * voxel.velocity.x),
            f64::from(voxel.mass * voxel.velocity.y),
            f64::from(voxel.mass * voxel.velocity.z),
        );
        sum + (scene.positions[index] - center).cross(momentum)
    })
}

fn kinetic_energy(scene: &Scene) -> f64 {
    scene.voxels.iter().fold(0.0, |sum, voxel| {
        sum + 0.5
            * f64::from(voxel.mass)
            * f64::from(voxel.velocity.length_squared())
    })
}

fn start_world_physics(world: &mut GeneratedWorld, character: &mut WorldCharacter,
    camera: &mut Camera, config: &SceneConfig) -> Result<PhysicsWorker, String> {
    let center = world.view_bounds.map_or(
        Vec3::new(character.position.x,character.position.y,0.0),
        |(min,max)| Vec3::new((min.x+max.x)*0.5,(min.y+max.y)*0.5,0.0));
    world.set_view_bounds(Some((center-Vec3::new(10.0,10.0,0.0),
        center+Vec3::new(10.0,10.0,512.0))));
    let region = world_physics::contact_experiment(world,center,config.clone())?;
    let scene = &region.scene;
    eprintln!("20m physics experiment: {} active 1cm REVs, dt={}s, substeps={}",
        scene.voxels.len(),scene.config.fixed_dt,scene.config.substeps);
    character.position = Vec3::new(center.x+0.9,center.y,
        world.terrain_height_at(center.x+0.9,center.y));
    camera.disable_freecam();
    camera.yaw = 0.0;
    camera.pitch = 0.95;
    camera.set_first_person(character.eye_position());
    PhysicsWorker::at_origin(region.scene,region.origin).map_err(|e| e.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let two_cube_demo = args.iter().any(|arg| arg == "--two-cubes");
    let friction_road_demo = args.iter().any(|arg| arg == "--friction-road");
    let pause_on_fracture = args.iter().any(|arg| arg == "--pause-on-fracture");
    let path = args
        .windows(2)
        .find(|p| p[0] == "--scene")
        .map(|p| p[1].as_str())
        .unwrap_or("scenes/mounted_axe.ron");
    let config = if two_cube_demo || friction_road_demo {
        let mut config = SceneConfig::default();
        if friction_road_demo {
            config.dimensions = [240, 9, 8];
            config.ground_dimensions = [240, 9, 1];
        } else {
            config.dimensions = [18, 2, 2];
            config.ground_dimensions = [24, 12, 1];
        }
        config
    } else {
        SceneConfig::load(path)?
    };
    let mut scene = if friction_road_demo {
        friction_heat_road_scene(config.clone())?
    } else if two_cube_demo {
        two_cube_collision_scene(config.clone())?
    } else {
        let mut scene = Scene::new(config.clone());
        fix_wood_base(&mut scene);
        scene
    };
    let mut ui = Ui::new(
        config.initial_speed,
        Vec3::from_array(config.tool_start),
        Vec3::from_array(config.tool_direction),
    );
    if two_cube_demo || friction_road_demo {
        ui.simulation_time_scale = 1.0;
    }
    if let Some(pair) = args.windows(2).find(|pair| pair[0] == "--time-scale") {
        let scale = pair[1].parse::<f32>()?;
        if !scale.is_finite() || scale < 0.0 {
            return Err("simulation time scale must be finite and nonnegative".into());
        }
        ui.simulation_time_scale = scale;
    }
    if let Some(index) = args.iter().position(|s| s == "--headless") {
        let steps: usize = args
            .get(index + 1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        let shape = selected_shape(&args).unwrap_or(Shape::Wedge);
        let mut tool = if two_cube_demo || friction_road_demo {
            None
        } else {
            Some(launch(shape, &scene, &ui))
        };
        let initial_momentum = total_momentum(&scene);
        let moving_mass: f64 = scene
            .voxels
            .iter()
            .filter(|voxel| !voxel.kinematic)
            .map(|voxel| f64::from(voxel.mass))
            .sum();
        let initial_kinetic_j = kinetic_energy(&scene);
        let initial_spin = angular_momentum(&scene, 0..8).length()
            + angular_momentum(&scene, 8..16).length();
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
        if two_cube_demo {
            let gravity_impulse = DVec3::new(
                0.0,
                0.0,
                moving_mass
                    * f64::from(config.gravity[2])
                    * f64::from(config.fixed_dt)
                    * steps as f64,
            );
            println!(
                "cube_centers_m=({:.4},{:.4},{:.4})/({:.4},{:.4},{:.4}) cube_velocity_m_s=({:.4},{:.4},{:.4})/({:.4},{:.4},{:.4}) pair_mu_oak_concrete={:.3} momentum_error_kg_m_s={:.3e} spin_L_kg_m2_s={:.3e}->{:.3e} kinetic_J={:.6}->{:.6} restitution_heat_J={:.6} friction_heat_J={:.6}",
                cube_center(&scene, 0..8).x,
                cube_center(&scene, 0..8).y,
                cube_center(&scene, 0..8).z,
                cube_center(&scene, 8..16).x,
                cube_center(&scene, 8..16).y,
                cube_center(&scene, 8..16).z,
                cube_velocity(&scene, 0..8).x,
                cube_velocity(&scene, 0..8).y,
                cube_velocity(&scene, 0..8).z,
                cube_velocity(&scene, 8..16).x,
                cube_velocity(&scene, 8..16).y,
                cube_velocity(&scene, 8..16).z,
                scene.config.fracture.friction
                    * (scene.properties[0].surface_friction_coefficient
                        + scene.properties[16].surface_friction_coefficient),
                (total_momentum(&scene) - initial_momentum - gravity_impulse).length(),
                initial_spin,
                angular_momentum(&scene, 0..8).length()
                    + angular_momentum(&scene, 8..16).length(),
                initial_kinetic_j,
                kinetic_energy(&scene),
                scene.energy.contact_restitution_heat,
                scene.energy.contact_friction_heat,
            );
        }
        if scene.positions.iter().any(|p| !p.is_finite()) {
            return Err("Non-finite solver output".into());
        }
        return Ok(());
    }
    // API checked with cargo check; opening a real window still needs display validation.
    let event_loop = EventLoop::new()?;
    let window = Arc::new(
        WindowBuilder::new()
        .with_title(if friction_road_demo {
            "Voxel lab | oak cube rolling on concrete road"
        } else if two_cube_demo {
            "Voxel lab | two 2x2x2 oak cubes collide"
        } else {
            "Voxel mechanics laboratory"
        })
            .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720))
            .build(&event_loop)?,
    );
    let mut renderer = pollster::block_on(Renderer::new(window.clone()))?;
    let mut camera = Camera::new();
    let height = config.dimensions[2] as f32 * scene::CELL;
    camera.target = if friction_road_demo {
        Vec3::new(-0.53, 0.0, 0.02)
    } else if two_cube_demo {
        Vec3::new(0.0, 0.0, 0.62)
    } else {
        Vec3::Z * (height * 0.5)
    };
    camera.radius = if friction_road_demo {
        0.38
    } else if two_cube_demo {
        0.90
    } else {
        (height * 1.5).max(
            config.ground_dimensions[0].max(config.ground_dimensions[1]) as f32
                * scene::CELL
                * 1.6,
        )
    };
    if two_cube_demo || friction_road_demo {
        camera.yaw = 0.55;
        camera.pitch = 0.35;
    }
    let initial_camera_radius = camera.radius;
    let initial_shape = selected_shape(&args);
    if let Some(shape) = initial_shape {
        ui.selected = shape;
    }
    let mut tool = if two_cube_demo || friction_road_demo {
        None
    } else {
        initial_shape.map(|shape| launch(shape, &scene, &ui))
    };
    let initial_seed = args
        .windows(2)
        .find(|pair| pair[0] == "--seed")
        .map(|pair| pair[1].as_str());
    // Reproduce vegetation-loading reports without waiting decades in the UI.
    let initial_age_years = args.windows(2)
        .find(|pair| pair[0] == "--world-age-years")
        .map(|pair| pair[1].parse::<f32>()).transpose()?.unwrap_or(0.0);
    if !initial_age_years.is_finite() || initial_age_years < 0.0 {
        return Err("initial world age must be finite and nonnegative".into());
    }
    let mut generated_world: Option<GeneratedWorld> = initial_seed.map(|seed| {
        ui.seed_text = seed.to_owned();
        ui.world_active = true;
        ui.paused = false;
        eprintln!("generating deterministic terrain and vegetation for seed {seed:?}");
        let mut world = GeneratedWorld::from_seed(seed);
        if initial_age_years > 0.0 {
            world.advance_growth(initial_age_years * (365.25 * 24.0 * 60.0 * 60.0));
        }
        world
    });
    let mut growth_scheduler = GrowthScheduler::default();
    let mut world_character = generated_world.as_ref().map(WorldCharacter::spawn);
    if args.iter().any(|arg| arg == "--near-tree") {
        if let (Some(world), Some(character)) = (&generated_world, &mut world_character) {
            if let Some(wood) = world.voxels.iter()
                .filter(|voxel| voxel.content == voxel_physics_lab::world::GeneratedCellContent::PlantWood)
                .min_by(|a, b| a.center.length_squared().total_cmp(&b.center.length_squared()))
            {
                let x = wood.center.x + 2.0;
                let y = wood.center.y;
                character.position = Vec3::new(x, y, world.terrain_height_at(x, y));
            }
        }
    }
    if world_character.is_some() {
        tool = None;
        camera.pitch = 0.0;
        camera.radius = 5.5;
        camera.set_first_person(world_character.as_ref().unwrap().eye_position());
        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
        window.set_cursor_visible(false);
    }
    let mut world_physics = None;
    if args.iter().any(|arg| arg == "--physics-world20") {
        let world = generated_world.as_mut().ok_or("--physics-world20 requires --seed")?;
        world_physics = Some(start_world_physics(world,world_character.as_mut().unwrap(),
            &mut camera,&config)?);
    }
    let mut pointer = (0.0f32, 0.0f32);
    let mut orbit = world_character.is_some() || two_cube_demo || friction_road_demo;
    let mut dragging = false;
    let mut keys_down = HashSet::new();
    let mut jump_queued = false;
    let mut last = Instant::now();
    let profile_world_geometry = std::env::var_os("VOXEL_PROFILE_WORLD_GEOMETRY").is_some();
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
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } if orbit && (camera.is_freecam() || camera.is_first_person()) => {
                camera.rotate_freecam(delta.0, delta.1)
            }
            Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
                WindowEvent::Focused(false) => {
                    orbit = false;
                    keys_down.clear();
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                WindowEvent::Focused(true) if camera.is_first_person() && !ui.menu_open => {
                    orbit = true;
                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                    window.set_cursor_visible(false);
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let next = (position.x as f32, position.y as f32);
                    if orbit && !camera.is_freecam() && !camera.is_first_person() {
                        camera.yaw -= (next.0 - pointer.0) * ui.look_sensitivity;
                        camera.pitch = (camera.pitch + (next.1 - pointer.1) * ui.look_sensitivity)
                            .clamp(-1.4, 1.4);
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
                } => {
                    camera.look_sensitivity = ui.look_sensitivity;
                    if camera.is_first_person() && !ui.menu_open {
                        orbit = true;
                    } else if camera.is_freecam() && !ui.menu_open {
                        orbit = state == ElementState::Pressed;
                        if orbit {
                            let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    } else if !camera.is_freecam() {
                        orbit = state == ElementState::Pressed;
                    }
                }
                WindowEvent::MouseInput {
                    button: MouseButton::Left,
                    state,
                    ..
                } => {
                    dragging = state == ElementState::Pressed
                        && (ui.menu_open || (!ui.world_active && pointer.0 < 295.0))
                        && pointer.1 >= 185.0;
                    if state == ElementState::Pressed {
                        match ui.click(pointer.0, pointer.1) {
                            Some(UiAction::Shape(shape)) => {
                                if mode == AppMode::Interactive && generated_world.is_none() {
                                    tool = Some(launch(shape, &scene, &ui));
                                }
                            }
                            Some(UiAction::GenerateWorld(seed)) => {
                                world_physics = None;
                                let generated = GeneratedWorld::from_seed(&seed);
                                eprintln!(
                                    "generated deterministic world: seed={seed:?}, area=10 km2, voxels={}, {}, simulation speed=1x",
                                    generated.voxels.len(),
                                    generated.vegetation_summary(),
                                );
                                generated_world = Some(generated);
                                growth_scheduler = GrowthScheduler::default();
                                world_character = generated_world.as_ref().map(WorldCharacter::spawn);
                                ui.world_active = true;
                                ui.paused = false;
                                ui.show_bonds = false;
                                tool = None;
                                mode = AppMode::Interactive;
                                playback_frames.clear();
                                playback_time_s = 0.0;
                                accumulator = 0.0;
                                camera.disable_freecam();
                                if let Some(character) = world_character.as_ref() {
                                    camera.pitch = 0.0;
                                    camera.radius = 5.5;
                                    camera.set_first_person(character.eye_position());
                                    orbit = true;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                }
                                last = Instant::now();
                            }
                            Some(UiAction::Resume | UiAction::Back | UiAction::Settings) => {
                                accumulator = 0.0;
                                last = Instant::now();
                                if !ui.menu_open && camera.is_first_person() {
                                    orbit = true;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                }
                            }
                            Some(UiAction::Exit) => target.exit(),
                            None => {}
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let amount = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                    };
                    if camera.is_freecam() {
                        camera.adjust_free_speed(amount);
                    } else if !camera.is_first_person() {
                        camera.radius = (camera.radius * (-amount * 0.1).exp())
                            .clamp(0.3, WORLD_SIDE_M * 1.5);
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Released =>
                {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        keys_down.remove(&key);
                    }
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed
                        && !event.repeat
                        && event.physical_key == PhysicalKey::Code(KeyCode::Escape) =>
                {
                    ui.toggle_pause_menu();
                    keys_down.clear();
                    jump_queued = false;
                    orbit = false;
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                    window.set_cursor_visible(true);
                    if !ui.menu_open && camera.is_first_person() {
                        orbit = true;
                        let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                        window.set_cursor_visible(false);
                    }
                    accumulator = 0.0;
                    last = Instant::now();
                }
                WindowEvent::KeyboardInput { event, .. }
                    if ui.menu_open && event.state == ElementState::Pressed =>
                {
                    // The pause/settings menu owns keyboard input until closed.
                }
                WindowEvent::KeyboardInput { event, .. }
                    if event.state == ElementState::Pressed && !event.repeat =>
                {
                    if let PhysicalKey::Code(key) = event.physical_key {
                        keys_down.insert(key);
                    }
                    let editing_seed = ui.seed_editing;
                    if editing_seed {
                        if event.physical_key == PhysicalKey::Code(KeyCode::Backspace) {
                            ui.erase_seed_character();
                        } else if let Some(text) = event.text.as_ref() {
                            ui.append_seed_text(text.as_str());
                        }
                    }
                    let shape = if editing_seed {
                        None
                    } else {
                        match event.physical_key {
                        PhysicalKey::Code(KeyCode::F6) => {
                            if let (Some(world),Some(character)) =
                                (generated_world.as_mut(),world_character.as_mut()) {
                                match start_world_physics(world,character,&mut camera,&config) {
                                    Ok(worker) => { world_physics = Some(worker); ui.paused = false; }
                                    Err(error) => eprintln!("world physics could not start: {error}"),
                                }
                            }
                            None
                        }
                        PhysicalKey::Code(KeyCode::KeyF) => {
                            if camera.is_freecam() {
                                camera.disable_freecam();
                                if let Some(character) = world_character.as_ref() {
                                    camera.set_first_person(character.eye_position());
                                    orbit = true;
                                    let _ = window.set_cursor_grab(CursorGrabMode::Locked);
                                    window.set_cursor_visible(false);
                                } else {
                                    orbit = false;
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                }
                            } else {
                                camera.enable_freecam();
                                orbit = false;
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                                window.set_cursor_visible(true);
                            }
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.x = (camera.target.x - pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.x = (camera.target.x + pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.y = (camera.target.y + pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            if generated_world.is_some() {
                                return;
                            }
                            let pan = (camera.radius * 0.08).clamp(0.1, 500.0);
                            camera.target.y = (camera.target.y - pan)
                                .clamp(-WORLD_SIDE_M * 0.5, WORLD_SIDE_M * 0.5);
                            None
                        }
                        PhysicalKey::Code(KeyCode::Digit1) => Some(Shape::Wedge),
                        PhysicalKey::Code(KeyCode::Digit2) => Some(Shape::Blade),
                        PhysicalKey::Code(KeyCode::Digit3) => Some(Shape::Rod),
                        PhysicalKey::Code(KeyCode::Digit4) => Some(Shape::Rounded),
                        PhysicalKey::Code(KeyCode::Space) => {
                            if world_character.is_some() && !camera.is_freecam() {
                                jump_queued = true;
                            } else {
                                ui.paused = !ui.paused;
                                if mode != AppMode::Playback {
                                    accumulator = 0.0;
                                }
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
                            generated_world = None;
                            world_character = None;
                            ui.world_active = false;
                            if friction_road_demo {
                                scene = friction_heat_road_scene(config.clone())
                                    .expect("the built-in friction road scene is valid");
                                camera.set_orbit(Vec3::new(-0.53, 0.0, 0.015), 0.38);
                                camera.yaw = 0.55;
                                camera.pitch = 0.35;
                            } else if two_cube_demo {
                                scene = two_cube_collision_scene(config.clone())
                                    .expect("the built-in cube collision scene is valid");
                                camera.set_orbit(Vec3::new(0.0, 0.0, 0.62), 0.90);
                                camera.yaw = 0.55;
                                camera.pitch = 0.35;
                            } else {
                                scene = Scene::new(config.clone());
                                fix_wood_base(&mut scene);
                                camera.set_orbit(Vec3::Z * (height * 0.5), initial_camera_radius);
                            }
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
                            if world_character.is_some() {
                                ui.paused = !ui.paused;
                                accumulator = 0.0;
                            } else { match mode {
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
                            }}
                            None
                        }
                        PhysicalKey::Code(KeyCode::Escape) => {
                            None
                        }
                        _ => None,
                    }};
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
                    if profile_world_geometry && generated_world.is_some() && elapsed > 0.05 {
                        eprintln!("world frame gap_ms={:.2}", elapsed * 1_000.0);
                    }

                    if camera.is_freecam() {
                        let pressed = |key| keys_down.contains(&key);
                        let right = (pressed(KeyCode::KeyD) || pressed(KeyCode::ArrowRight))
                            as u8 as f32
                            - (pressed(KeyCode::KeyA) || pressed(KeyCode::ArrowLeft)) as u8 as f32;
                        let forward = (pressed(KeyCode::KeyW) || pressed(KeyCode::ArrowUp)) as u8
                            as f32
                            - (pressed(KeyCode::KeyS) || pressed(KeyCode::ArrowDown)) as u8 as f32;
                        let up = f32::from(pressed(KeyCode::Space))
                            - f32::from(pressed(KeyCode::ControlLeft))
                            - f32::from(pressed(KeyCode::ControlRight));
                        let fast = pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight);
                        camera.move_freecam(right, forward, up, elapsed.min(0.1), fast);
                    }

                    if let Some(world) = generated_world.as_mut() {
                        growth_scheduler.advance(world, if ui.paused {
                            0.0
                        } else {
                            elapsed * ui.simulation_time_scale
                        });
                    }
                    if !ui.paused {
                        if let Some(world) = generated_world.as_mut() {
                            if !camera.is_freecam() {
                                if let Some(character) = world_character.as_mut() {
                                    let pressed = |key| keys_down.contains(&key);
                                    let right = f32::from(pressed(KeyCode::KeyD) || pressed(KeyCode::ArrowRight))
                                        - f32::from(pressed(KeyCode::KeyA) || pressed(KeyCode::ArrowLeft));
                                    let forward = f32::from(pressed(KeyCode::KeyW) || pressed(KeyCode::ArrowUp))
                                        - f32::from(pressed(KeyCode::KeyS) || pressed(KeyCode::ArrowDown));
                                    let sprint = pressed(KeyCode::ShiftLeft) || pressed(KeyCode::ShiftRight);
                                    character.set_crouching(world,
                                        pressed(KeyCode::ControlLeft) || pressed(KeyCode::ControlRight));
                                    character.advance(
                                        world,
                                        Vec3::new(right, forward, 0.0),
                                        camera.yaw,
                                        ui.walk_speed_mps,
                                        sprint,
                                        jump_queued,
                                        elapsed,
                                    );
                                    camera.set_first_person(character.eye_position());
                                }
                            }
                        }
                    }
                    jump_queued = false;
                    if let Some(physics) = world_physics.as_mut() {
                        physics.advance(f64::from(elapsed)*f64::from(ui.simulation_time_scale),ui.paused);
                    }

                    if generated_world.is_none()
                        && matches!(mode, AppMode::Interactive | AppMode::Recording)
                        && !ui.paused
                    {
                        live_wall_s += f64::from(elapsed);
                        accumulator += f64::from(elapsed) * f64::from(ui.simulation_time_scale);
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
                    let rects = ui.draw(
                        scene.broken,
                        scene.cluster_count,
                        fps,
                        sim_ms,
                        generated_world.as_ref().map(|world| world.age_years),
                    );
                    match renderer.draw(
                        &scene,
                        &tool,
                        generated_world.as_ref(),
                        world_physics.as_ref().map(|physics| &physics.frame),
                        &camera,
                        &rects,
                        ui.show_bonds,
                    ) {
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
                            .map_or(f32::NAN, |t| t.position.dot(ui.direction().as_dvec3()) as f32);
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
                        if let Some(physics) = world_physics.as_ref() {
                            window.set_title(&format!("Voxel lab | 20 x 20 m | 1cm physics | {:.2}s simulated | backlog {:.2}s | {:.1}ms/step | broken {} | {:.1} FPS | Ctrl crouch | F6 restart",
                                physics.frame.simulated_seconds,physics.backlog_seconds(),physics.frame.step_ms,
                                physics.frame.broken,fps));
                        } else if generated_world.is_some() {
                            let world = generated_world.as_ref().unwrap();
                            window.set_title(&format!(
                                "Voxel lab | 10 km2 map | seed {} | 1cm REV / overview LOD | age {:.0} y | {} | {:.1} FPS",
                                ui.seed_text,
                                world.age_years,
                                world.vegetation_summary(),
                                fps,
                            ));
                        } else if two_cube_demo {
                            window.set_title(&format!(
                                "Voxel lab | 2x2x2 oak cubes | {mode_title} | {:.2}s simulated | {:.1} FPS | broken {}",
                                live_simulated_s, fps, scene.broken
                            ));
                        } else if friction_road_demo {
                            window.set_title(&format!(
                                "Voxel lab | oak 3x3x3 rolling on concrete | {mode_title} | {:.2}s simulated | friction heat {:.3} mJ | {:.1} FPS",
                                live_simulated_s, scene.energy.contact_friction_heat * 1000.0, fps
                            ));
                        } else {
                            window.set_title(&format!(
                                "Voxel lab | {mode_title} | {:.1} FPS | {:.1} ms/step | tool {:.2} m/s axis {:.3} | broken {}",
                                fps, sim_ms, tool_speed, tool_axis_position, scene.broken
                            ));
                        }
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
