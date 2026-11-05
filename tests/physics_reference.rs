//! Analytic reference checks for WORLD_PARADIGM section 10 (10% tolerance).
//!
//! Each case compares one chosen observable with a closed-form continuum or
//! rigid-body result on the same material, mass, geometry and duration. Passing
//! these does not qualify every interaction: the coverage is what is listed here.
use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, SceneConfig, VoxelSpawn},
    solver,
};

const CELL: f64 = 0.01;
const TOLERANCE: f64 = 0.10;
const G: f64 = 9.81;

fn material(name: &str) -> (u16, Material) {
    let id = Material::component_id(name).unwrap();
    let table = SceneBuilder::standard()
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], id))
        .build()
        .unwrap();
    (id, table.properties[0].clone())
}

fn config(dt: f32, substeps: usize, gravity: [f32; 3]) -> SceneConfig {
    let mut config = SceneConfig::default();
    config.fixed_dt = dt;
    config.substeps = substeps;
    config.gravity = gravity;
    config.atmosphere.pressure_pa = 0.0;
    config
}

fn build(config: SceneConfig, cells: impl Iterator<Item = ([i32; 3], u16)>) -> Scene {
    let spawns = cells.map(|(cell, id)| VoxelSpawn::at_cell(cell, id)).collect();
    Scene::from_voxels(config, spawns).unwrap()
}

fn check(name: &str, expected: f64, measured: f64) {
    let error = (measured - expected).abs() / expected.abs();
    eprintln!("REF {name}: expected={expected:.6e} measured={measured:.6e} error={:.2}%", error * 100.0);
    assert!(
        error <= TOLERANCE,
        "{name}: expected {expected:.6e}, measured {measured:.6e}, error {:.2}%",
        error * 100.0
    );
}

fn run(scene: &mut Scene, steps: usize) {
    for _ in 0..steps {
        solver::step(scene, &mut None);
        assert!(scene.solver_residual.is_finite());
    }
}

#[test]
fn free_fall_matches_constant_acceleration() {
    let (iron, _) = material("iron");
    let mut scene = build(config(0.01, 4, [0.0, 0.0, -9.81]), [([0, 0, 0], iron)].into_iter());
    let start = scene.positions[0];
    run(&mut scene, 50);
    let t = 0.5;
    check("free_fall_speed", G * t, -f64::from(scene.voxels[0].velocity.z));
    check("free_fall_drop", 0.5 * G * t * t, start.z - scene.positions[0].z);
}

/// u = F (N-1) L / (E A) for a chain of N cells with one fixed end.
#[test]
fn axial_bar_static_elongation_matches_hooke() {
    let (iron, props) = material("iron");
    let n = 11;
    let mut scene = build(
        config(0.01, 4, [0.0; 3]),
        (0..n).map(|x| ([x, 0, 0], iron)),
    );
    scene.fix_voxels(&[0]).unwrap();
    let force = 200.0f64;
    let tip = (n - 1) as usize;
    let start = scene.positions[tip];
    solver::set_force(&mut scene, tip, Vec3::new(force as f32, 0.0, 0.0)).unwrap();
    run(&mut scene, 300);
    let length = (n - 1) as f64 * CELL;
    let expected = force * length / (f64::from(props.young) * CELL * CELL);
    check("axial_static_elongation", expected, scene.positions[tip].x - start.x);
}

/// Euler-Bernoulli plus Timoshenko shear: F L^3 / (3 E I) + F L / (k G A).
#[test]
fn cantilever_tip_deflection_matches_beam_theory() {
    let (iron, props) = material("iron");
    let (nx, ny, nz) = (31, 3, 3);
    let mut cells = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                cells.push(([x, y, z], iron));
            }
        }
    }
    let index = |x: i32, y: i32, z: i32| (x * ny * nz + y * nz + z) as usize;
    let mut scene = build(config(0.01, 4, [0.0; 3]), cells.into_iter());
    let root: Vec<_> = (0..ny)
        .flat_map(|y| (0..nz).map(move |z| (y, z)))
        .map(|(y, z)| index(0, y, z))
        .collect();
    scene.fix_voxels(&root).unwrap();
    let force = 90.0f64;
    let tip: Vec<_> = root.iter().map(|&i| i + ((nx - 1) * ny * nz) as usize).collect();
    let start: Vec<DVec3> = tip.iter().map(|&i| scene.positions[i]).collect();
    for &i in &tip {
        solver::set_force(&mut scene, i, Vec3::new(0.0, 0.0, (force / 9.0) as f32)).unwrap();
    }
    run(&mut scene, 400);
    let deflection = tip
        .iter()
        .zip(&start)
        .map(|(&i, s)| scene.positions[i].z - s.z)
        .sum::<f64>()
        / 9.0;
    let length = (nx - 1) as f64 * CELL;
    let side = f64::from(ny) * CELL;
    let inertia = side.powi(4) / 12.0;
    let area = side * side;
    let expected = force * length.powi(3) / (3.0 * f64::from(props.young) * inertia)
        + force * length / (5.0 / 6.0 * f64::from(props.shear) * area);
    check("cantilever_tip_deflection", expected, deflection);
}

/// First clamped-free longitudinal mode: f = sqrt(E/rho) / (4 L).
#[test]
fn bar_longitudinal_frequency_matches_wave_speed() {
    let (flesh, props) = material("flesh");
    let n = 21;
    let dt = 0.0005f32;
    let mut scene = build(config(dt, 1, [0.0; 3]), (0..n).map(|x| ([x, 0, 0], flesh)));
    scene.fix_voxels(&[0]).unwrap();
    let tip = (n - 1) as usize;
    let rest = scene.positions[tip].x;
    let force = Vec3::new(0.2, 0.0, 0.0);
    solver::set_force(&mut scene, tip, force).unwrap();
    run(&mut scene, 600);
    solver::set_force(&mut scene, tip, Vec3::ZERO).unwrap();
    // Crossing times of the released tip through its unloaded position.
    let mut crossings = Vec::new();
    let mut previous = scene.positions[tip].x - rest;
    for step in 0..2000 {
        solver::step(&mut scene, &mut None);
        let current = scene.positions[tip].x - rest;
        if previous * current < 0.0 {
            let fraction = previous.abs() / (previous.abs() + current.abs());
            crossings.push((step as f64 + fraction) * f64::from(dt));
        }
        previous = current;
    }
    assert!(crossings.len() >= 2, "tip stopped oscillating: {crossings:?}");
    let period = 2.0 * (crossings[1] - crossings[0]);
    // Total mass includes the fixed cell's half; L spans cell centres.
    let length = (n - 1) as f64 * CELL;
    let wave_speed = (f64::from(props.young) / f64::from(props.density)).sqrt();
    check("bar_first_mode_frequency", wave_speed / (4.0 * length), 1.0 / period);
}

/// A block sliding on a fixed floor stops after v^2 / (2 mu g).
#[test]
fn sliding_block_stops_at_coulomb_distance() {
    let (iron, iron_props) = material("iron");
    let (stone, stone_props) = material("stone");
    let config = config(0.01, 4, [0.0, 0.0, -9.81]);
    let friction_scale = f64::from(config.fracture.friction);
    let speed = 1.0f32;
    let mut spawns = Vec::new();
    for x in -60..60 {
        for y in -6..6 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let floor = spawns.len();
    for x in -2..2 {
        for y in -2..2 {
            spawns.push(VoxelSpawn::at_cell([x, y, 1], iron).with_velocity(Vec3::new(speed, 0.0, 0.0)));
        }
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    let start = scene.positions[floor].x;
    let mut stopped = false;
    for _ in 0..200 {
        solver::step(&mut scene, &mut None);
        if scene.voxels[floor].velocity.x.abs() < 1.0e-3 {
            stopped = true;
            break;
        }
    }
    assert!(stopped, "block did not stop, speed={}", scene.voxels[floor].velocity.x);
    let mu = friction_scale
        * f64::from(iron_props.surface_friction_coefficient + stone_props.surface_friction_coefficient);
    check(
        "sliding_stop_distance",
        f64::from(speed * speed) / (2.0 * mu * G),
        scene.positions[floor].x - start,
    );
}

fn iron_on_stone_mu(friction_scale: f64) -> f64 {
    let (_, iron) = material("iron");
    let (_, stone) = material("stone");
    friction_scale
        * f64::from(iron.surface_friction_coefficient + stone.surface_friction_coefficient)
}

/// Fixed stone floor with a free iron block `layers` thick resting on it.
fn block_on_floor(config: SceneConfig, layers: i32, velocity: Vec3) -> (Scene, usize) {
    let (iron, _) = material("iron");
    let (stone, _) = material("stone");
    let mut spawns = Vec::new();
    for x in -80..80 {
        for y in -6..6 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let floor = spawns.len();
    for z in 1..=layers {
        for x in -2..2 {
            for y in -2..2 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron).with_velocity(velocity));
            }
        }
    }
    (Scene::from_voxels(config, spawns).unwrap(), floor)
}

fn mean_velocity(scene: &Scene, from: usize) -> Vec3 {
    let moving = &scene.voxels[from..];
    moving.iter().map(|voxel| voxel.velocity).sum::<Vec3>() / moving.len() as f32
}

/// Coulomb friction does not depend on how many layers press on the contact.
#[test]
fn thick_block_stops_at_the_same_coulomb_distance() {
    let config = config(0.01, 4, [0.0, 0.0, -9.81]);
    let mu = iron_on_stone_mu(f64::from(config.fracture.friction));
    let speed = 1.0f32;
    let (mut scene, floor) = block_on_floor(config, 3, Vec3::new(speed, 0.0, 0.0));
    let start = scene.positions[floor].x;
    for _ in 0..200 {
        solver::step(&mut scene, &mut None);
        if mean_velocity(&scene, floor).x.abs() < 1.0e-3 {
            break;
        }
    }
    assert!(mean_velocity(&scene, floor).x.abs() < 1.0e-3, "block did not stop");
    check(
        "thick_block_stop_distance",
        f64::from(speed * speed) / (2.0 * mu * G),
        scene.positions[floor].x - start,
    );
}

/// Below the friction angle a block stays; above it, a = g_t - mu g_n.
#[test]
fn tilted_gravity_holds_below_and_accelerates_above_friction_angle() {
    let friction = f64::from(SceneConfig::default().fracture.friction);
    let mu = iron_on_stone_mu(friction);

    let holding = 0.5 * mu;
    let (mut scene, floor) = block_on_floor(
        config(0.01, 4, [(holding * G) as f32, 0.0, -9.81]),
        1,
        Vec3::ZERO,
    );
    let start = scene.positions[floor].x;
    run(&mut scene, 60);
    let crept = (scene.positions[floor].x - start).abs();
    eprintln!("REF friction_hold: tangential gravity {holding:.3} g crept {crept:.3e} m");
    assert!(crept < 1.0e-3, "block crept {crept} m below the friction angle");

    let pulling = 2.0 * mu;
    let (mut scene, floor) = block_on_floor(
        config(0.01, 4, [(pulling * G) as f32, 0.0, -9.81]),
        1,
        Vec3::ZERO,
    );
    run(&mut scene, 40);
    check(
        "friction_slide_speed",
        (pulling - mu) * G * 0.4,
        f64::from(mean_velocity(&scene, floor).x),
    );
}

/// A block with its centre of mass past the pivot edge tips about that edge.
/// Rigid pivot: angular acceleration = torque / inertia about the edge.
#[test]
fn block_past_its_tipping_angle_pivots_over_the_edge() {
    let (iron, _) = material("iron");
    let (stone, _) = material("stone");
    let gx = 1.2 * G;
    let mut config = config(0.01, 4, [gx as f32, 0.0, -9.81]);
    config.fracture.friction = 3.0;
    let mut spawns = Vec::new();
    for x in -20..20 {
        for y in -6..6 {
            spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
        }
    }
    let floor = spawns.len();
    for z in 1..=6 {
        for x in -2..2 {
            for y in -2..2 {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    for material in scene.properties.iter_mut() {
        material.thermal_expansion_parallel = 0.0;
        material.thermal_expansion_perpendicular = 0.0;
    }
    let count = (scene.voxels.len() - floor) as f64;
    let height = |scene: &Scene| scene.positions[floor..].iter().map(|p| p.z).sum::<f64>() / count;
    let start = height(&scene);
    run(&mut scene, 10);
    // Rigid pendulum about the front row, integrated exactly: the angle is large.
    let (half_w, half_h) = (1.5 * CELL, 2.5 * CELL);
    // Mass fills the 4 x 6 cell outline, whose inertia the lattice reproduces.
    let (side, tall) = (4.0 * CELL, 6.0 * CELL);
    let inertia = (side * side + tall * tall) / 12.0 + half_w * half_w + half_h * half_h;
    let (mut angle, mut rate) = (0.0f64, 0.0f64);
    let dt = 1.0e-5;
    for _ in 0..(0.1 / dt) as usize {
        let x = -half_w * angle.cos() + half_h * angle.sin();
        let z = half_w * angle.sin() + half_h * angle.cos();
        rate += (gx * z + G * x) / inertia * dt;
        angle += rate * dt;
    }
    let rise = half_w * angle.sin() + half_h * angle.cos() - half_h;
    check("pivot_com_rise", rise, height(&scene) - start);
}

/// A resting free body must not gain energy from solver noise.
#[test]
fn free_iron_block_at_rest_stays_at_rest() {
    let (iron, _) = material("iron");
    let mut cells = Vec::new();
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                cells.push(([x, y, z], iron));
            }
        }
    }
    let mut scene = build(config(0.01, 4, [0.0; 3]), cells.into_iter());
    let start = scene.positions.clone();
    run(&mut scene, 500);
    let mass: f64 = scene.voxels.iter().map(|v| f64::from(v.mass)).sum();
    let kinetic: f64 = scene
        .voxels
        .iter()
        .map(|v| 0.5 * f64::from(v.mass) * f64::from(v.velocity.length_squared()))
        .sum();
    let moved = scene
        .positions
        .iter()
        .zip(&start)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, f64::max);
    eprintln!("REF rest_noise: kinetic={:.3e} J/kg moved={moved:.3e} m", kinetic / mass);
    assert!(kinetic / mass < 1.0e-10, "resting block gained {} J/kg", kinetic / mass);
    assert!(moved < 1.0e-6, "resting block moved {moved} m");
}

/// Infinite-rod step solution: T(x) = Tm - dT/2 erf(x / (2 sqrt(alpha t))).
#[test]
fn rod_heat_step_matches_diffusion_solution() {
    let (iron, props) = material("iron");
    let n = 60i32;
    let (hot, cold) = (600.0f32, 300.0f32);
    let mut builder = SceneBuilder::new(config(0.1, 1, [0.0; 3]));
    for x in 0..n {
        let temperature = if x < n / 2 { hot } else { cold };
        builder = builder.add_voxel(VoxelSpawn::at_cell([x, 0, 0], iron).at_temperature(temperature));
    }
    let mut scene = builder.build().unwrap();
    let initial: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
    let seconds = 20.0f64;
    run(&mut scene, (seconds / 0.1) as usize);
    let alpha = f64::from(props.thermal_cond) / (f64::from(props.density) * f64::from(props.heat_capacity));
    let mean = 0.5 * f64::from(hot + cold);
    let half = 0.5 * f64::from(hot - cold);
    // First cell on the cold side: its centre lies half a cell from the step.
    for (name, cells_from_step) in [("heat_near", 0i32), ("heat_far", 4)] {
        let x = (f64::from(cells_from_step) + 0.5) * CELL;
        let expected = mean - half * erf(x / (2.0 * (alpha * seconds).sqrt()));
        let measured = f64::from(scene.temperature_kelvin((n / 2 + cells_from_step) as usize).unwrap());
        // Compare the temperature change, so the absolute kelvin offset cannot
        // hide an error in the transported heat.
        check(name, expected - f64::from(cold), measured - f64::from(cold));
    }
    let final_energy: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
    assert!((final_energy - initial).abs() <= initial * 1.0e-5, "heat energy drifted");
}

fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun 7.1.26, |error| <= 1.5e-7.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let polynomial = t * (0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    (1.0 - polynomial * (-x * x).exp()).copysign(x)
}

/// Heat needed to pass from solid at melting to all liquid is m * latent heat;
/// temperature holds during the transition.
#[test]
fn melting_holds_temperature_and_absorbs_latent_heat() {
    let (_, iron) = material("iron");
    let mass = 1.0f32;
    let start = iron.energy_at_temperature(mass, iron.melt_temp);
    let mut full_liquid = None;
    let mut energy = start;
    let step = iron.latent_fusion * mass / 200.0;
    for _ in 0..400 {
        energy += step;
        let phase = iron.phase_state(mass, energy);
        if phase.liquid_fraction < 0.999 {
            assert!((phase.temperature_kelvin - iron.melt_temp).abs() < 1.0, "melt plateau broke");
        } else if full_liquid.is_none() {
            full_liquid = Some(energy);
        }
    }
    let absorbed = f64::from(full_liquid.expect("iron never melted") - start);
    check("latent_heat_of_fusion", f64::from(iron.latent_fusion * mass), absorbed);
}
