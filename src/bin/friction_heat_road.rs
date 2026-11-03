//! Reproducible friction and heating run: a 3x3x3 oak REV cube rolls and slips
//! along a fixed concrete REV road. Outputs measured CSV data and an SVG plot.
use std::{fs, path::Path, time::Instant};

use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

const DT: f32 = 0.001;
const DURATION_S: f64 = 1.5;
const SAMPLE_INTERVAL: usize = 10;
const ROAD_LENGTH_CELLS: i32 = 240;
const ROAD_WIDTH_CELLS: i32 = 9;
const OAK_SIDE: i32 = 3;
const INITIAL_SPEED_M_S: f32 = 0.30;
const INITIAL_OMEGA_Y_RAD_S: f32 = 18.0;

#[derive(Clone, Copy)]
struct Sample {
    time_s: f64,
    x_m: f64,
    speed_m_s: f64,
    angular_speed_rad_s: f64,
    friction_heat_j: f64,
    mean_temperature_k: f64,
    broken_bonds: usize,
}

fn cube_state(scene: &Scene, indices: &[usize]) -> (DVec3, Vec3, f64, f64, f64) {
    let mut mass = 0.0_f64;
    let mut center = DVec3::ZERO;
    let mut momentum = DVec3::ZERO;
    let mut angular_momentum = DVec3::ZERO;
    let mut temperature_mass = 0.0_f64;
    for &i in indices {
        let voxel = &scene.voxels[i];
        let m = f64::from(voxel.mass);
        mass += m;
        center += scene.positions[i] * m;
        momentum += voxel.velocity.as_dvec3() * m;
        temperature_mass += f64::from(scene.temperature_kelvin(i).unwrap_or(293.15)) * m;
    }
    center /= mass.max(1e-30);
    for &i in indices {
        let voxel = &scene.voxels[i];
        let p = voxel.velocity.as_dvec3() * f64::from(voxel.mass);
        angular_momentum += (scene.positions[i] - center).cross(p);
    }
    let velocity = (momentum / mass.max(1e-30)).as_vec3();
    // For a uniform 3cm cube, the inertia tensor about the central axis is
    // approximately m*(a^2+a^2)/12, with a=3cm.
    let inertia_y = mass * 2.0 * (OAK_SIDE as f64 * 0.01).powi(2) / 12.0;
    let omega_y = angular_momentum.y.abs() / inertia_y.max(1e-30);
    (
        center,
        velocity,
        omega_y,
        temperature_mass / mass.max(1e-30),
        mass,
    )
}

fn kinetic_energy(scene: &Scene, indices: &[usize]) -> f64 {
    indices
        .iter()
        .map(|&i| {
            0.5 * f64::from(scene.voxels[i].mass)
                * f64::from(scene.voxels[i].velocity.length_squared())
        })
        .sum()
}

fn scene_internal_energy(scene: &Scene, range: std::ops::Range<usize>) -> f64 {
    range
        .map(|i| scene.voxels[i].total_internal_energy_j())
        .sum()
}

fn scene_kinetic_energy(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|voxel| 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared()))
        .sum()
}

fn gravitational_potential(scene: &Scene) -> f64 {
    let gravity_z = f64::from(scene.config.gravity[2]);
    scene
        .voxels
        .iter()
        .zip(&scene.positions)
        .map(|(voxel, position)| -f64::from(voxel.mass) * gravity_z * position.z)
        .sum()
}

fn svg_chart(samples: &[Sample], path: &Path) -> std::io::Result<()> {
    const W: f64 = 1100.0;
    const H: f64 = 680.0;
    const LEFT: f64 = 90.0;
    const RIGHT: f64 = 1030.0;
    const TOP1: f64 = 95.0;
    const BOT1: f64 = 340.0;
    const TOP2: f64 = 405.0;
    const BOT2: f64 = 625.0;
    const MID1: f64 = 217.5;
    const MID2: f64 = 515.0;
    const LABEL_Y: f64 = 665.0;
    let max_time = samples.last().map_or(DURATION_S, |s| s.time_s).max(1e-9);
    let max_heat = samples
        .iter()
        .map(|s| s.friction_heat_j)
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let t0 = samples.first().map_or(293.15, |s| s.mean_temperature_k);
    let max_temp_rise = samples
        .iter()
        .map(|s| (s.mean_temperature_k - t0).max(0.0))
        .fold(0.0_f64, f64::max)
        .max(1e-9);
    let polyline = |value: &dyn Fn(&Sample) -> f64, max_y: f64, top: f64, bottom: f64| {
        samples
            .iter()
            .map(|s| {
                let x = LEFT + (RIGHT - LEFT) * s.time_s / max_time;
                let y = bottom - (bottom - top) * (value(s) / max_y).clamp(0.0, 1.0);
                format!("{x:.2},{y:.2}")
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let heat = polyline(
        &|s| s.friction_heat_j * 1000.0,
        max_heat * 1000.0,
        TOP1,
        BOT1,
    );
    let temp = polyline(
        &|s| (s.mean_temperature_k - t0).max(0.0) * 1000.0,
        max_temp_rise * 1000.0,
        TOP2,
        BOT2,
    );
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<rect width="100%" height="100%" fill="#101923"/><style>text{{font-family:Segoe UI,Arial,sans-serif;fill:#e8eef4}} .muted{{fill:#a8bac9}} .grid{{stroke:#334655;stroke-width:1}} .axis{{stroke:#95a8b6;stroke-width:1.5}}</style>
<text x="{LEFT}" y="38" font-size="23" font-weight="700">Cube rolling on concrete: measured friction and heating</text>
<text x="{LEFT}" y="65" class="muted" font-size="14">Oak 3×3×3 cm REV cube · road 2.40 m × 0.09 m · dt 1 ms · sampled every 10 ms · μ pair from material table</text>
<text x="{LEFT}" y="87" font-size="15">Cumulative friction heat (mJ)</text>
<line x1="{LEFT}" y1="{BOT1}" x2="{RIGHT}" y2="{BOT1}" class="axis"/><line x1="{LEFT}" y1="{TOP1}" x2="{LEFT}" y2="{BOT1}" class="axis"/>
<line x1="{LEFT}" y1="{TOP1}" x2="{RIGHT}" y2="{TOP1}" class="grid"/><line x1="{LEFT}" y1="{MID1}" x2="{RIGHT}" y2="{MID1}" class="grid"/>
<text x="22" y="{MID1}" class="muted" font-size="13" transform="rotate(-90 22 {MID1})">heat (mJ)</text>
<polyline points="{heat}" fill="none" stroke="#ffad5c" stroke-width="3" stroke-linejoin="round"/>
<text x="{LEFT}" y="378" font-size="15">Mean cube temperature rise (mK)</text>
<line x1="{LEFT}" y1="{BOT2}" x2="{RIGHT}" y2="{BOT2}" class="axis"/><line x1="{LEFT}" y1="{TOP2}" x2="{LEFT}" y2="{BOT2}" class="axis"/>
<line x1="{LEFT}" y1="{TOP2}" x2="{RIGHT}" y2="{TOP2}" class="grid"/><line x1="{LEFT}" y1="{MID2}" x2="{RIGHT}" y2="{MID2}" class="grid"/>
<text x="22" y="{MID2}" class="muted" font-size="13" transform="rotate(-90 22 {MID2})">temperature rise (mK, max {max_temp_rise:.3})</text>
<polyline points="{temp}" fill="none" stroke="#ff6f76" stroke-width="3" stroke-linejoin="round"/>
<line x1="850" y1="365" x2="880" y2="365" stroke="#ff6f76" stroke-width="3"/><text x="888" y="370" class="muted" font-size="13">temperature rise · max {max_temp_rise:.3} mK</text>
<text x="{LEFT}" y="{LABEL_Y}" class="muted" font-size="13">time (s)</text><text x="{RIGHT}" y="{LABEL_Y}" text-anchor="end" class="muted" font-size="13">0 — {max_time:.2} s</text>
</svg>"##
    );
    // Scale labels show the actual extrema for each plotted quantity.
    fs::write(path, svg)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let oak = Material::component_id("oak").ok_or("oak material is unavailable")?;
    let concrete = Material::component_id("concrete").ok_or("concrete material is unavailable")?;
    let mut config = SceneConfig::default();
    config.dimensions = [ROAD_LENGTH_CELLS as usize, ROAD_WIDTH_CELLS as usize, 8];
    config.ground_dimensions = [ROAD_LENGTH_CELLS as usize, ROAD_WIDTH_CELLS as usize, 1];
    config.gravity = [0.0, 0.0, -9.81];
    config.fixed_dt = DT;
    config.substeps = 1;
    config.iterations = 48;
    config.corotation_iterations = 2;
    config.fracture.friction = 0.45;

    let mut spawns = Vec::with_capacity((ROAD_LENGTH_CELLS * ROAD_WIDTH_CELLS) as usize + 27);
    let road_start = -ROAD_LENGTH_CELLS / 2;
    for x in road_start..road_start + ROAD_LENGTH_CELLS {
        for y in -(ROAD_WIDTH_CELLS / 2)..=(ROAD_WIDTH_CELLS / 2) {
            spawns.push(VoxelSpawn::at_cell([x, y, -1], concrete).fixed());
        }
    }
    let center = DVec3::new(-0.555, 0.005, 0.015);
    let omega = Vec3::new(0.0, INITIAL_OMEGA_Y_RAD_S, 0.0);
    let cube_start_x = -57;
    let mut cube_indices = Vec::with_capacity(27);
    for z in 0..OAK_SIDE {
        for y in -1..=1 {
            for x in 0..OAK_SIDE {
                let cell = [cube_start_x + x, y, z];
                let pos = DVec3::new(
                    f64::from(cell[0]) * 0.01 + 0.005,
                    f64::from(cell[1]) * 0.01 + 0.005,
                    f64::from(cell[2]) * 0.01 + 0.005,
                );
                let velocity =
                    Vec3::new(INITIAL_SPEED_M_S, 0.0, 0.0) + omega.cross((pos - center).as_vec3());
                cube_indices.push(spawns.len());
                spawns.push(VoxelSpawn::at_cell(cell, oak).with_velocity(velocity));
            }
        }
    }
    let pair_mu = config.fracture.friction * (0.5 + 0.5);
    let mut scene = Scene::from_voxels(config, spawns)?;
    let initial_temperature: f64 = cube_indices
        .iter()
        .map(|&i| f64::from(scene.temperature_kelvin(i).unwrap_or(293.15)))
        .sum::<f64>()
        / cube_indices.len() as f64;
    let initial_kinetic = kinetic_energy(&scene, &cube_indices);
    let oak_start = cube_indices[0];
    let oak_end = oak_start + cube_indices.len();
    let initial_internal_oak = scene_internal_energy(&scene, oak_start..oak_end);
    let initial_internal_road = scene_internal_energy(&scene, 0..oak_start);
    let initial_internal_total = scene_internal_energy(&scene, 0..scene.voxels.len());
    let initial_total_energy = scene_kinetic_energy(&scene)
        + gravitational_potential(&scene)
        + initial_internal_total
        + scene.elastic_bond_energy_j()
        + scene.stored_plastic_bond_energy_j();
    let initial_scene_kinetic = scene_kinetic_energy(&scene);
    let initial_gravitational = gravitational_potential(&scene);
    let initial_elastic = scene.elastic_bond_energy_j();
    let initial_plastic_stored = scene.stored_plastic_bond_energy_j();
    let mut samples = Vec::with_capacity((DURATION_S * 100.0) as usize + 1);
    let steps = (DURATION_S / f64::from(DT)).round() as usize;
    let run_start = Instant::now();

    for step in 0..=steps {
        if step % SAMPLE_INTERVAL == 0 {
            let (center, velocity, angular_speed, temperature, _) =
                cube_state(&scene, &cube_indices);
            samples.push(Sample {
                time_s: step as f64 * f64::from(DT),
                x_m: center.x,
                speed_m_s: f64::from(velocity.x),
                angular_speed_rad_s: angular_speed,
                friction_heat_j: scene.energy.contact_friction_heat,
                mean_temperature_k: temperature,
                broken_bonds: scene.broken,
            });
        }
        if step < steps {
            solver::step(&mut scene, &mut None);
        }
    }

    fs::create_dir_all("validation")?;
    let csv_path = Path::new("validation/friction_heat_road.csv");
    let svg_path = Path::new("validation/friction_heat_road.svg");
    let mut csv = String::from("time_s,center_x_m,center_vx_m_s,angular_speed_rad_s,cumulative_friction_heat_j,mean_temperature_k,temperature_rise_mK,broken_bonds\n");
    for s in &samples {
        csv.push_str(&format!(
            "{:.3},{:.9},{:.9},{:.9},{:.12},{:.6},{:.6},{}\n",
            s.time_s,
            s.x_m,
            s.speed_m_s,
            s.angular_speed_rad_s,
            s.friction_heat_j,
            s.mean_temperature_k,
            (s.mean_temperature_k - initial_temperature) * 1000.0,
            s.broken_bonds
        ));
    }
    fs::write(csv_path, csv)?;
    svg_chart(&samples, svg_path)?;

    let last = samples.last().copied().ok_or("no samples collected")?;
    let initial_surface_speed = f64::from(INITIAL_OMEGA_Y_RAD_S.abs() * 0.015);
    let final_kinetic = kinetic_energy(&scene, &cube_indices);
    let final_internal_oak = scene_internal_energy(&scene, oak_start..oak_end);
    let final_internal_road = scene_internal_energy(&scene, 0..oak_start);
    let final_internal_total = scene_internal_energy(&scene, 0..scene.voxels.len());
    let final_total_energy = scene_kinetic_energy(&scene)
        + gravitational_potential(&scene)
        + final_internal_total
        + scene.elastic_bond_energy_j()
        + scene.stored_plastic_bond_energy_j();
    let final_scene_kinetic = scene_kinetic_energy(&scene);
    let final_gravitational = gravitational_potential(&scene);
    let final_elastic = scene.elastic_bond_energy_j();
    let final_plastic_stored = scene.stored_plastic_bond_energy_j();
    println!("friction_heat_road cube_revs=27 cube_size_m=0.03 road_m={:.2}x{:.2} dt_s={DT:.3} duration_s={DURATION_S:.2} samples={} pair_mu={pair_mu:.3}",
        ROAD_LENGTH_CELLS as f32*0.01, ROAD_WIDTH_CELLS as f32*0.01, samples.len());
    println!("speed_initial_m_s={INITIAL_SPEED_M_S:.4} spin_surface_speed_initial_m_s={initial_surface_speed:.4} speed_final_m_s={:.6} x_final_m={:.4} omega_final_rad_s={:.4}", last.speed_m_s, last.x_m, last.angular_speed_rad_s);
    println!("friction_heat_J={:.9e} mean_temperature_rise_mK={:.6} kinetic_J_initial={initial_kinetic:.9e} kinetic_J_final={final_kinetic:.9e} broken_bonds={} wall_s={:.3}",
        last.friction_heat_j, (last.mean_temperature_k-initial_temperature)*1000.0,
        last.broken_bonds, run_start.elapsed().as_secs_f64());
    println!("oak_internal_gain_J={:.9e} road_internal_gain_J={:.9e} total_internal_gain_J={:.9e} friction_ledger_J={:.9e} restitution_ledger_J={:.9e} solid_viscoelastic_heat_J={:.9e} plastic_heat_J={:.9e} damage_heat_J={:.9e} stored_plastic_J={:.9e} surface_work_J={:.9e} external_work_J={:.9e}",
        final_internal_oak-initial_internal_oak,
        final_internal_road-initial_internal_road,
        final_internal_total-initial_internal_total,
        scene.energy.contact_friction_heat,
        scene.energy.contact_restitution_heat, scene.energy.solid_viscoelastic_heat,
        scene.energy.plastic_heat, scene.energy.damage_heat, scene.energy.stored_plastic,
        scene.energy.surface_work, scene.energy.external_work);
    println!("total_energy_initial_J={initial_total_energy:.9e} total_energy_final_J={final_total_energy:.9e} total_energy_delta_J={:.9e} external_work_J={:.9e} balance_residual_J={:.9e}",
        final_total_energy-initial_total_energy,
        scene.energy.external_work,
        final_total_energy-initial_total_energy-scene.energy.external_work);
    println!("delta_kinetic_J={:.9e} delta_gravitational_J={:.9e} delta_elastic_bond_J={:.9e} delta_stored_plastic_bond_J={:.9e}",
        final_scene_kinetic-initial_scene_kinetic,
        final_gravitational-initial_gravitational,
        final_elastic-initial_elastic,
        final_plastic_stored-initial_plastic_stored);
    println!("csv={} svg={}", csv_path.display(), svg_path.display());
    Ok(())
}
