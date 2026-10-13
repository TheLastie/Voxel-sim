//! Physical impact capture for stress-free bond formation at a hot contact.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

const DT: f32 = 0.01;
const STEPS: usize = 1_760;
const FRAME_STRIDE: usize = 10;
const FRAMES: usize = STEPS / FRAME_STRIDE + 1;
const INITIAL_SPEED: f32 = 0.01;
const HOT_TEMPERATURE_K: f32 = 363.15;

fn kinetic(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
        .sum()
}
fn internal(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|v| v.total_internal_energy_j())
        .sum()
}
fn momentum(scene: &Scene) -> Vec3 {
    scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |p, v| p + v.velocity * v.mass)
}
fn write_frame(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    formed: bool,
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        writeln!(out, "V,{step},{i},{},{},{},0,0", p.x, p.y, p.z)?;
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{},0",
            p.x,
            p.y,
            p.z,
            scene.temperature_kelvin(i).unwrap_or(0.0)
        )?;
    }
    if formed && scene.bonds[0][0].neighbor != EMPTY {
        writeln!(out, "B,{step},0,1")?;
    }
    let mass: f64 = scene.voxels.iter().map(|v| v.mass as f64).sum();
    let p = momentum(scene);
    writeln!(
        out,
        "L,{step},2,0,{mass},{},{},{},{},{},{}",
        internal(scene),
        kinetic(scene),
        scene.energy.pending_release,
        p.x,
        p.y,
        p.z
    )?;
    // The viewer uses a non-empty tool row as a signal that a mechanical frame is ready.
    writeln!(out, "F,{step},0,0,0,0,0,0,1")?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/bond-formation.csv".into());
    let iron = Material::component_id("iron").ok_or("iron missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 2)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .fixed()
                .at_temperature(293.15),
        )
        .add_voxel(
            VoxelSpawn::at_cell([1, 0, 0], iron)
                .with_velocity(Vec3::NEG_X * INITIAL_SPEED)
                .at_temperature(293.15),
        )
        .build()?;
    // The paired REVs start as separate pieces because their initial supports differ.
    // Release the left REV before impact so the contact impulse conserves momentum.
    scene.positions[1].x += 0.005;
    scene.previous[1] = scene.positions[1];
    scene.release_voxels(&[0, 1])?;
    scene.voxels[0].velocity = Vec3::X * INITIAL_SPEED;
    scene.set_material_thermal_expansion("iron", 12.0e-6)?;
    let mut heater_input_j = 0.0f64;
    for i in 0..scene.voxels.len() {
        let hot_u =
            scene.properties[i].energy_at_temperature(scene.voxels[i].mass, HOT_TEMPERATURE_K);
        let increment = hot_u - scene.voxels[i].thermal_energy_f32();
        scene.voxels[i].add_internal_energy_j(increment as f64);
        heater_input_j += increment as f64;
    }
    scene.config.fracture.restitution = 0.0;
    let initial_mass = scene.total_mass_kg();
    let initial_internal = internal(&scene);
    let initial_momentum = momentum(&scene);
    let initial_kinetic = kinetic(&scene);
    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},2,mechanical")?;
    writeln!(
        out,
        "N,hot_contact_bonding,formation_offset_m,heater_input_J"
    )?;
    writeln!(out, "R,setup,{heater_input_j}")?;
    let mut formed_step = None;
    let start = Instant::now();
    write_frame(&mut out, &scene, 0, false)?;
    for step in 1..=STEPS {
        scene.tick();
        let relative_speed = (scene.voxels[1].velocity - scene.voxels[0].velocity).length();
        if formed_step.is_none()
            && (scene.positions[1] - scene.positions[0]).length() <= 0.01001
            && relative_speed <= 1e-5
        {
            let before_internal = internal(&scene);
            let before_momentum = momentum(&scene);
            scene.connect_adjacent_voxels(0, 1)?;
            let energy_jump = internal(&scene) - before_internal;
            let momentum_jump = (momentum(&scene) - before_momentum).length();
            if scene.bonds[0][0].neighbor != 1 || energy_jump != 0.0 || momentum_jump != 0.0 {
                return Err("bond formation changed energy/momentum or failed to connect".into());
            }
            formed_step = Some(step);
            writeln!(out, "E,{step},bond,formed_0_1")?;
            eprintln!(
                "bond formed at step={step}, t={:.3}s separation={:.9}m formation_offset={:?}",
                step as f32 * DT,
                (scene.positions[1] - scene.positions[0]).length(),
                scene.bonds[0][0].formation_offset
            );
        }
        if step % FRAME_STRIDE == 0 {
            write_frame(&mut out, &scene, step, formed_step.is_some())?;
        }
    }
    let final_total = internal(&scene) + kinetic(&scene);
    let expected_total = initial_internal + initial_kinetic;
    let energy_error = final_total - expected_total;
    let momentum_error = (momentum(&scene) - initial_momentum).length();
    let broken = scene.broken;
    eprintln!("bond_capture complete states={FRAMES} physical_s={:.2} formed_step={formed_step:?} heater={heater_input_j:.9}J mass={:.9}kg momentum_start={initial_momentum:?} momentum_end={:?} velocities_end={:?} momentum_error={momentum_error:.3e}kg*m/s energy_error={energy_error:.3e}J bonds_intact={} clusters={} wall_s={:.3}", STEPS as f32 * DT, initial_mass, momentum(&scene), scene.voxels.iter().map(|v| v.velocity).collect::<Vec<_>>(), scene.bonds[0][0].neighbor != EMPTY, scene.cluster_count, start.elapsed().as_secs_f64());
    if formed_step.is_none()
        || broken != 0
        || scene.bonds[0][0].neighbor == EMPTY
        || scene.total_mass_kg() != initial_mass
        || momentum_error > 1e-8
        || energy_error.abs() > expected_total.abs() * 2e-5
    {
        return Err(
            "hot-contact bond formation capture failed its topology or ledger checks".into(),
        );
    }
    Ok(())
}
