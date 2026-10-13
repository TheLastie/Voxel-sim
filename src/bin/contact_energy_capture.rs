//! Free REV collision capture for normal restitution and thermal accounting.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
    solver,
};

const STEPS: usize = 1_760;
const FRAMES: usize = 177;
const DT: f32 = 0.01;

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
    initial_u: &[f64],
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        let heat = (scene.voxels[i].total_internal_energy_j() - initial_u[i]).max(0.0);
        writeln!(out, "V,{step},{i},{},{},{},0,{heat}", p.x, p.y, p.z)?;
    }
    let mass: f64 = scene.voxels.iter().map(|v| v.mass as f64).sum();
    let p = momentum(scene);
    writeln!(
        out,
        "L,{step},2,0,{mass},{},{},0,{},{},{}",
        internal(scene),
        kinetic(scene),
        p.x,
        p.y,
        p.z
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/contact-energy.csv".into());
    let iron = Material::component_id("iron").ok_or("iron missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 4)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .at_temperature(293.15)
                .with_velocity(Vec3::X * 0.01),
        )
        .add_voxel(
            VoxelSpawn::at_cell([3, 0, 0], iron)
                .at_temperature(293.15)
                .with_velocity(Vec3::NEG_X * 0.01),
        )
        .build()?;
    scene.config.fracture.friction = 0.0;
    scene.config.fracture.restitution = 0.25;
    if scene.bonds.iter().flatten().any(|b| b.neighbor != u32::MAX) {
        return Err("collision fixture unexpectedly created a bond".into());
    }
    let initial_mass = scene.total_mass_kg();
    let initial_p = momentum(&scene);
    let initial_total = internal(&scene) + kinetic(&scene);
    let initial_u: Vec<_> = scene
        .voxels
        .iter()
        .map(|v| v.total_internal_energy_j())
        .collect();
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},mechanical")?;
    write_frame(&mut out, &scene, 0, &initial_u)?;
    let mut tool = None;
    let intervals = FRAMES - 1;
    let mut impact_step = None;
    let mut previous_heat = 0.0;
    let started = Instant::now();
    for step in 1..=STEPS {
        solver::step(&mut scene, &mut tool);
        let heat = scene.energy.contact_restitution_heat;
        if previous_heat == 0.0 && heat > 0.0 {
            impact_step = Some(step);
            writeln!(out, "E,{step},1,restitution_heat_{heat:.9e}_J")?;
        }
        previous_heat = heat;
        let sample = step == STEPS
            || (step as u128 * intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * intervals as u128 / STEPS as u128);
        if sample {
            write_frame(&mut out, &scene, step, &initial_u)?;
        }
        if step % 400 == 0 {
            eprintln!("collision step={step}/{STEPS} t={:.1}s restitution_heat={heat:.9e} J kinetic={:.9e} J energy_error={:.3e} J", step as f32 * DT, kinetic(&scene), (internal(&scene)+kinetic(&scene)-initial_total).abs());
        }
    }
    let final_total = internal(&scene) + kinetic(&scene);
    let energy_error = (final_total - initial_total).abs();
    let momentum_error = (momentum(&scene) - initial_p).length() as f64;
    eprintln!("collision complete steps={STEPS} frames={FRAMES} physical_s={:.2} impact_step={impact_step:?} restitution_heat={:.9e} J friction_heat={:.9e} J mass={:.9e} kg momentum_error={momentum_error:.3e} kg*m/s energy_error={energy_error:.3e} J wall={:.2}s", STEPS as f32 * DT, scene.energy.contact_restitution_heat, scene.energy.contact_friction_heat, scene.total_mass_kg(), started.elapsed().as_secs_f64());
    if impact_step.is_none()
        || scene.energy.contact_restitution_heat <= 0.0
        || scene.energy.contact_friction_heat != 0.0
        || (scene.total_mass_kg() - initial_mass).abs() > 1e-12
        || momentum_error > 1e-5
        || energy_error > initial_total.abs() * 2e-5
    {
        return Err("contact capture failed its mass, momentum, or energy balance".into());
    }
    Ok(())
}
