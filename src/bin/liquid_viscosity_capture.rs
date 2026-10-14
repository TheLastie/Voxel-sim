//! Physical capture of viscous shear between two liquid water REVs.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

const DT: f32 = 0.01;
const STEPS: usize = 1_760;
const STRIDE: usize = 10;
const FRAMES: usize = STEPS / STRIDE + 1;
const MU_PA_S: f32 = 1.002e-3;
const REFERENCE_TEMPERATURE_K: f32 = 293.15;
const TEMPERATURE_K: f32 = 333.15;
const ACTIVATION_ENERGY_J_MOL: f32 = 17_000.0;
const SLIP_SPEED: f32 = 5.0e-4;

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
fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        writeln!(out, "V,{step},{i},{},{},{},0,0", p.x, p.y, p.z)?;
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{},{}",
            p.x,
            p.y,
            p.z,
            scene.temperature_kelvin(i).unwrap_or(0.0),
            scene.phase_state(i).map_or(0.0, |s| s.liquid_fraction)
        )?;
    }
    let mass = scene.total_mass_kg();
    let p = momentum(scene);
    writeln!(
        out,
        "L,{step},{},{},{mass},{},{},{},{},{},{}",
        scene.voxels.len(),
        scene.broken,
        internal(scene),
        kinetic(scene),
        scene.energy.liquid_viscous_heat,
        p.x,
        p.y,
        p.z
    )?;
    writeln!(out, "F,{step},0,0,0,0,0,0,0")?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "validation/liquid-viscosity-temperature.csv".into());
    let interaction = args.next().unwrap_or_else(|| "liquid-liquid".into());
    if interaction != "liquid-liquid" && interaction != "liquid-solid" {
        return Err("interaction must be liquid-liquid or liquid-solid".into());
    }
    let water = Material::component_id("water").ok_or("water missing from material table")?;
    let second_material = if interaction == "liquid-solid" {
        Material::component_id("iron").ok_or("iron missing from material table")?
    } else {
        water
    };
    let second_velocity = if interaction == "liquid-solid" {
        Vec3::ZERO
    } else {
        Vec3::NEG_Y * SLIP_SPEED
    };
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 2)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], water)
                .with_velocity(Vec3::Y * SLIP_SPEED)
                .at_temperature(TEMPERATURE_K),
        )
        .add_voxel(
            VoxelSpawn::at_cell([1, 0, 0], second_material)
                .with_velocity(second_velocity)
                .at_temperature(TEMPERATURE_K),
        )
        .build()?;
    scene.set_material_dynamic_viscosity("water", MU_PA_S)?;
    scene.set_material_viscosity_temperature_model(
        "water",
        REFERENCE_TEMPERATURE_K,
        ACTIVATION_ENERGY_J_MOL,
    )?;
    let effective_viscosity = scene.properties[0].liquid_viscosity_at_temperature(TEMPERATURE_K);
    scene.config.fracture.restitution = 0.0;
    let initial_mass = scene.total_mass_kg();
    let initial_internal = internal(&scene);
    let initial_kinetic = kinetic(&scene);
    let initial_momentum = momentum(&scene);
    let initial_relative_speed = (scene.voxels[1].velocity - scene.voxels[0].velocity).length();
    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},2,mechanical")?;
    writeln!(
        out,
        "N,liquid_viscous_shear,reference_viscosity_Pa_s,activation_energy_J_mol,reference_temperature_K"
    )?;
    writeln!(
        out,
        "R,setup,{MU_PA_S},{SLIP_SPEED},{TEMPERATURE_K},{REFERENCE_TEMPERATURE_K},{ACTIVATION_ENERGY_J_MOL}"
    )?;
    writeln!(out, "R,interaction,{interaction}")?;
    let mut first_fluid_contact_step = None;
    let start = Instant::now();
    write_frame(&mut out, &scene, 0)?;
    for step in 1..=STEPS {
        scene.tick();
        if scene.energy.liquid_viscous_heat > 0.0 && first_fluid_contact_step.is_none() {
            first_fluid_contact_step = Some(step);
            writeln!(out, "E,{step},viscous_heat,liquid_contact")?;
        }
        if step % STRIDE == 0 {
            write_frame(&mut out, &scene, step)?;
        }
    }
    let final_relative_speed = (scene.voxels[1].velocity - scene.voxels[0].velocity).length();
    let energy_error = internal(&scene) + kinetic(&scene) - initial_internal - initial_kinetic;
    let momentum_error = (momentum(&scene) - initial_momentum).length();
    eprintln!("liquid_viscosity complete interaction={interaction} states={FRAMES} physical_s={:.2} temperature={TEMPERATURE_K:.2}K viscosity={effective_viscosity:.6e}Pa*s (reference {MU_PA_S:.6e}Pa*s at {REFERENCE_TEMPERATURE_K:.2}K; activation {ACTIVATION_ENERGY_J_MOL:.1}J/mol) liquid_contact_step={first_fluid_contact_step:?} relative_speed={initial_relative_speed:.8}->{final_relative_speed:.8}m/s viscous_heat={:.9e}J energy_error={energy_error:.3e}J momentum_error={momentum_error:.3e}kg*m/s mass={:.9}kg broken={} clusters={} wall_s={:.3}",
        STEPS as f32 * DT, scene.energy.liquid_viscous_heat, scene.total_mass_kg(), scene.broken,
        scene.cluster_count, start.elapsed().as_secs_f64());
    if first_fluid_contact_step.is_none()
        || final_relative_speed >= initial_relative_speed
        || scene.energy.liquid_viscous_heat <= 0.0
        || scene.total_mass_kg() != initial_mass
        || momentum_error > 1e-10
        || energy_error.abs() > (initial_internal + initial_kinetic).abs() * 1e-10
    {
        return Err("liquid viscosity capture failed conservation or shear checks".into());
    }
    Ok(())
}
