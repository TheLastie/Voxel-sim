//! Heated, displacement-controlled glass bond showing pre-melt strength loss.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, CELL},
    solver,
};

const DT: f32 = 0.001;
const STEPS: usize = 17_600;
const FRAMES: usize = 177;
const HEATING_STEPS: usize = 600;
const LOAD_RATIO: f32 = 0.8;

fn frame(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    heat_input: f64,
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        let damage = scene.bonds[0].get(i).map_or(0.0, |bond| bond.damage);
        writeln!(out, "V,{step},{i},{},{},{},{damage},0", p.x, p.y, p.z)?;
        let phase = scene.properties[i]
            .phase_state(scene.voxels[i].mass, scene.voxels[i].thermal_energy_f32());
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{},{}",
            p.x, p.y, p.z, phase.temperature_kelvin, phase.liquid_fraction
        )?;
    }
    if scene.bonds[0][0].neighbor != EMPTY {
        writeln!(out, "B,{step},0,1")?;
    }
    let material = &scene.properties[0];
    let phase = material.phase_state(scene.voxels[0].mass, scene.voxels[0].thermal_energy_f32());
    let limits = solver::bond_limits(scene, 0, 1, Vec3::X);
    let strain = (scene.positions[1].x - scene.positions[0].x - CELL) / CELL;
    let stress_ratio = if limits.tension > 0.0 {
        strain.abs() * material.young / limits.tension
    } else {
        f32::INFINITY
    };
    writeln!(
        out,
        "R,{step},{},{},{},{heat_input}",
        phase.temperature_kelvin,
        material.thermal_strength_scale(phase.temperature_kelvin),
        stress_ratio
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/thermal-strength.csv".into());
    let glass = Material::component_id("glass").ok_or("glass missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 1)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], glass))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], glass))
        .build()?;
    scene.set_material_strength_softening("glass", 0.7)?;
    if scene.bonds[0][0].neighbor != 1 {
        return Err("fixture did not create the expected glass bond".into());
    }
    let initial_position = scene.positions.clone();
    let material = &scene.properties[0];
    let target_temp =
        (material.melt_temp * material.strength_softening_start_ratio + material.melt_temp) * 0.5;
    let onset = material.melt_temp * material.strength_softening_start_ratio;
    let extension = LOAD_RATIO * material.tensile_strength / material.young * CELL;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},thermal")?;
    writeln!(
        out,
        "N,glass_pre_melt_softening,temperature_K,strength_factor,stress_to_limit,external_heat_J"
    )?;
    let mut heat_input = 0.0f64;
    frame(&mut out, &scene, 0, heat_input)?;
    let mut tool = None;
    let mut fracture_step = None;
    let intervals = FRAMES - 1;
    for step in 1..=STEPS {
        let target_temperature =
            300.0 + (target_temp - 300.0) * (step as f32 / HEATING_STEPS as f32).min(1.0);
        for i in 0..scene.voxels.len() {
            let target_energy =
                scene.properties[i].energy_at_temperature(scene.voxels[i].mass, target_temperature);
            let delta = target_energy - scene.voxels[i].thermal_energy_f32();
            scene.voxels[i].add_internal_energy_j(delta as f64);
            heat_input += delta.max(0.0) as f64;
        }
        let mut targets = [
            initial_position[0].to_array().map(Some),
            initial_position[1].to_array().map(Some),
        ];
        targets[1][0] = Some(initial_position[1].x + extension);
        solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
        if fracture_step.is_none() && scene.bonds[0][0].neighbor == EMPTY {
            fracture_step = Some(step);
            writeln!(out, "E,{step},1,thermal_strength_fracture")?;
            let temperature = scene.temperature_kelvin(0).unwrap_or(0.0);
            let limits = solver::bond_limits(&scene, 0, 1, Vec3::X);
            let strain = (scene.positions[1].x - scene.positions[0].x - CELL) / CELL;
            let ratio = if limits.tension > 0.0 {
                strain.abs() * scene.properties[0].young / limits.tension
            } else {
                f32::INFINITY
            };
            writeln!(
                out,
                "X,{step},{temperature},{},{ratio}",
                scene.properties[0].thermal_strength_scale(temperature)
            )?;
            eprintln!(
                "thermal-strength fracture step={step} T={temperature:.2}K factor={:.4} load_to_limit={ratio:.4}",
                scene.properties[0].thermal_strength_scale(temperature)
            );
        }
        let sample = step == STEPS
            || (step as u128 * intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * intervals as u128 / STEPS as u128);
        if sample {
            frame(&mut out, &scene, step, heat_input)?;
        }
        if step % 4_000 == 0 {
            eprintln!(
                "thermal-strength step={step}/{STEPS} t={:.2}s T={:.1}K factor={:.3} intact={} heat_in={heat_input:.6}J",
                step as f32 * DT,
                scene.temperature_kelvin(0).unwrap_or(0.0),
                scene.properties[0]
                    .thermal_strength_scale(scene.temperature_kelvin(0).unwrap_or(0.0)),
                scene.bonds[0][0].neighbor != EMPTY
            );
        }
    }
    let final_temp = scene.temperature_kelvin(0).unwrap_or(0.0);
    let factor = scene.properties[0].thermal_strength_scale(final_temp);
    eprintln!(
        "thermal-strength complete steps={STEPS} states={FRAMES} physical_s={:.2} onset={onset:.2}K target={target_temp:.2}K final={final_temp:.2}K factor={factor:.4} fracture_step={fracture_step:?} heat_input={heat_input:.6}J",
        STEPS as f32 * DT
    );
    if fracture_step.is_none() || final_temp >= scene.properties[0].melt_temp || factor <= 0.0 {
        return Err("thermal-strength capture failed to fracture a still-solid bond".into());
    }
    Ok(())
}
