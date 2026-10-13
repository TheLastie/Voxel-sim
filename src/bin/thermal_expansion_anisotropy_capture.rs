//! Compare restrained wood bars along and across their fiber axis.
use glam::Quat;
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    voxel::Microstructure,
};

const STEPS: usize = 1_760;
const DT: f32 = 0.01;
const SAMPLE_EVERY: usize = 10;
const FRAME_COUNT: usize = STEPS / SAMPLE_EVERY + 1;
const INITIAL_TEMPERATURE_K: f32 = 293.15;
const RISE_K: f32 = 70.0;
const RAMP_STEPS: usize = 60;
const ALPHA_PARALLEL: f32 = 3.0e-6;
const ALPHA_PERPENDICULAR: f32 = 30.0e-6;
const AREA_M2: f32 = 1.0e-4;

fn write_frame(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    heater_j: f64,
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, position) in scene.positions.iter().enumerate() {
        let temperature = scene.temperature_kelvin(i).unwrap_or(f32::NAN);
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{temperature},0",
            position.x, position.y, position.z
        )?;
    }
    let target = INITIAL_TEMPERATURE_K + RISE_K * (step as f32 / RAMP_STEPS as f32).min(1.0);
    let span0 = 0.02f32;
    let free_parallel = span0 * (1.0 + ALPHA_PARALLEL * (target - INITIAL_TEMPERATURE_K));
    let free_perpendicular = span0 * (1.0 + ALPHA_PERPENDICULAR * (target - INITIAL_TEMPERATURE_K));
    let actual_parallel = (scene.positions[5].z - scene.positions[3].z).abs();
    let actual_perpendicular = (scene.positions[2].x - scene.positions[0].x).abs();
    let force_parallel = scene.workspace[2][4].last_force.z.abs();
    let force_perpendicular = scene.workspace[0][1].last_force.x.abs();
    writeln!(
        out,
        "X,{step},{target},{},{},{},{},{heater_j},{free_parallel},{free_perpendicular},{actual_parallel},{actual_perpendicular},{},{},{},{}",
        scene.temperature_kelvin(0).unwrap_or(f32::NAN),
        scene.temperature_kelvin(1).unwrap_or(f32::NAN),
        scene.temperature_kelvin(3).unwrap_or(f32::NAN),
        scene.temperature_kelvin(4).unwrap_or(f32::NAN),
        force_parallel / AREA_M2,
        force_perpendicular / AREA_M2,
        scene.cluster_count,
        force_parallel,
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/thermal-expansion-anisotropy.csv".into());
    let wood = Material::component_id("wood").ok_or("wood is missing from material table")?;
    let microstructure = Microstructure {
        orientation: Quat::IDENTITY,
        anisotropy: 1.0,
    };
    let spawn = |cell| {
        VoxelSpawn::at_cell(cell, wood)
            .with_microstructure(microstructure)
            .at_temperature(INITIAL_TEMPERATURE_K)
    };
    let mut config = SceneConfig::default();
    config.gravity = [0.0; 3];
    config.fixed_dt = DT;
    config.substeps = 1;
    config.iterations = 80;
    config.corotation_iterations = 2;
    let mut scene = Scene::from_voxels(
        config,
        vec![
            spawn([-1, 0, 0]),
            spawn([0, 0, 0]),
            spawn([1, 0, 0]),
            spawn([4, 0, 0]),
            spawn([4, 0, 1]),
            spawn([4, 0, 2]),
        ],
    )?;
    scene.set_material_thermal_expansion_anisotropic(
        "wood",
        ALPHA_PARALLEL,
        ALPHA_PERPENDICULAR,
    )?;
    scene.fix_voxels(&[0, 2, 3, 5])?;

    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAME_COUNT},thermal")?;
    writeln!(out, "N,0,transverse_left_fixed")?;
    writeln!(out, "N,1,transverse_center")?;
    writeln!(out, "N,2,transverse_right_fixed")?;
    writeln!(out, "N,3,parallel_left_fixed")?;
    writeln!(out, "N,4,parallel_center")?;
    writeln!(out, "N,5,parallel_right_fixed")?;
    write_frame(&mut out, &scene, 0, 0.0)?;

    let mut heater_j = 0.0f64;
    for step in 1..=STEPS {
        let target = INITIAL_TEMPERATURE_K + RISE_K * (step as f32 / RAMP_STEPS as f32).min(1.0);
        for i in 0..scene.voxels.len() {
            let energy = scene.properties[i].energy_at_temperature(scene.voxels[i].mass, target);
            let delta = energy - scene.voxels[i].thermal_energy_f32();
            if !scene.add_heat_joules(i, delta) {
                return Err(format!("heater rejected REV {i} energy increment").into());
            }
            heater_j += delta as f64;
        }
        scene.tick();
        if step % SAMPLE_EVERY == 0 {
            write_frame(&mut out, &scene, step, heater_j)?;
        }
    }

    let parallel_stress = scene.workspace[2][4].last_force.z.abs() / AREA_M2;
    let perpendicular_stress = scene.workspace[0][1].last_force.x.abs() / AREA_M2;
    let actual_parallel = (scene.positions[5].z - scene.positions[3].z).abs();
    let actual_perpendicular = (scene.positions[2].x - scene.positions[0].x).abs();
    let expected_parallel = 0.02 * (1.0 + ALPHA_PARALLEL * RISE_K);
    let expected_perpendicular = 0.02 * (1.0 + ALPHA_PERPENDICULAR * RISE_K);
    eprintln!(
        "thermal_expansion_anisotropy complete states={FRAME_COUNT} simulated_s={:.3} parallel_free={expected_parallel:.9} m transverse_free={expected_perpendicular:.9} m restrained=[{actual_parallel:.9},{actual_perpendicular:.9}] m stresses=[{parallel_stress:.3},{perpendicular_stress:.3}] Pa heater={heater_j:.9} J broken={} clusters={}",
        STEPS as f64 * DT as f64, scene.broken, scene.cluster_count,
    );
    if scene.broken != 0
        || (actual_parallel - 0.02).abs() > 1.0e-7
        || (actual_perpendicular - 0.02).abs() > 1.0e-7
        || !(parallel_stress > 1.9e6 && parallel_stress < 2.3e6)
        || !(perpendicular_stress > 1.1e6 && perpendicular_stress < 1.4e6)
    {
        return Err("anisotropic thermal-expansion capture failed validation".into());
    }
    Ok(())
}
