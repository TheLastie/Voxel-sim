//! Physical capture of restrained thermal expansion through the public scene API.
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
};

const STEPS: usize = 1_760;
const DT: f32 = 0.01;
const SAMPLE_EVERY: usize = 10;
const FRAME_COUNT: usize = STEPS / SAMPLE_EVERY + 1;
const INITIAL_TEMPERATURE_K: f32 = 293.15;
const RISE_K: f32 = 70.0;
const RAMP_STEPS: usize = 60;
const ALPHA_PER_K: f32 = 12.0e-6;

fn write_state(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    heater_energy_j: f64,
    initial_span_m: f32,
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
    let target_t = INITIAL_TEMPERATURE_K + RISE_K * (step as f32 / RAMP_STEPS as f32).min(1.0);
    let free_span = initial_span_m * (1.0 + ALPHA_PER_K * (target_t - INITIAL_TEMPERATURE_K));
    let actual_span = (scene.positions[2] - scene.positions[0]).length();
    let force_n = scene.workspace[0][1].last_force.x.abs() as f64;
    let stress_pa = force_n / 1.0e-4;
    writeln!(
        out,
        "X,{step},{target_t},{},{},{},{heater_energy_j},{free_span},{actual_span},{stress_pa},{},{}",
        scene.temperature_kelvin(0).unwrap_or(f32::NAN),
        scene.temperature_kelvin(1).unwrap_or(f32::NAN),
        scene.temperature_kelvin(2).unwrap_or(f32::NAN),
        scene.cluster_count,
        force_n,
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/thermal-expansion.csv".into());
    let iron = Material::component_id("iron").ok_or("iron is missing from material table")?;
    let mut config = SceneConfig::default();
    config.gravity = [0.0; 3];
    config.fixed_dt = DT;
    config.substeps = 1;
    config.iterations = 80;
    config.corotation_iterations = 2;
    let mut scene = Scene::from_voxels(
        config,
        (0..3)
            .map(|x| VoxelSpawn::at_cell([x, 0, 0], iron).at_temperature(INITIAL_TEMPERATURE_K))
            .collect(),
    )?;
    scene.set_material_thermal_expansion("iron", ALPHA_PER_K)?;
    scene.fix_voxels(&[0, 2])?;
    let initial_span_m = (scene.positions[2] - scene.positions[0]).length();
    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAME_COUNT},thermal")?;
    writeln!(out, "N,0,fixed_left")?;
    writeln!(out, "N,1,heated_middle")?;
    writeln!(out, "N,2,fixed_right")?;
    write_state(&mut out, &scene, 0, 0.0, initial_span_m)?;

    let mut heater_energy_j = 0.0f64;
    for step in 1..=STEPS {
        let ramp = (step as f32 / RAMP_STEPS as f32).min(1.0);
        let target_t = INITIAL_TEMPERATURE_K + RISE_K * ramp;
        for i in 0..scene.voxels.len() {
            let target_energy =
                scene.properties[i].energy_at_temperature(scene.voxels[i].mass, target_t);
            let delta = target_energy - scene.voxels[i].thermal_energy_f32();
            if !scene.add_heat_joules(i, delta) {
                return Err(format!("heater rejected REV {i} energy increment").into());
            }
            heater_energy_j += delta as f64;
        }
        scene.tick();
        if step % SAMPLE_EVERY == 0 {
            write_state(&mut out, &scene, step, heater_energy_j, initial_span_m)?;
        }
        if step % 400 == 0 {
            eprintln!(
                "thermal_expansion step={step}/{STEPS} T={:.2} K stress={:.3} MPa heater={heater_energy_j:.6} J",
                scene.temperature_kelvin(1).unwrap_or(f32::NAN),
                scene.workspace[0][1].last_force.x.abs() / 1.0e-4 / 1.0e6,
            );
        }
    }

    let actual_span_m = (scene.positions[2] - scene.positions[0]).length();
    let expected_free_span_m = initial_span_m * (1.0 + ALPHA_PER_K * RISE_K);
    let stress_pa = scene.workspace[0][1].last_force.x.abs() / 1.0e-4;
    eprintln!(
        "thermal_expansion complete steps={STEPS} states={FRAME_COUNT} simulated_s={:.3} alpha={ALPHA_PER_K:.3e} 1/K delta_T={RISE_K:.2} K free_span={expected_free_span_m:.9} m restrained_span={actual_span_m:.9} m reaction_stress={:.6} MPa heater={heater_energy_j:.9} J broken={} clusters={} ",
        STEPS as f64 * DT as f64,
        stress_pa / 1.0e6,
        scene.broken,
        scene.cluster_count,
    );
    if scene.voxels.len() != 3
        || (actual_span_m - initial_span_m).abs() > 1.0e-7
        || !(stress_pa > 1.0e6 && stress_pa < 200.0e6)
        || scene.broken != 0
        || !heater_energy_j.is_finite()
    {
        return Err("restrained thermal-expansion validation failed".into());
    }
    Ok(())
}
