//! Real engine thermal transport capture on a hot/cold iron rod.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    material::Material,
    scene::{SceneBuilder, VoxelSpawn},
};

const VOXELS: usize = 24;
const HOT_VOXELS: usize = VOXELS / 2;
const STEPS: usize = 1_680;
const FRAMES: usize = 176;
const DT: f32 = 1.0;

fn write_frame(
    out: &mut BufWriter<File>,
    scene: &voxel_physics_lab::scene::Scene,
    step: usize,
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, position) in scene.positions.iter().enumerate() {
        let temperature = scene.temperature_kelvin(i).unwrap_or(f32::NAN);
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{temperature}",
            position.x, position.y, position.z
        )?;
    }
    out.flush()
}

fn energy_j(scene: &voxel_physics_lab::scene::Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|voxel| voxel.internal_energy as f64)
        .sum()
}

fn mean_temperature(scene: &voxel_physics_lab::scene::Scene, range: std::ops::Range<usize>) -> f64 {
    let count = range.end - range.start;
    range
        .map(|i| scene.temperature_kelvin(i).unwrap_or(f32::NAN) as f64)
        .sum::<f64>()
        / count as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/thermal-gradient.csv".into());
    let iron = Material::component_id("iron").ok_or("iron is missing from the material table")?;
    let mut builder = SceneBuilder::standard().with_gravity(Vec3::ZERO);
    for i in 0..VOXELS {
        let temperature = if i < HOT_VOXELS { 450.0 } else { 300.0 };
        builder = builder
            .add_voxel(VoxelSpawn::at_cell([i as i32, 0, 0], iron).at_temperature(temperature));
    }
    let mut scene = builder.build()?;
    let initial_energy = energy_j(&scene);
    let initial_hot = mean_temperature(&scene, 0..HOT_VOXELS);
    let initial_cold = mean_temperature(&scene, HOT_VOXELS..VOXELS);
    let initial_min = (0..VOXELS)
        .map(|i| scene.temperature_kelvin(i).unwrap())
        .fold(f32::INFINITY, f32::min);
    let initial_max = (0..VOXELS)
        .map(|i| scene.temperature_kelvin(i).unwrap())
        .fold(f32::NEG_INFINITY, f32::max);
    let bonds = scene.bonds[0]
        .iter()
        .filter(|bond| bond.neighbor != voxel_physics_lab::bond::EMPTY)
        .count();
    if bonds != VOXELS - 1 {
        return Err(format!(
            "expected {} connected thermal edges, found {bonds}",
            VOXELS - 1
        )
        .into());
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},thermal")?;
    let started = Instant::now();
    let mut max_substeps = 0;
    let mut max_energy_error_j = 0.0f64;
    let mut previous_min = initial_min;
    let mut previous_max = initial_max;
    write_frame(&mut out, &scene, 0)?;
    let intervals = FRAMES - 1;
    for step in 1..=STEPS {
        let thermal_substeps = scene.conduct_heat(DT)?;
        max_substeps = max_substeps.max(thermal_substeps);
        let energy_error = (energy_j(&scene) - initial_energy).abs();
        max_energy_error_j = max_energy_error_j.max(energy_error);
        let minimum = (0..VOXELS)
            .map(|i| scene.temperature_kelvin(i).unwrap())
            .fold(f32::INFINITY, f32::min);
        let maximum = (0..VOXELS)
            .map(|i| scene.temperature_kelvin(i).unwrap())
            .fold(f32::NEG_INFINITY, f32::max);
        if minimum + 1e-3 < previous_min || maximum - 1e-3 > previous_max {
            return Err(format!("temperature range expanded at step {step}: {previous_min:.3}..{previous_max:.3} K -> {minimum:.3}..{maximum:.3} K").into());
        }
        previous_min = minimum;
        previous_max = maximum;
        let is_sample = step == STEPS
            || (step as u128 * intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * intervals as u128 / STEPS as u128);
        if is_sample {
            write_frame(&mut out, &scene, step)?;
        }
        if step % 200 == 0 {
            eprintln!("thermal step={step}/{STEPS} wall={:.2}s substeps={thermal_substeps} range={minimum:.2}..{maximum:.2} K energy_error={energy_error:.3e} J", started.elapsed().as_secs_f64());
        }
    }
    let final_hot = mean_temperature(&scene, 0..HOT_VOXELS);
    let final_cold = mean_temperature(&scene, HOT_VOXELS..VOXELS);
    let final_energy = energy_j(&scene);
    let relative_error = (final_energy - initial_energy).abs() / initial_energy.abs().max(1e-30);
    eprintln!("thermal complete voxels={VOXELS} bonds={bonds} simulated_s={} frames={FRAMES} max_substeps={max_substeps} hot_mean={initial_hot:.3}->{final_hot:.3} K cold_mean={initial_cold:.3}->{final_cold:.3} K initial_energy={initial_energy:.9} J final_energy={final_energy:.9} J max_energy_error={max_energy_error_j:.3e} J relative_error={relative_error:.3e} wall={:.2}s", STEPS as f64 * DT as f64, started.elapsed().as_secs_f64());
    if !(final_hot < initial_hot && final_cold > initial_cold && relative_error < 2e-5) {
        return Err(
            "thermal transport did not show expected equilibration and energy conservation".into(),
        );
    }
    Ok(())
}
