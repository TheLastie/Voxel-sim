//! Quasi-static voltage ramp across a glass REV capacitor network.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    bond::EMPTY,
    electrical::{self, VoltageBoundary},
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

const DT: f64 = 0.01;
const STEPS: usize = 1_760;
const FRAMES: usize = 177;
const MAX_VOLTAGE_V: f64 = 1.0;

fn write_frame(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    voltage_v: f64,
    charge_c: f64,
    capacitance_f: f64,
    stored_energy_j: f64,
    source_work_j: f64,
) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, position) in scene.positions.iter().enumerate() {
        writeln!(
            out,
            "V,{step},{i},{},{},{},0,0",
            position.x, position.y, position.z
        )?;
        let phase = scene.properties[i]
            .phase_state(scene.voxels[i].mass, scene.voxels[i].thermal_energy_f32());
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{},{}",
            position.x, position.y, position.z, phase.temperature_kelvin, phase.liquid_fraction
        )?;
        for axis in 0..3 {
            let neighbor = scene.bonds[axis][i].neighbor;
            if neighbor != EMPTY {
                writeln!(out, "B,{step},{i},{neighbor}")?;
            }
        }
    }
    writeln!(
        out,
        "R,{step},{voltage_v},{charge_c},{capacitance_f},{stored_energy_j},{source_work_j}"
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/dielectric-capacitance.csv".into());
    let glass = Material::component_id("glass").ok_or("glass missing from material table")?;
    let scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], glass))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], glass))
        .add_voxel(VoxelSpawn::at_cell([2, 0, 0], glass))
        .build()?;
    if scene.bonds[0][0].neighbor != 1 || scene.bonds[0][1].neighbor != 2 {
        return Err("capture fixture did not create a three-REV dielectric".into());
    }
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},electrical")?;
    writeln!(
        out,
        "N,glass_capacitor,voltage_V,charge_C,capacitance_F,stored_energy_J,source_work_J"
    )?;
    let mut previous_voltage = 0.0;
    let mut previous_charge = 0.0;
    let mut source_work_j = 0.0;
    write_frame(&mut out, &scene, 0, 0.0, 0.0, 0.0, 0.0, 0.0)?;
    for step in 1..=STEPS {
        let voltage_v = MAX_VOLTAGE_V * step as f64 / STEPS as f64;
        let state = electrical::solve_dielectric_state(
            &scene,
            &[
                VoltageBoundary {
                    voxel: 0,
                    voltage_v: 0.0,
                },
                VoltageBoundary {
                    voxel: 2,
                    voltage_v,
                },
            ],
        )?;
        let charge_c = state.boundary_charge_c[1];
        let source_increment = 0.5 * (previous_voltage + voltage_v) * (charge_c - previous_charge);
        source_work_j += source_increment;
        previous_voltage = voltage_v;
        previous_charge = charge_c;
        let capacitance_f = charge_c / voltage_v;
        let sample = step == STEPS || step % (STEPS / (FRAMES - 1)) == 0;
        if sample {
            write_frame(
                &mut out,
                &scene,
                step,
                voltage_v,
                charge_c,
                capacitance_f,
                state.stored_energy_j,
                source_work_j,
            )?;
        }
        if step % 400 == 0 {
            eprintln!("dielectric step={step}/{STEPS} t={:.2}s V={voltage_v:.3} charge={charge_c:.6e}C C={capacitance_f:.6e}F U={:.6e}J", step as f64 * DT, state.stored_energy_j);
        }
    }
    let energy_error = source_work_j - 0.5 * MAX_VOLTAGE_V * previous_charge;
    eprintln!("dielectric complete steps={STEPS} states={FRAMES} physical_s={:.2} voltage={MAX_VOLTAGE_V:.3}V charge={previous_charge:.9e}C capacitance={:.9e}F stored_energy={:.9e}J source_work={source_work_j:.9e}J energy_error={energy_error:.3e}J", STEPS as f64 * DT, previous_charge / MAX_VOLTAGE_V, 0.5 * MAX_VOLTAGE_V * previous_charge);
    if !energy_error.is_finite() || energy_error.abs() > 1e-24 || previous_charge <= 0.0 {
        return Err("dielectric charge/energy verification failed".into());
    }
    Ok(())
}
