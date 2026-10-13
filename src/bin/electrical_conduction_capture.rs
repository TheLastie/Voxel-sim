//! Constant-voltage conduction and Joule heating in a three-REV copper wire.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    electrical::{self, VoltageBoundary},
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

const DT: f64 = 0.01;
const STEPS: usize = 1_760;
const FRAMES: usize = 177;
const VOLTAGE_V: f64 = 1.0e-3;

fn write_frame(
    out: &mut BufWriter<File>,
    scene: &Scene,
    step: usize,
    current_a: f64,
    resistance_ohm: f64,
    power_w: f64,
    source_energy_j: f64,
    heat_j: f64,
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
            if neighbor != voxel_physics_lab::bond::EMPTY {
                writeln!(out, "B,{step},{i},{neighbor}")?;
            }
        }
    }
    writeln!(
        out,
        "R,{step},{current_a},{resistance_ohm},{power_w},{source_energy_j},{heat_j}"
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/electrical-conduction.csv".into());
    let copper = Material::component_id("copper").ok_or("copper missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], copper))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper))
        .add_voxel(VoxelSpawn::at_cell([2, 0, 0], copper))
        .build()?;
    if scene.bonds[0][0].neighbor != 1 || scene.bonds[0][1].neighbor != 2 {
        return Err("capture fixture did not create a three-REV connected wire".into());
    }
    let boundaries = [
        VoltageBoundary {
            voxel: 0,
            voltage_v: 0.0,
        },
        VoltageBoundary {
            voxel: 2,
            voltage_v: VOLTAGE_V,
        },
    ];
    let initial_internal_energy: f64 = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .sum();
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},thermal")?;
    writeln!(
        out,
        "N,copper_wire,current_A,resistance_ohm,power_W,source_energy_J,joule_heat_J"
    )?;
    let mut accumulated_source_j = 0.0;
    let mut accumulated_heat_j = 0.0;
    let mut latest_current_a = 0.0;
    let mut latest_resistance_ohm = 0.0;
    let mut latest_power_w = 0.0;
    write_frame(&mut out, &scene, 0, 0.0, 0.0, 0.0, 0.0, 0.0)?;
    for step in 1..=STEPS {
        let result = electrical::solve_and_heat(&mut scene, &boundaries, DT)?;
        let terminal_current = result.boundary_currents_a[1];
        latest_current_a = terminal_current;
        latest_resistance_ohm = VOLTAGE_V / terminal_current;
        latest_power_w = result.joule_heat_j / DT;
        accumulated_source_j += result.source_energy_j;
        accumulated_heat_j += result.joule_heat_j;
        let sample = step == STEPS || step % (STEPS / (FRAMES - 1)) == 0;
        if sample {
            write_frame(
                &mut out,
                &scene,
                step,
                latest_current_a,
                latest_resistance_ohm,
                latest_power_w,
                accumulated_source_j,
                accumulated_heat_j,
            )?;
        }
        if step % 400 == 0 {
            eprintln!("electrical step={step}/{STEPS} t={:.2}s V={VOLTAGE_V:.6}I={latest_current_a:.6}A R={latest_resistance_ohm:.6e}Ohm P={latest_power_w:.6e}W heat={accumulated_heat_j:.6}J", step as f64 * DT);
        }
    }
    let final_internal_energy: f64 = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .sum();
    let energy_error = final_internal_energy - initial_internal_energy - accumulated_heat_j;
    let relative_budget_error =
        (accumulated_source_j - accumulated_heat_j).abs() / accumulated_heat_j.max(1e-30);
    eprintln!("electrical complete steps={STEPS} states={FRAMES} physical_s={:.2} current={latest_current_a:.6}A resistance={latest_resistance_ohm:.6e}Ohm power={latest_power_w:.6e}W source={accumulated_source_j:.9}J heat={accumulated_heat_j:.9}J thermal_error={energy_error:.3e}J source_heat_relative_error={relative_budget_error:.3e}", STEPS as f64 * DT);
    if energy_error.abs() > 1e-9 || relative_budget_error > 1e-9 || latest_current_a <= 0.0 {
        return Err("electrical energy/current verification failed".into());
    }
    Ok(())
}
