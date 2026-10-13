//! Physical scene capture showing a fully molten REV losing its solid bonds.
use std::{
    fs::File,
    io::{BufWriter, Write},
};

use glam::Vec3;
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

const SIDE: usize = 2;
const STEPS: usize = 1_680;
const FRAMES: usize = 176;
const MELT_TEMPERATURE_K: f32 = 2_200.0;

fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, position) in scene.positions.iter().enumerate() {
        let temperature = scene.temperature_kelvin(i).unwrap_or(f32::NAN);
        let liquid = scene
            .phase_state(i)
            .map_or(0.0, |phase| phase.liquid_fraction);
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{temperature},{liquid}",
            position.x, position.y, position.z
        )?;
    }
    for axis in 0..3 {
        for (i, bond) in scene.bonds[axis].iter().enumerate() {
            if bond.neighbor != EMPTY {
                writeln!(out, "B,{step},{i},{}", bond.neighbor)?;
            }
        }
    }
    out.flush()
}

fn count_bonds(scene: &Scene) -> usize {
    scene
        .bonds
        .iter()
        .flatten()
        .filter(|bond| bond.neighbor != EMPTY)
        .count()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/melting.csv".into());
    let iron = Material::component_id("iron").ok_or("iron is missing from the material table")?;
    let mut builder = SceneBuilder::standard().with_gravity(Vec3::ZERO);
    for z in 0..SIDE {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let temperature = if [x, y, z] == [0, 0, 0] {
                    MELT_TEMPERATURE_K
                } else {
                    300.0
                };
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([x as i32, y as i32, z as i32], iron)
                        .at_temperature(temperature),
                );
            }
        }
    }

    let mut scene = builder.build()?;
    let initial_bonds = count_bonds(&scene);
    let molten = scene
        .phase_state(0)
        .ok_or("missing initial phase state")?
        .liquid_fraction;
    if initial_bonds != 12 || molten < 1.0 {
        return Err(
            format!("invalid melt scene: bonds={initial_bonds}, liquid_fraction={molten}").into(),
        );
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,{},{STEPS},{FRAMES},thermal", scene.config.fixed_dt)?;
    write_frame(&mut out, &scene, 0)?;
    let intervals = FRAMES - 1;
    for step in 1..=STEPS {
        scene.tick();
        let sample = step == STEPS
            || (step as u128 * intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * intervals as u128 / STEPS as u128);
        if sample {
            write_frame(&mut out, &scene, step)?;
        }
    }

    let final_bonds = count_bonds(&scene);
    let final_liquid = scene.phase_state(0).unwrap().liquid_fraction;
    eprintln!(
        "melting complete voxels={} initial_bonds={initial_bonds} final_bonds={final_bonds} broken={} clusters={} hot_liquid={molten:.3}->{final_liquid:.3} simulated_s={:.3} frames={FRAMES}",
        scene.voxels.len(), scene.broken, scene.cluster_count,
        scene.config.fixed_dt as f64 * STEPS as f64,
    );
    if final_bonds >= initial_bonds || scene.broken < 3 || final_liquid < 1.0 {
        return Err("fully molten REV did not release its solid bonds".into());
    }
    Ok(())
}
