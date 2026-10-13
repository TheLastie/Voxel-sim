//! Physical solver scaling capture using the public arbitrary-voxel scene API.
//! A gravity-loaded iron block is held at its bottom face; no scripted motion is
//! applied during the run. The CSV records the real outer surface at each state.

use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
};

const NX: usize = 36;
const NY: usize = 32;
const NZ: usize = 32;
const DT: f32 = 0.001;

fn surface_state(scene: &Scene, x: usize, y: usize, z: usize) -> (f32, usize) {
    let index = (z * NY + y) * NX + x;
    let strides = [1, NX, NX * NY];
    let coordinates = [x, y, z];
    let extents = [NX, NY, NZ];
    let mut max_damage = 0.0f32;
    let mut lost = 0;
    for axis in 0..3 {
        if coordinates[axis] + 1 < extents[axis] {
            let bond = scene.bonds[axis][index];
            if bond.neighbor == EMPTY {
                lost += 1;
            } else {
                max_damage = max_damage.max(bond.damage);
            }
        }
        if coordinates[axis] > 0 {
            let bond = scene.bonds[axis][index - strides[axis]];
            if bond.neighbor == EMPTY {
                lost += 1;
            } else {
                max_damage = max_damage.max(bond.damage);
            }
        }
    }
    (max_damage, lost)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/solver-scale-36k.csv".into());
    let steps: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "176".into())
        .parse()?;
    let saved_frames: usize = std::env::args()
        .nth(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(steps + 1);
    if steps == 0 || saved_frames < 2 || saved_frames > steps + 1 {
        return Err("steps must be positive and saved_frames must be in 2..=steps+1".into());
    }

    let iron = Material::component_id("iron").ok_or("iron material is unavailable")?;
    let mut config = SceneConfig::default();
    config.dimensions = [NX, NY, NZ];
    config.fixed_dt = DT;
    // Keep the standard four substeps and 80-iteration IC(0) solver settings.
    let mut spawns = Vec::with_capacity(NX * NY * NZ);
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let mut voxel = VoxelSpawn::at_cell([x as i32, y as i32, z as i32], iron)
                    .at_temperature(293.15);
                if z == 0 {
                    voxel = voxel.fixed();
                }
                spawns.push(voxel);
            }
        }
    }
    let mut scene = Scene::from_voxels(config, spawns)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    let active = scene.voxels.iter().filter(|voxel| !voxel.kinematic).count();
    let surface_count = NX * NY * NZ - (NX - 2) * (NY - 2) * (NZ - 2);
    // The browser's current mechanical palette reserves IDs >= 4320 for a
    // separate coupon. Keep these sample IDs below that range and carry the
    // real scene index as an additional CSV field for offline analysis.
    let mut surface = Vec::with_capacity(surface_count * 2 / 3);
    for z in 0..NZ {
        for y in 0..NY {
            for x in 0..NX {
                let index = (z * NY + y) * NX + x;
                let boundary =
                    x == 0 || x + 1 == NX || y == 0 || y + 1 == NY || z == 0 || z + 1 == NZ;
                if boundary && index % 3 != 0 {
                    surface.push((surface.len(), index, x, y, z));
                }
            }
        }
    }
    eprintln!(
        "solver scale capture: {} REV, {active} active, {surface_count} surface REV ({} sampled), dt={DT} s, substeps={}, steps={steps}, states={saved_frames}, threads={} cutoff={}",
        scene.voxels.len(),
        surface.len(),
        scene.config.substeps,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into()),
        std::env::var("VOXEL_PARALLEL_MIN_VOXELS").unwrap_or_else(|_| "default".into()),
    );

    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    writeln!(out, "H,{DT},{steps},{saved_frames}")?;
    let clock = std::time::Instant::now();
    let intervals = saved_frames - 1;
    for frame in 0..=steps {
        if frame > 0 {
            scene.tick();
        }
        if frame % 10 == 0 || frame == steps {
            eprintln!(
                "frame={frame}/{steps} wall={:.2}s broken={} clusters={} residual={:.3e}",
                clock.elapsed().as_secs_f64(),
                scene.broken,
                scene.cluster_count,
                scene.solver_residual,
            );
        }
        let save = frame == 0
            || frame == steps
            || (frame as u128 * intervals as u128 / steps as u128)
                > ((frame - 1) as u128 * intervals as u128 / steps as u128);
        if !save {
            continue;
        }
        writeln!(
            out,
            "F,{frame},0,0,0,0,0,0,1,0,0,{},{},0",
            scene.broken, scene.cluster_count
        )?;
        for &(sample_id, index, x, y, z) in &surface {
            let position = scene.positions[index];
            let (damage, lost) = surface_state(&scene, x, y, z);
            writeln!(
                out,
                "V,{frame},{sample_id},{},{},{},{damage},{lost},{index}",
                position.x, position.y, position.z
            )?;
        }
        out.flush()?;
    }
    eprintln!("complete wall={:.2}s", clock.elapsed().as_secs_f64());
    Ok(())
}
