//! Physical capture of engine world-boundary removal and its conservation ledger.
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

const STEPS: usize = 1_760;
const DT: f32 = 0.01;
const SAMPLE_EVERY: usize = 10;
const FRAMES: usize = STEPS / SAMPLE_EVERY + 1;

#[derive(Clone, Copy)]
struct Probe {
    name: &'static str,
    start: Vec3,
    velocity: Vec3,
    temperature_k: f32,
}

const PROBES: [Probe; 8] = [
    Probe {
        name: "x_min",
        start: Vec3::new(-0.195, -0.095, -0.095),
        velocity: Vec3::new(-0.0032, 0.0, 0.0),
        temperature_k: 300.0,
    },
    Probe {
        name: "x_max",
        start: Vec3::new(0.195, -0.095, 0.095),
        velocity: Vec3::new(0.004, 0.0, 0.0),
        temperature_k: 360.0,
    },
    Probe {
        name: "y_min",
        start: Vec3::new(-0.095, -0.195, 0.095),
        velocity: Vec3::new(0.0, -0.005, 0.0),
        temperature_k: 420.0,
    },
    Probe {
        name: "y_max",
        start: Vec3::new(0.095, 0.195, -0.095),
        velocity: Vec3::new(0.0, 0.006, 0.0),
        temperature_k: 480.0,
    },
    Probe {
        name: "z_min",
        start: Vec3::new(0.095, -0.095, -0.195),
        velocity: Vec3::new(0.0, 0.0, -0.007),
        temperature_k: 540.0,
    },
    Probe {
        name: "z_max",
        start: Vec3::new(-0.095, 0.095, 0.195),
        velocity: Vec3::new(0.0, 0.0, 0.008),
        temperature_k: 600.0,
    },
    Probe {
        name: "x_max_bond_a",
        start: Vec3::new(0.215, 0.205, 0.205),
        velocity: Vec3::new(0.006, 0.0, 0.0),
        temperature_k: 300.0,
    },
    Probe {
        name: "x_max_bond_b",
        start: Vec3::new(0.225, 0.205, 0.205),
        velocity: Vec3::new(0.0060001, 0.0, 0.0),
        temperature_k: 300.0,
    },
];

fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, (position, voxel)) in scene.positions.iter().zip(&scene.voxels).enumerate() {
        let temperature = scene.temperature_kelvin(i).unwrap_or(f32::NAN);
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{temperature},0",
            position.x, position.y, position.z
        )?;
        // Extra diagnostics are ignored by the live 3D parser and retained in CSV.
        writeln!(
            out,
            "U,{step},{i},{},{},{},{}",
            voxel.velocity.x, voxel.velocity.y, voxel.velocity.z, voxel.mass
        )?;
    }
    let ledger = scene.boundary_removals();
    writeln!(
        out,
        "L,{step},{},{},{},{},{},{},{},{},{},{},{}",
        scene.voxels.len(),
        ledger.voxel_count,
        ledger.mass_kg,
        ledger.internal_energy_j,
        ledger.kinetic_energy_j,
        ledger.pending_bond_energy_j,
        ledger.momentum_kg_m_s[0],
        ledger.momentum_kg_m_s[1],
        ledger.momentum_kg_m_s[2],
        scene.cluster_count,
        ledger.elastic_bond_energy_j
    )?;
    out.flush()
}

fn total_energy_j(scene: &Scene) -> f64 {
    let voxel_energy: f64 = scene
        .voxels
        .iter()
        .map(|voxel| {
            voxel.total_internal_energy_j()
                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
        })
        .sum();
    voxel_energy + scene.elastic_bond_energy_j() + scene.energy.pending_release
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/world-bounds.csv".into());
    let iron = Material::component_id("iron").ok_or("iron is missing from material table")?;
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 2);
    for probe in PROBES {
        builder = builder.add_voxel(
            VoxelSpawn::new(probe.start, iron)
                .with_velocity(probe.velocity)
                .at_temperature(probe.temperature_k),
        );
    }
    let mut scene = builder.build()?;
    let initial_mass = scene.total_mass_kg();
    let initial_energy = total_energy_j(&scene);
    let initial_thermal_energy: f64 = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .sum();
    let initial_bonds: usize = scene
        .bonds
        .iter()
        .flatten()
        .filter(|bond| bond.neighbor != voxel_physics_lab::bond::EMPTY)
        .count();
    if initial_bonds != 1 {
        return Err(format!("expected one strained probe bond, found {initial_bonds}").into());
    }
    let removed_at_setup = scene.set_world_bounds(Vec3::splat(-0.25), Vec3::splat(0.25))?;
    if removed_at_setup != 0 {
        return Err("a probe starts outside configured bounds".into());
    }

    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},thermal")?;
    writeln!(out, "W,-0.25,-0.25,-0.25,0.25,0.25,0.25")?;
    for (i, probe) in PROBES.iter().enumerate() {
        writeln!(out, "N,{i},{}", probe.name)?;
    }
    let started = Instant::now();
    write_frame(&mut out, &scene, 0)?;
    let mut active_ids: Vec<usize> = (0..PROBES.len()).collect();
    let mut crossed_faces = [0usize; 6];
    for step in 1..=STEPS {
        let old_positions = scene.positions.clone();
        let old_velocities: Vec<Vec3> = scene.voxels.iter().map(|voxel| voxel.velocity).collect();
        let old_count = scene.voxels.len();
        scene.tick();
        if scene.voxels.len() < old_count {
            let mut kept_ids = Vec::with_capacity(scene.voxels.len());
            let bounds = 0.25f32;
            for (&id, (&position, &velocity)) in active_ids
                .iter()
                .zip(old_positions.iter().zip(&old_velocities))
            {
                let next = position + velocity * DT;
                let exit = if next.x < -bounds {
                    Some((0, "x_min"))
                } else if next.x > bounds {
                    Some((1, "x_max"))
                } else if next.y < -bounds {
                    Some((2, "y_min"))
                } else if next.y > bounds {
                    Some((3, "y_max"))
                } else if next.z < -bounds {
                    Some((4, "z_min"))
                } else if next.z > bounds {
                    Some((5, "z_max"))
                } else {
                    None
                };
                if let Some((face, name)) = exit {
                    crossed_faces[face] += 1;
                    writeln!(out, "E,{step},{id},{name}")?;
                } else {
                    kept_ids.push(id);
                }
            }
            active_ids = kept_ids;
        }
        if step % SAMPLE_EVERY == 0 {
            write_frame(&mut out, &scene, step)?;
        }
        if step % 400 == 0 {
            let ledger = scene.boundary_removals();
            eprintln!("world_bounds step={step}/{STEPS} remaining={} removed={} mass={:.6e} kg wall={:.2}s",
                scene.voxels.len(), ledger.voxel_count, ledger.mass_kg, started.elapsed().as_secs_f64());
        }
    }

    let ledger = scene.boundary_removals();
    let final_mass = scene.total_mass_kg();
    let final_energy = total_energy_j(&scene);
    let mass_error = (initial_mass - final_mass - ledger.mass_kg).abs();
    let energy_error = (initial_energy
        - final_energy
        - ledger.internal_energy_j
        - ledger.kinetic_energy_j
        - ledger.elastic_bond_energy_j
        - ledger.pending_bond_energy_j)
        .abs();
    let energy_tolerance = (initial_energy - initial_thermal_energy).abs().max(1e-9) * 1e-3;
    eprintln!("world_bounds complete probes={} frames={FRAMES} steps={STEPS} simulated_s={:.3} remaining={} removed={} ledger_mass={:.9e} kg ledger_internal_energy={:.9e} J ledger_kinetic_energy={:.9e} J elastic_bond_energy={:.9e} J pending_bond_energy={:.9e} J ledger_momentum=[{:.9e},{:.9e},{:.9e}] kg m/s mass_balance_error={:.3e} kg total_energy_balance_error={:.3e} J energy_tolerance={:.3e} J wall={:.2}s",
        PROBES.len(), STEPS as f64 * DT as f64, scene.voxels.len(), ledger.voxel_count,
        ledger.mass_kg, ledger.internal_energy_j, ledger.kinetic_energy_j,
        ledger.elastic_bond_energy_j, ledger.pending_bond_energy_j, ledger.momentum_kg_m_s[0],
        ledger.momentum_kg_m_s[1], ledger.momentum_kg_m_s[2], mass_error,
        energy_error, energy_tolerance, started.elapsed().as_secs_f64());
    if ledger.voxel_count != PROBES.len() as u64
        || scene.voxels.len() != 0
        || crossed_faces != [1, 3, 1, 1, 1, 1]
        || mass_error > initial_mass * 1e-6
        || energy_error > energy_tolerance
        || ledger.kinetic_energy_j <= 0.0
        || ledger.elastic_bond_energy_j <= 0.0
    {
        return Err("world-boundary removal or mass/energy ledger validation failed".into());
    }
    Ok(())
}
