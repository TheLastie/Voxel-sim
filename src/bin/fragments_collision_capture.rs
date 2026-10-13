//! Physical fatigue-fracture and fragment-collision capture using engine APIs.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, CELL},
    solver,
};

const STEPS: usize = 17_600;
const SAMPLE_EVERY: usize = 100;
const FRAMES: usize = STEPS / SAMPLE_EVERY + 1;
const DT: f32 = 0.001;
const CYCLE_STEPS: usize = 10;

fn momentum(scene: &Scene) -> Vec3 {
    scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |sum, voxel| sum + voxel.velocity * voxel.mass)
}

fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, position) in scene.positions.iter().enumerate() {
        let (damage, lost) = scene.incident_bond_damage()[i];
        let temperature = scene.temperature_kelvin(i).unwrap_or(293.15);
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{temperature},{lost}",
            position.x, position.y, position.z
        )?;
        writeln!(
            out,
            "V,{step},{i},{},{},{},{damage},{lost}",
            position.x, position.y, position.z
        )?;
    }
    for bonds in &scene.bonds {
        for (i, bond) in bonds.iter().enumerate() {
            if bond.neighbor != EMPTY {
                writeln!(out, "B,{step},{i},{}", bond.neighbor)?;
            }
        }
    }
    let internal: f64 = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .sum();
    let kinetic: f64 = scene
        .voxels
        .iter()
        .map(|voxel| 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64)
        .sum();
    let p = momentum(scene);
    writeln!(
        out,
        "L,{step},{},{},{},{internal},{kinetic},0,{},{},{}",
        scene.voxels.len(),
        scene.cluster_count,
        scene.total_mass_kg(),
        p.x,
        p.y,
        p.z
    )?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/fracture-fragments-collision.csv".into());
    let iron = Material::component_id("iron").ok_or("iron missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 1)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).at_temperature(293.15))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron).at_temperature(293.15))
        .add_voxel(VoxelSpawn::at_cell([2, 0, 0], iron).at_temperature(293.15))
        .build()?;
    scene.config.fracture.fatigue_cycles = 30.0;
    scene.config.fracture.fatigue_exponent = 1.0;
    scene.config.fracture.fatigue_threshold = 0.1;
    scene.config.fracture.restitution = 0.25;
    scene.config.fracture.friction = 0.0;
    if scene.cluster_count != 1
        || scene.bonds[0][0].neighbor != 1
        || scene.bonds[0][1].neighbor != 2
    {
        return Err("three-REV fatigue specimen did not form its expected bonds".into());
    }

    let initial_mass = scene.total_mass_kg();
    let strength = scene.properties[0].tensile_strength;
    let young = scene.properties[0].young;
    let amplitude = 0.8 * strength / young * CELL;
    let fixed_base = scene.positions[0];
    let driven_end = scene.positions[2];
    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},{FRAMES},thermal")?;
    writeln!(out, "N,0,fatigue fragment A")?;
    writeln!(out, "N,1,fatigue fragment B")?;
    writeln!(out, "N,2,fatigue fragment C")?;
    write_frame(&mut out, &scene, 0)?;

    let started = Instant::now();
    let mut tool = None;
    let mut fracture_step = None;
    let mut collision_step = None;
    let mut max_stress_ratio = 0.0f32;
    let mut impulse_momentum = None;
    let mut energy_before_collision = None;
    let intervals = FRAMES - 1;
    for step in 1..=STEPS {
        if fracture_step.is_none() {
            let phase = std::f32::consts::TAU * (step % CYCLE_STEPS) as f32 / CYCLE_STEPS as f32;
            let targets = [
                fixed_base.to_array().map(Some),
                [None; 3],
                [
                    Some(driven_end.x + amplitude * phase.sin()),
                    Some(driven_end.y),
                    Some(driven_end.z),
                ],
            ];
            solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
            for i in 0..2 {
                let rest = scene.rest_positions[i + 1].x - scene.rest_positions[i].x;
                let strain = (scene.positions[i + 1].x - scene.positions[i].x - rest) / CELL;
                max_stress_ratio = max_stress_ratio.max((strain * young / strength).abs());
            }
            if scene.bonds[0][1].neighbor == EMPTY {
                fracture_step = Some(step);
                if scene.cluster_count != 3 || scene.bonds[0][0].neighbor != EMPTY {
                    return Err(
                        "fatigue event did not split the specimen into three fragments".into(),
                    );
                }
                writeln!(out, "E,{step},2,fatigue_fracture_three_clusters")?;
                // A controlled impulse starts the recorded fragment impact; all
                // fracture, topology, motion and contact response stay in engine APIs.
                let impulse = scene.voxels[2].mass * 0.02;
                solver::apply_impulse(&mut scene, 2, Vec3::NEG_X * impulse)?;
                impulse_momentum = Some(momentum(&scene));
                energy_before_collision = Some(
                    scene
                        .voxels
                        .iter()
                        .map(|voxel| {
                            voxel.total_internal_energy_j()
                                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
                        })
                        .sum::<f64>(),
                );
                writeln!(out, "E,{step},2,controlled_impulse_toward_fragment_B")?;
            }
        } else {
            solver::step(&mut scene, &mut tool);
            if collision_step.is_none() && scene.energy.contact_restitution_heat > 0.0 {
                collision_step = Some(step);
                writeln!(
                    out,
                    "E,{step},2,fragment_collision_heat_{:.9e}_J",
                    scene.energy.contact_restitution_heat
                )?;
            }
        }

        let sample = step == STEPS
            || (step as u128 * intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * intervals as u128 / STEPS as u128);
        if sample {
            write_frame(&mut out, &scene, step)?;
        }
        if step % 4_000 == 0 {
            eprintln!(
                "fragment_capture step={step}/{STEPS} clusters={} broken={} collision={collision_step:?} wall={:.2}s",
                scene.cluster_count,
                scene.broken,
                started.elapsed().as_secs_f64()
            );
        }
    }

    let final_momentum = momentum(&scene);
    let momentum_error = impulse_momentum
        .map(|before| (final_momentum - before).length() as f64)
        .unwrap_or(f64::INFINITY);
    let final_energy: f64 = scene
        .voxels
        .iter()
        .map(|voxel| {
            voxel.total_internal_energy_j()
                + 0.5 * voxel.mass as f64 * voxel.velocity.length_squared() as f64
        })
        .sum();
    let energy_error = energy_before_collision
        .map(|before| (final_energy - before).abs())
        .unwrap_or(f64::INFINITY);
    eprintln!(
        "fragment_capture complete steps={STEPS} states={FRAMES} physical_s={:.3} fracture_step={fracture_step:?} collision_step={collision_step:?} max_stress_ratio={max_stress_ratio:.4} clusters={} mass={:.9e} kg momentum_error={momentum_error:.3e} kg*m/s post_impulse_energy_error={energy_error:.3e} J restitution_heat={:.9e} J wall={:.2}s",
        STEPS as f64 * DT as f64,
        scene.cluster_count,
        scene.total_mass_kg(),
        scene.energy.contact_restitution_heat,
        started.elapsed().as_secs_f64()
    );
    if fracture_step.is_none()
        || collision_step.is_none()
        || scene.cluster_count != 3
        || (scene.total_mass_kg() - initial_mass).abs() > initial_mass * 1e-6
        || momentum_error > 1e-5
        || energy_error > 1e-6
        || max_stress_ratio >= 1.0
    {
        return Err("fragment collision capture failed its topology or conservation checks".into());
    }
    Ok(())
}
