//! Displacement-controlled cyclic endurance capture using the engine API.
//! The prescribed end displacement is the loading fixture; bond fatigue and
//! failure remain entirely in the shared constitutive update.
use glam::Vec3;
use std::{
    f32::consts::TAU,
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, CELL},
    solver,
};

const STEPS: usize = 1_760;
const FRAMES: usize = 177;
const STEPS_PER_CYCLE: usize = 10;
const STRAIN_RATIO: f32 = 0.4;

fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        let damage = scene.bonds[0].get(i).map_or(0.0, |bond| bond.damage);
        let fatigue = scene.bonds[0]
            .get(i)
            .map_or(0.0, |bond| bond.fatigue_damage);
        writeln!(
            out,
            "V,{step},{i},{},{},{},{damage},{fatigue}",
            p.x, p.y, p.z
        )?;
    }
    for (i, bond) in scene.bonds[0].iter().enumerate() {
        if bond.neighbor != EMPTY {
            writeln!(out, "B,{step},{i},{}", bond.neighbor)?;
        }
    }
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/fatigue-cycles.csv".into());
    let iron = Material::component_id("iron").ok_or("iron missing from material table")?;
    let builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 4)
        .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron));
    let mut scene = builder.build()?;
    scene.config.fracture.fatigue_cycles = 30.0;
    scene.config.fracture.fatigue_exponent = 1.0;
    scene.config.fracture.fatigue_threshold = 0.1;
    let strength = scene.properties[0].tensile_strength;
    let young = scene.properties[0].young;
    let displacement = STRAIN_RATIO * strength / young * CELL;
    let base = scene.positions[0];
    let end = scene.positions[1];
    if scene.bonds[0][0].neighbor != 1 {
        return Err("fixture did not create the expected iron bond".into());
    }

    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "H,0.01,{STEPS},{FRAMES},mechanical")?;
    write_frame(&mut out, &scene, 0)?;
    let mut tool = None;
    let mut previous_bond = true;
    let mut break_step = None;
    let mut max_ratio = 0.0f32;
    let frame_intervals = FRAMES - 1;
    for step in 1..=STEPS {
        let phase = TAU * (step % STEPS_PER_CYCLE) as f32 / STEPS_PER_CYCLE as f32;
        let target_x = end.x + displacement * phase.sin();
        let targets = [
            base.to_array().map(Some),
            [Some(target_x), Some(end.y), Some(end.z)],
        ];
        solver::step_with_axis_targets(&mut scene, &mut tool, &targets);
        let bond = &scene.bonds[0][0];
        let strain = (scene.positions[1].x - scene.positions[0].x - CELL) / CELL;
        let signed_ratio = strain * young / strength;
        max_ratio = max_ratio.max(signed_ratio.abs());
        if previous_bond && bond.neighbor == EMPTY {
            break_step = Some(step);
            writeln!(out, "E,{step},1,fatigue_fracture")?;
            previous_bond = false;
        }
        let sample = step == STEPS
            || (step as u128 * frame_intervals as u128 / STEPS as u128)
                > ((step - 1) as u128 * frame_intervals as u128 / STEPS as u128);
        if sample {
            write_frame(&mut out, &scene, step)?;
        }
        if step % 200 == 0 {
            eprintln!("fatigue step={step}/{STEPS} t={:.2}s cycle={:.0} bond={} damage={:.4} fatigue={:.4} strain_stress_ratio={:.3}",
                step as f32 * scene.config.fixed_dt,
                step as f32 / STEPS_PER_CYCLE as f32,
                bond.neighbor != EMPTY, bond.damage, bond.fatigue_damage, signed_ratio);
        }
    }
    let final_fatigue = scene.bonds[0][0].fatigue_damage;
    let final_broken = scene.bonds[0][0].neighbor == EMPTY;
    eprintln!("fatigue complete steps={STEPS} states={FRAMES} physical_s={:.2} max_stress_ratio={max_ratio:.4} break_step={break_step:?} final_bond_intact={} damage={:.4} fatigue={final_fatigue:.4}",
        STEPS as f32 * scene.config.fixed_dt, !final_broken, scene.bonds[0][0].damage);
    if !final_broken || break_step.is_none() || max_ratio >= 1.0 {
        return Err("fatigue capture failed: bond did not break below static strength".into());
    }
    Ok(())
}
