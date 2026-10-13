//! Conservation capture of a hot iron/copper REV fusion using the Scene API.
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
    solver,
};

const DT: f32 = 0.01;
const STEPS: usize = 1_760;
const STRIDE: usize = 10;
const FRAMES: usize = STEPS / STRIDE + 1;
const HOT_TEMPERATURE_K: f32 = 363.15;
const IMPACT_SPEED: f32 = 0.01;
const TANGENTIAL_SPEED: f32 = 0.005;
const COMPRESSION_FORCE_N: f32 = 0.1;
const FUSION_TEMPERATURE_K: f32 = 350.0;
const FUSION_PRESSURE_PA: f32 = 900.0;
const FUSION_DWELL_S: f32 = 0.1;
const ENERGY_RESIDUAL_FRACTION_OF_WORK: f64 = 1.0e-5;

fn kinetic(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
        .sum()
}
fn internal(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .map(|v| v.total_internal_energy_j())
        .sum()
}
fn momentum(scene: &Scene) -> Vec3 {
    scene
        .voxels
        .iter()
        .fold(Vec3::ZERO, |p, v| p + v.velocity * v.mass)
}
fn write_frame(out: &mut BufWriter<File>, scene: &Scene, step: usize) -> std::io::Result<()> {
    writeln!(out, "S,{step}")?;
    for (i, p) in scene.positions.iter().enumerate() {
        writeln!(out, "V,{step},{i},{},{},{},0,0", p.x, p.y, p.z)?;
        writeln!(
            out,
            "Q,{step},{i},{},{},{},{},0",
            p.x,
            p.y,
            p.z,
            scene.temperature_kelvin(i).unwrap_or(0.0)
        )?;
        for &(component, fraction) in &scene.voxels[i].composition.components {
            writeln!(
                out,
                "C,{step},{i},{component},{fraction},{}",
                scene.voxels[i].mass
            )?;
        }
    }
    let p = momentum(scene);
    writeln!(
        out,
        "L,{step},{},0,{},{},{},{},{},{},{},{},{},{},{},{},{}",
        scene.voxels.len(),
        scene.total_mass_kg(),
        internal(scene),
        kinetic(scene),
        scene.energy.pending_release,
        p.x,
        p.y,
        p.z,
        scene.energy.fusion_heat,
        scene.energy.contact_restitution_heat,
        scene.energy.contact_friction_heat,
        scene.energy.plastic_heat,
        scene.energy.damage_heat,
        scene.energy.external_work
    )?;
    // Identity tool pose keeps the shared live mechanical viewer active.
    writeln!(out, "F,{step},0,0,0,0,0,0,1,0,0,0,0,0,{},0", scene.broken)?;
    out.flush()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "validation/fusion.csv".into());
    let iron = Material::component_id("iron").ok_or("iron missing from material table")?;
    let copper = Material::component_id("copper").ok_or("copper missing from material table")?;
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(DT, 2)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .fixed()
                .at_temperature(HOT_TEMPERATURE_K),
        )
        .add_voxel(VoxelSpawn::at_cell([1, 0, 0], copper).at_temperature(HOT_TEMPERATURE_K))
        .build()?;
    // Start the unbonded pair apart. Releasing the first REV preserves a
    // topology-free pair while allowing the ordinary solver to create contact.
    scene.positions[1].x += 0.005;
    scene.previous[1] = scene.positions[1];
    scene.release_voxel(0)?;
    scene.voxels[0].velocity = Vec3::new(IMPACT_SPEED, TANGENTIAL_SPEED, 0.0);
    scene.voxels[1].velocity = Vec3::new(-IMPACT_SPEED, -TANGENTIAL_SPEED, 0.0);
    scene.config.fracture.restitution = 0.0;
    scene.config.fracture.friction = 0.0;
    solver::set_force(&mut scene, 0, Vec3::X * COMPRESSION_FORCE_N)?;
    solver::set_force(&mut scene, 1, Vec3::NEG_X * COMPRESSION_FORCE_N)?;
    scene.set_thermal_fusion_rule(FUSION_TEMPERATURE_K, FUSION_PRESSURE_PA, FUSION_DWELL_S)?;

    let initial_mass = scene.total_mass_kg();
    let initial_internal = internal(&scene);
    let initial_kinetic = kinetic(&scene);
    let initial_momentum = momentum(&scene);
    let mut out = BufWriter::new(File::create(&path)?);
    writeln!(out, "H,{DT},{STEPS},2,mechanical")?;
    writeln!(out, "N,iron_copper_fusion,mass_fraction,temperature_K")?;
    writeln!(
        out,
        "R,fusion_rule,{FUSION_TEMPERATURE_K},{FUSION_PRESSURE_PA},{FUSION_DWELL_S}"
    )?;
    writeln!(out, "R,compression_force,{COMPRESSION_FORCE_N}")?;
    writeln!(
        out,
        "R,external_work,midpoint_velocity_free_or_new_impact;projected_displacement_persistent_contact"
    )?;
    writeln!(out, "R,fixed_step_diagnostics,U,K,work,Q_restitution,Q_friction,Q_plastic,Q_damage,Q_fusion,residual")?;
    let mut fused_at = None;
    let initial_external_work_j = scene.energy.external_work;
    let start = Instant::now();
    write_frame(&mut out, &scene, 0)?;
    writeln!(out, "W,0,0,0")?;
    for step in 1..=STEPS {
        scene.tick();
        let external_work_j = scene.energy.external_work - initial_external_work_j;
        let step_internal = internal(&scene);
        let step_kinetic = kinetic(&scene);
        let residual =
            step_internal + step_kinetic - initial_internal - initial_kinetic - external_work_j;
        writeln!(out, "D,{step},{step_internal:.12e},{step_kinetic:.12e},{external_work_j:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{residual:.12e}",
            scene.energy.contact_restitution_heat,
            scene.energy.contact_friction_heat,
            scene.energy.plastic_heat,
            scene.energy.damage_heat,
            scene.energy.fusion_heat)?;
        if fused_at.is_none() && scene.voxels.len() == 1 {
            fused_at = Some(step);
            writeln!(out, "E,{step},fusion,iron_copper")?;
            eprintln!("automatic thermal fusion formed at step={step}, t={:.3}s; mass={:.9}kg composition={:?}",
                step as f32 * DT, scene.voxels[0].mass, scene.voxels[0].composition.components);
        }
        if step % STRIDE == 0 {
            write_frame(&mut out, &scene, step)?;
            writeln!(out, "W,{step},{external_work_j:.12e},{residual:.12e}")?;
        }
    }

    let external_work_j = scene.energy.external_work - initial_external_work_j;
    let energy_error =
        internal(&scene) + kinetic(&scene) - initial_internal - initial_kinetic - external_work_j;
    let energy_tolerance_j = external_work_j.abs() * ENERGY_RESIDUAL_FRACTION_OF_WORK;
    let momentum_error = (momentum(&scene) - initial_momentum).length();
    let composition_ok = scene.voxels.len() == 1
        && scene.voxels[0].composition.components.len() == 2
        && scene.voxels[0]
            .composition
            .components
            .iter()
            .all(|(_, fraction)| *fraction > 0.0);
    eprintln!("fusion_capture complete states={FRAMES} physical_s={:.2} fused_step={fused_at:?} mass={:.9}kg momentum_error={momentum_error:.3e}kg*m/s energy_residual={energy_error:.3e}J external_work={external_work_j:.9e}J residual_over_work={:.3e} restitution_heat={:.9e}J friction_heat={:.9e}J plastic_heat={:.9e}J damage_heat={:.9e}J fusion_heat={:.9e}J composition_ok={composition_ok} clusters={} wall_s={:.3}",
        STEPS as f32 * DT, scene.total_mass_kg(),
        energy_error.abs() / external_work_j.abs().max(f64::MIN_POSITIVE),
        scene.energy.contact_restitution_heat,
        scene.energy.contact_friction_heat,
        scene.energy.plastic_heat,
        scene.energy.damage_heat,
        scene.energy.fusion_heat,
        scene.cluster_count,
        start.elapsed().as_secs_f64());
    if fused_at.is_none()
        || scene.voxels.len() != 1
        || !composition_ok
        || (scene.total_mass_kg() - initial_mass).abs() > 1e-9
        || momentum_error > 1e-7
        || scene.energy.fusion_heat <= 0.0
        || energy_error.abs() > energy_tolerance_j
    {
        return Err(format!(
            "fusion capture failed: topology/mass/momentum/heat or energy residual {:.3e} J exceeds {:.3e} J",
            energy_error.abs(), energy_tolerance_j
        ).into());
    }
    Ok(())
}
