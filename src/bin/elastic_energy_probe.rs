//! Measure elastic oscillator energy drift without adding compensating heat.
use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{SceneBuilder, SceneConfig, VoxelSpawn, CELL},
    solver,
};

fn value<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(default);
    };
    args.get(index + 1)
        .ok_or_else(|| format!("missing value after {name}"))?
        .parse()
        .map_err(|error| format!("invalid value for {name}: {error}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dt_s: f32 = value(&args, "--dt", 0.000001)?;
    let substeps: usize = value(&args, "--substeps", 1)?;
    let steps: usize = value(&args, "--steps", 20)?;
    let relative_speed_m_s: f32 = value(&args, "--speed", 0.01)?;
    if !dt_s.is_finite()
        || dt_s <= 0.0
        || substeps == 0
        || steps == 0
        || !relative_speed_m_s.is_finite()
        || relative_speed_m_s <= 0.0
    {
        return Err("dt, substeps, steps, and speed must be positive and finite".into());
    }

    let iron = Material::component_id("iron").ok_or("built-in iron material is missing")?;
    let mut config = SceneConfig::default();
    config.gravity = [0.0; 3];
    config.fixed_dt = dt_s;
    config.substeps = substeps;
    // Zero ambient pressure disables aerodynamic drag for this isolated mode.
    config.atmosphere.pressure_pa = 0.0;
    let mut scene = SceneBuilder::new(config)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .with_velocity(Vec3::X * (0.5 * relative_speed_m_s)),
        )
        .add_voxel(
            VoxelSpawn::at_cell([1, 0, 0], iron)
                .with_velocity(Vec3::NEG_X * (0.5 * relative_speed_m_s)),
        )
        .build()?;
    if scene.bonds.iter().flatten().filter(|bond| bond.neighbor != u32::MAX).count() != 1 {
        return Err("expected exactly one intact bond in the oscillator".into());
    }
    let axial_stiffness_n_m =
        f64::from(solver::stiffness_matrix(&scene, 0, 1, glam::Mat3::IDENTITY).x_axis.x);
    let reduced_mass_kg = f64::from(scene.voxels[0].mass) * f64::from(scene.voxels[1].mass)
        / f64::from(scene.voxels[0].mass + scene.voxels[1].mass);
    let omega_rad_s = (axial_stiffness_n_m / reduced_mass_kg).sqrt();
    let period_s = std::f64::consts::TAU / omega_rad_s;

    let energy = |scene: &voxel_physics_lab::scene::Scene| {
        scene.voxels.iter().fold(0.0, |sum, voxel| {
            sum + 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared())
        })
    };
    let initial_kinetic_j = energy(&scene);
    let initial_elastic_j = scene.elastic_bond_energy_j();
    let initial_internal_j: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
    let initial_plastic_store_j = scene.stored_plastic_bond_energy_j();
    let mut tool = None;
    println!("step,time_s,kinetic_j,elastic_j,internal_delta_j,solid_damping_heat_cumulative_j,plastic_heat_cumulative_j,damage_heat_cumulative_j,contact_heat_cumulative_j,plastic_store_delta_j,mechanical_balance_delta_j,peak_stress_pa,broken,residual");

    let mut final_balance_j = 0.0;
    let mut peak_residual = 0.0_f32;
    let mut last_step_ms = 0.0;
    for step in 1..=steps {
        let started = std::time::Instant::now();
        solver::step(&mut scene, &mut tool);
        last_step_ms = started.elapsed().as_secs_f64() * 1000.0;
        let kinetic_j = energy(&scene);
        let elastic_j = scene.elastic_bond_energy_j();
        let internal_delta_j = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum::<f64>()
            - initial_internal_j;
        let plastic_store_delta_j = scene.stored_plastic_bond_energy_j() - initial_plastic_store_j;
        final_balance_j = (kinetic_j - initial_kinetic_j)
            + (elastic_j - initial_elastic_j)
            + internal_delta_j
            + plastic_store_delta_j;
        let peak_stress_pa = scene.workspace.iter().flatten()
            .map(|workspace| workspace.last_force.length() / (CELL * CELL))
            .fold(0.0_f32, f32::max);
        let contact_heat_j = scene.energy.contact_friction_heat + scene.energy.contact_restitution_heat;
        peak_residual = peak_residual.max(scene.solver_residual);
        println!(
            "{step},{:.9},{kinetic_j:.12e},{elastic_j:.12e},{internal_delta_j:.12e},{:.12e},{:.12e},{:.12e},{contact_heat_j:.12e},{plastic_store_delta_j:.12e},{final_balance_j:.12e},{peak_stress_pa:.6e},{},{}",
            f64::from(step as f32 * dt_s),
            scene.energy.solid_viscoelastic_heat,
            scene.energy.plastic_heat,
            scene.energy.damage_heat,
            scene.broken,
            scene.solver_residual,
        );
        if scene.broken != 0 || scene.positions.iter().any(|position| !position.is_finite()) {
            return Err(format!("oscillator left the intact finite regime at step {step}").into());
        }
    }

    eprintln!(
        "elastic_energy_probe complete mode=axial voxels={} bonds=1 dt_s={dt_s} substeps={substeps} steps={steps} simulated_s={:.9e} relative_speed_m_s={relative_speed_m_s} linearized_omega_rad_s={omega_rad_s:.9e} linearized_period_s={period_s:.9e} initial_kinetic_j={initial_kinetic_j:.9e} initial_elastic_j={initial_elastic_j:.9e} damping_heat_j={:.9e} plastic_heat_j={:.9e} contact_heat_j={:.9e} signed_balance_delta_j={final_balance_j:.9e} peak_stress_pa={:.6e} broken={} peak_residual={:.3e}",
        scene.voxels.len(),
        f64::from(dt_s) * steps as f64,
        scene.energy.solid_viscoelastic_heat,
        scene.energy.plastic_heat,
        scene.energy.contact_friction_heat + scene.energy.contact_restitution_heat,
        scene.workspace.iter().flatten().map(|workspace| workspace.last_force.length() / (CELL * CELL)).fold(0.0_f32, f32::max),
        scene.broken,
        peak_residual,
    );
    if voxel_physics_lab::live_tests::enabled() {
        voxel_physics_lab::live_tests::observe_scene_force(
            &scene, "", scene.simulated_seconds, last_step_ms,
        );
        if !voxel_physics_lab::live_tests::flush(std::time::Duration::from_secs(2)) {
            eprintln!("live viewer: final frame was not confirmed written");
        }
    }
    Ok(())
}
