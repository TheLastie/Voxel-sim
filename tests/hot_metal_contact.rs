use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, DCELL},
    solver,
};

const CUBE_SIDE: i32 = 4;
const RIGHT_ORIGIN_X: i32 = 12; // Eight empty cells between the cube faces.
const STEPS: usize = 100;

fn kinetic(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .filter(|voxel| !voxel.kinematic)
        .map(|voxel| 0.5 * f64::from(voxel.mass) * voxel.velocity.as_dvec3().length_squared())
        .sum()
}

fn cube_members(initial_positions: &[DVec3], left: bool) -> Vec<bool> {
    initial_positions
        .iter()
        .map(|position| (position.x < 0.08) == left)
        .collect()
}

fn original_internal_bonds(scene: &Scene, members: &[bool]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    for axis in 0..3 {
        for (index, bond) in scene.bonds[axis].iter().enumerate() {
            if bond.neighbor == u32::MAX {
                continue;
            }
            let neighbor = bond.neighbor as usize;
            if members[index] && members[neighbor] {
                result.push((axis, index));
            }
        }
    }
    result
}

fn intact_count(scene: &Scene, original: &[(usize, usize)]) -> usize {
    original
        .iter()
        .filter(|&&(axis, index)| scene.bonds[axis][index].neighbor != u32::MAX)
        .count()
}

fn maximum_original_bond_stress_pa(scene: &Scene, original: &[(usize, usize)]) -> f32 {
    original
        .iter()
        .map(|&(axis, index)| scene.workspace[axis][index].last_force.length() / (DCELL as f32).powi(2))
        .fold(0.0, f32::max)
}

fn maximum_plastic_strain(scene: &Scene, original: &[(usize, usize)]) -> f32 {
    original
        .iter()
        .map(|&(axis, index)| scene.bonds[axis][index].equivalent_plastic_strain)
        .fold(0.0, f32::max)
}

fn run_case(hot_temperature: f32, label: &str, trace: bool) -> Vec<String> {
    let iron = Material::component_id("iron").unwrap();
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 4);
    for x in 0..CUBE_SIDE {
        for y in 0..CUBE_SIDE {
            for z in 0..CUBE_SIDE {
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([x, y, z], iron)
                        .with_velocity(Vec3::new(0.2, 0.0, 0.0))
                        .at_temperature(hot_temperature),
                );
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([RIGHT_ORIGIN_X + x, y, z], iron)
                        .with_velocity(Vec3::new(-0.2, 0.0, 0.0))
                        .at_temperature(293.15),
                );
            }
        }
    }
    let mut scene = builder.build().unwrap();
    scene.config.atmosphere.pressure_pa = 0.0;
    let initial_positions = scene.positions.clone();
    let hot_members = cube_members(&initial_positions, true);
    let cold_members = cube_members(&initial_positions, false);
    let hot_bonds = original_internal_bonds(&scene, &hot_members);
    let cold_bonds = original_internal_bonds(&scene, &cold_members);
    let initial_mass = scene.total_mass_kg();
    let initial_hot_bonds = hot_bonds.len();
    let initial_cold_bonds = cold_bonds.len();
    let initial_internal: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
    let initial_accounted = kinetic(&scene)
        + initial_internal
        + scene.elastic_bond_energy_j()
        + scene.stored_plastic_bond_energy_j()
        + scene.energy.surface_work
        + scene.energy.pending_release;
    // A large thermal reservoir must not hide spurious mechanical energy.
    let energy_tolerance = (32.0 * f64::from(f32::EPSILON) * kinetic(&scene))
        .max(64.0 * f64::EPSILON * initial_internal.abs())
        .max(1.0e-9);
    let mut peak_kinetic = kinetic(&scene);
    let mut maximum_speed = 0.0_f32;
    let mut maximum_cold_temperature = 0.0_f32;
    let mut maximum_hot_gas_fraction = 0.0_f32;
    let mut first_cold_break_step = None;
    let mut first_cold_break_stress_pa = 0.0_f32;
    let mut first_cold_break_plastic_strain = 0.0_f32;
    let mut maximum_cold_stress = 0.0_f32;
    let mut maximum_cold_plastic_strain = 0.0_f32;
    let mut maximum_accounted_energy_excess = 0.0_f64;
    let mut maximum_energy_excess_step = 0usize;

    for step in 1..=STEPS {
        solver::step(&mut scene, &mut None);
        assert_eq!(scene.voxels.len(), 128, "{label}: voxel count changed at step {step}");
        assert_eq!(scene.total_mass_kg(), initial_mass, "{label}: mass changed at step {step}");
        assert!(scene.positions.iter().all(|p| p.is_finite()), "{label}: non-finite position at step {step}");
        assert!(scene.voxels.iter().all(|v| v.velocity.is_finite()), "{label}: non-finite velocity at step {step}");

        let mut step_max_speed = 0.0_f32;
        let mut step_max_cold_temperature = 0.0_f32;
        let mut step_max_hot_gas_fraction = 0.0_f32;
        for index in 0..scene.voxels.len() {
            step_max_speed = step_max_speed.max(scene.voxels[index].velocity.length());
            let phase = scene.properties[index]
                .phase_state(scene.voxels[index].mass, scene.voxels[index].thermal_energy_f32());
            if cold_members[index] {
                step_max_cold_temperature = step_max_cold_temperature.max(phase.temperature_kelvin);
            }
            if hot_members[index] {
                step_max_hot_gas_fraction = step_max_hot_gas_fraction.max(phase.gas_fraction);
            }
        }
        maximum_speed = maximum_speed.max(step_max_speed);
        peak_kinetic = peak_kinetic.max(kinetic(&scene));
        maximum_cold_temperature = maximum_cold_temperature.max(step_max_cold_temperature);
        maximum_hot_gas_fraction = maximum_hot_gas_fraction.max(step_max_hot_gas_fraction);

        let cold_intact = intact_count(&scene, &cold_bonds);
        let internal_total: f64 = scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum();
        let hot_internal: f64 = scene.voxels.iter().enumerate()
            .filter(|(index, _)| hot_members[*index])
            .map(|(_, voxel)| voxel.total_internal_energy_j()).sum();
        let cold_internal: f64 = scene.voxels.iter().enumerate()
            .filter(|(index, _)| cold_members[*index])
            .map(|(_, voxel)| voxel.total_internal_energy_j()).sum();
        let recoverable_elastic = scene.elastic_bond_energy_j();
        let stored_plastic = scene.stored_plastic_bond_energy_j();
        let accounted_energy = kinetic(&scene)
            + internal_total
            + recoverable_elastic
            + stored_plastic
            + scene.energy.surface_work
            + scene.energy.pending_release;
        let energy_excess = accounted_energy - initial_accounted;
        if energy_excess > maximum_accounted_energy_excess {
            maximum_accounted_energy_excess = energy_excess;
            maximum_energy_excess_step = step;
        }
        let step_cold_stress = maximum_original_bond_stress_pa(&scene, &cold_bonds);
        let step_cold_plastic_strain = maximum_plastic_strain(&scene, &cold_bonds);
        maximum_cold_stress = maximum_cold_stress.max(step_cold_stress);
        maximum_cold_plastic_strain = maximum_cold_plastic_strain.max(step_cold_plastic_strain);
        if cold_intact < initial_cold_bonds && first_cold_break_step.is_none() {
            first_cold_break_step = Some(step);
            first_cold_break_stress_pa = step_cold_stress;
            first_cold_break_plastic_strain = step_cold_plastic_strain;
        }
        if trace {
            eprintln!(
                "HOT_METAL_TRACE case={label} step={step} t={:.3} max_speed={step_max_speed:.6e} kinetic={:.6e} internal_total={internal_total:.9e} internal_hot={hot_internal:.9e} internal_cold={cold_internal:.9e} recoverable_elastic={recoverable_elastic:.9e} stored_plastic={stored_plastic:.9e} accounted_excess={energy_excess:.9e} implicit_solver_heat={:.9e} restitution_heat={:.9e} damage_heat={:.9e} plastic_heat={:.9e} surface_work={:.9e} pending_release={:.9e} cold_max_K={step_max_cold_temperature:.6e} hot_gas_fraction={step_max_hot_gas_fraction:.6e} hot_bonds={}/{} cold_bonds={cold_intact}/{initial_cold_bonds} cold_stress_Pa={step_cold_stress:.6e} cold_plastic_strain={step_cold_plastic_strain:.6e} broken={} clusters={}",
                scene.simulated_seconds,
                kinetic(&scene),
                scene.energy.implicit_solver_heat,
                scene.energy.contact_restitution_heat,
                scene.energy.damage_heat,
                scene.energy.plastic_heat,
                scene.energy.surface_work,
                scene.energy.pending_release,
                intact_count(&scene, &hot_bonds),
                initial_hot_bonds,
                scene.broken,
                scene.cluster_count,
            );
        }
    }

    eprintln!(
        "HOT_METAL_SUMMARY case={label} steps={STEPS} final_t={:.3} max_speed={maximum_speed:.6e} peak_kinetic={peak_kinetic:.6e} max_cold_temperature_K={maximum_cold_temperature:.6e} max_hot_gas_fraction={maximum_hot_gas_fraction:.6e} hot_bonds={}/{} cold_bonds={}/{} first_cold_break_step={first_cold_break_step:?} first_break_stress_Pa={first_cold_break_stress_pa:.6e} first_break_plastic_strain={first_cold_break_plastic_strain:.6e} peak_cold_bond_stress_Pa={maximum_cold_stress:.6e} peak_cold_plastic_strain={maximum_cold_plastic_strain:.6e} max_energy_excess={maximum_accounted_energy_excess:.9e} at_step={maximum_energy_excess_step} tolerance={energy_tolerance:.9e} internal_total={:.9e} implicit_solver_heat={:.9e} restitution_heat={:.9e} damage_heat={:.9e} plastic_heat={:.9e} surface_work={:.9e} pending_release={:.9e} broken={} clusters={}",
        scene.simulated_seconds,
        intact_count(&scene, &hot_bonds),
        initial_hot_bonds,
        intact_count(&scene, &cold_bonds),
        initial_cold_bonds,
        scene.voxels.iter().map(|v| v.total_internal_energy_j()).sum::<f64>(),
        scene.energy.implicit_solver_heat,
        scene.energy.contact_restitution_heat,
        scene.energy.damage_heat,
        scene.energy.plastic_heat,
        scene.energy.surface_work,
        scene.energy.pending_release,
        scene.broken,
        scene.cluster_count,
    );
    let mut failures = Vec::new();
    let expected_hot_bonds = if hot_temperature >= 9000.0 { 0 } else { initial_hot_bonds };
    if intact_count(&scene, &hot_bonds) != expected_hot_bonds {
        failures.push(format!("{label}: hot cube retained an unexpected number of solid bonds"));
    }
    if hot_temperature >= 9000.0 && maximum_hot_gas_fraction < 0.999 {
        failures.push(format!("{label}: vaporized iron did not publish its gas phase"));
    }
    if intact_count(&scene, &cold_bonds) < initial_cold_bonds {
        failures.push(format!(
            "{label}: cold cube lost an original internal bond; first_step={first_cold_break_step:?} max_stress={maximum_cold_stress:.6e} Pa max_plastic_strain={maximum_cold_plastic_strain:.6e}"
        ));
    }
    if maximum_accounted_energy_excess > energy_tolerance {
        failures.push(format!(
            "{label}: accounted energy exceeded initial by {maximum_accounted_energy_excess:.9e} J at step {maximum_energy_excess_step}, tolerance={energy_tolerance:.9e} J"
        ));
    }
    failures
}

#[test]
fn hot_iron_cube_contact_does_not_break_cold_cube() {
    let trace = std::env::var_os("VOXEL_TRACE_HOT_METAL_CONTACT").is_some();
    let failures = [("ambient", 293.15), ("warm", 1500.0), ("gas", 9000.0)]
        .into_iter()
        .flat_map(|(label, temperature)| run_case(temperature, label, trace))
        .collect::<Vec<_>>();
    assert!(failures.is_empty(), "cold-cube bond failures: {failures:?}");
}
