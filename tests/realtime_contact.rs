use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
    solver,
};

fn kinetic(scene: &Scene) -> f64 {
    scene
        .voxels
        .iter()
        .filter(|v| !v.kinematic)
        .map(|v| 0.5 * f64::from(v.mass) * v.velocity.as_dvec3().length_squared())
        .sum()
}

#[test]
fn swept_bonded_layers_contact_the_floor_without_being_pinned_into_one_layer() {
    let stone = Material::component_id("stone").unwrap();
    let oak = Material::component_id("oak").unwrap();
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.01, 1);
    for x in 0..3 {
        for y in 0..3 {
            for z in -1..=0 {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], stone).fixed());
            }
            for z in 3..9 {
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([x, y, z], oak).with_velocity(Vec3::new(0.0, 0.0, -3.0)),
                );
            }
        }
    }
    let mut scene = builder.build().unwrap();
    scene.config.atmosphere.pressure_pa = 0.0;
    let mass = scene.total_mass_kg();
    let initial_kinetic = kinetic(&scene);
    let initial_internal: f64 = scene
        .voxels
        .iter()
        .map(|v| v.total_internal_energy_j())
        .sum();
    for _ in 0..12 {
        solver::step(&mut scene, &mut None);
        assert_eq!(
            scene.broken, 0,
            "a modest floor impact must not tear apart the beam"
        );
        assert_eq!(scene.total_mass_kg(), mass);
        let positions: Vec<_> = scene
            .voxels
            .iter()
            .zip(&scene.positions)
            .filter_map(|(v, p)| (!v.kinematic).then_some(*p))
            .collect();
        assert!(positions
            .iter()
            .all(|p| p.is_finite() && p.z >= 0.015 - 2e-6));
        let bottom = positions.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        let top = positions
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            top - bottom > 0.049,
            "the six-layer beam collapsed: height={} m",
            top - bottom
        );
        let heat = scene
            .voxels
            .iter()
            .map(|v| v.total_internal_energy_j())
            .sum::<f64>()
            - initial_internal;
        let accounted = kinetic(&scene)
            + scene.elastic_bond_energy_j()
            + scene.stored_plastic_bond_energy_j()
            + heat
            + scene.energy.surface_work
            + scene.energy.pending_release;
        assert!(
            accounted <= initial_kinetic + 1e-5,
            "contact created energy: before={initial_kinetic} J after={accounted} J"
        );
    }
}

fn falling_cube(adaptive: bool) -> Scene {
    let oak = Material::component_id("oak").unwrap();
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::new(0.0, 0.0, -9.81))
        .with_timestep(0.01, 40);
    for x in 0..4 {
        for y in 0..4 {
            for z in 100..104 {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], oak));
            }
        }
    }
    let mut scene = builder.build().unwrap();
    scene.set_adaptive_substeps(adaptive);
    scene
}

#[test]
fn adaptive_free_fall_retains_revs_and_matches_fine_reference() {
    let mut fine = falling_cube(false);
    let mut adaptive = falling_cube(true);
    let mass = fine.total_mass_kg();
    for _ in 0..8 {
        solver::step(&mut fine, &mut None);
        solver::step(&mut adaptive, &mut None);
        assert_eq!(adaptive.last_mechanical_substeps(), 1);
        assert_eq!(adaptive.voxels.len(), 64);
        assert_eq!(adaptive.total_mass_kg(), mass);
        assert_eq!(adaptive.broken, 0);
        for ((a, b), (va, vb)) in adaptive
            .positions
            .iter()
            .zip(&fine.positions)
            .zip(adaptive.voxels.iter().zip(&fine.voxels))
        {
            assert!(
                (*a - *b).length() < 1e-6,
                "position mismatch: {:?}",
                *a - *b
            );
            assert!(
                (va.velocity - vb.velocity).length() < 1e-4,
                "velocity mismatch: {:?}",
                va.velocity - vb.velocity
            );
        }
    }
}

#[test]
fn explicit_force_refines_adaptive_body_before_integration() {
    let mut scene = falling_cube(true);
    solver::step(&mut scene, &mut None);
    assert_eq!(scene.last_mechanical_substeps(), 1);
    solver::set_force(&mut scene, 0, Vec3::X * 0.001).unwrap();
    solver::step(&mut scene, &mut None);
    assert_eq!(scene.last_mechanical_substeps(), 40);
    assert_eq!(scene.voxels.len(), 64);
    assert_eq!(scene.broken, 0);
    let momentum_x: f64 = scene
        .voxels
        .iter()
        .map(|v| f64::from(v.mass) * f64::from(v.velocity.x))
        .sum();
    assert!((momentum_x - 1e-5).abs() < 1e-7, "momentum={momentum_x}");
}

#[test]
fn adaptive_fall_reaches_static_floor_without_losing_voxels() {
    let stone = Material::component_id("stone").unwrap();
    let oak = Material::component_id("oak").unwrap();
    let build = || {
        let mut builder = SceneBuilder::standard()
            .with_gravity(Vec3::new(0.0, 0.0, -9.81))
            .with_timestep(0.01, 40);
        for x in 0..4 {
            for y in 0..4 {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
                for z in 8..12 {
                    builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], oak));
                }
            }
        }
        builder.build().unwrap()
    };
    let mut fine = build();
    let mut scene = build();
    scene.set_adaptive_substeps(true);
    let internal = |scene: &Scene| {
        scene
            .voxels
            .iter()
            .map(|v| v.total_internal_energy_j())
            .sum::<f64>()
    };
    let initial_internal = internal(&scene);
    let mass = scene.total_mass_kg();
    let mut fine_steps = 0;
    for _ in 0..60 {
        solver::step(&mut fine, &mut None);
        solver::step(&mut scene, &mut None);
        fine_steps += usize::from(scene.last_mechanical_substeps() == 40);
        assert_eq!(scene.total_mass_kg(), mass);
        assert_eq!(scene.voxels.len(), 80);
        assert_eq!(scene.broken, 0);
        assert!(scene
            .voxels
            .iter()
            .zip(&scene.positions)
            .all(|(v, p)| p.is_finite()
                && v.velocity.is_finite()
                && (v.kinematic || p.z >= 0.015 - 2e-6)));
    }
    let expected_time = 60.0 * f64::from(scene.config.fixed_dt);
    assert!((scene.simulated_seconds - expected_time).abs() < 1e-12);
    assert_eq!(scene.simulated_seconds, fine.simulated_seconds);
    let fine_heat = internal(&fine) - initial_internal;
    let adaptive_heat = internal(&scene) - initial_internal;
    assert!(
        (adaptive_heat - fine_heat).abs() <= 0.1 * fine_heat.abs().max(1e-8),
        "impact heating must match fine reference: fine={fine_heat} J adaptive={adaptive_heat} J"
    );
    assert!(fine_steps > 0, "contact must restore the fine solve");
    assert_eq!(scene.last_mechanical_substeps(), 0, "settled support should sleep; max_speed={} residual={} max_damage={} max_plastic={} max_force={}", scene.voxels.iter().map(|v| v.velocity.length()).fold(0.0_f32,f32::max), scene.solver_residual, scene.bonds.iter().flatten().map(|b| b.damage).fold(0.0_f32,f32::max), scene.bonds.iter().flatten().map(|b| b.equivalent_plastic_strain).fold(0.0_f32,f32::max), scene.workspace.iter().flatten().map(|w| w.last_force.length()).fold(0.0_f32,f32::max));
    let heat = f64::from(scene.voxels[1].mass * scene.properties[1].heat_capacity) * 0.01;
    scene.voxels[1].add_internal_energy_j(heat);
    solver::step(&mut scene, &mut None);
    assert!((scene.simulated_seconds - expected_time - f64::from(scene.config.fixed_dt)).abs() < 1e-12);
    assert_eq!(
        scene.last_mechanical_substeps(),
        40,
        "thermal change must wake mechanics"
    );
}
