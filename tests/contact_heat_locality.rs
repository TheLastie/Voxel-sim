use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    bond::EMPTY,
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn},
};

fn momentum(scene: &Scene) -> DVec3 {
    scene.voxels.iter().fold(DVec3::ZERO, |total, voxel| {
        total + voxel.velocity.as_dvec3() * f64::from(voxel.mass)
    })
}

fn pair_internal_energy(scene: &Scene, pair: [usize; 2]) -> f64 {
    pair.into_iter()
        .map(|i| scene.voxels[i].total_internal_energy_j())
        .sum()
}

#[test]
fn impact_heat_stays_with_impact_pair_when_an_unrelated_pair_is_resting() {
    let iron = Material::component_id("iron").expect("iron material is registered");
    let mut scene = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(0.001, 2)
        .add_voxel(
            VoxelSpawn::at_cell([0, 0, 0], iron)
                .with_velocity(Vec3::X)
                .at_temperature(293.15),
        )
        .add_voxel(
            VoxelSpawn::at_cell([1, 0, 0], iron)
                .with_velocity(Vec3::NEG_X)
                .at_temperature(293.15),
        )
        .add_voxel(VoxelSpawn::at_cell([5, 0, 0], iron).at_temperature(293.15))
        .add_voxel(VoxelSpawn::at_cell([6, 0, 0], iron).at_temperature(293.15))
        .build()
        .expect("scene builds");
    scene.config.atmosphere.pressure_pa = 0.0;

    // SceneBuilder authors ordinary lattice bonds for both adjacent pairs.
    // Start this regression with free, unbonded contact pairs instead.
    scene.bonds[0][0].neighbor = EMPTY;
    scene.bonds[0][2].neighbor = EMPTY;
    voxel_physics_lab::solver::rebuild_clusters(&mut scene);

    let resting_initial = pair_internal_energy(&scene, [2, 3]);
    let impact_initial = pair_internal_energy(&scene, [0, 1]);
    let momentum_initial = momentum(&scene);
    let mass_initial: f64 = scene.voxels.iter().map(|voxel| f64::from(voxel.mass)).sum();

    scene.tick();

    let resting_final = pair_internal_energy(&scene, [2, 3]);
    let impact_final = pair_internal_energy(&scene, [0, 1]);
    let momentum_final = momentum(&scene);
    let mass_final: f64 = scene.voxels.iter().map(|voxel| f64::from(voxel.mass)).sum();

    assert!(impact_final > impact_initial, "the impact pair dissipates heat");
    assert_eq!(scene.voxels.len(), 4);
    assert!(scene.voxels.iter().all(|voxel| voxel.mass.is_finite() && voxel.mass > 0.0));
    assert!(momentum_final.is_finite());
    assert!(mass_final.is_finite());
    assert!((mass_final - mass_initial).abs() <= 1.0e-12);
    assert!((momentum_final - momentum_initial).length() <= 1.0e-10);
    assert!(scene.voxels[2..].iter().all(|voxel| voxel.velocity.length() <= 1.0e-10));
    assert!(
        (resting_final - resting_initial).abs() <= 1.0e-12,
        "unrelated resting pair gained heat: {resting_initial:.12e} -> {resting_final:.12e} J"
    );
}
