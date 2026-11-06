//! Reactions run at the fixed-step boundary when a scene enables chemistry, and not
//! otherwise: composition changes at constant mass and constant stored energy.
use smallvec::smallvec;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
    voxel::Composition,
};

fn burning_cell(chemistry: bool) -> Scene {
    let wood = Material::component_id("dry_wood").expect("dry_wood component");
    let oxygen = Material::component_id("oxygen").expect("oxygen component");
    let mut config = SceneConfig::default();
    config.gravity = [0.0; 3];
    config.atmosphere.pressure_pa = 0.0;
    let composition = Composition { components: smallvec![(wood, 0.4), (oxygen, 0.6)] };
    let spawn = VoxelSpawn::at_cell([0, 0, 0], wood)
        .with_composition(composition)
        .at_temperature(700.0);
    let mut scene = Scene::from_voxels(config, vec![spawn]).unwrap();
    scene.chemistry_enabled = chemistry;
    scene
}

fn fuel_fraction(scene: &Scene) -> f32 {
    let wood = Material::component_id("dry_wood").unwrap();
    scene.voxels[0]
        .composition
        .components
        .iter()
        .find(|(id, _)| *id == wood)
        .map_or(0.0, |(_, fraction)| *fraction)
}

#[test]
fn chemistry_runs_in_the_step_only_when_enabled() {
    let mut off = burning_cell(false);
    let mut on = burning_cell(true);
    let mass = on.voxels[0].mass;
    let energy = on.voxels[0].total_internal_energy_j();
    let temperature_before = on.temperature_kelvin(0).unwrap();
    for _ in 0..50 {
        solver::step(&mut off, &mut None);
        solver::step(&mut on, &mut None);
    }
    assert!((fuel_fraction(&off) - 0.4).abs() < 1.0e-6, "chemistry off changed the fuel");
    assert!(fuel_fraction(&on) < 0.2, "fuel not consumed: {}", fuel_fraction(&on));
    assert_eq!(on.voxels[0].mass, mass, "REV mass changed");
    assert!(on.temperature_kelvin(0).unwrap() > temperature_before + 100.0, "no heating from the burn");
    // Stored internal energy of a closed cell is unchanged by composition change.
    assert!((on.voxels[0].total_internal_energy_j() - energy).abs() <= 1.0e-6 * energy.abs());
}
