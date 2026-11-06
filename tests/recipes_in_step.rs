//! The fixed-step chemistry hook honours the scene's interval and recipe set.
use smallvec::smallvec;
use voxel_physics_lab::{
    material::Material,
    recipes::RecipeRegistry,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
    voxel::Composition,
};

fn burning_cell() -> Scene {
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
    scene.chemistry_enabled = true;
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
fn the_step_hook_waits_for_the_chemistry_interval() {
    let mut idle = burning_cell();
    idle.set_chemistry_interval(1.0e3).unwrap();
    let mut timed = burning_cell();
    timed.set_chemistry_interval(0.025).unwrap();
    let mut every_step = burning_cell();
    let (mass, energy) = (timed.voxels[0].mass, timed.voxels[0].total_internal_energy_j());
    for _ in 0..50 {
        solver::step(&mut idle, &mut None);
        solver::step(&mut timed, &mut None);
        solver::step(&mut every_step, &mut None);
    }
    // 50 ms elapsed. The long interval has not been reached; the 25 ms
    // interval fired twice and burned the fuel; both end like the per-step hook.
    assert!((fuel_fraction(&idle) - 0.4).abs() < 1.0e-6);
    assert!(fuel_fraction(&timed) < 0.2, "{}", fuel_fraction(&timed));
    assert!(fuel_fraction(&every_step) < 0.2, "{}", fuel_fraction(&every_step));
    assert_eq!(timed.voxels[0].mass, mass);
    assert!((timed.voxels[0].total_internal_energy_j() - energy).abs() <= 1.0e-6 * energy.abs());
}

#[test]
fn the_step_hook_uses_the_scene_registry_when_one_is_set() {
    let mut scene = burning_cell();
    let empty = RecipeRegistry::new(&scene.materials, Vec::new(), None).unwrap();
    scene.set_chemistry_registry(empty).unwrap();
    for _ in 0..50 {
        solver::step(&mut scene, &mut None);
    }
    assert!((fuel_fraction(&scene) - 0.4).abs() < 1.0e-6, "an empty recipe set must not burn anything");
}
