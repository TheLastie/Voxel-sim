//! A slender part of a scene stepped as a beam region (tier T2) must follow the same
//! physics as the cells it replaces: static deflection under gravity, an energy ledger
//! that balances, and a lossless hand-back to the cell solver.
use glam::DVec3;
use voxel_physics_lab::{
    beam::{reduce_lattice_along_x, BeamMaterial},
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

const CELL: f64 = 0.01;
/// Strong enough for a deflection of about 1 mm on the 31 cm iron cantilever (validity
/// about half), weak enough to stay inside the beam model's range.
const GRAVITY: f32 = -2.0e3;

fn cantilever(nx: i32, side: i32) -> (Scene, Vec<usize>, Vec<usize>) {
    let iron = Material::component_id("iron").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.gravity = [0.0, 0.0, GRAVITY];
    config.atmosphere.pressure_pa = 0.0;
    let mut spawns = Vec::new();
    for x in 0..nx {
        for y in 0..side {
            for z in 0..side {
                spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
            }
        }
    }
    let index = |x: i32, y: i32, z: i32| (x * side * side + y * side + z) as usize;
    let root: Vec<usize> = (0..side).flat_map(|y| (0..side).map(move |z| (y, z))).map(|(y, z)| index(0, y, z)).collect();
    let tip: Vec<usize> = root.iter().map(|&i| i + ((nx - 1) * side * side) as usize).collect();
    let mut scene = Scene::from_voxels(config, spawns).unwrap();
    scene.fix_voxels(&root).unwrap();
    (scene, root, tip)
}

fn iron_material(scene: &Scene) -> BeamMaterial {
    let m = &scene.properties[0];
    BeamMaterial {
        young: f64::from(m.young),
        shear: f64::from(m.shear),
        density: f64::from(m.density),
        yield_stress: f64::from(m.tensile_strength),
    }
}

fn mean_tip_z(scene: &Scene, tip: &[usize]) -> f64 {
    tip.iter().map(|&i| scene.positions[i].z).sum::<f64>() / tip.len() as f64
}

/// Static tip deflection under gravity from the beam model alone.
fn static_deflection(scene: &Scene, rest: &[DVec3]) -> f64 {
    let (mut beam, _) = reduce_lattice_along_x(rest, CELL, iron_material(scene)).unwrap();
    let loads: Vec<[f64; 6]> = (0..beam.node_count())
        .map(|node| [0.0, 0.0, f64::from(GRAVITY) * beam.node_mass(node), 0.0, 0.0, 0.0])
        .collect();
    beam.solve_static(&loads);
    beam.displacement.last().unwrap()[2]
}

#[test]
fn a_beam_region_sways_about_the_static_deflection_and_balances_energy() {
    let (mut scene, _root, tip) = cantilever(31, 3);
    let rest = scene.positions.clone();
    let tip_rest = mean_tip_z(&scene, &tip);
    let expected = static_deflection(&scene, &rest);
    let cells: Vec<usize> = (0..scene.positions.len()).collect();
    let id = scene.reduce_to_beam(&cells, 40).expect("reduce");

    let mut sum = 0.0;
    let mut peak = 0.0f64;
    let steps = 200;
    for _ in 0..steps {
        solver::step(&mut scene, &mut None);
        let d = mean_tip_z(&scene, &tip) - tip_rest;
        sum += d;
        peak = peak.max(d.abs());
    }
    let mean = sum / f64::from(steps);
    let energy = scene.beam_region_energy_j(id).unwrap();
    let work = scene.energy.external_work;
    eprintln!(
        "BEAM REGION static={expected:.4e} mean={mean:.4e} peak={peak:.4e} energy={energy:.4e} work={work:.4e} validity={:.3}",
        scene.beam_region_validity(id).unwrap()
    );
    assert!(expected < 0.0, "gravity should bend the beam down");
    // An undamped beam released from rest swings about the static shape, to twice it.
    assert!((mean / expected - 1.0).abs() < 0.15, "mean {mean:e} against static {expected:e}");
    assert!(peak > 1.5 * expected.abs() && peak < 2.1 * expected.abs(), "peak {peak:e} against static {expected:e}");
    assert!((energy / work.abs() - 1.0).abs() < 1.0e-3, "beam energy {energy:e} against work {work:e}");
    assert!(scene.beam_region_validity(id).unwrap() < 1.0);
}

#[test]
fn an_expanded_region_continues_in_the_cell_solver() {
    let (mut scene, _root, tip) = cantilever(31, 3);
    let tip_rest = mean_tip_z(&scene, &tip);
    let cells: Vec<usize> = (0..scene.positions.len()).collect();
    let id = scene.reduce_to_beam(&cells, 40).expect("reduce");
    for _ in 0..30 {
        solver::step(&mut scene, &mut None);
    }
    let before = mean_tip_z(&scene, &tip) - tip_rest;
    scene.expand_beam(id).expect("expand");
    assert!(scene.beam_region_energy_j(id).is_none());
    assert!(scene.voxels.iter().enumerate().all(|(i, v)| v.kinematic == (i < 9)));
    for _ in 0..5 {
        solver::step(&mut scene, &mut None);
    }
    let after = mean_tip_z(&scene, &tip) - tip_rest;
    eprintln!("BEAM EXPAND tip before={before:.4e} after={after:.4e}");
    assert!(scene.positions.iter().all(|p| p.is_finite()));
    // The tip moves continuously across the hand-back: no jump larger than the swing.
    assert!((after - before).abs() < 2.0 * before.abs() + 1.0e-6);
}

#[test]
fn a_region_with_a_free_first_slice_is_refused() {
    let (mut scene, root, _tip) = cantilever(8, 3);
    scene.release_voxels(&root).unwrap();
    let cells: Vec<usize> = (0..scene.positions.len()).collect();
    assert!(scene.reduce_to_beam(&cells, 10).is_err());
}
