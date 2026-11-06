//! The reduced beam must agree with the fine cell solver within the 10% tolerance of
//! the paradigm before it may stand in for cells (INTEGRATION_PLAN section 7.6).
use glam::{DVec3, Vec3};
use voxel_physics_lab::{
    beam::{reduce_lattice_along_x, BeamMaterial},
    material::Material,
    scene::{Scene, SceneConfig, VoxelSpawn},
    solver,
};

const CELL: f64 = 0.01;

fn cantilever_scene(nx: i32, side: i32) -> (Scene, Vec<usize>, Vec<usize>) {
    let iron = Material::component_id("iron").unwrap();
    let mut config = SceneConfig::default();
    config.fixed_dt = 0.01;
    config.substeps = 4;
    config.gravity = [0.0; 3];
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
    (Scene::from_voxels(config, spawns).unwrap(), root, tip)
}

fn iron_material() -> BeamMaterial {
    let table = Scene::from_voxels(SceneConfig::default(), vec![VoxelSpawn::at_cell([0, 0, 0], Material::component_id("iron").unwrap())]).unwrap();
    let m = &table.properties[0];
    BeamMaterial {
        young: f64::from(m.young),
        shear: f64::from(m.shear),
        density: f64::from(m.density),
        yield_stress: f64::from(m.tensile_strength),
    }
}

#[test]
fn beam_tip_deflection_agrees_with_the_cell_solver() {
    let (mut scene, root, tip) = cantilever_scene(31, 3);
    let rest: Vec<DVec3> = scene.positions.clone();
    scene.fix_voxels(&root).unwrap();
    let force = 90.0f32;
    let start: Vec<DVec3> = tip.iter().map(|&i| scene.positions[i]).collect();
    for &i in &tip {
        solver::set_force(&mut scene, i, Vec3::new(0.0, 0.0, force / tip.len() as f32)).unwrap();
    }
    for _ in 0..400 {
        solver::step(&mut scene, &mut None);
    }
    let cells = tip.iter().zip(&start).map(|(&i, s)| scene.positions[i].z - s.z).sum::<f64>() / tip.len() as f64;

    let (mut beam, groups) = reduce_lattice_along_x(&rest, CELL, iron_material()).expect("a chain of slices");
    assert_eq!(groups.len(), 31);
    let mut loads = vec![[0.0; 6]; beam.node_count()];
    loads[beam.node_count() - 1] = [0.0, 0.0, f64::from(force), 0.0, 0.0, 0.0];
    beam.solve_static(&loads);
    let reduced = beam.displacement.last().unwrap()[2];
    eprintln!("BEAM EQUIVALENCE cells={cells:.4e} beam={reduced:.4e} ratio={:.4}", reduced / cells);
    assert!((reduced / cells - 1.0).abs() < 0.10, "beam {reduced:e} against cells {cells:e}");
}

#[test]
fn a_lattice_that_is_not_a_chain_is_refused() {
    let positions = vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(3.0 * CELL, 0.0, 0.0)];
    assert!(reduce_lattice_along_x(&positions, CELL, iron_material()).is_none());
}

/// Demoting cells to a beam and promoting it back must reproduce the cell state and its
/// energy within the tolerance, so a region can change tier without losing or inventing
/// energy.
#[test]
fn a_loaded_cantilever_survives_a_round_trip_between_cells_and_beam() {
    let (mut scene, root, tip) = cantilever_scene(31, 3);
    let rest: Vec<DVec3> = scene.positions.clone();
    scene.fix_voxels(&root).unwrap();
    let force = 90.0f32;
    for &i in &tip {
        solver::set_force(&mut scene, i, Vec3::new(0.0, 0.0, force / tip.len() as f32)).unwrap();
    }
    for _ in 0..400 {
        solver::step(&mut scene, &mut None);
    }
    let velocities: Vec<DVec3> = scene.voxels.iter().map(|v| v.velocity.as_dvec3()).collect();
    let cell_energy = scene.elastic_bond_energy_j();

    let (mut beam, groups) = reduce_lattice_along_x(&rest, CELL, iron_material()).unwrap();
    beam.set_from_cells(&groups, &rest, &scene.positions, &velocities);
    let promoted = beam.expand_positions(&groups, &rest);

    let deflection = scene.positions[tip[0]].z - rest[tip[0]].z;
    let worst = promoted
        .iter()
        .zip(&scene.positions)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0, f64::max);
    let beam_energy = beam.strain_energy();
    eprintln!(
        "ROUND TRIP worst position error {:.2}% of the tip deflection, strain energy cells={cell_energy:.3e} beam={beam_energy:.3e}",
        100.0 * worst / deflection.abs()
    );
    assert!(worst < 0.10 * deflection.abs(), "positions differ by {worst:e} against deflection {deflection:e}");
    assert!((beam_energy / cell_energy - 1.0).abs() < 0.15, "energy {beam_energy:e} against {cell_energy:e}");
    // A resting cantilever has no velocity to carry across.
    let carried = beam.expand_velocities(&groups, &rest);
    assert!(carried.iter().all(|v| v.length() < 1.0e-3));
}
