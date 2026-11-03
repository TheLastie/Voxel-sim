use std::time::{Duration, Instant};

use glam::Vec3;
use voxel_physics_lab::{
    material::Material,
    scene::{Scene, SceneConfig},
    voxel::Composition,
    world::{GeneratedCellContent, GeneratedCellPhysics, GeneratedWorld},
};

const SECONDS_PER_YEAR: f32 = 365.25 * 24.0 * 60.0 * 60.0;
const REPEATS: usize = 3;

fn median_ms(samples: &[Duration]) -> f64 {
    let mut values = samples
        .iter()
        .map(Duration::as_secs_f64)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2] * 1_000.0
}

fn scene_signature(scene: &Scene) -> (usize, u64, usize, u64) {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for position in &scene.positions {
        for coordinate in position.to_array() {
            mix(coordinate.to_bits());
        }
    }
    for axis in &scene.bonds {
        for bond in axis {
            mix(u64::from(bond.neighbor));
        }
    }
    let bonds = scene
        .bonds
        .iter()
        .flat_map(|axis| axis.iter())
        .filter(|bond| bond.neighbor != voxel_physics_lab::bond::EMPTY)
        .count();
    (
        scene.voxels.len(),
        scene.total_mass_kg().to_bits(),
        bonds,
        hash,
    )
}

fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let seed = args
        .first()
        .map(String::as_str)
        .unwrap_or("oak-patch-bench-20261031");
    let age_years = args
        .get(1)
        .map(|value| value.parse::<f32>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(20.0);
    let radius_m = args
        .get(2)
        .map(|value| value.parse::<f32>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(1.5);

    let start = Instant::now();
    let mut world = GeneratedWorld::from_seed(seed);
    let generation_ms = start.elapsed().as_secs_f64() * 1_000.0;
    let maturation_start = Instant::now();
    world.advance_growth(age_years * SECONDS_PER_YEAR);
    let maturation_ms = maturation_start.elapsed().as_secs_f64() * 1_000.0;

    let map_center = Vec3::ZERO;
    let center = world
        .voxels
        .iter()
        .filter(|voxel| voxel.content == GeneratedCellContent::PlantWood)
        .min_by(|a, b| {
            a.center
                .distance_squared(map_center)
                .total_cmp(&b.center.distance_squared(map_center))
        })
        .map(|voxel| voxel.center)
        .ok_or_else(|| format!("no rendered plant wood found at age {age_years} y"))?;

    let query_start = Instant::now();
    let cells = world.plant_wood_rev_cells_near(center, radius_m, 250_000)?;
    let setup_query_ms = query_start.elapsed().as_secs_f64() * 1_000.0;
    if cells.is_empty() {
        return Err("wood query returned no 1 cm REVs".into());
    }

    let oak = Material::component_id("oak").ok_or("oak material missing")?;
    let origin_cell = cells[0];
    let mut sample_times = Vec::with_capacity(REPEATS);
    let mut spawn_times = Vec::with_capacity(REPEATS);
    let mut scene_times = Vec::with_capacity(REPEATS);
    let mut legacy_query_times = Vec::with_capacity(REPEATS);
    let mut legacy_patch_times = Vec::with_capacity(REPEATS);
    let mut legacy_total_times = Vec::with_capacity(REPEATS);
    let mut generated_query_times = Vec::with_capacity(REPEATS);
    let mut generated_patch_times = Vec::with_capacity(REPEATS);
    let mut legacy_checksums = Vec::with_capacity(REPEATS);
    let mut generated_checksums = Vec::with_capacity(REPEATS);
    let mut optimized_total_times = Vec::with_capacity(REPEATS);
    let mut checksums = Vec::with_capacity(REPEATS);

    for _ in 0..REPEATS {
        let sample_start = Instant::now();
        let generated = cells
            .iter()
            .map(|&cell| {
                world
                    .sample_cell(cell)
                    .filter(|sample| sample.content == GeneratedCellContent::PlantWood)
                    .ok_or_else(|| format!("wood REV {cell:?} failed to sample as wood"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        sample_times.push(sample_start.elapsed());

        let spawn_start = Instant::now();
        let spawns = generated
            .iter()
            .map(|&cell| {
                cell.to_local_voxel_spawn(
                    origin_cell,
                    Composition::pure(oak),
                    1.0,
                    Vec3::ZERO,
                    false,
                )
                .map_err(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        spawn_times.push(spawn_start.elapsed());

        let scene_start = Instant::now();
        let scene = Scene::from_voxels(SceneConfig::default(), spawns)?;
        scene_times.push(scene_start.elapsed());
        checksums.push(scene_signature(&scene));

        drop(scene);

        let legacy_query_start = Instant::now();
        let legacy_cells = world.plant_wood_rev_cells_near(center, radius_m, 250_000)?;
        let legacy_query_elapsed = legacy_query_start.elapsed();
        legacy_query_times.push(legacy_query_elapsed);
        if legacy_cells != cells {
            return Err("repeated legacy cell-ID query changed its result".into());
        }

        let legacy_patch_start = Instant::now();
        let legacy_patch = world.build_physical_patch_from_cells(
            SceneConfig::default(),
            origin_cell,
            legacy_cells,
            |_| {
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(oak),
                    anisotropy: 1.0,
                    velocity: Vec3::ZERO,
                    kinematic: false,
                }))
            },
        )?;
        let legacy_patch_elapsed = legacy_patch_start.elapsed();
        legacy_patch_times.push(legacy_patch_elapsed);
        legacy_total_times.push(legacy_query_elapsed + legacy_patch_elapsed);
        legacy_checksums.push(scene_signature(&legacy_patch.scene));

        let generated_query_start = Instant::now();
        let generated_cells = world.plant_wood_generated_cells_near(center, radius_m, 250_000)?;
        generated_query_times.push(generated_query_start.elapsed());
        if generated_cells
            .iter()
            .map(|cell| cell.cell)
            .ne(cells.iter().copied())
        {
            return Err("sampled-cell query differs from the legacy cell-ID query".into());
        }

        let generated_patch_start = Instant::now();
        let generated_patch = world.build_physical_patch_from_generated_cells(
            SceneConfig::default(),
            origin_cell,
            generated_cells,
            |_| {
                Ok(Some(GeneratedCellPhysics {
                    composition: Composition::pure(oak),
                    anisotropy: 1.0,
                    velocity: Vec3::ZERO,
                    kinematic: false,
                }))
            },
        )?;
        generated_patch_times.push(generated_patch_start.elapsed());
        optimized_total_times.push(
            generated_query_times.last().copied().unwrap()
                + generated_patch_times.last().copied().unwrap(),
        );
        generated_checksums.push(scene_signature(&generated_patch.scene));
    }

    let first_checksum = checksums[0];
    if checksums.iter().any(|&checksum| checksum != first_checksum) {
        return Err("physical scene mass/count checksum changed between runs".into());
    }
    if legacy_checksums
        .iter()
        .any(|&checksum| checksum != legacy_checksums[0])
        || generated_checksums
            .iter()
            .any(|&checksum| checksum != generated_checksums[0])
        || legacy_checksums[0] != generated_checksums[0]
        || legacy_checksums[0] != first_checksum
    {
        return Err("legacy and sampled-state physical patch results differ".into());
    }

    println!("seed={seed} age_y={age_years:.2} map_km2=10.0 rev_cm=1 oak_moisture=12% radius_m={radius_m:.2}");
    println!("world_generation_ms={generation_ms:.3} world_growth_rebuild_ms={maturation_ms:.3}");
    println!(
        "target_center_world_m=[{:.2},{:.2},{:.2}] cells={} repeats={REPEATS}",
        center.x,
        center.y,
        center.z,
        cells.len()
    );
    println!("setup_query_ms={setup_query_ms:.3}");
    println!("sample_cell_median_ms={:.3}", median_ms(&sample_times));
    println!("cell_to_spawn_median_ms={:.3}", median_ms(&spawn_times));
    println!("scene_from_voxels_median_ms={:.3}", median_ms(&scene_times));
    println!(
        "legacy_query_median_ms={:.3}",
        median_ms(&legacy_query_times)
    );
    println!(
        "legacy_patch_builder_median_ms={:.3}",
        median_ms(&legacy_patch_times)
    );
    println!(
        "sampled_state_query_median_ms={:.3}",
        median_ms(&generated_query_times)
    );
    println!(
        "sampled_state_patch_builder_median_ms={:.3}",
        median_ms(&generated_patch_times)
    );
    println!(
        "legacy_query_plus_patch_median_ms={:.3}",
        median_ms(&legacy_total_times)
    );
    println!(
        "sampled_query_plus_patch_median_ms={:.3}",
        median_ms(&optimized_total_times)
    );
    println!(
        "activation_speedup={:.3}x",
        median_ms(&legacy_total_times) / median_ms(&optimized_total_times)
    );
    println!(
        "scene_voxels={} mass_kg={:.6} bonds={} fingerprint={:016x}",
        first_checksum.0,
        f64::from_bits(first_checksum.1),
        first_checksum.2,
        first_checksum.3
    );
    println!("scope=assembly_only; no ground, solver step, or tree-ground interaction measured");
    Ok(())
}
