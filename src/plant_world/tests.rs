use super::*;
use crate::world::{GeneratedCellContent, GeneratedWorld, GrowthModel, GrowthScheduler, WORLD_SIDE_M};

const YEAR_S: f32 = 365.25 * 24.0 * 3600.0;

// ---------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= *b as u64;
        *hash = hash.wrapping_mul(0x100_0000_01b3);
    }
}

/// Hash of the overview geometry and of a 1 cm region around the first seed/plant.
fn world_hash(world: &GeneratedWorld) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for v in &world.voxels {
        for c in v.center.to_array().iter().chain(v.size.to_array().iter()).chain(v.color.iter()) {
            fnv(&mut hash, &c.to_bits().to_le_bytes());
        }
        fnv(&mut hash, &[v.content as u8]);
    }
    let anchor = world
        .voxels
        .iter()
        .find(|v| v.content == GeneratedCellContent::Seed)
        .or_else(|| world.voxels.iter().find(|v| v.content == GeneratedCellContent::PlantWood));
    if let Some(anchor) = anchor {
        let half = WORLD_SIDE_M * 0.5;
        let cell = [
            ((anchor.center.x + half) / 0.01).floor() as i32,
            ((anchor.center.y + half) / 0.01).floor() as i32,
            ((anchor.center.z) / 0.01).floor() as i32,
        ];
        let min = [cell[0] - 20, cell[1] - 20, (cell[2] - 10).max(0)];
        let max = [cell[0] + 20, cell[1] + 20, cell[2] + 30];
        for c in world.sample_region(min, max).expect("region within work limit") {
            fnv(&mut hash, &[c.content as u8]);
        }
    }
    hash
}

fn base_genome(id: u64) -> SeedGenome {
    SeedGenome {
        individual_id: id,
        morphotype_id: 0,
        maximum_height_m: 18.0,
        crown_spread_m: 5.0,
        mature_trunk_radius_m: 0.34,
        shade_tolerance: 0.5,
        water_demand: 0.5,
        nutrient_demand: 0.4,
        longevity_years: 90.0,
        moisture_optimum: 0.5,
        temperature_optimum_kelvin: 288.0,
        root_count: 7,
        root_extent_m: 1.3,
        root_depth_factor: 0.7,
        trunk_height_fraction: 0.75,
        branching_rate: 0.6,
        apical_dominance: 0.5,
        branch_spacing_m: 0.9,
        branch_slope: 0.3,
        branch_width: 0.3,
        leaf_retention: 0.5,
        mycorrhizal_affinity: 0.5,
        germination_delay_years: 0.5,
        mutation_probability: 1e-5,
    }
}

/// A genome that varies deterministically with `id` around `base_genome`.
fn varied_genome(id: u64) -> SeedGenome {
    let u = |salt: u64| unit_hash(77, id, salt);
    let mut g = base_genome(id);
    g.maximum_height_m = 14.0 + 10.0 * u(1) as f32;
    g.crown_spread_m = 4.0 + 2.5 * u(2) as f32;
    g.shade_tolerance = 0.2 + 0.6 * u(3) as f32;
    g.water_demand = 0.3 + 0.4 * u(4) as f32;
    g.nutrient_demand = 0.3 + 0.3 * u(5) as f32;
    g.longevity_years = 70.0 + 70.0 * u(6) as f32;
    g.temperature_optimum_kelvin = 284.0 + 8.0 * u(7) as f32;
    g.branching_rate = 0.4 + 0.5 * u(8) as f32;
    g.leaf_retention = u(9) as f32;
    g
}

fn site(id: u64, x: f64, y: f64, germination_years: f64) -> SiteInput {
    SiteInput {
        genome: base_genome(id),
        x_m: x,
        y_m: y,
        moisture: 0.5,
        fertility: 0.55,
        temperature_k: 288.0,
        germination_years,
    }
}

fn varied_site(id: u64, x: f64, y: f64, germination_years: f64) -> SiteInput {
    SiteInput { genome: varied_genome(id), ..site(id, x, y, germination_years) }
}

/// Soil cells so small that every plant of these tests has its own.
fn separate_soils() -> PopulationConfig {
    PopulationConfig { soil_cell_m: 0.5, soil_area_m2: Some(64.0), ..PopulationConfig::default() }
}

fn state_bits(pop: &PlantPopulation) -> Vec<u64> {
    let mut bits = Vec::new();
    for i in 0..pop.len() {
        let p = pop.plant(i);
        for x in [
            p.stores.sugar,
            p.stores.water,
            p.stores.minerals,
            p.pools.roots,
            p.pools.trunk,
            p.pools.branches,
            p.pools.leaves,
            p.pools.fruit,
            p.damage,
            p.time_days,
            pop.light_fraction(i),
        ] {
            bits.push(x.to_bits());
        }
        bits.extend(p.rings().iter().flat_map(|r| [r.mass_kg.to_bits(), r.volume_m3.to_bits()]));
    }
    bits.push(pop.ledger().closed_mass_kg.to_bits());
    bits
}

fn assert_ledger_closed(pop: &PlantPopulation) {
    let ledger = pop.ledger();
    let drift = pop.mass_drift_kg();
    assert!(
        drift.abs() <= 1e-12 * ledger.closed_mass_kg.abs().max(1.0),
        "mass drift {drift} kg of {} kg",
        ledger.closed_mass_kg
    );
    let scale = ledger.radiation_in_j.max(ledger.imported_energy_j).max(1.0);
    assert!(ledger.energy_residual_j.abs() <= 1e-9 * scale, "energy residual {} J of {scale} J", ledger.energy_residual_j);
}

// ---------------------------------------------------------------------------------
// Old worlds are unchanged
// ---------------------------------------------------------------------------------

/// Hashes of the generated geometry for seed "1001" taken on the commit before the
/// resource-based model existed (`from_seed`, ages 0, 12 and 40 years).
const LEGACY_HASHES: [(f32, usize, u64); 3] = [
    (0.0, 10270, 10614768561515818392),
    (12.0, 97229, 17211152562056706053),
    (40.0, 621079, 2780495442977497277),
];

#[test]
fn legacy_worlds_generate_exactly_what_they_did_before() {
    for (age, count, hash) in LEGACY_HASHES {
        for explicit in [false, true] {
            let mut world = if explicit {
                GeneratedWorld::from_seed_with_model("1001", GrowthModel::Legacy)
            } else {
                GeneratedWorld::from_seed("1001")
            };
            assert_eq!(world.growth_model(), GrowthModel::Legacy);
            assert!(world.plant_population().is_none());
            if age > 0.0 {
                world.advance_growth(age * YEAR_S);
            }
            assert_eq!(world.voxels.len(), count, "age {age} explicit {explicit}");
            assert_eq!(world_hash(&world), hash, "age {age} explicit {explicit}");
        }
    }
}

// ---------------------------------------------------------------------------------
// Genome mapping, determinism
// ---------------------------------------------------------------------------------

#[test]
fn world_genome_maps_to_the_plant_genome_with_habit_defaults() {
    let seed = varied_genome(9);
    let genome = plant_genome_from_seed(&seed);
    assert!(genome.in_bounds());
    let tree = PlantGenome::preset(HabitPreset::Tree, 9);
    assert_eq!(genome.habit, tree.habit, "unmapped genes take the tree habit defaults");
    assert_eq!(genome.seed.individual_id, 9);
    assert_eq!(genome.seed.maximum_height_m, seed.maximum_height_m);
    assert_eq!(genome.seed.shade_tolerance, seed.shade_tolerance);
    assert_eq!(genome.seed.branching_rate, seed.branching_rate);
    assert_eq!(genome.seed.crown_spread_m, 2.0 * seed.crown_spread_m, "world radius becomes plant diameter");
    let mut huge = seed;
    huge.crown_spread_m = 28.0;
    assert!(plant_genome_from_seed(&huge).in_bounds());
}

#[test]
fn a_seed_gives_the_same_population_and_another_seed_a_different_one() {
    let sites: Vec<SiteInput> = (0..6).map(|i| varied_site(100 + i, 3.0 * i as f64, 0.5 * i as f64, 0.2 * i as f64)).collect();
    let run = |seed: u64| {
        let mut pop = PlantPopulation::new(seed, &sites, PopulationConfig::default());
        pop.advance_years(12.0);
        pop
    };
    let a = run(5);
    let b = run(5);
    assert_eq!(state_bits(&a), state_bits(&b), "same seed, same state to the bit");
    for i in 0..a.len() {
        assert_eq!(a.geometry(i), b.geometry(i));
    }
    let c = run(6);
    assert_ne!(state_bits(&a), state_bits(&c), "the world seed changes the seeds' masses");
    // The parallel schedule gives identical results.
    let mut parallel = PlantPopulation::new(5, &sites, PopulationConfig { parallel: true, ..PopulationConfig::default() });
    parallel.advance_years(12.0);
    assert_eq!(state_bits(&a), state_bits(&parallel));
}

// ---------------------------------------------------------------------------------
// Mass ledger, events vs small steps
// ---------------------------------------------------------------------------------

fn mixed_sites() -> Vec<SiteInput> {
    let mut sites = Vec::new();
    // Six isolated individuals and a cluster of four sharing one soil cell.
    for i in 0..6u64 {
        sites.push(varied_site(10 + i, 40.0 * i as f64, 0.0, 0.1 + 0.15 * i as f64));
    }
    for k in 0..4u64 {
        sites.push(varied_site(50 + k, 200.0 + 1.5 * (k % 2) as f64, 200.0 + 1.5 * (k / 2) as f64, 0.3));
    }
    sites
}

#[test]
fn the_world_ledger_closes_through_rain_litter_shedding_and_a_cut() {
    let mut pop = PlantPopulation::new(11, &mixed_sites(), PopulationConfig::default());
    assert_eq!(pop.soil_count(), 7, "the cluster shares one soil cell");
    let initial = pop.ledger();
    assert_ledger_closed(&pop);
    pop.advance_years(15.0);
    assert_ledger_closed(&pop);
    let after_growth = pop.ledger();
    assert!(after_growth.plant_mass_kg > 20.0 * initial.plant_mass_kg, "the plants took biomass from air and soil");
    assert!(after_growth.imported_mass_kg > initial.imported_mass_kg, "rain is an import");
    let litter: f64 = (0..pop.soil_count()).map(|s| pop.soil_environment(s).litter_organic_kg).sum();
    assert!(litter > 0.0, "turnover reached the litter ledger");
    // Cut the biggest tree, go on.
    let biggest = (0..pop.len()).max_by(|&a, &b| pop.plant(a).biomass_kg().total_cmp(&pop.plant(b).biomass_kg())).unwrap();
    let before = pop.plant(biggest).total_mass_kg();
    let part = pop.cut(biggest, 0.7).expect("something above ground to cut");
    let (owner, severed) = pop.severed_part(part);
    assert_eq!(owner, biggest);
    assert!((severed.total_mass_kg() + pop.plant(biggest).total_mass_kg() - before).abs() < 1e-9 * before);
    assert_ledger_closed(&pop);
    pop.advance_years(10.0);
    assert_ledger_closed(&pop);
    println!(
        "ledger after 25 y: plants {:.1} kg, severed {:.1} kg, environment {:.1} kg, closed drift {:e} kg",
        pop.ledger().plant_mass_kg,
        pop.ledger().severed_mass_kg,
        pop.ledger().environment_mass_kg,
        pop.mass_drift_kg()
    );
}

#[test]
fn fifty_years_in_one_call_match_many_small_calls() {
    let sites = mixed_sites();
    let mut one = PlantPopulation::new(3, &sites, PopulationConfig::default());
    one.advance_years(50.0);
    let mut many = PlantPopulation::new(3, &sites, PopulationConfig::default());
    let calls = (50.0 * YEAR_DAYS / 30.0).ceil() as usize;
    for k in 1..=calls {
        many.advance_to_days((k as f64 * 30.0).min(50.0 * YEAR_DAYS));
    }
    assert!((one.time_days() - many.time_days()).abs() < 1e-6);
    assert_ledger_closed(&one);
    assert_ledger_closed(&many);
    let (a, b) = (one.total_biomass_kg(), many.total_biomass_kg());
    println!("50 y: one call {a:.2} kg, {calls} calls {b:.2} kg, total difference {:.2}%", 100.0 * (a - b) / b);
    assert!(a > 100.0, "the stand grew: {a}");
    assert!(((a - b) / b).abs() < 0.10, "total biomass {a} vs {b}");
    let mut worst: f64 = 0.0;
    for i in 0..sites.len() {
        let (x, y) = (one.plant(i).biomass_kg(), many.plant(i).biomass_kg());
        if x.max(y) > 1.0 {
            worst = worst.max(((x - y) / y).abs());
        }
        assert_eq!(one.plant(i).is_dead(), many.plant(i).is_dead(), "plant {i}");
    }
    println!("worst single-plant difference {:.2}%", 100.0 * worst);
    assert!(worst < 0.10, "single plant differs by {worst}");
}

#[test]
fn an_event_step_and_fine_daily_steps_agree_for_three_years() {
    let sites = vec![site(1, 0.0, 0.0, 0.0), site(2, 30.0, 0.0, 0.0)];
    let mut long = PlantPopulation::new(1, &sites, separate_soils());
    long.advance_years(3.0);
    let mut daily = PlantPopulation::new(1, &sites, separate_soils());
    for d in 1..=(3 * 365) {
        daily.advance_to_days(d as f64);
    }
    for i in 0..2 {
        let (a, b) = (long.plant(i).biomass_kg(), daily.plant(i).biomass_kg());
        println!("3 y plant {i}: event steps {a:.3} kg, daily {b:.3} kg");
        assert!(a > 0.5);
        assert!(((a - b) / b).abs() < 0.10, "{a} vs {b}");
    }
    assert_ledger_closed(&long);
    assert_ledger_closed(&daily);
}

// ---------------------------------------------------------------------------------
// Light and soil
// ---------------------------------------------------------------------------------

#[test]
fn neighbours_shade_a_plant_and_slow_it() {
    // Every plant has its own soil, so only light differs between the two runs.
    let target_germination = 3.0;
    let alone_sites = vec![site(1, 2.0, 2.0, target_germination)];
    let mut alone = PlantPopulation::new(1, &alone_sites, separate_soils());
    let mut crowded_sites = alone_sites.clone();
    for (k, (dx, dy)) in [(3.0, 0.0), (0.0, 3.0), (-3.0, 0.0), (0.0, -3.0)].into_iter().enumerate() {
        crowded_sites.push(site(10 + k as u64, 2.0 + dx, 2.0 + dy, 0.0));
    }
    let mut crowded = PlantPopulation::new(1, &crowded_sites, separate_soils());
    alone.advance_years(20.0);
    crowded.advance_years(20.0);
    let (lone, shaded) = (alone.plant(0).biomass_kg(), crowded.plant(0).biomass_kg());
    println!(
        "target after 20 y: alone {lone:.2} kg (mean light {:.2}), among 4 neighbours {shaded:.2} kg (mean light {:.2}, neighbours {:.1} m high)",
        alone.mean_light_fraction(0),
        crowded.mean_light_fraction(0),
        crowded.geometry(1).height_m
    );
    assert_eq!(alone.mean_light_fraction(0), 1.0);
    assert!(crowded.mean_light_fraction(0) < 0.6, "light {}", crowded.mean_light_fraction(0));
    assert!(shaded < 0.8 * lone, "shaded {shaded} vs alone {lone}");
    assert!(shaded > 0.0);
    assert_ledger_closed(&crowded);
}

#[test]
fn plants_in_one_soil_cell_share_its_water_and_minerals() {
    let thin = |cell: f64| PopulationConfig { soil_cell_m: cell, soil_area_m2: Some(64.0), soil_depth_m: 0.12, ..PopulationConfig::default() };
    let positions = [(0.2, 0.2), (1.2, 0.2)];
    let sites: Vec<SiteInput> = positions.iter().enumerate().map(|(i, p)| site(1 + i as u64, p.0, p.1, 0.0)).collect();
    let mut shared = PlantPopulation::new(1, &sites, thin(8.0));
    let mut apart = PlantPopulation::new(1, &sites, thin(0.5));
    assert_eq!(shared.soil_count(), 1);
    assert_eq!(apart.soil_count(), 2);
    assert_eq!(shared.soil_index(0), shared.soil_index(1));
    shared.advance_years(15.0);
    apart.advance_years(15.0);
    let (s, a) = (shared.total_biomass_kg(), apart.total_biomass_kg());
    println!("15 y: shared soil {s:.2} kg, separate soils {a:.2} kg");
    assert!(s < 0.85 * a, "a shared finite soil gives less in total: {s} vs {a}");
    assert!(s > 0.0);
    let env = shared.soil_environment(0);
    assert!(env.soil_water_kg >= 0.05 * env.soil_water_capacity_kg - 1e-6, "no uptake below the wilting point");
    assert_ledger_closed(&shared);
    assert_ledger_closed(&apart);
}

#[test]
fn geometry_is_computed_from_pools_and_responds_to_the_environment() {
    let mut wet = site(1, 0.0, 0.0, 0.0);
    wet.moisture = 0.9;
    wet.temperature_k = 289.0;
    let mut cold = site(1, 100.0, 0.0, 0.0);
    cold.temperature_k = 281.0;
    let mut pop = PlantPopulation::new(1, &[wet, cold], separate_soils());
    let seed_state = PlantPopulation::new(1, &[site(1, 0.0, 0.0, 1.0)], separate_soils());
    assert_eq!(seed_state.geometry(0), PlantGeometry::default(), "a seed has no dimensions yet");
    pop.advance_years(20.0);
    let (g_wet, g_dry) = (pop.geometry(0), pop.geometry(1));
    println!("wet {g_wet:?}\ndry {g_dry:?}");
    assert!(g_wet.germinated && g_wet.height_m > 1.0 && g_wet.trunk_radius_m > 0.01 && g_wet.crown_radius_m > 0.5);
    assert!(g_wet.height_m > g_dry.height_m && g_wet.crown_radius_m > g_dry.crown_radius_m && g_wet.biomass_kg > g_dry.biomass_kg);
    // Same quantities straight from the plant: pools and rings, no table.
    let plant = pop.plant(0);
    let morph = plant.morphology();
    assert_eq!(g_wet.height_m, morph.height_m as f32);
    assert_eq!(g_wet.trunk_radius_m, (0.5 * morph.trunk_diameter_m) as f32);
    assert_eq!(g_wet.leaf_area_m2, morph.leaf_area_m2 as f32);
    assert_eq!(g_wet.branch_volume_m3, (plant.pools.branches / plant.wood_density()) as f32);
    assert!(g_wet.branch_volume_m3 > 0.0 && g_wet.branch_count > 0);
    // Growth is monotonic in size while the plant is healthy.
    pop.advance_years(5.0);
    let next = pop.geometry(0);
    assert!(next.height_m >= g_wet.height_m * 0.999 && next.trunk_radius_m >= g_wet.trunk_radius_m * 0.999);
}

// ---------------------------------------------------------------------------------
// Cutting
// ---------------------------------------------------------------------------------

#[test]
fn a_cut_tree_splits_and_the_severed_part_keeps_its_mass_and_stops_growing() {
    let mut pop = PlantPopulation::new(2, &[site(1, 0.0, 0.0, 0.0)], PopulationConfig::default());
    // Cut at 10.4 y (warm season). With the model's provisional coefficients a heavy
    // cut often starves the stump (no leaves, almost no sugar reserve); at this age
    // and fraction the stump survives with damage and regrows.
    pop.advance_years(10.4);
    let before = pop.plant(0).total_mass_kg();
    let height_before = pop.geometry(0).height_m;
    let reference = pop.ledger().closed_mass_kg;
    let part_index = pop.cut(0, 0.8).unwrap();
    assert!(pop.cut(0, 0.0).is_none() && pop.cut(5, 0.5).is_none());
    let (_, part) = pop.severed_part(part_index);
    let part_mass = part.total_mass_kg();
    let part_pools = part.pools;
    assert!(part_mass > 0.5 * before, "most of the mass left with the crown");
    assert!(!part.can_take_up_water());
    assert!((part_mass + pop.plant(0).total_mass_kg() - before).abs() < 1e-9 * before);
    assert!((pop.ledger().closed_mass_kg - reference).abs() < 1e-9 * reference);
    assert!(pop.geometry(0).height_m < height_before, "the rooted remainder is shorter");
    pop.advance_years(8.0);
    let (_, part_later) = pop.severed_part(part_index);
    assert_eq!(part_later.pools, part_pools, "the severed part grows nothing and loses no tissue");
    assert!(part_later.total_mass_kg() <= part_mass && part_later.total_mass_kg() > 0.9 * part_mass, "it only respires some sugar");
    assert!(pop.plant(0).biomass_kg() > 0.0 && !pop.plant(0).is_dead(), "the stump regrows from roots and reserves");
    assert_ledger_closed(&pop);
}

// ---------------------------------------------------------------------------------
// The generated world
// ---------------------------------------------------------------------------------

fn resource_world(age_years: f32) -> GeneratedWorld {
    let mut world = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    if age_years > 0.0 {
        world.advance_growth(age_years * YEAR_S);
        world.sync_plants();
    }
    world
}

#[test]
fn the_world_builds_one_resource_plant_per_vegetation_individual() {
    let legacy = GeneratedWorld::from_seed("1001");
    let world = resource_world(0.0);
    assert_eq!(world.growth_model(), GrowthModel::ResourceBased);
    let population = world.plant_population().expect("population");
    assert_eq!(population.len(), world.plant_count());
    assert_eq!(world.plant_count(), legacy.plant_count(), "same sites and genomes as the legacy world");
    for i in 0..population.len() {
        let (x, y) = world.plant_position(i).unwrap();
        assert_eq!(legacy.plant_position(i), Some((x, y)));
        assert_eq!(population.position(i), (x as f64, y as f64));
    }
}

#[test]
fn generator_dimensions_come_from_the_plants_state() {
    let world = resource_world(40.0);
    let population = world.plant_population().unwrap();
    assert!((population.time_days() - 40.0 * YEAR_DAYS).abs() < 1.0);
    assert_ledger_closed(population);
    let mut checked = 0;
    let mut volume_ratios = Vec::new();
    for i in 0..world.plant_count() {
        let state = population.geometry(i);
        let built = world.built_plant_geometry(i).unwrap();
        if state.height_m <= 0.5 {
            continue;
        }
        checked += 1;
        assert!(!built.stale, "plant {i}");
        assert_eq!(built.height_m, state.height_m);
        assert_eq!(built.trunk_radius_m, state.trunk_radius_m);
        assert_eq!(built.trunk_height_m, state.trunk_height_m);
        assert_eq!(built.crown_radius_m, state.crown_radius_m);
        if state.branch_volume_m3 > 0.0 && built.branch_wood_volume_m3 > 0.0 {
            volume_ratios.push(built.branch_wood_volume_m3 / state.branch_volume_m3);
        }
    }
    println!("plants with geometry: {checked}; generated/pool branch volume ratios {volume_ratios:.3?}");
    assert!(checked >= 5, "{checked}");
    let exact = volume_ratios.iter().filter(|r| (**r - 1.0).abs() < 0.02).count();
    assert!(exact * 2 >= volume_ratios.len(), "branch wood volume follows the branch pool where the radius scale is not clamped");
    // The centimetre path really sees wood and the overview really has crowns.
    let wood = world.voxels.iter().filter(|v| v.content == GeneratedCellContent::PlantWood).count();
    assert!(wood > 20, "{wood}");
    let tallest = (0..world.plant_count())
        .max_by(|&a, &b| population.geometry(a).height_m.total_cmp(&population.geometry(b).height_m))
        .unwrap();
    let (x, y) = world.plant_position(tallest).unwrap();
    let half = WORLD_SIDE_M * 0.5;
    let column = [((x + half) / 0.01).floor() as i32, ((y + half) / 0.01).floor() as i32];
    let terrain = world.terrain_height_at(x, y);
    let z0 = (terrain / 0.01).floor() as i32;
    let cells = world
        .sample_region([column[0] - 12, column[1] - 12, z0], [column[0] + 12, column[1] + 12, z0 + 25])
        .unwrap();
    assert!(cells.iter().any(|c| c.content == GeneratedCellContent::PlantWood), "trunk cells at the base of the tallest tree");
}

#[test]
fn old_and_new_models_agree_on_the_unchanged_parts_of_the_world() {
    let legacy = GeneratedWorld::from_seed("1001");
    let resource = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    assert_eq!(legacy.terrain_height_at(10.0, 10.0), resource.terrain_height_at(10.0, 10.0));
    let terrain = |w: &GeneratedWorld| w.voxels.iter().filter(|v| v.content == GeneratedCellContent::Terrain).count();
    assert_eq!(terrain(&legacy), terrain(&resource));
}

#[test]
fn the_world_clock_advances_by_events_in_one_call_or_many() {
    let mut one = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    one.advance_growth(50.0 * YEAR_S);
    one.sync_plants();
    let mut many = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    for _ in 0..200 {
        many.advance_growth(0.25 * YEAR_S);
    }
    many.sync_plants();
    let (a, b) = (one.plant_population().unwrap(), many.plant_population().unwrap());
    let (ma, mb) = (a.total_biomass_kg(), b.total_biomass_kg());
    println!("world, 50 y: one call {ma:.1} kg, 200 quarter calls {mb:.1} kg, difference {:.2}%", 100.0 * (ma - mb) / mb);
    assert!(ma > 100.0);
    assert!(((ma - mb) / mb).abs() < 0.10);
    assert_ledger_closed(a);
    assert_ledger_closed(b);
    assert!((one.age_years - many.age_years).abs() < 1e-3);
}

#[test]
fn growth_outside_the_loaded_region_runs_without_geometry_and_loads_in_chunks() {
    let mut world = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    let count = world.plant_count();
    assert!(count >= 6);
    let (cx, cy) = world.plant_position(0).unwrap();
    world.set_loaded_region(Some((cx, cy, 5.0)));
    let seed_biomass: Vec<f32> = (0..count).map(|i| world.plant_population().unwrap().plant(i).biomass_kg() as f32).collect();
    world.advance_growth(25.0 * YEAR_S);
    world.sync_plants();
    let population = world.plant_population().unwrap();
    // State advanced everywhere; geometry only where the region is.
    let grown = (0..count).filter(|&i| population.plant(i).biomass_kg() > 1.0).count();
    assert!(grown * 4 >= count * 3, "{grown} of {count} grew");
    assert!(!world.built_plant_geometry(0).unwrap().stale);
    assert!(world.built_plant_geometry(0).unwrap().height_m > 1.0);
    assert_eq!(world.stale_geometry_count(), 0, "unloaded plants are not stale: they are not wanted");
    for i in 1..count {
        let built = world.built_plant_geometry(i).unwrap();
        assert_eq!(built.height_m, 0.0, "plant {i} outside the region got no geometry");
        assert!(built.stale, "plant {i} cached geometry is older than its state");
        assert!(population.plant(i).biomass_kg() as f32 > seed_biomass[i]);
    }
    // Load everything: the backlog is worked off in bounded chunks.
    world.set_loaded_region(None);
    assert_eq!(world.stale_geometry_count(), count - 1);
    let mut remaining = count - 1;
    let mut calls = 0;
    while world.stale_geometry_count() > 0 {
        let done = world.refresh_geometry_chunk(2);
        assert!((1..=2).contains(&done), "each call rebuilds at most its budget: {done}");
        let now = world.stale_geometry_count();
        assert_eq!(now, remaining - done);
        remaining = now;
        calls += 1;
        assert!(calls < 100);
    }
    assert!(calls >= count / 2);
    for i in 0..count {
        let built = world.built_plant_geometry(i).unwrap();
        assert!(!built.stale);
        let state = world.plant_population().unwrap().geometry(i);
        assert_eq!(built.height_m, state.height_m);
    }
}

#[test]
fn cutting_a_world_tree_splits_its_state_and_geometry_follows_in_a_chunk() {
    let mut world = resource_world(35.0);
    let population = world.plant_population().unwrap();
    let tallest = (0..world.plant_count())
        .max_by(|&a, &b| population.geometry(a).height_m.total_cmp(&population.geometry(b).height_m))
        .unwrap();
    let height = world.built_plant_geometry(tallest).unwrap().height_m;
    let before = world.plant_population().unwrap().plant(tallest).total_mass_kg();
    let reference = world.plant_population().unwrap().ledger().closed_mass_kg;
    let part = world.cut_plant(tallest, 0.75).expect("cut");
    let population = world.plant_population().unwrap();
    let (owner, severed) = population.severed_part(part);
    assert_eq!(owner, tallest);
    assert!((severed.total_mass_kg() + population.plant(tallest).total_mass_kg() - before).abs() < 1e-9 * before.max(1.0));
    assert!((population.ledger().closed_mass_kg - reference).abs() < 1e-9 * reference);
    let pools = severed.pools;
    assert!(world.built_plant_geometry(tallest).unwrap().stale);
    assert_eq!(world.stale_geometry_count(), 1);
    assert_eq!(world.refresh_geometry_chunk(8), 1);
    let after = world.built_plant_geometry(tallest).unwrap();
    assert!(after.height_m < height, "{} -> {}", height, after.height_m);
    // The severed part stops: five more years change nothing in its pools.
    world.advance_growth(5.0 * YEAR_S);
    world.sync_plants();
    let (_, later) = world.plant_population().unwrap().severed_part(part);
    assert_eq!(later.pools, pools);
    assert_ledger_closed(world.plant_population().unwrap());
}

#[test]
fn the_growth_scheduler_replays_increments_for_the_resource_model() {
    let mut expected = GeneratedWorld::from_seed_with_model("1001", GrowthModel::ResourceBased);
    expected.advance_growth(0.24 * YEAR_S);
    let mut actual = expected.clone();
    let mut scheduler = GrowthScheduler::default();
    for years in [0.02, 0.001, 0.03, 0.24, 0.005, 0.6] {
        let elapsed = years * YEAR_S;
        expected.advance_growth(elapsed);
        scheduler.advance(&mut actual, elapsed);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while scheduler.is_busy() {
        assert!(std::time::Instant::now() < deadline, "growth worker did not finish");
        scheduler.advance(&mut actual, 0.0);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    expected.sync_plants();
    actual.sync_plants();
    assert_eq!(actual.age_years.to_bits(), expected.age_years.to_bits());
    assert_eq!(actual.render_cache_key(), expected.render_cache_key());
    assert_eq!(actual.voxels.len(), expected.voxels.len());
    assert_eq!(state_bits(actual.plant_population().unwrap()), state_bits(expected.plant_population().unwrap()));
}

// ---------------------------------------------------------------------------------
// Performance probe
// ---------------------------------------------------------------------------------

fn forest(side: usize, spacing: f64) -> Vec<SiteInput> {
    let mut sites = Vec::with_capacity(side * side);
    for gy in 0..side {
        for gx in 0..side {
            let id = (gy * side + gx) as u64 + 1;
            let jitter = |salt: u64| (unit_hash(5, id, salt) - 0.5) * 0.6 * spacing;
            sites.push(SiteInput {
                moisture: 0.3 + 0.5 * unit_hash(5, id, 90),
                fertility: 0.3 + 0.6 * unit_hash(5, id, 91),
                temperature_k: 284.0 + 8.0 * unit_hash(5, id, 92),
                ..varied_site(id, gx as f64 * spacing + jitter(1), gy as f64 * spacing + jitter(2), 0.0)
            });
        }
    }
    sites
}

/// 10 000 plants for one year, single-threaded, debug-profile (opt-level 2) build.
/// Run with: cargo test --no-default-features --lib plant_world::tests::ten_thousand -- --ignored --nocapture
#[test]
#[ignore = "performance probe, a few seconds"]
fn ten_thousand_plants_advance_one_year() {
    let mut results = Vec::new();
    for side in [50usize, 100] {
        let sites = forest(side, 6.0);
        let build = std::time::Instant::now();
        let mut pop = PlantPopulation::new(1, &sites, PopulationConfig::default());
        let build_ms = build.elapsed().as_secs_f64() * 1e3;
        let start = std::time::Instant::now();
        pop.advance_years(1.0);
        let wall = start.elapsed().as_secs_f64();
        assert_ledger_closed(&pop);
        println!(
            "{} plants, {} soil cells: build {build_ms:.1} ms, advance 1 y {:.3} s = {:.1} us per plant-year, biomass {:.1} kg",
            sites.len(),
            pop.soil_count(),
            wall,
            wall * 1e6 / sites.len() as f64,
            pop.total_biomass_kg()
        );
        results.push((sites.len(), wall));
    }
    // No hidden quadratic loop: 4x the plants must cost about 4x (allow 6x).
    let ratio = results[1].1 / results[0].1;
    println!("time ratio for 4x plants: {ratio:.2}");
    assert!(ratio < 6.0, "scaling ratio {ratio}");
    // The same 10 000 plants on the rayon pool give the same answer.
    let sites = forest(100, 6.0);
    let mut parallel = PlantPopulation::new(1, &sites, PopulationConfig { parallel: true, ..PopulationConfig::default() });
    let start = std::time::Instant::now();
    parallel.advance_years(1.0);
    println!("10000 plants, rayon pool: {:.3} s", start.elapsed().as_secs_f64());
    let mut serial = PlantPopulation::new(1, &sites, PopulationConfig::default());
    serial.advance_years(1.0);
    assert_eq!(state_bits(&serial), state_bits(&parallel));
}
