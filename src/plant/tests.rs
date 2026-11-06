use super::*;

const ALL_HABITS: [HabitPreset; 5] = [
    HabitPreset::Tree,
    HabitPreset::Shrub,
    HabitPreset::Grass,
    HabitPreset::Vine,
    HabitPreset::Mushroom,
];

fn rich_env() -> Environment {
    Environment::new(2.0e5, 5.0e3)
}

fn seed_mass(habit: HabitPreset) -> f64 {
    match habit {
        HabitPreset::Tree => 0.05,
        HabitPreset::Shrub => 0.02,
        HabitPreset::Grass => 0.002,
        HabitPreset::Vine => 0.01,
        HabitPreset::Mushroom => 0.0005,
    }
}

fn sprout(habit: HabitPreset, env: &mut Environment) -> Plant {
    Plant::from_seed(PlantGenome::preset(habit, 1), seed_mass(habit), 0.0, env)
}

/// Seed germinating at the start of the warm season of `Climate::seasonal(.., 90.0, 200.0)`.
fn sprout_in_spring(habit: HabitPreset, env: &mut Environment) -> Plant {
    Plant::from_seed(PlantGenome::preset(habit, 1), seed_mass(habit), 90.0, env)
}

fn closed_mass(env: &Environment, plants: &[&Plant], parts: &[&SeveredPart]) -> f64 {
    let mass: f64 = plants.iter().map(|p| p.total_mass_kg()).sum::<f64>()
        + parts.iter().map(|p| p.total_mass_kg()).sum::<f64>();
    env.closed_mass(mass)
}

fn assert_closed(env: &Environment, plants: &[&Plant], parts: &[&SeveredPart], reference: f64) {
    let drift = closed_mass(env, plants, parts) - reference;
    assert!(drift.abs() <= 1e-12 * reference.abs().max(1.0), "mass drift {drift} kg on {reference} kg");
}

fn assert_energy_closed(env: &Environment, plants: &[&Plant], parts: &[&SeveredPart]) {
    let chemical: f64 = plants.iter().map(|p| p.chemical_energy_j()).sum::<f64>()
        + parts.iter().map(|p| p.chemical_energy_j()).sum::<f64>();
    let residual = env.energy_residual(chemical);
    let scale = env.radiation_in_j.max(env.imported_energy_j).max(1.0);
    assert!(residual.abs() <= 1e-9 * scale, "energy residual {residual} J of {scale} J");
}

fn assert_rings_match_pools(plant: &Plant) {
    let ring_mass: f64 = plant.rings().iter().map(|r| r.mass_kg).sum();
    let wood = plant.pools.trunk + plant.pools.branches;
    assert!((ring_mass - wood).abs() <= 1e-9 * wood.max(1e-9), "rings {ring_mass} vs wood {wood}");
}

// ---------------------------------------------------------------------------------
// Conservation
// ---------------------------------------------------------------------------------

#[test]
fn mass_and_energy_are_conserved_through_seasons_and_shedding() {
    let mut env = rich_env();
    let mut plant = sprout_in_spring(HabitPreset::Tree, &mut env);
    let reference = closed_mass(&env, &[&plant], &[]);
    let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 90.0, 200.0);
    for _ in 0..(4 * 365) {
        plant.advance(&mut env, &climate, 1.0);
    }
    assert_closed(&env, &[&plant], &[], reference);
    assert_energy_closed(&env, &[&plant], &[]);
    assert!(plant.biomass_kg() > 0.1, "the plant should have grown: {}", plant.biomass_kg());
    assert!(env.litter_organic_kg > 0.0, "deciduous leaf loss must reach the litter ledger");
    assert!(env.radiation_in_j > env.heat_out_j && env.heat_out_j > 0.0);
    assert_rings_match_pools(&plant);
    println!("mass drift after 4 years: {:e} kg of {:e}", closed_mass(&env, &[&plant], &[]) - reference, reference);
}

#[test]
fn every_habit_conserves_mass_and_energy() {
    for habit in ALL_HABITS {
        let mut env = rich_env();
        env.add_litter(40.0, 1.0);
        let mut plant = sprout(habit, &mut env);
        let reference = closed_mass(&env, &[&plant], &[]);
        let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 60.0, 220.0);
        plant.advance(&mut env, &climate, 3.0 * YEAR_DAYS);
        assert_closed(&env, &[&plant], &[], reference);
        assert_energy_closed(&env, &[&plant], &[]);
        assert_rings_match_pools(&plant);
        assert!(plant.stores.sugar >= 0.0 && plant.stores.water >= 0.0 && plant.stores.minerals >= 0.0, "{habit:?} negative store");
    }
}

#[test]
fn seed_mass_is_imported_and_biomass_comes_from_air_and_soil() {
    let mut env = rich_env();
    let co2_before = env.co2_kg;
    let soil_before = env.soil_water_kg + env.soil_minerals_kg;
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    assert!((env.imported_mass_kg - 0.05).abs() < 1e-15);
    let start = plant.total_mass_kg();
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 2.0 * YEAR_DAYS);
    assert!(plant.total_mass_kg() > 3.0 * start);
    assert!(env.co2_kg < co2_before, "carbon must come from the air ledger");
    assert!(env.soil_water_kg + env.soil_minerals_kg < soil_before, "water and minerals come from the soil");
}

// ---------------------------------------------------------------------------------
// Events instead of frames
// ---------------------------------------------------------------------------------

#[test]
fn ten_year_event_step_matches_3650_daily_steps() {
    let climate = Climate::constant(Conditions::SUMMER);
    let mut env_long = rich_env();
    let mut long = sprout(HabitPreset::Tree, &mut env_long);
    let reference_long = closed_mass(&env_long, &[&long], &[]);
    let report_long = long.advance(&mut env_long, &climate, 10.0 * YEAR_DAYS);

    let mut env_daily = rich_env();
    let mut daily = sprout(HabitPreset::Tree, &mut env_daily);
    let reference_daily = closed_mass(&env_daily, &[&daily], &[]);
    let mut substeps_daily = 0;
    for _ in 0..3650 {
        substeps_daily += daily.advance(&mut env_daily, &climate, 1.0).substeps;
    }

    assert_closed(&env_long, &[&long], &[], reference_long);
    assert_closed(&env_daily, &[&daily], &[], reference_daily);
    assert_energy_closed(&env_long, &[&long], &[]);
    assert_energy_closed(&env_daily, &[&daily], &[]);
    assert!((long.time_days - 3650.0).abs() < 1e-9 && (daily.time_days - 3650.0).abs() < 1e-6);
    let (a, b) = (long.biomass_kg(), daily.biomass_kg());
    println!(
        "10 y: long-step biomass {a:.4} kg in {} substeps, daily biomass {b:.4} kg in {substeps_daily} substeps; mass drift {:e} / {:e}",
        report_long.substeps,
        closed_mass(&env_long, &[&long], &[]) - reference_long,
        closed_mass(&env_daily, &[&daily], &[]) - reference_daily
    );
    assert!(a > 5.0 && b > 5.0, "both runs must actually grow: {a} {b}");
    assert!(((a - b) / b).abs() < 0.10, "biomass long {a} vs daily {b}");
    for (x, y) in [
        (long.pools.roots, daily.pools.roots),
        (long.pools.trunk, daily.pools.trunk),
        (long.pools.leaves, daily.pools.leaves),
    ] {
        assert!(((x - y) / y).abs() < 0.15, "pool {x} vs {y}");
    }
    assert!(report_long.substeps * 3 < substeps_daily, "{} vs {}", report_long.substeps, substeps_daily);
}

#[test]
fn a_long_span_is_integrated_not_skipped() {
    // Same total span in one call and in 40 calls: nearly the same plant.
    let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 90.0, 200.0);
    let mut env_a = rich_env();
    let mut a = sprout_in_spring(HabitPreset::Shrub, &mut env_a);
    a.advance(&mut env_a, &climate, 1600.0);
    let mut env_b = rich_env();
    let mut b = sprout_in_spring(HabitPreset::Shrub, &mut env_b);
    for _ in 0..40 {
        b.advance(&mut env_b, &climate, 40.0);
    }
    println!("long span: {} vs {}", a.biomass_kg(), b.biomass_kg());
    assert!(a.biomass_kg() > 0.3);
    assert!(((a.biomass_kg() - b.biomass_kg()) / b.biomass_kg()).abs() < 0.10, "{} vs {}", a.biomass_kg(), b.biomass_kg());
    assert!(a.age_days() > 1599.0 && (a.age_days() - b.age_days()).abs() < 1e-6);
}

#[test]
fn events_are_reported_at_their_times_and_do_not_depend_on_step_size() {
    let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 90.0, 200.0);
    let mut env_a = rich_env();
    let mut a = sprout(HabitPreset::Grass, &mut env_a);
    let report = a.advance(&mut env_a, &climate, 2.0 * YEAR_DAYS);
    let seasons: Vec<f64> = report.events.iter().filter(|e| e.1 == EventKind::SeasonChange).map(|e| e.0).collect();
    assert_eq!(seasons.len(), 4, "{seasons:?}");
    for (got, want) in seasons.iter().zip([90.0, 290.0, 455.0, 655.0]) {
        assert!((got - want).abs() < 1e-6, "{seasons:?}");
    }
    let maturity: Vec<f64> = report.events.iter().filter(|e| e.1 == EventKind::Maturation).map(|e| e.0).collect();
    assert_eq!(maturity.len(), 1);
    assert!((maturity[0] - (0.2_f32 as f64) * 3.0 * YEAR_DAYS).abs() < 1e-6, "{maturity:?}");
    assert!(a.is_mature());

    // The same events when the span is cut into pieces.
    let mut env_b = rich_env();
    let mut b = sprout(HabitPreset::Grass, &mut env_b);
    let mut pieces = Vec::new();
    for _ in 0..73 {
        pieces.extend(b.advance(&mut env_b, &climate, 10.0).events);
    }
    let seasons_b: Vec<f64> = pieces.iter().filter(|e| e.1 == EventKind::SeasonChange).map(|e| e.0).collect();
    assert_eq!(seasons_b.len(), 4);
    for (x, y) in seasons.iter().zip(&seasons_b) {
        assert!((x - y).abs() < 1e-6);
    }
}

#[test]
fn branching_orders_are_events_timed_by_the_integration() {
    let mut env = rich_env();
    let mut shrub = sprout(HabitPreset::Shrub, &mut env);
    let report = shrub.advance(&mut env, &Climate::constant(Conditions::SUMMER), 3.0 * YEAR_DAYS);
    let times: Vec<f64> = report.events.iter().filter(|e| e.1 == EventKind::Branching).map(|e| e.0).collect();
    assert!(!times.is_empty() && report.new_branches as usize >= times.len());
    assert!(times.windows(2).all(|w| w[0] <= w[1]), "{times:?}");
    assert_eq!(shrub.morphology().branch_count, report.new_branches, "every new order was reported");
    assert!(shrub.morphology().branch_count > 0);
}

#[test]
fn next_event_time_is_state_and_matches_the_climate() {
    let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 90.0, 200.0);
    assert_eq!(climate.at(10.0), Conditions::WINTER);
    assert_eq!(climate.at(100.0), Conditions::SUMMER);
    assert!((climate.next_change_after(0.0) - 90.0).abs() < 1e-9);
    assert!((climate.next_change_after(90.0) - 290.0).abs() < 1e-9);
    assert!((climate.next_change_after(300.0) - 455.0).abs() < 1e-9);
    assert!(Climate::constant(Conditions::SUMMER).next_change_after(5.0).is_infinite());

    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    plant.advance(&mut env, &climate, 40.0);
    assert!(plant.next_event_day > 40.0 && plant.next_event_day <= 90.0 + 1e-9, "{}", plant.next_event_day);
    let (time, _kind) = plant.next_event(&env, &climate, true);
    assert!((time - plant.next_event_day).abs() < 1e-9);
}

// ---------------------------------------------------------------------------------
// Light, water, dormancy
// ---------------------------------------------------------------------------------

#[test]
fn a_shaded_plant_grows_slower_than_a_lit_one() {
    let mut results = Vec::new();
    for light in [1.0, 0.5, 0.15] {
        let mut env = rich_env();
        let mut plant = sprout(HabitPreset::Tree, &mut env);
        let reference = closed_mass(&env, &[&plant], &[]);
        plant.advance(&mut env, &Climate::constant(Conditions::SUMMER.with_light_fraction(light)), 4.0 * YEAR_DAYS);
        assert_closed(&env, &[&plant], &[], reference);
        results.push(plant.biomass_kg());
    }
    println!("biomass after 4 y at light 1.0/0.5/0.15: {results:?}");
    assert!(results[0] > results[1] && results[1] > results[2], "{results:?}");
    assert!(results[2] < 0.6 * results[0], "{results:?}");
    assert!(results[2] > 0.0);
}

#[test]
fn shade_tolerance_reduces_the_cost_of_shade() {
    let mut tolerant = PlantGenome::preset(HabitPreset::Tree, 3);
    tolerant.seed.shade_tolerance = 0.9;
    let mut intolerant = tolerant;
    intolerant.seed.shade_tolerance = 0.1;
    let mut out = Vec::new();
    for genome in [tolerant, intolerant] {
        let mut env = rich_env();
        let mut plant = Plant::from_seed(genome, 0.05, 0.0, &mut env);
        plant.advance(&mut env, &Climate::constant(Conditions::SUMMER.with_light_fraction(0.1)), 4.0 * YEAR_DAYS);
        out.push(plant.biomass_kg());
    }
    assert!(out[0] > out[1], "tolerant {} vs intolerant {}", out[0], out[1]);
}

#[test]
fn drought_stalls_growth_and_mass_is_still_conserved() {
    // A tree grows for three years with water, then the soil is drained to the
    // wilting range and no rain falls: it stalls. The control keeps its water.
    let climate = Climate::constant(Conditions::SUMMER);
    let mut env = Environment::new(2.0e5, 5.0e3);
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    plant.advance(&mut env, &climate, 3.0 * YEAR_DAYS);
    let mut control = plant.clone();
    let mut control_env = env.clone();

    env.drain_soil_water(env.soil_water_kg - 0.055 * env.soil_water_capacity_kg);
    let reference = closed_mass(&env, &[&plant], &[]);
    let wilting_kg = 0.05 * env.soil_water_capacity_kg;
    let before = plant.biomass_kg();
    let leaves_before = plant.pools.leaves;
    plant.advance(&mut env, &climate, 400.0);
    control.advance(&mut control_env, &climate, 400.0);
    let after = plant.biomass_kg();
    assert_closed(&env, &[&plant], &[], reference);
    assert_energy_closed(&env, &[&plant], &[]);
    println!(
        "drought: biomass {before:.4} -> {after:.4} (watered control {:.4}), leaves {leaves_before:.4} -> {:.4}, damage {:.3}",
        control.biomass_kg(),
        plant.pools.leaves,
        plant.damage
    );
    assert!(after <= before * 1.02, "the stalled plant does not keep growing: {before} -> {after}");
    assert!(control.biomass_kg() > 1.4 * before, "the watered plant does grow");
    assert!(plant.biomass_kg() < 0.7 * control.biomass_kg());
    assert!(env.soil_water_kg >= wilting_kg - 1e-6, "roots cannot take water below the wilting point");
    assert!(plant.pools.leaves < leaves_before, "drought sheds leaves");
    assert!(plant.damage > 0.0 || plant.stores.water < 0.5 * plant.water_capacity());
}

#[test]
fn rain_restarts_a_stalled_plant() {
    let climate = Climate::constant(Conditions::SUMMER);
    let mut env = Environment::new(2.0e5, 5.0e3);
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    plant.advance(&mut env, &climate, 2.0 * YEAR_DAYS);
    env.drain_soil_water(env.soil_water_kg - 0.055 * env.soil_water_capacity_kg);
    plant.advance(&mut env, &climate, 12.0);
    assert!(!plant.is_dead());
    println!("before rain: leaves {} sugar {} damage {} water {}", plant.pools.leaves, plant.stores.sugar, plant.damage, plant.stores.water);
    let stalled = plant.biomass_kg();
    let reference = closed_mass(&env, &[&plant], &[]);
    env.add_rain(1.9e5); // an import: the invariant subtracts it
    plant.advance(&mut env, &climate, YEAR_DAYS);
    assert!(plant.biomass_kg() > 1.3 * stalled, "{stalled} -> {}", plant.biomass_kg());
    assert_closed(&env, &[&plant], &[], reference);
}

#[test]
fn no_light_means_no_photosynthesis_and_the_plant_lives_on_reserves_then_dies() {
    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Shrub, &mut env);
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 2.0 * YEAR_DAYS);
    let reference = closed_mass(&env, &[&plant], &[]);
    let dark = Climate::constant(Conditions::SUMMER.with_light_fraction(0.0));
    let radiation = env.radiation_in_j;
    let sugar_start = plant.stores.sugar;
    let mut last_sugar = sugar_start;
    for _ in 0..20 {
        plant.advance(&mut env, &dark, 3.0);
        assert!(plant.stores.sugar <= last_sugar + 1e-15, "sugar must not rise in the dark");
        last_sugar = plant.stores.sugar;
    }
    assert_eq!(env.radiation_in_j, radiation, "no light is absorbed");
    assert!(plant.stores.sugar < sugar_start);
    assert!(!plant.is_dead(), "reserves carry the plant for a while");
    plant.advance(&mut env, &dark, 3.0 * YEAR_DAYS);
    assert!(plant.is_dead(), "damage {}", plant.damage);
    assert_eq!(plant.vitality(), 0.0);
    assert_closed(&env, &[&plant], &[], reference);
    assert_energy_closed(&env, &[&plant], &[]);
}

#[test]
fn dormancy_is_derived_from_conditions_and_stops_growth() {
    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), YEAR_DAYS);
    assert!(!plant.is_dormant(&Conditions::SUMMER));
    assert!(plant.is_dormant(&Conditions::WINTER));
    let leaves = plant.pools.leaves;
    let biomass = plant.biomass_kg();
    let reference = closed_mass(&env, &[&plant], &[]);
    plant.advance(&mut env, &Climate::constant(Conditions::WINTER), 60.0);
    assert!(plant.biomass_kg() < biomass, "no new tissue in winter, only loss");
    assert!(plant.pools.leaves < 0.2 * leaves, "a deciduous plant sheds its leaves: {} -> {}", leaves, plant.pools.leaves);
    assert_closed(&env, &[&plant], &[], reference);
    // Spring: growth resumes from the reserves with no stored flag to clear.
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 120.0);
    assert!(plant.pools.leaves > 0.0);
}

#[test]
fn water_scarcity_moves_allocation_toward_roots() {
    let climate = Climate::constant(Conditions::SUMMER);
    let mut shares = Vec::new();
    for water in [2.0e5, 600.0] {
        let mut env = Environment::new(water, 5.0e3);
        env.soil_water_capacity_kg = 2.0e5;
        let mut plant = sprout(HabitPreset::Shrub, &mut env);
        plant.advance(&mut env, &climate, 400.0);
        shares.push(plant.pools.roots / plant.pools.total());
    }
    println!("root share, wet vs dry: {shares:?}");
    assert!(shares[1] > shares[0], "{shares:?}");
}

#[test]
fn neighbours_compete_for_one_finite_soil() {
    let climate = Climate::constant(Conditions::SUMMER);
    let make_env = || {
        let mut env = Environment::new(400.0, 5.0e3);
        env.soil_water_capacity_kg = 1000.0;
        env
    };
    let mut alone_env = make_env();
    let mut alone = sprout(HabitPreset::Shrub, &mut alone_env);
    alone.advance(&mut alone_env, &climate, 3.0 * YEAR_DAYS);

    let mut env = make_env();
    let mut a = sprout(HabitPreset::Shrub, &mut env);
    let mut b = Plant::from_seed(PlantGenome::preset(HabitPreset::Shrub, 2), 0.02, 0.0, &mut env);
    let reference = closed_mass(&env, &[&a, &b], &[]);
    for _ in 0..(3 * 73) {
        a.advance(&mut env, &climate, 5.0);
        b.advance(&mut env, &climate, 5.0);
    }
    assert_closed(&env, &[&a, &b], &[], reference);
    assert!(env.soil_water_kg >= 50.0 - 1e-6, "the soil is not drawn below the wilting point");
    println!("alone {:.3} kg, with a neighbour {:.3} + {:.3} kg", alone.biomass_kg(), a.biomass_kg(), b.biomass_kg());
    assert!(a.biomass_kg() < 0.8 * alone.biomass_kg(), "a shared soil gives each plant less");
    assert!(b.biomass_kg() < 0.8 * alone.biomass_kg());
    assert!(a.biomass_kg() + b.biomass_kg() < 1.6 * alone.biomass_kg(), "the two together do not double the harvest");
}

// ---------------------------------------------------------------------------------
// Forms from the genome
// ---------------------------------------------------------------------------------

#[derive(Debug)]
struct Form {
    density: f64,
    slenderness: f64,
    leaf_fraction: f64,
    leaf_to_stem: f64,
    root_fraction: f64,
    height: f64,
    leaves: f64,
}

fn grow_form(habit: HabitPreset) -> Form {
    let mut env = rich_env();
    env.add_litter(300.0, 6.0);
    let mut plant = sprout(habit, &mut env);
    let reference = closed_mass(&env, &[&plant], &[]);
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 2.5 * YEAR_DAYS);
    assert_closed(&env, &[&plant], &[], reference);
    let morph = plant.morphology();
    let total = plant.pools.total();
    Form {
        density: plant.wood_density(),
        slenderness: if morph.trunk_diameter_m > 0.0 { morph.height_m / morph.trunk_diameter_m } else { 0.0 },
        leaf_fraction: plant.pools.leaves / total,
        leaf_to_stem: plant.pools.leaves / (plant.pools.trunk + plant.pools.branches).max(1e-12),
        root_fraction: plant.pools.roots / total,
        height: morph.height_m,
        leaves: plant.pools.leaves,
    }
}

#[test]
fn different_habit_genes_give_different_forms_under_identical_conditions() {
    let forms: Vec<(HabitPreset, Form)> = ALL_HABITS.iter().map(|h| (*h, grow_form(*h))).collect();
    for (habit, form) in &forms {
        println!("{habit:?}: {form:?}");
    }
    let get = |h: HabitPreset| &forms.iter().find(|f| f.0 == h).unwrap().1;
    let (tree, shrub, grass, vine, mushroom) = (
        get(HabitPreset::Tree),
        get(HabitPreset::Shrub),
        get(HabitPreset::Grass),
        get(HabitPreset::Vine),
        get(HabitPreset::Mushroom),
    );
    // Wood: lignified plants lay dense tissue, herbs and fungi do not.
    assert!(tree.density > shrub.density && shrub.density > vine.density && vine.density > grass.density);
    assert!(grass.density >= mushroom.density);
    // Slenderness: climbing vines are far more slender than self-supporting trees.
    assert!(vine.slenderness > tree.slenderness && tree.slenderness > mushroom.slenderness);
    assert!(vine.slenderness > 1.5 * tree.slenderness, "{} vs {}", vine.slenderness, tree.slenderness);
    // Allocation: grass is mostly leaf, a mushroom has none and is mostly mycelium.
    assert!(grass.leaf_to_stem > 3.0 * tree.leaf_to_stem, "{} vs {}", grass.leaf_to_stem, tree.leaf_to_stem);
    assert_eq!(mushroom.leaves, 0.0);
    assert!(mushroom.root_fraction > tree.root_fraction && mushroom.root_fraction > grass.root_fraction);
    // Size: the habit genes cap the height of each form.
    assert!(grass.height < 0.7 && mushroom.height < 0.2 && vine.height > grass.height && tree.height > grass.height);
    // No two forms coincide in the descriptor space.
    for i in 0..forms.len() {
        for j in (i + 1)..forms.len() {
            let (a, b) = (&forms[i].1, &forms[j].1);
            let distance = ((a.density - b.density) / 500.0).powi(2)
                + ((a.slenderness - b.slenderness) / 100.0).powi(2)
                + (a.leaf_fraction - b.leaf_fraction).powi(2)
                + (a.root_fraction - b.root_fraction).powi(2);
            assert!(distance.sqrt() > 0.05, "{:?} and {:?} are too similar", forms[i].0, forms[j].0);
        }
    }
}

#[test]
fn a_mushroom_like_form_lives_on_soil_organic_matter_and_not_on_light() {
    let climate = Climate::constant(Conditions::SUMMER.with_light_fraction(0.0));
    let mut env = rich_env();
    env.add_litter(40.0, 0.8);
    let mut fungus = sprout(HabitPreset::Mushroom, &mut env);
    let reference = closed_mass(&env, &[&fungus], &[]);
    let litter = env.litter_organic_kg;
    fungus.advance(&mut env, &climate, YEAR_DAYS);
    assert!(fungus.biomass_kg() > 100.0 * 0.0005, "grown from litter: {}", fungus.biomass_kg());
    assert!(env.litter_organic_kg < litter, "carbon comes out of the litter ledger");
    assert_eq!(env.radiation_in_j, 0.0);
    assert_closed(&env, &[&fungus], &[], reference);
    assert_energy_closed(&env, &[&fungus], &[]);
    // The same genes in an environment without litter starve.
    let mut bare = rich_env();
    let mut starving = sprout(HabitPreset::Mushroom, &mut bare);
    starving.advance(&mut bare, &climate, 2.0 * YEAR_DAYS);
    assert!(starving.biomass_kg() < 0.3 * fungus.biomass_kg());
}

#[test]
fn the_habit_space_is_continuous_between_tree_and_grass() {
    let tree = PlantGenome::preset(HabitPreset::Tree, 1);
    let grass = PlantGenome::preset(HabitPreset::Grass, 2);
    let mut densities = Vec::new();
    for step in 0..=4 {
        let w = step as f64 / 4.0;
        let (a, b) = (tree.genes(), grass.genes());
        let mut mix = a;
        for i in 0..GENE_COUNT {
            mix[i] = a[i] * (1.0 - w) + b[i] * w;
        }
        let genome = tree.with_genes(&mix);
        assert!(genome.in_bounds());
        densities.push(Params::from_genome(&genome).base_density);
    }
    assert!(densities.windows(2).all(|p| p[0] > p[1]), "{densities:?}");
}

// ---------------------------------------------------------------------------------
// Wood from state
// ---------------------------------------------------------------------------------

#[test]
fn wood_density_and_stiffness_follow_the_growth_history() {
    let climate_for = |light: f64| Climate::constant(Conditions::SUMMER.with_light_fraction(light));
    let grow = |genome: PlantGenome, light: f64| {
        let mut env = rich_env();
        let mut plant = Plant::from_seed(genome, 0.05, 0.0, &mut env);
        plant.advance(&mut env, &climate_for(light), 5.0 * YEAR_DAYS);
        plant
    };
    let ring_porous = PlantGenome::preset(HabitPreset::Tree, 1);
    let mut conifer = ring_porous;
    conifer.habit.ring_response = -0.15;

    let lit = grow(ring_porous, 1.0);
    let shaded = grow(ring_porous, 0.3);
    // Positive response: faster growth, denser wood, stiffer wood.
    let (wl, ws) = (lit.wood_properties(), shaded.wood_properties());
    println!("ring-porous lit {wl:?}\nring-porous shaded {ws:?}");
    assert!(lit.biomass_kg() > shaded.biomass_kg());
    assert!(wl.density_kg_m3 > ws.density_kg_m3, "{} vs {}", wl.density_kg_m3, ws.density_kg_m3);
    assert!(wl.young_modulus_pa > ws.young_modulus_pa);
    // Opposite sign of the gene reverses the dependence on growth rate.
    let lit_conifer = grow(conifer, 1.0);
    let shaded_conifer = grow(conifer, 0.3);
    assert!(lit_conifer.wood_density() < shaded_conifer.wood_density());
    // Stiffness is a function of density through one law with the oak reference point.
    assert!((young_modulus_from_density(625.0) - 12.088e9).abs() < 1.0);
    assert!(young_modulus_from_density(700.0) > young_modulus_from_density(500.0));
    assert!((wl.young_modulus_pa - young_modulus_from_density(wl.density_kg_m3)).abs() < 1.0);
    // Rings: one layer per year of growth, densities differ between years, EI positive.
    assert!(wl.ring_count >= 4 && wl.ring_count <= 6, "{}", wl.ring_count);
    let ring_densities: Vec<f64> = lit.rings().iter().filter(|r| r.mass_kg > 0.0).map(|r| r.density()).collect();
    assert!(ring_densities.windows(2).any(|p| (p[0] - p[1]).abs() > 1.0), "{ring_densities:?}");
    assert!(wl.bending_stiffness_ei > 0.0);
    let d = wl.trunk_diameter_m;
    let uniform = young_modulus_from_density(wl.density_kg_m3) * std::f64::consts::PI * d.powi(4) / 64.0;
    assert!((wl.bending_stiffness_ei / uniform - 1.0).abs() < 0.2, "{} vs {uniform}", wl.bending_stiffness_ei);
    assert_rings_match_pools(&lit);
    // Engine material: oak profile with this plant's density and Young modulus.
    let table = Material::table();
    let oak = &table[Material::component_id("oak").unwrap() as usize];
    let derived = wl.to_material(oak);
    assert_eq!(derived.density, wl.density_kg_m3 as f32);
    assert_eq!(derived.young, wl.young_modulus_pa as f32);
    assert_eq!(derived.name, oak.name);
}

#[test]
fn stem_height_is_limited_by_the_buckling_of_the_wood_that_was_built() {
    // Equal stem volume, different wood: stiffer, denser wood carries a taller stem.
    let (h_herb, _) = stem_geometry(0.01, 160.0, 400.0, 0.0);
    let (h_wood, _) = stem_geometry(0.01, 650.0, 400.0, 0.0);
    let (h_climb, _) = stem_geometry(0.01, 160.0, 400.0, 1.0);
    assert!(h_wood > h_herb);
    assert!(h_climb > h_herb, "a climber is not limited by buckling");
    // Slender enough genes are not changed by the limit.
    let (h, d) = stem_geometry(0.001, 650.0, 10.0, 0.0);
    assert!((h / d - 10.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------------
// Cutting
// ---------------------------------------------------------------------------------

#[test]
fn cutting_splits_the_pools_conserves_mass_and_stops_the_severed_part() {
    let climate = Climate::constant(Conditions::SUMMER);
    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Tree, &mut env);
    plant.advance(&mut env, &climate, 6.0 * YEAR_DAYS);
    let reference = closed_mass(&env, &[&plant], &[]);
    let before = plant.pools;
    let stores_before = plant.stores;
    let mass_before = plant.total_mass_kg();
    let energy_before = plant.chemical_energy_j();

    let mut part = plant.cut_above(0.6);

    // Split by mass fraction, exactly.
    assert!((part.pools.trunk - 0.6 * before.trunk).abs() < 1e-12);
    assert!((plant.pools.trunk - 0.4 * before.trunk).abs() < 1e-12);
    assert!((part.pools.leaves + plant.pools.leaves - before.leaves).abs() < 1e-12);
    assert_eq!(part.pools.roots, 0.0);
    assert_eq!(plant.pools.roots, before.roots);
    assert!((part.stores.sugar + plant.stores.sugar - stores_before.sugar).abs() < 1e-12);
    assert!((part.stores.water + plant.stores.water - stores_before.water).abs() < 1e-12);
    assert!((part.total_mass_kg() + plant.total_mass_kg() - mass_before).abs() < 1e-12 * mass_before);
    assert!((part.chemical_energy_j() + plant.chemical_energy_j() - energy_before).abs() < 1e-9 * energy_before);
    assert_closed(&env, &[&plant], &[&part], reference);
    assert_rings_match_pools(&plant);
    let ring_mass: f64 = part.rings().iter().map(|r| r.mass_kg).sum();
    assert!((ring_mass - (part.pools.trunk + part.pools.branches)).abs() < 1e-9);
    assert!((part.wood_density() - before_density(&plant, &before)).abs() < 1e-6 || part.wood_density() > 0.0);

    // The severed part keeps its mass: no water, no light, no new tissue.
    assert!(!part.can_take_up_water());
    let part_mass = part.total_mass_kg();
    let part_pools = part.pools;
    let water = part.stores.water;
    let mut last_sugar = part.stores.sugar;
    for _ in 0..60 {
        part.advance(&mut env, &Conditions::SUMMER, 1.0);
        assert!(part.stores.sugar <= last_sugar + 1e-15);
        last_sugar = part.stores.sugar;
    }
    assert_eq!(part.pools, part_pools);
    assert_eq!(part.stores.water, water);
    assert!(part.total_mass_kg() <= part_mass);
    assert!(part.total_mass_kg() > 0.95 * part_mass, "the mass is kept");
    let radiation = env.radiation_in_j;
    assert_closed(&env, &[&plant], &[&part], reference);
    assert_energy_closed(&env, &[&plant], &[&part]);

    // The rooted remainder goes on living and regrows from roots and reserves.
    let remaining = plant.biomass_kg();
    plant.advance(&mut env, &climate, 2.0 * YEAR_DAYS);
    assert!(!plant.is_dead());
    assert!(plant.biomass_kg() > remaining);
    assert!(env.radiation_in_j > radiation);
    assert_closed(&env, &[&plant], &[&part], reference);
    assert_energy_closed(&env, &[&plant], &[&part]);
}

fn before_density(plant: &Plant, _before: &Pools) -> f64 {
    plant.wood_density()
}

#[test]
fn cutting_everything_above_ground_leaves_a_root_stump() {
    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Shrub, &mut env);
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 3.0 * YEAR_DAYS);
    let reference = closed_mass(&env, &[&plant], &[]);
    let part = plant.cut_above(1.0);
    assert_eq!(plant.pools.trunk + plant.pools.branches + plant.pools.leaves + plant.pools.fruit, 0.0);
    assert!(plant.pools.roots > 0.0);
    assert_eq!(plant.morphology().height_m, 0.0);
    assert_closed(&env, &[&plant], &[&part], reference);
}

// ---------------------------------------------------------------------------------
// Determinism and genome bounds
// ---------------------------------------------------------------------------------

fn state_bits(plant: &Plant) -> Vec<u64> {
    let mut bits: Vec<u64> = [
        plant.stores.sugar,
        plant.stores.water,
        plant.stores.minerals,
        plant.pools.roots,
        plant.pools.trunk,
        plant.pools.branches,
        plant.pools.leaves,
        plant.pools.fruit,
        plant.damage,
        plant.time_days,
        plant.next_event_day,
    ]
    .iter()
    .map(|x| x.to_bits())
    .collect();
    bits.extend(plant.rings().iter().flat_map(|r| [r.mass_kg.to_bits(), r.volume_m3.to_bits()]));
    bits
}

#[test]
fn same_genome_and_conditions_give_the_same_plant() {
    let climate = Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 90.0, 200.0);
    let run = |genome: PlantGenome| {
        let mut env = rich_env();
        let mut plant = Plant::from_seed(genome, 0.05, 0.0, &mut env);
        plant.advance(&mut env, &climate, 3.0 * YEAR_DAYS);
        (state_bits(&plant), env.heat_out_j.to_bits())
    };
    let genome = PlantGenome::preset(HabitPreset::Tree, 7);
    assert_eq!(run(genome), run(genome));
    let mut other = genome;
    other.seed.apical_dominance = 0.3;
    assert_ne!(run(genome), run(other));
}

#[test]
fn genome_roundtrip_and_presets_are_inside_bounds() {
    for habit in ALL_HABITS {
        let genome = PlantGenome::preset(habit, 5);
        assert!(genome.in_bounds(), "{habit:?}");
        assert_eq!(genome.with_genes(&genome.genes()), genome);
    }
    assert_eq!(GENE_SPECS.len(), GENE_COUNT);
    assert!(GENE_SPECS.iter().all(|s| s.min < s.max));
    let mut broken = PlantGenome::preset(HabitPreset::Tree, 1).genes();
    broken[0] = 1.0e9;
    broken[3] = -4.0;
    broken[5] = f64::NAN;
    assert!(PlantGenome::preset(HabitPreset::Tree, 1).with_genes(&broken).in_bounds());
}

#[test]
fn mutation_is_deterministic_bounded_and_mostly_small_with_rare_extremes() {
    let parent = PlantGenome::preset(HabitPreset::Tree, 11);
    assert_eq!(parent.mutate(9, 100, 1, Some(1.0)), parent.mutate(9, 100, 1, Some(1.0)));
    assert_ne!(parent.mutate(9, 100, 1, Some(1.0)), parent.mutate(9, 101, 1, Some(1.0)));
    assert_ne!(parent.mutate(9, 100, 1, Some(1.0)), parent.mutate(10, 100, 1, Some(1.0)));

    let parent_genes = parent.genes();
    let mut small = 0usize;
    let mut extreme = 0usize;
    let mut total = 0usize;
    let mut ids = std::collections::HashSet::new();
    for child_id in 0..3000u64 {
        let child = parent.mutate(42, child_id, 1, Some(1.0));
        assert!(child.in_bounds(), "child {child_id} left the bounds");
        ids.insert(child.seed.individual_id);
        for (locus, (c, p)) in child.genes().iter().zip(parent_genes.iter()).enumerate() {
            if locus == 9 {
                continue; // integer gene
            }
            let range = GENE_SPECS[locus].max - GENE_SPECS[locus].min;
            let step = (c - p).abs() / range;
            total += 1;
            if step < 0.15 {
                small += 1;
            }
            if step > 0.4 {
                extreme += 1;
            }
        }
    }
    let small_fraction = small as f64 / total as f64;
    let extreme_fraction = extreme as f64 / total as f64;
    println!("mutation steps: {small_fraction:.4} small, {extreme_fraction:.4} extreme");
    assert!(small_fraction > 0.9, "{small_fraction}");
    assert!(extreme_fraction > 0.001 && extreme_fraction < 0.03, "{extreme_fraction}");
    assert_eq!(ids.len(), 3000, "offspring identities are distinct");

    // With the genome's own tiny rate almost every child equals the parent.
    let unchanged = (0..2000u64)
        .filter(|id| {
            let child = parent.mutate(5, *id, 1, None);
            child.genes() == parent_genes
        })
        .count();
    assert!(unchanged >= 1950, "{unchanged}");
}

#[test]
fn crossover_is_deterministic_bounded_and_mixes_both_parents() {
    let a = PlantGenome::preset(HabitPreset::Tree, 1);
    let b = PlantGenome::preset(HabitPreset::Grass, 2);
    let child = PlantGenome::crossover(&a, &b, 3, 50, 1);
    assert_eq!(child, PlantGenome::crossover(&a, &b, 3, 50, 1));
    assert_ne!(child, PlantGenome::crossover(&a, &b, 3, 51, 1));
    let (ga, gb, gc) = (a.genes(), b.genes(), child.genes());
    assert!(child.in_bounds());
    let mut from_a = 0;
    let mut from_b = 0;
    for i in 0..GENE_COUNT {
        let (lo, hi) = (ga[i].min(gb[i]), ga[i].max(gb[i]));
        assert!(gc[i] >= lo - 1e-4 && gc[i] <= hi + 1e-4, "gene {} outside parents", GENE_SPECS[i].name);
        if gc[i] == ga[i] && ga[i] != gb[i] {
            from_a += 1;
        }
        if gc[i] == gb[i] && ga[i] != gb[i] {
            from_b += 1;
        }
    }
    assert!(from_a > 3 && from_b > 3, "{from_a} {from_b}");
    // The offspring grows: a viable organism from mixed genes.
    let mut env = rich_env();
    let mut plant = Plant::from_seed(child, 0.02, 0.0, &mut env);
    let reference = closed_mass(&env, &[&plant], &[]);
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 2.0 * YEAR_DAYS);
    assert_closed(&env, &[&plant], &[], reference);
}

#[test]
fn offspring_of_many_mutations_still_grow_without_breaking_the_ledger() {
    let parent = PlantGenome::preset(HabitPreset::Shrub, 3);
    for id in 0..12u64 {
        let genome = parent.mutate(77, id, 2, Some(0.3));
        let mut env = rich_env();
        env.add_litter(5.0, 0.1);
        let mut plant = Plant::from_seed(genome, 0.02, 0.0, &mut env);
        let reference = closed_mass(&env, &[&plant], &[]);
        plant.advance(&mut env, &Climate::seasonal(Conditions::SUMMER, Conditions::WINTER, 60.0, 220.0), 2.0 * YEAR_DAYS);
        assert_closed(&env, &[&plant], &[], reference);
        assert_energy_closed(&env, &[&plant], &[]);
        assert!(plant.stores.sugar >= 0.0 && plant.stores.water >= 0.0 && plant.stores.minerals >= 0.0);
        assert!(plant.pools.to_array().iter().all(|x| *x >= 0.0 && x.is_finite()));
    }
}

#[test]
fn derived_flags_are_not_stored() {
    // Dead, dormant, mature and senescent are all recomputed from state and input.
    let mut env = rich_env();
    let mut plant = sprout(HabitPreset::Grass, &mut env);
    assert!(!plant.is_mature() && !plant.is_senescent() && !plant.is_dead());
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 300.0);
    assert!(plant.is_mature() && !plant.is_senescent());
    plant.advance(&mut env, &Climate::constant(Conditions::SUMMER), 3.0 * YEAR_DAYS);
    assert!(plant.is_senescent());
    let mut copy = plant.clone();
    copy.damage = 1.0;
    assert!(copy.is_dead());
    copy.damage = 0.0;
    assert!(!copy.is_dead() || copy.pools.total() <= 1e-12);
}
