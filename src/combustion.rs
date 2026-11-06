//! Combustion as material reactions (WORLD_PARADIGM.md 7.7, WORLD_RUNTIME_PLAN.md).
//!
//! There is no "burning" flag and no prescribed flame temperature. Combustion
//! is one recipe of the data-driven registry in `crate::recipes`: reactants,
//! products and mass coefficients `nu_k` per kilogram of fuel with
//! `sum(nu_k) = 0`; one step computes an extent of reaction `xi` (kg of fuel
//! consumed) and changes every component mass by `nu_k * xi`. Chemical energy
//! lives in the equation of state as a per-kilogram formation energy
//! (`Material::formation_energy_j_per_kg`), so the heat of the reaction appears
//! as a rise of temperature at *constant internal energy*: the step never adds
//! or removes energy. Temperature is derived from `internal energy - formation
//! offset of the current composition`.
//!
//! Rate model (all SI): `d(xi)/dt = k * m_fuel`, with
//! `k = A * exp(-Ea / (R T)) * (Y_O2 / Y_ref)` for `T >= T_decomposition` and
//! zero below it. The extent is limited by the available fuel and oxidizer
//! (stoichiometry), and integrated with an exponential limiter and adaptive
//! sub-steps (at most 25 K of heating and 25% of the fuel per sub-step).
//!
//! Moisture is the second reaction `water -> water_vapor`. Boiling is limited by
//! heat supply, not by a rate constant: liquid water in a host whose own equation of state has no
//! vaporization plateau can only evaporate above 373.15 K, absorbs its latent
//! heat from the sensible energy (`formation(vapor) - formation(water)`), and
//! cannot cool the REV below the boiling point. Until the water is gone the
//! temperature stays at the boiling point, which delays ignition.
//!
//! [`CombustionModel`] is the single-recipe view kept for direct use and for
//! the combustion tests; [`crate::recipes::RecipeRegistry::builtin`] runs the
//! same recipe together with tanning and decay, and `Scene::react` is the one
//! scene hook for all of them.
//!
//! Scope, stated plainly: the REV is a *closed* reactor. Products stay in the
//! REV; there is no transport of gas, oxygen or heat of products between REVs
//! here (heat conduction between REVs is the existing thermal update), no
//! pressure or volume model, no flame front, no surface-area dependence, no
//! pyrolysis without oxygen, no condensation, and no radiation. Constant heat
//! capacities per component (no temperature-dependent cp, no dissociation).

use crate::material::Material;
pub use crate::recipes::{
    ReactionExtent, ReactionOutcome, ReactionReport, Recipe, RecipeRegistry,
};
#[allow(unused_imports)]
use crate::recipes::{
    add_mass, component_by_name, composition_from_masses, mass_of, set_mass, state_of,
};
use crate::recipes::{Dependence, Evaporation, RateLaw};
use crate::voxel::Composition;

/// Reaction-rate parameters. Illustrative values, not calibrated data.
#[derive(Clone, Copy, Debug)]
pub struct Kinetics {
    /// Arrhenius pre-exponential factor, 1/s.
    pub pre_exponential_per_s: f64,
    /// Activation energy, J/mol.
    pub activation_energy_j_mol: f64,
    /// Below this temperature (K) the fuel does not decompose or ignite.
    pub decomposition_temperature_k: f64,
    /// Oxidizer mass fraction at which the oxygen factor equals one (air).
    pub reference_oxidizer_mass_fraction: f64,
}

impl Default for Kinetics {
    fn default() -> Self {
        Self {
            pre_exponential_per_s: 2.0e8,
            activation_energy_j_mol: 100_000.0,
            // Cellulose pyrolysis onset, roughly 470-570 K.
            decomposition_temperature_k: 500.0,
            reference_oxidizer_mass_fraction: 0.232,
        }
    }
}

impl Kinetics {
    /// The same law as a generic rate law with an unbounded oxidizer factor.
    pub fn rate_law(&self, oxidizer: u16) -> RateLaw {
        let mut law =
            RateLaw::arrhenius(self.pre_exponential_per_s, self.activation_energy_j_mol);
        law.minimum_temperature_k = self.decomposition_temperature_k;
        law.dependencies = vec![Dependence {
            component: oxidizer,
            reference_mass_fraction: self.reference_oxidizer_mass_fraction,
            cap: f64::INFINITY,
        }];
        law.max_fraction_per_substep = 0.25;
        law.max_heating_per_substep_k = 25.0;
        law
    }
}

/// The built-in combustion recipe: cellulose-like fuel + oxidizer -> CO2 +
/// water vapour + ash, with ids resolved from a material table (`dry_wood`,
/// `oxygen`, `carbon_dioxide`, `water_vapor`, `ash`). `None` when one is missing.
pub fn combustion_recipe(materials: &[Material]) -> Option<Recipe> {
    let (fuel, oxidizer, co2, vapor, ash) = (
        component_by_name(materials, "dry_wood")?,
        component_by_name(materials, "oxygen")?,
        component_by_name(materials, "carbon_dioxide")?,
        component_by_name(materials, "water_vapor")?,
        component_by_name(materials, "ash")?,
    );
    // Fuel: 99% cellulose C6H10O5 + 1% mineral ash by mass.
    // C6H10O5 + 6 O2 -> 6 CO2 + 5 H2O, from atomic masses (g/mol).
    let (c, h, o) = (12.011, 1.008, 15.999);
    let cellulose = 6.0 * c + 10.0 * h + 5.0 * o;
    let (o2, co2_m, h2o) = (2.0 * o, c + 2.0 * o, 2.0 * h + o);
    let moles_per_kg_fuel = 0.99 / cellulose;
    let nu_o2 = 6.0 * moles_per_kg_fuel * o2;
    let nu_co2 = 6.0 * moles_per_kg_fuel * co2_m;
    let nu_h2o = 5.0 * moles_per_kg_fuel * h2o;
    // The ash coefficient closes the mass balance exactly (about 0.01).
    let nu_ash = 1.0 + nu_o2 - nu_co2 - nu_h2o;
    let recipe = Recipe {
        name: "dry_wood + oxygen -> carbon_dioxide + water_vapor + ash".into(),
        key: fuel,
        terms: vec![
            (fuel, -1.0),
            (oxidizer, -nu_o2),
            (co2, nu_co2),
            (vapor, nu_h2o),
            (ash, nu_ash),
        ],
        rate: Kinetics::default().rate_law(oxidizer),
        is_combustion: true,
    };
    recipe.validate().ok()?;
    Some(recipe)
}

/// Combustion + evaporation with the component ids resolved from a material
/// table; a single-recipe view over [`RecipeRegistry`].
#[derive(Clone, Debug)]
pub struct CombustionModel {
    pub recipe: Recipe,
    pub kinetics: Kinetics,
    pub water: u16,
    pub water_vapor: u16,
    oxidizer: u16,
    heat_of_combustion_j_per_kg: f64,
    evaporation_enthalpy_j_per_kg: f64,
    evaporation_cp_change: f64,
}

impl CombustionModel {
    /// Resolve `dry_wood`, `oxygen`, `carbon_dioxide`, `water_vapor`, `ash` and
    /// `water` by name. `None` when the table lacks any of them.
    pub fn for_materials(materials: &[Material]) -> Option<Self> {
        let recipe = combustion_recipe(materials)?;
        let evaporation = Evaporation::for_materials(materials)?;
        let heat = recipe.heat_of_reaction_j_per_kg(materials);
        let oxidizer = recipe.terms[1].0;
        Some(Self {
            recipe,
            kinetics: Kinetics::default(),
            water: evaporation.water,
            water_vapor: evaporation.vapor,
            oxidizer,
            heat_of_combustion_j_per_kg: heat,
            evaporation_enthalpy_j_per_kg: evaporation.enthalpy_j_per_kg,
            evaporation_cp_change: evaporation.cp_change,
        })
    }

    /// Heat released per kilogram of dry fuel, J/kg.
    pub fn heat_of_combustion_j_per_kg(&self) -> f64 {
        self.heat_of_combustion_j_per_kg
    }

    /// The registry holding this model's recipe (with the current `kinetics`)
    /// and evaporation.
    pub fn registry(&self, materials: &[Material]) -> Result<RecipeRegistry, &'static str> {
        let mut recipe = self.recipe.clone();
        recipe.rate = self.kinetics.rate_law(self.oxidizer);
        let evaporation = Evaporation {
            water: self.water,
            vapor: self.water_vapor,
            enthalpy_j_per_kg: self.evaporation_enthalpy_j_per_kg,
            cp_change: self.evaporation_cp_change,
            boiling_k: crate::recipes::WATER_BOILING_K,
        };
        RecipeRegistry::new(materials, vec![recipe], Some(evaporation))
    }

    /// Advance one REV by `dt_s`. `composition` holds mass fractions of the
    /// REV of `mass_kg`; `internal_energy_j` is its total internal energy,
    /// including the chemical part. Mass is unchanged by construction and the
    /// energy is returned unchanged; only the composition (and therefore the
    /// derived temperature) changes.
    pub fn step(
        &self,
        materials: &[Material],
        composition: &Composition,
        mass_kg: f32,
        internal_energy_j: f64,
        dt_s: f64,
    ) -> Result<ReactionOutcome, &'static str> {
        let registry = self.registry(materials)?;
        let outcome = registry.step(materials, composition, mass_kg, internal_energy_j, dt_s)?;
        Ok(ReactionOutcome {
            composition: outcome.composition,
            internal_energy_j: outcome.internal_energy_j,
            extent: registry.legacy_extent(&outcome.extent),
            temperature_before_k: outcome.temperature_before_k,
            temperature_after_k: outcome.temperature_after_k,
        })
    }

    /// Core integrator on component masses in kg (f64). The sum of `masses` is
    /// conserved to floating-point rounding. Entries reaching exactly zero stay
    /// in the list with mass 0.
    pub fn step_masses(
        &self,
        materials: &[Material],
        masses: &mut Vec<(u16, f64)>,
        internal_energy_j: f64,
        dt_s: f64,
    ) -> Result<ReactionExtent, &'static str> {
        let registry = self.registry(materials)?;
        let extent = registry.step_masses(materials, masses, internal_energy_j, dt_s)?;
        Ok(registry.legacy_extent(&extent))
    }

    fn state(
        &self,
        materials: &[Material],
        masses: &[(u16, f64)],
        internal_energy_j: f64,
    ) -> Result<(f64, Material), &'static str> {
        state_of(materials, masses, internal_energy_j)
    }
}

#[cfg(test)]
use crate::scene::Scene;


#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneConfig, VoxelSpawn};

    fn id(table: &[Material], name: &str) -> u16 {
        table.iter().position(|m| m.name == name).unwrap() as u16
    }

    fn composition(table: &[Material], parts: &[(&str, f64)]) -> Composition {
        let total: f64 = parts.iter().map(|p| p.1).sum();
        Composition {
            components: parts
                .iter()
                .map(|&(name, mass)| (id(table, name), (mass / total) as f32))
                .collect(),
        }
    }

    fn masses(table: &[Material], parts: &[(&str, f64)]) -> Vec<(u16, f64)> {
        parts.iter().map(|&(n, m)| (id(table, n), m)).collect()
    }

    fn energy_at(table: &[Material], masses: &[(u16, f64)], kelvin: f32) -> f64 {
        let total: f64 = masses.iter().map(|m| m.1).sum();
        let mixed = Material::mixed(&composition_from_masses(masses), table);
        mixed.energy_at_temperature(total as f32, kelvin) as f64
    }

    fn total(masses: &[(u16, f64)]) -> f64 {
        masses.iter().map(|m| m.1).sum()
    }

    #[test]
    fn recipe_conserves_mass_and_releases_about_17_mj_per_kg() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        model.recipe.validate().unwrap();
        let sum: f64 = model.recipe.terms.iter().map(|t| t.1).sum();
        assert!(sum.abs() < 1e-12, "mass coefficients sum {sum}");
        let heat = model.heat_of_combustion_j_per_kg();
        assert!((heat - 17.0e6).abs() < 0.17e6, "heat of combustion {heat}");
        for material in &table {
            assert!(material.formation_energy_j_per_kg >= 0.0);
        }
        // Evaporation enthalpy equals the built-in latent heat of vaporization.
        let water = &table[id(&table, "water") as usize];
        assert_eq!(
            model.evaporation_enthalpy_j_per_kg as f32,
            water.latent_vaporization
        );
    }

    #[test]
    fn burn_conserves_mass_and_energy_and_heats_by_the_chemical_energy() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        let mut state = masses(&table, &[("dry_wood", 1.0e-3), ("oxygen", 1.5e-3)]);
        let energy = energy_at(&table, &state, 700.0);
        let mass_before = total(&state);
        let before = Material::mixed(&composition_from_masses(&state), &table);
        let t_before = before.temperature(mass_before as f32, energy as f32);

        let extent = model.step_masses(&table, &mut state, energy, 1.0).unwrap();
        assert!(extent.fuel_burned_kg > 0.99e-3, "fuel burned {extent:?}");
        // Component masses move by exactly nu * xi, and the total is conserved.
        assert!((total(&state) - mass_before).abs() <= 1e-12 * mass_before);
        let burned = extent.fuel_burned_kg;
        for &(component, nu) in &model.recipe.terms {
            let initial = match component {
                c if c == id(&table, "dry_wood") => 1.0e-3,
                c if c == id(&table, "oxygen") => 1.5e-3,
                _ => 0.0,
            };
            let expected = initial + nu * burned;
            assert!((mass_of(&state, component) - expected).abs() < 1e-15);
        }

        // The stored energy is unchanged; temperature rises by the heat of
        // reaction over the heat capacity: M (cp2 T2 - cp1 T1) = xi Q.
        let after = Material::mixed(&composition_from_masses(&state), &table);
        let mass_after = total(&state);
        let t_after = after.temperature(mass_after as f32, energy as f32);
        let heat = burned * model.heat_of_combustion_j_per_kg();
        assert!((extent.combustion_heat_j - heat).abs() <= 1e-9 * heat);
        let sensible_gain = mass_after
            * (after.heat_capacity as f64 * t_after as f64
                - before.heat_capacity as f64 * t_before as f64);
        assert!(
            (sensible_gain - heat).abs() < 2e-3 * heat,
            "sensible gain {sensible_gain} vs released {heat}"
        );
        // heat / (m cp) is only a first estimate: cp changes from reactants to
        // products (the exact identity is the check above), so allow 10%.
        let cp_mean = 0.5 * (before.heat_capacity + after.heat_capacity) as f64;
        let estimate = heat / (mass_after * cp_mean);
        let rise = (t_after - t_before) as f64;
        assert!(
            (rise - estimate).abs() < 0.10 * estimate,
            "rise {rise} K vs heat/(m cp) {estimate} K"
        );
        // Total energy rebuilt from the new composition equals the old energy.
        let rebuilt = after.energy_at_temperature(mass_after as f32, t_after) as f64;
        assert!((rebuilt - energy).abs() < 1e-5 * energy);
    }

    #[test]
    fn composition_path_returns_unchanged_energy_and_normalized_fractions() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        let parts = [("dry_wood", 1.0e-3), ("oxygen", 1.0e-3)];
        let composition = composition(&table, &parts);
        let state = masses(&table, &parts);
        let energy = energy_at(&table, &state, 800.0);
        let outcome = model
            .step(&table, &composition, 2.0e-3, energy, 0.5)
            .unwrap();
        assert_eq!(outcome.internal_energy_j.to_bits(), energy.to_bits());
        outcome.composition.validate(table.len()).unwrap();
        assert!(outcome.temperature_after_k > outcome.temperature_before_k + 100.0);
    }

    #[test]
    fn no_ignition_when_cold_or_below_decomposition_temperature() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        for kelvin in [293.15_f32, 450.0, 499.0] {
            let mut state = masses(&table, &[("dry_wood", 1.0e-3), ("oxygen", 1.5e-3)]);
            let energy = energy_at(&table, &state, kelvin);
            let initial = state.clone();
            let extent = model.step_masses(&table, &mut state, energy, 1.0e4).unwrap();
            assert!(extent.is_empty(), "{kelvin} K reacted: {extent:?}");
            assert_eq!(state, initial);
        }
    }

    #[test]
    fn no_reaction_without_oxidizer() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        for parts in [
            vec![("dry_wood", 1.0e-3)],
            vec![("dry_wood", 1.0e-3), ("carbon_dioxide", 1.0e-3)],
        ] {
            let mut state = masses(&table, &parts);
            let energy = energy_at(&table, &state, 1200.0);
            let initial = state.clone();
            let extent = model.step_masses(&table, &mut state, energy, 100.0).unwrap();
            assert!(extent.is_empty());
            assert_eq!(state, initial);
        }
    }

    /// Heat a REV with a constant external power and report when 1% of the
    /// fuel has burned. The heater is test input; the reaction adds no energy.
    fn ignition_time(table: &[Material], water_kg: f64, max_temperature_while_wet: &mut f64) -> f64 {
        let model = CombustionModel::for_materials(table).unwrap();
        let mut parts = vec![("dry_wood", 1.0e-3), ("oxygen", 1.5e-3)];
        if water_kg > 0.0 {
            parts.push(("water", water_kg));
        }
        let mut state = masses(table, &parts);
        let mut energy = energy_at(table, &state, 293.15);
        let (power_w, dt) = (2.0, 0.01);
        let fuel0 = mass_of(&state, id(table, "dry_wood"));
        for step in 0..200_000 {
            energy += power_w * dt;
            model.step_masses(table, &mut state, energy, dt).unwrap();
            if mass_of(&state, id(table, "water")) > 0.0 {
                let (t, _) = model.state(table, &state, energy).unwrap();
                *max_temperature_while_wet = max_temperature_while_wet.max(t);
            }
            if mass_of(&state, id(table, "dry_wood")) < 0.99 * fuel0 {
                return (step + 1) as f64 * dt;
            }
        }
        panic!("never ignited");
    }

    #[test]
    fn wet_fuel_ignites_later_by_about_the_latent_heat_of_its_moisture() {
        let table = Material::table();
        let mut ignored = 0.0;
        let dry = ignition_time(&table, 0.0, &mut ignored);
        let water = 5.0e-5;
        let mut wet_peak = 0.0;
        let wet = ignition_time(&table, water, &mut wet_peak);
        let latent_time = water * 2_256_000.0 / 2.0;
        assert!(wet > dry + 0.9 * latent_time, "dry {dry} s, wet {wet} s");
        assert!(wet_peak < 373.15 + 1.0, "wet fuel exceeded boiling: {wet_peak}");
    }

    #[test]
    fn oxidizer_exhaustion_stops_the_burn_and_leaves_the_fuel_excess() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        let mut state = masses(&table, &[("dry_wood", 7.0e-4), ("oxygen", 3.0e-4)]);
        let energy = energy_at(&table, &state, 700.0);
        let first = model.step_masses(&table, &mut state, energy, 2.0).unwrap();
        let nu_ox = -model.recipe.terms[1].1;
        assert_eq!(mass_of(&state, id(&table, "oxygen")), 0.0);
        assert!((first.fuel_burned_kg - 3.0e-4 / nu_ox).abs() < 1e-12);
        assert!(mass_of(&state, id(&table, "dry_wood")) > 0.0);
        // Dies down: nothing further happens, however hot it is.
        let settled = state.clone();
        let later = model.step_masses(&table, &mut state, energy, 1.0e3).unwrap();
        assert!(later.is_empty());
        assert_eq!(state, settled);
    }

    #[test]
    fn water_boils_off_at_constant_energy_and_pins_the_temperature() {
        let table = Material::table();
        let model = CombustionModel::for_materials(&table).unwrap();
        let mut state = masses(&table, &[("dry_wood", 1.0e-3), ("water", 1.0e-4)]);
        let energy = energy_at(&table, &state, 420.0);
        let before = total(&state);
        let extent = model.step_masses(&table, &mut state, energy, 1.0).unwrap();
        assert!(extent.water_evaporated_kg > 0.0);
        assert!((total(&state) - before).abs() < 1e-15);
        let (t, _) = model.state(&table, &state, energy).unwrap();
        assert!((t - 373.15).abs() < 0.2 || mass_of(&state, id(&table, "water")) == 0.0);
        // Latent heat came out of the sensible part, not out of nowhere.
        assert!(extent.evaporation_heat_j > 0.0);
    }

    #[test]
    fn scene_react_burns_a_closed_rev_and_keeps_mass_and_energy_bit_exact() {
        let table = Material::table();
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.001;
        let mut spawn = VoxelSpawn::at_cell([0, 0, 0], id(&table, "dry_wood"));
        spawn.composition = composition(&table, &[("dry_wood", 1.0), ("oxygen", 1.0)]);
        spawn.temperature_kelvin = 700.0;
        let mut scene = Scene::from_voxels(config, vec![spawn]).unwrap();
        let mass = scene.voxels[0].mass;
        let energy = (
            scene.voxels[0].internal_energy.to_bits(),
            scene.voxels[0].internal_energy_remainder_j.to_bits(),
        );
        let t0 = scene.temperature_kelvin(0).unwrap();
        let mut burned = 0.0;
        for _ in 0..3000 {
            burned += scene.react(0.001).unwrap().extent.fuel_burned_kg;
        }
        assert!(burned > 0.0);
        assert_eq!(scene.voxels[0].mass.to_bits(), mass.to_bits());
        assert_eq!(
            (
                scene.voxels[0].internal_energy.to_bits(),
                scene.voxels[0].internal_energy_remainder_j.to_bits()
            ),
            energy
        );
        assert!(scene.temperature_kelvin(0).unwrap() > t0 + 500.0);
        scene.voxels[0]
            .composition
            .validate(scene.materials.len())
            .unwrap();
        let mixed = Material::mixed(&scene.voxels[0].composition, &scene.materials);
        assert_eq!(
            scene.properties[0].heat_capacity.to_bits(),
            mixed.heat_capacity.to_bits()
        );
        // Closed volume: the oxidizer is gone and a further call does nothing.
        let oxygen = id(&scene.materials, "oxygen");
        assert!(scene.voxels[0]
            .composition
            .components
            .iter()
            .all(|&(c, f)| c != oxygen || f < 1e-6));
        let burned_in_total = burned;
        let again = scene.react(1.0).unwrap().extent.fuel_burned_kg;
        assert!(again < 1e-3 * burned_in_total, "still burning: {again}");
    }

    #[test]
    fn scene_without_chemistry_is_untouched() {
        let mut scene = Scene::new(SceneConfig::default());
        let report = scene.react(0.001).unwrap();
        assert_eq!(report, ReactionReport::default());
        assert!(scene.react(f32::NAN).is_err());
    }
}
