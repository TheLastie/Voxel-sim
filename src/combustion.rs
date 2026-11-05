//! Combustion as material reactions (WORLD_PARADIGM.md 7.7, WORLD_RUNTIME_PLAN.md).
//!
//! There is no "burning" flag and no prescribed flame temperature. A recipe has
//! reactants, products and mass coefficients `nu_k` per kilogram of fuel with
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
//! Scope, stated plainly: the REV is a *closed* reactor. Products stay in the
//! REV; there is no transport of gas, oxygen or heat of products between REVs
//! here (heat conduction between REVs is the existing thermal update), no
//! pressure or volume model, no flame front, no surface-area dependence, no
//! pyrolysis without oxygen, no condensation, and no radiation. Constant heat
//! capacities per component (no temperature-dependent cp, no dissociation).

use crate::material::Material;
use crate::scene::Scene;
use crate::voxel::Composition;

const GAS_CONSTANT_J_MOL_K: f64 = 8.314_462_618_153_24;
/// Boiling point of water at 1 atm, kelvin; matches the built-in `water` entry.
const WATER_BOILING_K: f64 = 373.15;
const MAX_SUBSTEPS: usize = 256;
const MAX_FUEL_FRACTION_PER_SUBSTEP: f64 = 0.25;
const MAX_HEATING_PER_SUBSTEP_K: f64 = 25.0;

/// One reaction: component mass coefficients per kilogram of fuel.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub name: &'static str,
    pub fuel: u16,
    pub oxidizer: u16,
    /// `(component, nu)` in kg per kg of fuel; reactants are negative and the
    /// fuel is exactly `-1`. The sum is zero (mass is conserved).
    pub terms: Vec<(u16, f64)>,
}

impl Recipe {
    pub fn validate(&self) -> Result<(), &'static str> {
        let sum: f64 = self.terms.iter().map(|&(_, nu)| nu).sum();
        let fuel = self.terms.iter().find(|&&(id, _)| id == self.fuel);
        let oxidizer = self.terms.iter().find(|&&(id, _)| id == self.oxidizer);
        match (fuel, oxidizer) {
            (Some(&(_, f)), Some(&(_, o))) if f == -1.0 && o < 0.0 => {}
            _ => return Err("recipe needs fuel with coefficient -1 and a consumed oxidizer"),
        }
        if sum.abs() > 1e-12 {
            return Err("recipe mass coefficients must sum to zero");
        }
        Ok(())
    }

    /// Heat released per kilogram of fuel, J/kg, from the materials' formation
    /// energies: `-sum(nu_k * h_k)`.
    pub fn heat_of_reaction_j_per_kg(&self, materials: &[Material]) -> f64 {
        -self
            .terms
            .iter()
            .map(|&(id, nu)| nu * materials[id as usize].formation_energy_j_per_kg as f64)
            .sum::<f64>()
    }
}

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

/// What one REV step did. Energy is returned unchanged: reactions only move
/// energy between the chemical and thermal parts of the same stored energy.
#[derive(Clone, Debug)]
pub struct ReactionOutcome {
    pub composition: Composition,
    /// Total internal energy of the REV, identical to the input (J).
    pub internal_energy_j: f64,
    pub extent: ReactionExtent,
    pub temperature_before_k: f32,
    pub temperature_after_k: f32,
}

/// Integrated amounts of one step (f64, SI).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReactionExtent {
    /// Dry fuel consumed, kg (the extent of reaction).
    pub fuel_burned_kg: f64,
    pub water_evaporated_kg: f64,
    /// Chemical energy turned into sensible heat by combustion, J.
    pub combustion_heat_j: f64,
    /// Sensible heat turned into latent chemical energy by evaporation, J.
    pub evaporation_heat_j: f64,
    pub substeps: usize,
}

impl ReactionExtent {
    pub fn is_empty(&self) -> bool {
        self.fuel_burned_kg == 0.0 && self.water_evaporated_kg == 0.0
    }

    fn accumulate(&mut self, other: &Self) {
        self.fuel_burned_kg += other.fuel_burned_kg;
        self.water_evaporated_kg += other.water_evaporated_kg;
        self.combustion_heat_j += other.combustion_heat_j;
        self.evaporation_heat_j += other.evaporation_heat_j;
        self.substeps += other.substeps;
    }
}

/// Totals of one [`Scene::react`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReactionReport {
    pub reacting_revs: usize,
    pub extent: ReactionExtent,
}

/// Cellulose-like fuel + oxidizer -> CO2 + water vapour + ash, with the
/// component ids resolved from a material table.
#[derive(Clone, Debug)]
pub struct CombustionModel {
    pub recipe: Recipe,
    pub kinetics: Kinetics,
    pub water: u16,
    pub water_vapor: u16,
    heat_of_combustion_j_per_kg: f64,
    evaporation_enthalpy_j_per_kg: f64,
    evaporation_cp_change: f64,
}

impl CombustionModel {
    /// Resolve `dry_wood`, `oxygen`, `carbon_dioxide`, `water_vapor`, `ash` and
    /// `water` by name. `None` when the table lacks any of them.
    pub fn for_materials(materials: &[Material]) -> Option<Self> {
        let id = |name: &str| {
            materials
                .iter()
                .position(|m| m.name == name)
                .and_then(|i| u16::try_from(i).ok())
        };
        let (fuel, oxidizer, co2, vapor, ash, water) = (
            id("dry_wood")?,
            id("oxygen")?,
            id("carbon_dioxide")?,
            id("water_vapor")?,
            id("ash")?,
            id("water")?,
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
            name: "dry_wood + oxygen -> carbon_dioxide + water_vapor + ash",
            fuel,
            oxidizer,
            terms: vec![
                (fuel, -1.0),
                (oxidizer, -nu_o2),
                (co2, nu_co2),
                (vapor, nu_h2o),
                (ash, nu_ash),
            ],
        };
        recipe.validate().ok()?;
        let heat = recipe.heat_of_reaction_j_per_kg(materials);
        let (w, v) = (&materials[water as usize], &materials[vapor as usize]);
        Some(Self {
            recipe,
            kinetics: Kinetics::default(),
            water,
            water_vapor: vapor,
            heat_of_combustion_j_per_kg: heat,
            evaporation_enthalpy_j_per_kg: (v.formation_energy_j_per_kg
                - w.formation_energy_j_per_kg) as f64,
            evaporation_cp_change: (v.heat_capacity - w.heat_capacity) as f64,
        })
    }

    /// Heat released per kilogram of dry fuel, J/kg.
    pub fn heat_of_combustion_j_per_kg(&self) -> f64 {
        self.heat_of_combustion_j_per_kg
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
        if !mass_kg.is_finite() || mass_kg <= 0.0 {
            return Err("REV mass must be positive and finite");
        }
        composition
            .validate(materials.len())
            .map_err(|_| "invalid composition")?;
        let mut masses: Vec<(u16, f64)> = composition
            .components
            .iter()
            .map(|&(id, fraction)| (id, fraction as f64 * mass_kg as f64))
            .collect();
        let before = self.temperature(materials, &masses, internal_energy_j)?;
        let extent = self.step_masses(materials, &mut masses, internal_energy_j, dt_s)?;
        let after = self.temperature(materials, &masses, internal_energy_j)?;
        Ok(ReactionOutcome {
            composition: composition_from_masses(&masses),
            internal_energy_j,
            extent,
            temperature_before_k: before,
            temperature_after_k: after,
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
        if !dt_s.is_finite() || dt_s < 0.0 {
            return Err("reaction time step must be finite and nonnegative");
        }
        if !internal_energy_j.is_finite() || internal_energy_j < 0.0 {
            return Err("internal energy must be finite and nonnegative");
        }
        if masses.iter().any(|&(_, m)| !m.is_finite() || m < 0.0)
            || masses.iter().map(|&(_, m)| m).sum::<f64>() <= 0.0
        {
            return Err("component masses must be finite, nonnegative and not all zero");
        }
        let k = self.kinetics;
        let heat = self.heat_of_combustion_j_per_kg;
        let nu_ox = -self
            .recipe
            .terms
            .iter()
            .find(|&&(id, _)| id == self.recipe.oxidizer)
            .map(|&(_, nu)| nu)
            .unwrap_or(-1.0);
        let mut total = ReactionExtent::default();
        let mut remaining = dt_s;
        while remaining > 0.0 {
            let m_fuel = mass_of(masses, self.recipe.fuel);
            let m_ox = mass_of(masses, self.recipe.oxidizer);
            let m_water = mass_of(masses, self.water);
            let can_burn = m_fuel > 0.0 && m_ox > 0.0;
            if !can_burn && m_water <= 0.0 {
                break;
            }
            let (mut temp, mixed) = self.state(materials, masses, internal_energy_j)?;
            let sum: f64 = masses.iter().map(|&(_, m)| m).sum();

            // Sub-step length from the combustion rate at the start state.
            let rate = |temp: f64, masses: &[(u16, f64)]| -> f64 {
                if temp < k.decomposition_temperature_k {
                    return 0.0;
                }
                let oxygen_factor = (mass_of(masses, self.recipe.oxidizer) / sum)
                    / k.reference_oxidizer_mass_fraction;
                k.pre_exponential_per_s
                    * (-k.activation_energy_j_mol / (GAS_CONSTANT_J_MOL_K * temp)).exp()
                    * oxygen_factor
            };
            let mut rate_per_s = if can_burn { rate(temp, masses) } else { 0.0 };
            let mut h = remaining;
            if rate_per_s > 0.0 && total.substeps + 1 < MAX_SUBSTEPS {
                let heating_w = rate_per_s * m_fuel * heat;
                h = h
                    .min(MAX_FUEL_FRACTION_PER_SUBSTEP / rate_per_s)
                    .min(MAX_HEATING_PER_SUBSTEP_K * sum * mixed.heat_capacity as f64 / heating_w);
            }
            let h = h.max(remaining * 1e-12).min(remaining);
            let mut active = rate_per_s > 0.0;

            // 1. Moisture: boil off at constant energy, never below boiling.
            if m_water > 0.0 && mixed.boil_temp.is_infinite() && temp > WATER_BOILING_K {
                let cp = mixed.heat_capacity as f64;
                let denominator =
                    self.evaporation_enthalpy_j_per_kg + WATER_BOILING_K * self.evaporation_cp_change;
                let energy_limit = if denominator > 0.0 {
                    cp * (temp - WATER_BOILING_K) / denominator * sum
                } else {
                    0.0
                };
                let mut evaporated = energy_limit.min(m_water);
                if evaporated > 0.0 {
                    if evaporated >= m_water {
                        evaporated = m_water;
                        set_mass(masses, self.water, 0.0);
                    } else {
                        add_mass(masses, self.water, -evaporated);
                    }
                    add_mass(masses, self.water_vapor, evaporated);
                    total.water_evaporated_kg += evaporated;
                    total.evaporation_heat_j += evaporated * self.evaporation_enthalpy_j_per_kg;
                    active = true;
                    // The latent heat lowered the temperature; refresh it.
                    let state = self.state(materials, masses, internal_energy_j)?;
                    temp = state.0;
                    rate_per_s = if can_burn { rate(temp, masses) } else { 0.0 };
                }
            }

            // 2. Fuel + oxidizer -> products. Extent limited by both reactants.
            if rate_per_s > 0.0 {
                let raw = m_fuel * -(-rate_per_s * h).exp_m1();
                let by_oxidizer = m_ox / nu_ox;
                let oxidizer_limited = raw >= by_oxidizer;
                let extent = if oxidizer_limited { by_oxidizer } else { raw }.min(m_fuel);
                if extent > 0.0 {
                    for &(id, nu) in &self.recipe.terms {
                        if id == self.recipe.fuel && extent >= m_fuel {
                            set_mass(masses, id, 0.0);
                        } else if id == self.recipe.oxidizer && oxidizer_limited {
                            set_mass(masses, id, 0.0);
                        } else {
                            add_mass(masses, id, nu * extent);
                        }
                    }
                    total.fuel_burned_kg += extent;
                    total.combustion_heat_j += extent * heat;
                    active = true;
                }
            }
            total.substeps += 1;
            if !active {
                break;
            }
            // Past the sub-step budget `h` is the whole remainder: one limited
            // step; reactants stay conserved, only the thermal feedback is coarser.
            remaining -= h;
            if remaining <= dt_s * 1e-12 {
                break;
            }
        }
        Ok(total)
    }

    fn state(
        &self,
        materials: &[Material],
        masses: &[(u16, f64)],
        internal_energy_j: f64,
    ) -> Result<(f64, Material), &'static str> {
        let composition = composition_from_masses(masses);
        if composition.components.is_empty() {
            return Err("composition has no mass");
        }
        let total: f64 = masses.iter().map(|&(_, m)| m).sum();
        let mixed = Material::mixed(&composition, materials);
        let temperature = mixed.temperature(total as f32, internal_energy_j as f32) as f64;
        Ok((temperature, mixed))
    }

    fn temperature(
        &self,
        materials: &[Material],
        masses: &[(u16, f64)],
        internal_energy_j: f64,
    ) -> Result<f32, &'static str> {
        Ok(self.state(materials, masses, internal_energy_j)?.0 as f32)
    }
}

fn mass_of(masses: &[(u16, f64)], id: u16) -> f64 {
    masses
        .iter()
        .find(|&&(component, _)| component == id)
        .map_or(0.0, |&(_, m)| m)
}

fn set_mass(masses: &mut Vec<(u16, f64)>, id: u16, value: f64) {
    match masses.iter_mut().find(|(component, _)| *component == id) {
        Some(entry) => entry.1 = value,
        None => masses.push((id, value)),
    }
}

fn add_mass(masses: &mut Vec<(u16, f64)>, id: u16, delta: f64) {
    match masses.iter_mut().find(|(component, _)| *component == id) {
        Some(entry) => entry.1 = (entry.1 + delta).max(0.0),
        None => masses.push((id, delta.max(0.0))),
    }
}

/// Mass fractions of `masses`; components of exactly zero mass are dropped.
fn composition_from_masses(masses: &[(u16, f64)]) -> Composition {
    let total: f64 = masses.iter().map(|&(_, m)| m).sum();
    let mut composition = Composition {
        components: Default::default(),
    };
    for &(id, mass) in masses {
        let fraction = (mass / total) as f32;
        if mass > 0.0 && fraction > 0.0 {
            composition.components.push((id, fraction));
        }
    }
    composition
}

impl Scene {
    /// Advance chemical reactions of every REV by `dt` seconds. Call it once
    /// per fixed physical step, after the step (like the thermal update it
    /// works on the stored energy); it is not run by `tick` so scenes without
    /// fuel pay nothing. Each REV is a closed reactor: component masses change
    /// by the recipe coefficients, the REV mass and stored internal energy are
    /// untouched, and `properties` is recomputed from the new composition, so
    /// the heat appears as derived temperature. Returns the totals.
    ///
    /// Products do not leave the REV (no transport yet), which also means a
    /// closed oxygen supply runs out and the burn dies down.
    pub fn react(&mut self, dt: f32) -> Result<ReactionReport, &'static str> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("reaction time step must be finite and nonnegative");
        }
        let mut report = ReactionReport::default();
        let Some(model) = CombustionModel::for_materials(&self.materials) else {
            return Ok(report);
        };
        if dt == 0.0 {
            return Ok(report);
        }
        let k = model.kinetics;
        let mut changed = false;
        for index in 0..self.voxels.len() {
            let voxel = &self.voxels[index];
            let (mut fuel, mut oxygen, mut water) = (false, false, false);
            for &(id, _) in &voxel.composition.components {
                fuel |= id == model.recipe.fuel;
                oxygen |= id == model.recipe.oxidizer;
                water |= id == model.water;
            }
            let burnable = fuel && oxygen;
            let wet = water && voxel.composition.components.len() > 1;
            if !burnable && !wet {
                continue;
            }
            let temperature = self.properties[index]
                .temperature(voxel.mass, voxel.thermal_energy_f32()) as f64;
            let threshold = match (burnable, wet) {
                (true, true) => k.decomposition_temperature_k.min(WATER_BOILING_K),
                (true, false) => k.decomposition_temperature_k,
                _ => WATER_BOILING_K,
            };
            if temperature < threshold {
                continue;
            }
            let energy = voxel.total_internal_energy_j();
            let outcome = model
                .step(&self.materials, &voxel.composition, voxel.mass, energy, dt as f64)?;
            if outcome.extent.is_empty() {
                continue;
            }
            self.voxels[index].composition = outcome.composition;
            self.properties[index] =
                Material::mixed(&self.voxels[index].composition, &self.materials);
            report.reacting_revs += 1;
            report.extent.accumulate(&outcome.extent);
            changed = true;
        }
        if changed {
            self.mark_render_changed();
        }
        Ok(report)
    }
}

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
