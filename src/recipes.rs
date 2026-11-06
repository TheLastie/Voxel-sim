//! Data-driven reaction recipes (WORLD_PARADIGM.md sections 4.2, 5 and 7.7,
//! DECISIONS_2026-11-05.md item 18).
//!
//! A [`Recipe`] is data: signed mass coefficients `nu_k` per kilogram of the
//! key reactant (reactants negative, products positive, key exactly `-1`,
//! `sum(nu_k) = 0`) and a [`RateLaw`]. One step computes an extent of reaction
//! `xi` (kg of key reactant consumed) and moves every component mass by
//! `nu_k * xi`. Chemical energy is part of the equation of state
//! (`Material::formation_energy_j_per_kg`), so the heat of the reaction,
//! `-sum(nu_k * h_k)` per kg of key reactant, appears as a change of derived
//! temperature at *constant internal energy*. A step never adds or removes
//! energy and never changes the REV mass.
//!
//! Rate law (SI): `d(xi)/dt = k * m_key` with
//! `k = A exp(-Ea/(R T)) * g(T) * prod_j min(Y_j / Y_ref_j, cap_j) * w(moisture)`.
//! `g` is an optional logistic high-temperature deactivation (microbes,
//! denaturing), `Y_j` are mass fractions of named reactants in the REV, and
//! `w` is a clamped linear function of the water-to-key mass ratio. Hard
//! thresholds: no reaction below `minimum_temperature_k` or above
//! `maximum_temperature_k`. The extent is limited by every consumed reactant
//! (the exhausted reactant goes to exactly zero) and integrated with an
//! exponential limiter and adaptive sub-steps (limited fraction of the key per
//! sub-step and limited heating per sub-step).
//!
//! Built-in registry ([`RecipeRegistry::builtin`]): combustion (see
//! `crate::combustion`), water evaporation (heat-limited, not a rate law),
//! vegetable tanning (`hide + tannin -> leather`), and aerobic decay
//! (`tissue + oxygen -> humus + carbon_dioxide + water`) for hide, flesh,
//! muscle and, thousands of times slower, leather. All rate constants,
//! stoichiometries and component properties of tanning and decay are
//! illustrative engineering values, not calibrated data.
//!
//! Scope, stated plainly: every REV is a *closed* reactor. There is no
//! transport of oxygen, water, tannin, products or microbes between REVs and
//! no microbial population state (the temperature/moisture response of a rate
//! law stands in for it), no pH, no surface-area or diffusion limit, no
//! pressure or volume model. Recipes that share a key reactant split the key
//! in proportion to their rates; recipes that share another reactant (e.g.
//! oxygen) are served in registry order, so the first one wins when it runs
//! short. The tissue components `flesh` and `muscle` are unchanged built-ins
//! with formation energy zero, so tissue carries no chemical energy and the
//! heat of rot comes from the oxygen and carbon-dioxide terms only (about
//! 8 MJ per kg of tissue consumed, the same order as composting). Composition
//! fractions are stored as `f32`; a step that moves less than a few `f32`
//! epsilons of the REV mass is not applied and is counted in
//! [`ReactionReport::deferred_revs`] (call slow chemistry at its own cadence,
//! see [`ReactionClock`]).
//!
//! Heat and heat capacity: the equation of state is `H = cp*T + h_f` from 0 K,
//! so the heat a recipe reports, `-sum(nu_k * h_k)`, is the change of the
//! chemical part only; at constant total energy the derived temperature also
//! moves when `cp` of the mixture changes (the sensible enthalpy `m*cp*T` is
//! what rises by exactly the reported heat). Components meant to react
//! without temperature change must therefore have matching `cp` (leather has
//! the mass-weighted mean `cp` of hide and tannin).
//!
//! Recipes as data: [`crate::recipe_data`] reads and writes a versioned JSON
//! file ([`RecipeSet`](crate::recipe_data::RecipeSet), example
//! `recipes/wood_rot.json`) that names components, so new reactions need no
//! recompiling. `Scene::chemistry_enabled` runs the built-in registry at the
//! fixed-step boundary; [`Scene::react_with`] takes any registry.

use crate::material::Material;
use crate::scene::Scene;
use crate::voxel::Composition;

pub const GAS_CONSTANT_J_MOL_K: f64 = 8.314_462_618_153_24;
/// Boiling point of water at 1 atm, kelvin; matches the built-in `water` entry.
pub const WATER_BOILING_K: f64 = 373.15;
const MAX_SUBSTEPS: usize = 256;
/// A REV step moving less than this fraction of the REV mass would be lost in
/// the `f32` composition fractions and is deferred instead.
const MIN_RELATIVE_EXTENT: f64 = 4.0 * f32::EPSILON as f64;
const MASS_BALANCE_TOLERANCE: f64 = 1e-9;

/// Logistic high-temperature loss of activity: `1 / (1 + exp((T - mid) / w))`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Deactivation {
    pub midpoint_k: f64,
    pub width_k: f64,
}

/// Rate factor `min(Y_component / Y_reference, cap)` from the mass fraction of
/// a named component in the REV. `cap = INFINITY` keeps it unbounded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dependence {
    pub component: u16,
    pub reference_mass_fraction: f64,
    pub cap: f64,
}

/// Moisture factor from the water-to-key mass ratio `r`:
/// `clamp((r - minimum) / (reference - minimum), 0, 1)`; zero when dry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moisture {
    pub water: u16,
    pub minimum_ratio: f64,
    pub reference_ratio: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RateLaw {
    /// Arrhenius pre-exponential factor, 1/s.
    pub pre_exponential_per_s: f64,
    /// Activation energy, J/mol.
    pub activation_energy_j_mol: f64,
    /// Below this temperature (K) the reaction does not run.
    pub minimum_temperature_k: f64,
    /// Above this temperature (K) the reaction does not run (sterilisation,
    /// denaturing). `INFINITY` disables the cutoff.
    pub maximum_temperature_k: f64,
    pub deactivation: Option<Deactivation>,
    pub dependencies: Vec<Dependence>,
    pub moisture: Option<Moisture>,
    /// Largest fraction of the key reactant consumed in one sub-step.
    pub max_fraction_per_substep: f64,
    /// Largest heating or cooling (K) one sub-step may cause.
    pub max_heating_per_substep_k: f64,
}

impl RateLaw {
    pub fn arrhenius(pre_exponential_per_s: f64, activation_energy_j_mol: f64) -> Self {
        Self {
            pre_exponential_per_s,
            activation_energy_j_mol,
            minimum_temperature_k: 0.0,
            maximum_temperature_k: f64::INFINITY,
            deactivation: None,
            dependencies: Vec::new(),
            moisture: None,
            max_fraction_per_substep: 0.25,
            max_heating_per_substep_k: 25.0,
        }
    }

    /// Arrhenius law with the pre-exponential factor chosen so that the plain
    /// Arrhenius rate equals `rate_at_reference_per_s` at `reference_k`.
    pub fn arrhenius_from_reference(
        rate_at_reference_per_s: f64,
        reference_k: f64,
        activation_energy_j_mol: f64,
    ) -> Self {
        let pre =
            rate_at_reference_per_s * (activation_energy_j_mol / (GAS_CONSTANT_J_MOL_K * reference_k)).exp();
        Self::arrhenius(pre, activation_energy_j_mol)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        let finite_pos = |x: f64| x.is_finite() && x > 0.0;
        if !finite_pos(self.pre_exponential_per_s)
            || !self.activation_energy_j_mol.is_finite()
            || self.activation_energy_j_mol < 0.0
        {
            return Err("rate law needs a positive pre-exponential factor and activation energy >= 0");
        }
        if !self.minimum_temperature_k.is_finite()
            || self.minimum_temperature_k < 0.0
            || self.maximum_temperature_k.is_nan()
            || self.maximum_temperature_k <= self.minimum_temperature_k
        {
            return Err("rate law temperature thresholds must satisfy 0 <= minimum < maximum");
        }
        if let Some(d) = self.deactivation {
            if !d.midpoint_k.is_finite() || !finite_pos(d.width_k) {
                return Err("deactivation needs a finite midpoint and positive width");
            }
        }
        for d in &self.dependencies {
            if !finite_pos(d.reference_mass_fraction) || d.cap.is_nan() || d.cap <= 0.0 {
                return Err("reactant dependence needs a positive reference fraction and cap");
            }
        }
        if let Some(m) = self.moisture {
            if !m.minimum_ratio.is_finite()
                || m.minimum_ratio < 0.0
                || !m.reference_ratio.is_finite()
                || m.reference_ratio <= m.minimum_ratio
            {
                return Err("moisture factor needs 0 <= minimum < reference");
            }
        }
        if !(self.max_fraction_per_substep > 0.0 && self.max_fraction_per_substep <= 1.0)
            || !finite_pos(self.max_heating_per_substep_k)
        {
            return Err("sub-step limits must be positive (fraction at most one)");
        }
        Ok(())
    }
}

/// One reaction: signed component mass coefficients per kilogram of the key
/// reactant, plus its rate law.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub name: String,
    /// Key reactant: coefficient exactly `-1`; the rate is proportional to its mass.
    pub key: u16,
    /// `(component, nu)` in kg per kg of key; reactants negative, products
    /// positive. The sum is zero (mass is conserved).
    pub terms: Vec<(u16, f64)>,
    pub rate: RateLaw,
    /// Reported in the legacy `fuel_burned_kg` / `combustion_heat_j` totals.
    pub is_combustion: bool,
}

impl Recipe {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.terms.iter().any(|&(_, nu)| !nu.is_finite() || nu == 0.0) {
            return Err("recipe coefficients must be finite and nonzero");
        }
        for (n, &(id, _)) in self.terms.iter().enumerate() {
            if self.terms[..n].iter().any(|&(other, _)| other == id) {
                return Err("recipe lists a component twice");
            }
        }
        match self.terms.iter().find(|&&(id, _)| id == self.key) {
            Some(&(_, nu)) if nu == -1.0 => {}
            _ => return Err("recipe key reactant needs coefficient -1"),
        }
        if !self.terms.iter().any(|&(_, nu)| nu > 0.0) {
            return Err("recipe needs at least one product");
        }
        let sum: f64 = self.terms.iter().map(|&(_, nu)| nu).sum();
        if sum.abs() > 1e-12 {
            return Err("recipe mass coefficients must sum to zero");
        }
        self.rate.validate()
    }

    pub fn reactants(&self) -> impl Iterator<Item = (u16, f64)> + '_ {
        self.terms.iter().copied().filter(|&(_, nu)| nu < 0.0)
    }

    pub fn products(&self) -> impl Iterator<Item = (u16, f64)> + '_ {
        self.terms.iter().copied().filter(|&(_, nu)| nu > 0.0)
    }

    /// Heat released per kilogram of key reactant, J/kg, from the materials'
    /// formation energies: `-sum(nu_k * h_k)`. Negative means endothermic.
    pub fn heat_of_reaction_j_per_kg(&self, materials: &[Material]) -> f64 {
        -self
            .terms
            .iter()
            .map(|&(id, nu)| nu * materials[id as usize].formation_energy_j_per_kg as f64)
            .sum::<f64>()
    }

    /// Every consumed reactant has mass.
    pub fn can_react(&self, masses: &[(u16, f64)]) -> bool {
        self.reactants().all(|(id, _)| mass_of(masses, id) > 0.0)
    }

    /// Every consumed reactant is a component of `composition`.
    pub fn reactants_present(&self, composition: &Composition) -> bool {
        self.reactants()
            .all(|(id, _)| composition.components.iter().any(|&(c, _)| c == id))
    }

    /// Rate constant `k` in 1/s at `temperature_k` for the given component
    /// masses (`total_kg` is their sum). Zero outside the temperature
    /// thresholds, without the key reactant, or when a factor is zero.
    pub fn rate_per_s(&self, temperature_k: f64, masses: &[(u16, f64)], total_kg: f64) -> f64 {
        let law = &self.rate;
        if !(temperature_k >= law.minimum_temperature_k) || temperature_k > law.maximum_temperature_k {
            return 0.0;
        }
        let key_kg = mass_of(masses, self.key);
        if key_kg <= 0.0 {
            return 0.0;
        }
        let mut k = law.pre_exponential_per_s
            * (-law.activation_energy_j_mol / (GAS_CONSTANT_J_MOL_K * temperature_k)).exp();
        if let Some(d) = law.deactivation {
            k /= 1.0 + ((temperature_k - d.midpoint_k) / d.width_k).exp();
        }
        for d in &law.dependencies {
            let fraction = mass_of(masses, d.component) / total_kg;
            k *= (fraction / d.reference_mass_fraction).min(d.cap);
        }
        if let Some(m) = law.moisture {
            let ratio = mass_of(masses, m.water) / key_kg;
            k *= ((ratio - m.minimum_ratio) / (m.reference_ratio - m.minimum_ratio)).clamp(0.0, 1.0);
        }
        k
    }
}

/// Water -> vapour as a heat-limited process (not a rate law): liquid water in
/// a host whose own equation of state has no vaporization plateau can only
/// evaporate above the boiling point, takes its latent heat
/// (`formation(vapor) - formation(water)`) from the sensible energy and cannot
/// cool the REV below the boiling point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evaporation {
    pub water: u16,
    pub vapor: u16,
    pub enthalpy_j_per_kg: f64,
    pub cp_change: f64,
    pub boiling_k: f64,
}

impl Evaporation {
    pub fn for_materials(materials: &[Material]) -> Option<Self> {
        let water = component_by_name(materials, "water")?;
        let vapor = component_by_name(materials, "water_vapor")?;
        Some(Self::for_components(materials, water, vapor))
    }

    /// Evaporation of component `water` into `vapor` (ids valid in `materials`),
    /// latent heat from the formation energies, boiling point 373.15 K.
    pub fn for_components(materials: &[Material], water: u16, vapor: u16) -> Self {
        let (w, v) = (&materials[water as usize], &materials[vapor as usize]);
        Self {
            water,
            vapor,
            enthalpy_j_per_kg: (v.formation_energy_j_per_kg - w.formation_energy_j_per_kg) as f64,
            cp_change: (v.heat_capacity - w.heat_capacity) as f64,
            boiling_k: WATER_BOILING_K,
        }
    }
}

/// Amounts of one recipe over a step (f64, SI).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RecipeExtent {
    /// Key reactant consumed, kg.
    pub extent_kg: f64,
    /// Chemical energy turned into sensible heat (negative: the reverse), J.
    pub heat_j: f64,
}

/// Integrated amounts of one REV step for every recipe of a registry.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistryExtent {
    pub recipes: Vec<RecipeExtent>,
    pub water_evaporated_kg: f64,
    /// Sensible heat turned into latent chemical energy by evaporation, J.
    pub evaporation_heat_j: f64,
    pub substeps: usize,
}

impl RegistryExtent {
    fn new(recipes: usize) -> Self {
        Self {
            recipes: vec![RecipeExtent::default(); recipes],
            water_evaporated_kg: 0.0,
            evaporation_heat_j: 0.0,
            substeps: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.water_evaporated_kg == 0.0 && self.recipes.iter().all(|r| r.extent_kg == 0.0)
    }

    /// Total mass moved by recipes and evaporation, kg.
    pub fn moved_kg(&self) -> f64 {
        self.water_evaporated_kg + self.recipes.iter().map(|r| r.extent_kg).sum::<f64>()
    }
}

/// What one REV step of a registry did.
#[derive(Clone, Debug)]
pub struct RegistryOutcome {
    pub composition: Composition,
    /// Total internal energy of the REV, identical to the input (J).
    pub internal_energy_j: f64,
    pub extent: RegistryExtent,
    pub temperature_before_k: f32,
    pub temperature_after_k: f32,
}

/// What one combustion REV step did (legacy view of [`RegistryOutcome`]).
#[derive(Clone, Debug)]
pub struct ReactionOutcome {
    pub composition: Composition,
    /// Total internal energy of the REV, identical to the input (J).
    pub internal_energy_j: f64,
    pub extent: ReactionExtent,
    pub temperature_before_k: f32,
    pub temperature_after_k: f32,
}

/// Integrated amounts of the combustion recipe and of evaporation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReactionExtent {
    /// Dry fuel consumed, kg (the extent of the combustion recipe).
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

    pub(crate) fn accumulate(&mut self, other: &Self) {
        self.fuel_burned_kg += other.fuel_burned_kg;
        self.water_evaporated_kg += other.water_evaporated_kg;
        self.combustion_heat_j += other.combustion_heat_j;
        self.evaporation_heat_j += other.evaporation_heat_j;
        self.substeps += other.substeps;
    }
}

/// Per-recipe totals of one [`Scene::react`] call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecipeTotal {
    pub name: String,
    pub extent_kg: f64,
    pub heat_j: f64,
}

/// Totals of one [`Scene::react`] call.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReactionReport {
    pub reacting_revs: usize,
    /// Combustion and evaporation totals (the original fields).
    pub extent: ReactionExtent,
    /// Every recipe that moved mass, in registry order.
    pub recipes: Vec<RecipeTotal>,
    /// REVs whose step was below `f32` composition resolution and was not
    /// applied; the caller should react less often with a larger `dt`.
    pub deferred_revs: usize,
}

/// Recipes plus the evaporation process, with heats of reaction derived from
/// the material table they were built for.
#[derive(Clone, Debug)]
pub struct RecipeRegistry {
    recipes: Vec<Recipe>,
    heats_j_per_kg: Vec<f64>,
    evaporation: Option<Evaporation>,
    relevant: Vec<bool>,
}

impl RecipeRegistry {
    pub fn new(
        materials: &[Material],
        recipes: Vec<Recipe>,
        evaporation: Option<Evaporation>,
    ) -> Result<Self, &'static str> {
        let mut relevant = vec![false; materials.len()];
        for (n, recipe) in recipes.iter().enumerate() {
            recipe.validate()?;
            if recipes[..n].iter().any(|other| other.name == recipe.name) {
                return Err("recipe names must be unique");
            }
            if recipe.terms.iter().any(|&(id, _)| id as usize >= materials.len())
                || recipe.rate.dependencies.iter().any(|d| d.component as usize >= materials.len())
                || recipe.rate.moisture.is_some_and(|m| m.water as usize >= materials.len())
            {
                return Err("recipe refers to a component missing from the material table");
            }
            for (id, _) in recipe.reactants() {
                relevant[id as usize] = true;
            }
        }
        if let Some(e) = evaporation {
            if e.water as usize >= materials.len() || e.vapor as usize >= materials.len() {
                return Err("evaporation refers to a component missing from the material table");
            }
            relevant[e.water as usize] = true;
        }
        let heats_j_per_kg = recipes
            .iter()
            .map(|r| r.heat_of_reaction_j_per_kg(materials))
            .collect();
        Ok(Self {
            recipes,
            heats_j_per_kg,
            evaporation,
            relevant,
        })
    }

    /// Every built-in recipe whose components exist in `materials`:
    /// combustion, tanning and decay, plus evaporation.
    pub fn builtin(materials: &[Material]) -> Self {
        let mut recipes = Vec::new();
        recipes.extend(crate::combustion::combustion_recipe(materials));
        recipes.extend(tanning_recipe(materials));
        recipes.extend(decay_recipes(materials));
        let evaporation = Evaporation::for_materials(materials);
        Self::new(materials, recipes, evaporation).unwrap_or_else(|_| {
            Self::new(materials, Vec::new(), evaporation).unwrap_or(Self {
                recipes: Vec::new(),
                heats_j_per_kg: Vec::new(),
                evaporation: None,
                relevant: vec![false; materials.len()],
            })
        })
    }

    pub fn recipes(&self) -> &[Recipe] {
        &self.recipes
    }

    pub fn evaporation(&self) -> Option<&Evaporation> {
        self.evaporation.as_ref()
    }

    pub fn is_empty(&self) -> bool {
        self.recipes.is_empty() && self.evaporation.is_none()
    }

    pub fn recipe_index(&self, name: &str) -> Option<usize> {
        self.recipes.iter().position(|r| r.name == name)
    }

    pub fn heat_of_reaction_j_per_kg(&self, index: usize) -> f64 {
        self.heats_j_per_kg[index]
    }

    fn is_relevant(&self, id: u16) -> bool {
        self.relevant.get(id as usize).copied().unwrap_or(false)
    }

    /// Combustion/evaporation view of a registry extent.
    pub fn legacy_extent(&self, extent: &RegistryExtent) -> ReactionExtent {
        let mut legacy = ReactionExtent {
            fuel_burned_kg: 0.0,
            water_evaporated_kg: extent.water_evaporated_kg,
            combustion_heat_j: 0.0,
            evaporation_heat_j: extent.evaporation_heat_j,
            substeps: extent.substeps,
        };
        for (recipe, amount) in self.recipes.iter().zip(&extent.recipes) {
            if recipe.is_combustion {
                legacy.fuel_burned_kg += amount.extent_kg;
                legacy.combustion_heat_j += amount.heat_j;
            }
        }
        legacy
    }

    /// Advance one REV. `composition` holds mass fractions of the REV of
    /// `mass_kg`; `internal_energy_j` is its total internal energy including
    /// the chemical part. The REV mass is unchanged and the energy is returned
    /// unchanged; only the composition (and so the derived temperature) moves.
    pub fn step(
        &self,
        materials: &[Material],
        composition: &Composition,
        mass_kg: f32,
        internal_energy_j: f64,
        dt_s: f64,
    ) -> Result<RegistryOutcome, &'static str> {
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
        let before = state_of(materials, &masses, internal_energy_j)?.0 as f32;
        let extent = self.step_masses(materials, &mut masses, internal_energy_j, dt_s)?;
        let after = state_of(materials, &masses, internal_energy_j)?.0 as f32;
        Ok(RegistryOutcome {
            composition: composition_from_masses(&masses),
            internal_energy_j,
            extent,
            temperature_before_k: before,
            temperature_after_k: after,
        })
    }

    /// Core integrator on component masses in kg (f64). The sum of `masses` is
    /// conserved to floating-point rounding (checked; a violation is an
    /// error, never a silent loss). Entries reaching exactly zero stay in the
    /// list with mass 0.
    pub fn step_masses(
        &self,
        materials: &[Material],
        masses: &mut Vec<(u16, f64)>,
        internal_energy_j: f64,
        dt_s: f64,
    ) -> Result<RegistryExtent, &'static str> {
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
        let mass_before: f64 = masses.iter().map(|&(_, m)| m).sum();
        let n = self.recipes.len();
        let mut total = RegistryExtent::new(n);
        let mut rates = vec![0.0_f64; n];
        let mut key_start = vec![0.0_f64; n];
        let mut remaining = dt_s;
        while remaining > 0.0 {
            let m_water = self.evaporation.map_or(0.0, |e| mass_of(masses, e.water));
            let any_can = self.recipes.iter().any(|r| r.can_react(masses));
            if !any_can && m_water <= 0.0 {
                break;
            }
            let (mut temp, mixed) = state_of(materials, masses, internal_energy_j)?;
            let sum: f64 = masses.iter().map(|&(_, m)| m).sum();
            for (i, recipe) in self.recipes.iter().enumerate() {
                rates[i] = if recipe.can_react(masses) {
                    recipe.rate_per_s(temp, masses, sum)
                } else {
                    0.0
                };
            }

            // Sub-step length from the rates at the start state.
            let mut h = remaining;
            if total.substeps + 1 < MAX_SUBSTEPS {
                let (mut heating_w, mut limit_k) = (0.0_f64, f64::INFINITY);
                for (i, recipe) in self.recipes.iter().enumerate() {
                    if rates[i] > 0.0 {
                        h = h.min(recipe.rate.max_fraction_per_substep / rates[i]);
                        heating_w += rates[i] * mass_of(masses, recipe.key) * self.heats_j_per_kg[i].abs();
                        limit_k = limit_k.min(recipe.rate.max_heating_per_substep_k);
                    }
                }
                if heating_w > 0.0 {
                    h = h.min(limit_k * sum * mixed.heat_capacity as f64 / heating_w);
                }
            }
            let h = h.max(remaining * 1e-12).min(remaining);
            let mut active = rates.iter().any(|&r| r > 0.0);

            // 1. Moisture: boil off at constant energy, never below boiling.
            if let Some(e) = self.evaporation {
                if m_water > 0.0 && mixed.boil_temp.is_infinite() && temp > e.boiling_k {
                    let cp = mixed.heat_capacity as f64;
                    let denominator = e.enthalpy_j_per_kg + e.boiling_k * e.cp_change;
                    let energy_limit = if denominator > 0.0 {
                        cp * (temp - e.boiling_k) / denominator * sum
                    } else {
                        0.0
                    };
                    let mut evaporated = energy_limit.min(m_water);
                    if evaporated > 0.0 {
                        if evaporated >= m_water {
                            evaporated = m_water;
                            set_mass(masses, e.water, 0.0);
                        } else {
                            add_mass(masses, e.water, -evaporated);
                        }
                        add_mass(masses, e.vapor, evaporated);
                        total.water_evaporated_kg += evaporated;
                        total.evaporation_heat_j += evaporated * e.enthalpy_j_per_kg;
                        active = true;
                        // The latent heat lowered the temperature; refresh the rates.
                        temp = state_of(materials, masses, internal_energy_j)?.0;
                        for (i, recipe) in self.recipes.iter().enumerate() {
                            rates[i] = if recipe.can_react(masses) {
                                recipe.rate_per_s(temp, masses, sum)
                            } else {
                                0.0
                            };
                        }
                    }
                }
            }

            // 2. Recipes. Recipes sharing a key split its loss `1 - exp(-R h)`
            // in proportion to their rates; every consumed reactant limits the
            // extent and the exhausted one becomes exactly zero.
            for (i, recipe) in self.recipes.iter().enumerate() {
                key_start[i] = mass_of(masses, recipe.key);
            }
            for i in 0..n {
                if rates[i] <= 0.0 {
                    continue;
                }
                let recipe = &self.recipes[i];
                let rate_sum: f64 = (0..n)
                    .filter(|&j| self.recipes[j].key == recipe.key)
                    .map(|j| rates[j])
                    .sum();
                let raw = key_start[i] * -(-rate_sum * h).exp_m1() * (rates[i] / rate_sum);
                let mut extent = raw;
                for (id, nu) in recipe.reactants() {
                    extent = extent.min(mass_of(masses, id) / -nu);
                }
                if extent <= 0.0 {
                    continue;
                }
                let exhausted: Vec<u16> = recipe
                    .reactants()
                    .filter(|&(id, nu)| mass_of(masses, id) / -nu <= extent)
                    .map(|(id, _)| id)
                    .collect();
                for &(id, nu) in &recipe.terms {
                    if exhausted.contains(&id) {
                        set_mass(masses, id, 0.0);
                    } else {
                        add_mass(masses, id, nu * extent);
                    }
                }
                total.recipes[i].extent_kg += extent;
                total.recipes[i].heat_j += extent * self.heats_j_per_kg[i];
                active = true;
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
        let mass_after: f64 = masses.iter().map(|&(_, m)| m).sum();
        if (mass_after - mass_before).abs() > MASS_BALANCE_TOLERANCE * mass_before {
            return Err("reaction step violated the mass balance");
        }
        Ok(total)
    }
}

/// Derived temperature (K) and mixed material of component masses at a given
/// total internal energy (chemical part included).
pub(crate) fn state_of(
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

pub(crate) fn component_by_name(materials: &[Material], name: &str) -> Option<u16> {
    materials
        .iter()
        .position(|m| m.name == name)
        .and_then(|i| u16::try_from(i).ok())
}

pub(crate) fn mass_of(masses: &[(u16, f64)], id: u16) -> f64 {
    masses
        .iter()
        .find(|&&(component, _)| component == id)
        .map_or(0.0, |&(_, m)| m)
}

pub(crate) fn set_mass(masses: &mut Vec<(u16, f64)>, id: u16, value: f64) {
    match masses.iter_mut().find(|(component, _)| *component == id) {
        Some(entry) => entry.1 = value,
        None => masses.push((id, value)),
    }
}

pub(crate) fn add_mass(masses: &mut Vec<(u16, f64)>, id: u16, delta: f64) {
    match masses.iter_mut().find(|(component, _)| *component == id) {
        Some(entry) => entry.1 = (entry.1 + delta).max(0.0),
        None => masses.push((id, delta.max(0.0))),
    }
}

/// Mass fractions of `masses`; components of exactly zero mass are dropped.
pub(crate) fn composition_from_masses(masses: &[(u16, f64)]) -> Composition {
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

/// Vegetable tanning: `hide + tannin -> leather`, 0.35 kg of tannin bound per
/// kg of hide (leather gains weight, as real vegetable-tanned leather does).
/// Slow: about 10 hours half-life at 30 C with tannin and moisture present;
/// activation energy 50 kJ/mol (diffusion-limited uptake); runs from 2 C and
/// stops above 65 C where collagen shrinks and denatures, with the rate
/// already falling above about 55 C (logistic midpoint). Needs water (the hide must be wet for
/// the tannin to penetrate) and tannin (rate proportional to its mass
/// fraction up to 5%). No formation energy is tabulated for the tissue or the
/// tannin: all three components sit at the same reference value (zero), so
/// the derived heat of reaction is exactly zero. Illustrative, not calibrated.
pub fn tanning_recipe(materials: &[Material]) -> Option<Recipe> {
    let (hide, tannin, leather, water) = (
        component_by_name(materials, "hide")?,
        component_by_name(materials, "tannin")?,
        component_by_name(materials, "leather")?,
        component_by_name(materials, "water")?,
    );
    let nu_tannin = 0.35;
    let mut rate = RateLaw::arrhenius_from_reference(2.0e-5, 303.15, 50_000.0);
    rate.minimum_temperature_k = 275.0;
    rate.maximum_temperature_k = 338.0;
    rate.deactivation = Some(Deactivation {
        midpoint_k: 328.0,
        width_k: 3.0,
    });
    rate.dependencies = vec![Dependence {
        component: tannin,
        reference_mass_fraction: 0.05,
        cap: 1.0,
    }];
    rate.moisture = Some(Moisture {
        water,
        minimum_ratio: 0.25,
        reference_ratio: 0.8,
    });
    rate.max_heating_per_substep_k = 5.0;
    Some(Recipe {
        name: "tanning: hide + tannin -> leather".into(),
        key: hide,
        terms: vec![(hide, -1.0), (tannin, -nu_tannin), (leather, 1.0 + nu_tannin)],
        rate,
        is_combustion: false,
    })
}

/// Aerobic decay of organic tissue: `tissue + oxygen -> humus + carbon_dioxide
/// + water`, per kg of tissue (about half of the carbon mineralised, 0.40 kg
/// of humus left, the water coefficient closes the mass balance). Built for
/// hide, flesh and muscle at the same rate order (`1.0e-5` to `1.5e-5` per
/// second at 27 C, half-life 13 to 19 h) and for leather at 1/400 of the hide
/// rate (half-life about a year). Microbial rate response: activation energy
/// 60 kJ/mol (Q10 about 2.2), logistic loss of activity centred at 45 C,
/// no activity at or below 1 C (frozen) or above 70 C (sterilised), and zero
/// activity when dry (water below 0.15 kg per kg of tissue; full rate from
/// 0.6). Oxygen enters as a rate factor up to the air fraction (0.232) and as
/// a consumed reactant. Heating per sub-step is limited to 1 K. The heat comes
/// from the formation energies of oxygen and carbon dioxide because every
/// tissue component and humus sit at the same reference value (zero).
/// Illustrative values, not calibrated; no microbial population state.
pub fn decay_recipes(materials: &[Material]) -> Vec<Recipe> {
    let ids = (
        component_by_name(materials, "oxygen"),
        component_by_name(materials, "carbon_dioxide"),
        component_by_name(materials, "humus"),
        component_by_name(materials, "water"),
    );
    let (Some(oxygen), Some(co2), Some(humus), Some(water)) = ids else {
        return Vec::new();
    };
    let (nu_o2, nu_co2, nu_humus) = (0.93, 0.92, 0.40);
    let nu_water = 1.0 + nu_o2 - nu_co2 - nu_humus;
    let mut recipes = Vec::new();
    for (substrate, k_300_per_s) in [
        ("hide", 1.0e-5),
        ("flesh", 1.5e-5),
        ("muscle", 1.5e-5),
        ("leather", 2.5e-8),
    ] {
        let Some(key) = component_by_name(materials, substrate) else {
            continue;
        };
        let mut rate = RateLaw::arrhenius_from_reference(k_300_per_s, 300.0, 60_000.0);
        rate.minimum_temperature_k = 274.15;
        rate.maximum_temperature_k = 343.15;
        rate.deactivation = Some(Deactivation {
            midpoint_k: 318.0,
            width_k: 4.0,
        });
        rate.dependencies = vec![Dependence {
            component: oxygen,
            reference_mass_fraction: 0.232,
            cap: 1.0,
        }];
        rate.moisture = Some(Moisture {
            water,
            minimum_ratio: 0.15,
            reference_ratio: 0.6,
        });
        rate.max_heating_per_substep_k = 1.0;
        recipes.push(Recipe {
            name: format!("decay of {substrate}"),
            key,
            terms: vec![
                (key, -1.0),
                (oxygen, -nu_o2),
                (co2, nu_co2),
                (humus, nu_humus),
                (water, nu_water),
            ],
            rate,
            is_combustion: false,
        });
    }
    recipes
}

/// Accumulates fixed physical steps so slow chemistry can be advanced at its
/// own cadence: feed every fixed step to [`ReactionClock::advance`] and call
/// [`Scene::react`] with the returned time whenever it is `Some`.
#[derive(Clone, Debug)]
pub struct ReactionClock {
    interval_s: f64,
    pending_s: f64,
}

impl ReactionClock {
    pub fn new(interval_s: f64) -> Result<Self, &'static str> {
        if !interval_s.is_finite() || interval_s <= 0.0 {
            return Err("reaction interval must be positive and finite");
        }
        Ok(Self {
            interval_s,
            pending_s: 0.0,
        })
    }

    /// Add `dt_s` of simulated time; return the accumulated time once at
    /// least one interval has passed (and reset), otherwise `None`.
    pub fn advance(&mut self, dt_s: f64) -> Option<f64> {
        if !dt_s.is_finite() || dt_s < 0.0 {
            return None;
        }
        self.pending_s += dt_s;
        if self.pending_s >= self.interval_s {
            Some(std::mem::take(&mut self.pending_s))
        } else {
            None
        }
    }

    pub fn pending_s(&self) -> f64 {
        self.pending_s
    }
}

impl Scene {
    /// Advance chemical reactions of every REV by `dt` seconds with the
    /// built-in registry (combustion, evaporation, tanning, decay). Call it
    /// once per chemistry step after the physical step (like the thermal
    /// update it works on the stored energy); it is not run by `tick`, so
    /// scenes without reactive components pay a scan only. Each REV is a closed
    /// reactor: component masses change by the recipe coefficients, the REV
    /// mass and stored internal energy are untouched, and `properties` is
    /// recomputed from the new composition, so the heat appears as derived
    /// temperature. Returns the totals.
    ///
    /// Products do not leave the REV (no transport yet), so a closed oxygen
    /// supply runs out. Slow recipes need `dt` large enough to be resolved by
    /// the `f32` composition fractions; see [`ReactionClock`].
    pub fn react(&mut self, dt: f32) -> Result<ReactionReport, &'static str> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("reaction time step must be finite and nonnegative");
        }
        if dt == 0.0 {
            return Ok(ReactionReport::default());
        }
        let registry = RecipeRegistry::builtin(&self.materials);
        self.react_with(&registry, dt)
    }

    /// Like [`Scene::react`] with a caller-supplied registry (built for this
    /// scene's material table).
    pub fn react_with(
        &mut self,
        registry: &RecipeRegistry,
        dt: f32,
    ) -> Result<ReactionReport, &'static str> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("reaction time step must be finite and nonnegative");
        }
        let mut report = ReactionReport::default();
        if registry.is_empty() || dt == 0.0 {
            return Ok(report);
        }
        let mut totals = vec![RecipeExtent::default(); registry.recipes.len()];
        let mut changed = false;
        for index in 0..self.voxels.len() {
            let voxel = &self.voxels[index];
            let components = &voxel.composition.components;
            if !components.iter().any(|&(id, _)| registry.is_relevant(id)) {
                continue;
            }
            // Lowest temperature at which anything present could react.
            let mut threshold = f64::INFINITY;
            for recipe in &registry.recipes {
                if recipe.reactants_present(&voxel.composition) {
                    threshold = threshold.min(recipe.rate.minimum_temperature_k);
                }
            }
            if let Some(e) = &registry.evaporation {
                if components.len() > 1 && components.iter().any(|&(id, _)| id == e.water) {
                    threshold = threshold.min(e.boiling_k);
                }
            }
            if !threshold.is_finite() {
                continue;
            }
            let temperature = self.properties[index]
                .temperature(voxel.mass, voxel.thermal_energy_f32()) as f64;
            if temperature < threshold {
                continue;
            }
            let energy = voxel.total_internal_energy_j();
            let outcome =
                registry.step(&self.materials, &voxel.composition, voxel.mass, energy, dt as f64)?;
            if outcome.extent.is_empty() {
                continue;
            }
            if outcome.extent.moved_kg() < MIN_RELATIVE_EXTENT * voxel.mass as f64 {
                report.deferred_revs += 1;
                continue;
            }
            self.voxels[index].composition = outcome.composition;
            self.properties[index] =
                Material::mixed(&self.voxels[index].composition, &self.materials);
            report.reacting_revs += 1;
            report.extent.accumulate(&registry.legacy_extent(&outcome.extent));
            for (total, amount) in totals.iter_mut().zip(&outcome.extent.recipes) {
                total.extent_kg += amount.extent_kg;
                total.heat_j += amount.heat_j;
            }
            changed = true;
        }
        report.recipes = registry
            .recipes
            .iter()
            .zip(&totals)
            .filter(|(_, t)| t.extent_kg > 0.0)
            .map(|(recipe, t)| RecipeTotal {
                name: recipe.name.clone(),
                extent_kg: t.extent_kg,
                heat_j: t.heat_j,
            })
            .collect();
        if changed {
            self.mark_render_changed();
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combustion::CombustionModel;
    use crate::scene::{SceneConfig, VoxelSpawn};

    const DAY_S: f64 = 86_400.0;

    fn id(table: &[Material], name: &str) -> u16 {
        component_by_name(table, name).unwrap()
    }

    fn masses(table: &[Material], parts: &[(&str, f64)]) -> Vec<(u16, f64)> {
        parts.iter().map(|&(n, m)| (id(table, n), m)).collect()
    }

    fn total(masses: &[(u16, f64)]) -> f64 {
        masses.iter().map(|m| m.1).sum()
    }

    fn energy_at(table: &[Material], masses: &[(u16, f64)], kelvin: f64) -> f64 {
        let mixed = Material::mixed(&composition_from_masses(masses), table);
        mixed.energy_at_temperature(total(masses) as f32, kelvin as f32) as f64
    }

    fn temperature(table: &[Material], masses: &[(u16, f64)], energy: f64) -> f64 {
        state_of(table, masses, energy).unwrap().0
    }

    fn recipe<'a>(registry: &'a RecipeRegistry, name_part: &str) -> (usize, &'a Recipe) {
        let index = registry
            .recipes()
            .iter()
            .position(|r| r.name.contains(name_part))
            .unwrap_or_else(|| panic!("no recipe {name_part}"));
        (index, &registry.recipes()[index])
    }

    /// Run `days` of chemistry on a REV held at `kelvin` by an external heat
    /// sink (test input, like the heater in the combustion ignition test): the
    /// reactions themselves never change the stored energy.
    fn isothermal(
        table: &[Material],
        registry: &RecipeRegistry,
        state: &mut Vec<(u16, f64)>,
        kelvin: f64,
        days: f64,
        step_s: f64,
    ) {
        let steps = (days * DAY_S / step_s).round() as usize;
        for _ in 0..steps {
            let energy = energy_at(table, state, kelvin);
            registry.step_masses(table, state, energy, step_s).unwrap();
        }
    }

    #[test]
    fn builtin_registry_holds_combustion_tanning_decay_and_evaporation() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        assert!(registry.evaporation().is_some());
        let names: Vec<&str> = registry.recipes().iter().map(|r| r.name.as_str()).collect();
        for part in [
            "dry_wood",
            "tanning",
            "decay of hide",
            "decay of flesh",
            "decay of muscle",
            "decay of leather",
        ] {
            assert!(names.iter().any(|n| n.contains(part)), "{part} missing from {names:?}");
        }
        for recipe in registry.recipes() {
            recipe.validate().unwrap();
            let sum: f64 = recipe.terms.iter().map(|t| t.1).sum();
            assert!(sum.abs() < 1e-12, "{}: {sum}", recipe.name);
            assert!(recipe.reactants().count() >= 2 && recipe.products().count() >= 1);
        }
        // A table without the organic components still builds (older scenes).
        let old: Vec<Material> = table[..21].to_vec();
        let old_registry = RecipeRegistry::builtin(&old);
        assert_eq!(old_registry.recipes().len(), 1);
    }

    #[test]
    fn invalid_recipes_are_rejected() {
        let table = Material::table();
        let good = tanning_recipe(&table).unwrap();
        let mut bad = good.clone();
        bad.terms[2].1 += 0.01;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.terms[0].1 = -2.0;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.rate.pre_exponential_per_s = 0.0;
        assert!(bad.validate().is_err());
        let mut bad = good.clone();
        bad.rate.moisture = Some(Moisture {
            water: 4,
            minimum_ratio: 1.0,
            reference_ratio: 0.5,
        });
        assert!(bad.validate().is_err());
        let mut bad = good;
        bad.terms[2].0 = 9999;
        assert!(RecipeRegistry::new(&table, vec![bad], None).is_err());
    }

    #[test]
    fn heat_of_reaction_comes_from_formation_energies() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (wood, _) = recipe(&registry, "dry_wood");
        let q = registry.heat_of_reaction_j_per_kg(wood);
        assert!((q - 17.0e6).abs() < 0.17e6, "combustion {q}");
        // Tanning: tissue, tannin and leather share one reference value, so
        // the derived heat is exactly zero.
        let (tan, tanning) = recipe(&registry, "tanning");
        assert_eq!(registry.heat_of_reaction_j_per_kg(tan), 0.0);
        assert_eq!(tanning.heat_of_reaction_j_per_kg(&table), 0.0);
        // Decay: exothermic, from the O2 and CO2 terms: several MJ per kg of
        // tissue, well below combustion of dry wood.
        for name in ["decay of hide", "decay of flesh", "decay of muscle", "decay of leather"] {
            let (i, r) = recipe(&registry, name);
            let q = registry.heat_of_reaction_j_per_kg(i);
            let expected = 0.93 * 15_678_100.0 - 0.92 * 6_736_800.0;
            assert!((q - expected).abs() < 1.0e3, "{name}: {q}");
            assert!(q > 5.0e6 && q < 12.0e6);
            assert!(r.rate.max_heating_per_substep_k <= 1.0);
        }
    }

    #[test]
    fn every_recipe_step_conserves_mass_moves_masses_by_nu_xi_and_keeps_energy() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        // (recipe, REV contents with only that recipe's reactants, K, dt)
        let cases: Vec<(&str, Vec<(&str, f64)>, f64, f64)> = vec![
            ("dry_wood", vec![("dry_wood", 1.0e-3), ("oxygen", 1.5e-3)], 700.0, 1.0),
            (
                "tanning",
                vec![("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 1.0e-3)],
                303.0,
                0.5 * DAY_S,
            ),
            (
                "decay of hide",
                vec![("hide", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                2.0 * DAY_S,
            ),
            (
                "decay of flesh",
                vec![("flesh", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                2.0 * DAY_S,
            ),
            (
                "decay of muscle",
                vec![("muscle", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                2.0 * DAY_S,
            ),
            (
                "decay of leather",
                vec![("leather", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                400.0 * DAY_S,
            ),
        ];
        for (name, parts, kelvin, dt) in cases {
            let (index, r) = recipe(&registry, name);
            let initial = masses(&table, &parts);
            let mut state = initial.clone();
            let energy = energy_at(&table, &state, kelvin);
            let mass_before = total(&state);
            let extent = registry.step_masses(&table, &mut state, energy, dt).unwrap();
            let xi = extent.recipes[index].extent_kg;
            assert!(xi > 0.0, "{name} did not react");
            for (i, e) in extent.recipes.iter().enumerate() {
                assert!(i == index || e.extent_kg == 0.0, "{name}: recipe {i} also ran");
            }
            assert_eq!(extent.water_evaporated_kg, 0.0);
            assert!((total(&state) - mass_before).abs() <= 1e-12 * mass_before, "{name}");
            for &(component, nu) in &r.terms {
                let expected = mass_of(&initial, component) + nu * xi;
                assert!(
                    (mass_of(&state, component) - expected).abs() <= 1e-12 * mass_before,
                    "{name}: component {component}"
                );
            }
            // Energy is untouched: rebuilding it from the new composition at
            // the new derived temperature returns the stored energy, and the
            // sensible gain equals the reported heat.
            let t_after = temperature(&table, &state, energy);
            let rebuilt = energy_at(&table, &state, t_after);
            assert!((rebuilt - energy).abs() <= 1e-5 * energy, "{name}: {rebuilt} vs {energy}");
            let heat = extent.recipes[index].heat_j;
            let expected_heat = xi * registry.heat_of_reaction_j_per_kg(index);
            assert!((heat - expected_heat).abs() <= 1e-9 * heat.abs().max(1.0));
            let mixed_before = Material::mixed(&composition_from_masses(&initial), &table);
            let mixed_after = Material::mixed(&composition_from_masses(&state), &table);
            let gain = mass_before
                * (mixed_after.heat_capacity as f64 * t_after
                    - mixed_before.heat_capacity as f64 * kelvin);
            assert!(
                (gain - heat).abs() <= 3e-3 * heat.abs() + 1e-6 * energy,
                "{name}: sensible gain {gain} vs released {heat}"
            );
        }
    }

    #[test]
    fn hide_tans_only_with_tannin_and_only_while_wet() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let leather = id(&table, "leather");
        let run = |parts: &[(&str, f64)], kelvin: f64| {
            let mut state = masses(&table, parts);
            isothermal(&table, &registry, &mut state, kelvin, 3.0, 3600.0);
            mass_of(&state, leather)
        };
        let with = run(&[("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 1.0e-3)], 303.0);
        let without = run(&[("hide", 1.0e-3), ("water", 1.0e-3)], 303.0);
        let dry = run(&[("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 5.0e-5)], 303.0);
        assert!(with > 2.0e-4, "leather made {with}");
        assert_eq!(without, 0.0);
        assert_eq!(dry, 0.0);
        // Not instantaneous: hours to days. After 20 minutes only a few percent.
        let mut state = masses(&table, &[("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 1.0e-3)]);
        isothermal(&table, &registry, &mut state, 303.0, 20.0 / 1440.0, 60.0);
        let early = mass_of(&state, leather);
        assert!(early > 0.0 && early < 0.05 * 1.35e-3, "after 20 min {early}");
    }

    #[test]
    fn tanning_is_faster_when_warm_but_stops_when_hot() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (_, tanning) = recipe(&registry, "tanning");
        let state = masses(&table, &[("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 1.0e-3)]);
        let sum = total(&state);
        let rate = |k: f64| tanning.rate_per_s(k, &state, sum);
        assert_eq!(rate(270.0), 0.0);
        assert!(rate(283.0) > 0.0);
        assert!(rate(303.0) > 2.0 * rate(283.0));
        assert!(rate(313.0) > rate(303.0));
        // Hot: falls off and then stops (collagen shrinks above ~65 C).
        assert!(rate(335.0) < 0.5 * rate(320.0));
        assert_eq!(rate(339.0), 0.0);
        assert_eq!(rate(345.0), 0.0);
        // The same ordering in integrated leather mass over six hours.
        let leather = id(&table, "leather");
        let made = |kelvin: f64| {
            let mut s = state.clone();
            let energy = energy_at(&table, &s, kelvin);
            registry.step_masses(&table, &mut s, energy, 6.0 * 3600.0).unwrap();
            mass_of(&s, leather)
        };
        let (cold, warm, hot) = (made(283.0), made(308.0), made(345.0));
        assert!(warm > 2.0 * cold && cold > 0.0, "cold {cold} warm {warm}");
        assert_eq!(hot, 0.0);
    }

    #[test]
    fn tanned_leather_survives_what_rots_untanned_hide() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (hide, leather) = (id(&table, "hide"), id(&table, "leather"));
        // Two identical wet, aerated REVs at 27 C for ten days; one holds
        // hide, the other the same mass of leather.
        let mut raw = masses(&table, &[("hide", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let mut tanned = masses(&table, &[("leather", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let (mass_raw, mass_tanned) = (total(&raw), total(&tanned));
        isothermal(&table, &registry, &mut raw, 300.0, 10.0, 3600.0);
        isothermal(&table, &registry, &mut tanned, 300.0, 10.0, 3600.0);
        let hide_left = mass_of(&raw, hide) / 1.0e-3;
        let leather_left = mass_of(&tanned, leather) / 1.0e-3;
        assert!(hide_left < 0.05, "hide left {hide_left}");
        assert!(leather_left > 0.97, "leather left {leather_left}");
        // Rot product, CO2 and water appear; mass is conserved.
        assert!(mass_of(&raw, id(&table, "humus")) > 0.3 * 0.95e-3);
        assert!(mass_of(&raw, id(&table, "carbon_dioxide")) > 0.5e-3);
        assert!((total(&raw) - mass_raw).abs() <= 1e-12 * mass_raw);
        assert!((total(&tanned) - mass_tanned).abs() <= 1e-12 * mass_tanned);
        // Tan first, then the same ten days: the tanned hide survives.
        let mut both = masses(
            &table,
            &[("hide", 1.0e-3), ("tannin", 4.0e-4), ("oxygen", 3.0e-3), ("water", 1.0e-3)],
        );
        isothermal(&table, &registry, &mut both, 303.0, 3.0, 3600.0);
        let hide_after_tanning = mass_of(&both, hide);
        assert!(hide_after_tanning < 0.05e-3, "hide left after tanning {hide_after_tanning}");
        let leather_made = mass_of(&both, leather);
        isothermal(&table, &registry, &mut both, 300.0, 10.0, 3600.0);
        assert!(mass_of(&both, leather) > 0.95 * leather_made);
    }

    #[test]
    fn frozen_or_dried_tissue_does_not_rot() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        for tissue in ["hide", "flesh", "muscle", "leather"] {
            // Frozen, wet, aerated.
            let mut frozen = masses(&table, &[(tissue, 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
            let initial = frozen.clone();
            let energy = energy_at(&table, &frozen, 265.0);
            let extent = registry.step_masses(&table, &mut frozen, energy, 400.0 * DAY_S).unwrap();
            assert!(extent.is_empty(), "{tissue} rotted frozen");
            assert_eq!(frozen, initial);
            // Dried: warm, aerated, almost no water.
            let mut dried = masses(&table, &[(tissue, 1.0e-3), ("oxygen", 2.0e-3), ("water", 8.0e-5)]);
            let initial = dried.clone();
            let energy = energy_at(&table, &dried, 300.0);
            let extent = registry.step_masses(&table, &mut dried, energy, 400.0 * DAY_S).unwrap();
            assert!(extent.is_empty(), "{tissue} rotted dry");
            assert_eq!(dried, initial);
            // No oxygen: no aerobic decay.
            let mut sealed = masses(&table, &[(tissue, 1.0e-3), ("water", 1.0e-3)]);
            let initial = sealed.clone();
            let energy = energy_at(&table, &sealed, 300.0);
            registry.step_masses(&table, &mut sealed, energy, 400.0 * DAY_S).unwrap();
            assert_eq!(sealed, initial);
        }
    }

    #[test]
    fn microbial_rate_peaks_at_mild_temperature_and_vanishes_when_cold_or_sterilising() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (_, decay) = recipe(&registry, "decay of hide");
        let state = masses(&table, &[("hide", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let sum = total(&state);
        let rate = |k: f64| decay.rate_per_s(k, &state, sum);
        let temps: Vec<f64> = (2750..=3500).step_by(10).map(|t| t as f64 / 10.0).collect();
        let peak = temps
            .iter()
            .copied()
            .fold(temps[0], |best, t| if rate(t) > rate(best) { t } else { best });
        assert!((295.0..=322.0).contains(&peak), "peak at {peak} K");
        assert_eq!(rate(250.0), 0.0);
        assert_eq!(rate(273.15), 0.0);
        assert!(rate(285.0) > 0.0 && rate(285.0) < 0.5 * rate(peak));
        assert!(rate(335.0) < 0.1 * rate(peak));
        assert_eq!(rate(343.2), 0.0);
        assert_eq!(rate(400.0), 0.0);
        // Hide rots on a day scale at 27 C: half-life in hours.
        let half_life_h = std::f64::consts::LN_2 / rate(300.0) / 3600.0;
        assert!((8.0..40.0).contains(&half_life_h), "half-life {half_life_h} h");
        // Leather is at least 100 times slower.
        let (_, leather) = recipe(&registry, "decay of leather");
        let lstate = masses(&table, &[("leather", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        assert!(rate(300.0) > 100.0 * leather.rate_per_s(300.0, &lstate, total(&lstate)));
    }

    #[test]
    fn decay_self_heating_is_bounded_by_sterilisation() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (index, decay) = recipe(&registry, "decay of hide");
        let max_k = decay.rate.maximum_temperature_k;
        // An adiabatic closed REV: no heat leaves, so rot warms it until the
        // microbes die at the cutoff; it must never run away.
        let mut state = masses(&table, &[("hide", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let initial = state.clone();
        let hide = id(&table, "hide");
        let energy = energy_at(&table, &state, 300.0);
        let mut peak = 300.0_f64;
        let mut consumed = 0.0;
        for _ in 0..60 {
            let extent = registry.step_masses(&table, &mut state, energy, DAY_S).unwrap();
            consumed += extent.recipes[index].extent_kg;
            peak = peak.max(temperature(&table, &state, energy));
            assert!(extent.substeps < MAX_SUBSTEPS, "sub-step budget exhausted");
        }
        assert!(peak > 305.0, "no self-heating at all: {peak}");
        assert!(peak <= max_k + 1.5, "peak {peak} K above sterilisation {max_k} K");
        assert!(consumed < 0.1 * mass_of(&initial, hide), "consumed {consumed}");
        // Afterwards the process is dead: nothing more happens.
        let more = registry.step_masses(&table, &mut state, energy, 30.0 * DAY_S).unwrap();
        assert!(more.recipes[index].extent_kg < 1e-3 * consumed, "still rotting {more:?}");
        // One single 30-day call behaves the same (no overshoot from a long dt).
        let mut once = initial.clone();
        registry.step_masses(&table, &mut once, energy, 30.0 * DAY_S).unwrap();
        let t_once = temperature(&table, &once, energy);
        assert!(t_once <= max_k + 1.5, "one-call peak {t_once}");
        // Leather in the same closed REV for the same 30 days: bounded by the
        // same cutoff, and only a small fraction of it rots.
        let mut leather = masses(&table, &[("leather", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let energy = energy_at(&table, &leather, 300.0);
        registry.step_masses(&table, &mut leather, energy, 30.0 * DAY_S).unwrap();
        assert!(temperature(&table, &leather, energy) <= max_k + 1.5);
        assert!(mass_of(&leather, id(&table, "leather")) > 0.9e-3);
    }

    #[test]
    fn combustion_runs_unchanged_as_one_recipe_of_the_registry() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let model = CombustionModel::for_materials(&table).unwrap();
        let (index, _) = recipe(&registry, "dry_wood");
        let parts = [("dry_wood", 1.0e-3), ("oxygen", 1.5e-3), ("water", 2.0e-5)];
        let mut via_registry = masses(&table, &parts);
        let mut via_model = via_registry.clone();
        let energy = energy_at(&table, &via_registry, 600.0);
        let a = registry.step_masses(&table, &mut via_registry, energy, 1.0).unwrap();
        let b = model.step_masses(&table, &mut via_model, energy, 1.0).unwrap();
        assert!(b.fuel_burned_kg > 0.0 && b.water_evaporated_kg > 0.0);
        assert_eq!(a.recipes[index].extent_kg.to_bits(), b.fuel_burned_kg.to_bits());
        assert_eq!(a.water_evaporated_kg.to_bits(), b.water_evaporated_kg.to_bits());
        assert_eq!(via_registry, via_model);
        // Another recipe in the same REV does not disturb it.
        let mut mixed = masses(
            &table,
            &[("dry_wood", 1.0e-3), ("oxygen", 1.5e-3), ("hide", 1.0e-3), ("water", 1.0e-3)],
        );
        let energy = energy_at(&table, &mixed, 700.0);
        let before = total(&mixed);
        let extent = registry.step_masses(&table, &mut mixed, energy, 1.0).unwrap();
        assert!(extent.recipes[index].extent_kg > 0.0);
        assert!((total(&mixed) - before).abs() <= 1e-12 * before);
    }

    #[test]
    fn recipes_sharing_a_key_reactant_split_it_and_conserve_mass() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let (tan, _) = recipe(&registry, "tanning");
        let (rot, _) = recipe(&registry, "decay of hide");
        let mut state = masses(
            &table,
            &[("hide", 1.0e-3), ("tannin", 4.0e-4), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
        );
        let before = total(&state);
        let hide0 = mass_of(&state, id(&table, "hide"));
        let energy = energy_at(&table, &state, 303.0);
        let extent = registry.step_masses(&table, &mut state, energy, 3.0 * 3600.0).unwrap();
        let (a, b) = (extent.recipes[tan].extent_kg, extent.recipes[rot].extent_kg);
        assert!(a > 0.0 && b > 0.0 && a > 2.0 * b, "tanning {a}, rot {b}");
        assert!((hide0 - mass_of(&state, id(&table, "hide")) - (a + b)).abs() < 1e-12 * before);
        assert!((total(&state) - before).abs() <= 1e-12 * before);
    }

    #[test]
    fn scene_react_runs_all_recipes_on_a_few_closed_revs() {
        let table = Material::table();
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.001;
        let make = |cell: i32, parts: &[(&str, f64)], kelvin: f32| {
            let sum: f64 = parts.iter().map(|p| p.1).sum();
            let mut spawn = VoxelSpawn::at_cell([cell, 0, 0], id(&table, parts[0].0));
            spawn.composition = Composition {
                components: parts
                    .iter()
                    .map(|&(n, m)| (id(&table, n), (m / sum) as f32))
                    .collect(),
            };
            spawn.temperature_kelvin = kelvin;
            spawn
        };
        let spawns = vec![
            make(0, &[("hide", 1.0), ("oxygen", 2.0), ("water", 1.0)], 300.0), // rots
            make(2, &[("leather", 1.0), ("oxygen", 2.0), ("water", 1.0)], 300.0), // resists
            make(4, &[("hide", 1.0), ("tannin", 0.4), ("water", 1.0)], 305.0), // tans
            make(6, &[("hide", 1.0), ("tannin", 0.4), ("oxygen", 2.0)], 265.0), // cold and dry
            make(8, &[("iron", 1.0)], 300.0),                                  // inert
        ];
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        let ledger = |scene: &Scene| -> Vec<(u32, u32, u64)> {
            scene
                .voxels
                .iter()
                .map(|v| {
                    (
                        v.mass.to_bits(),
                        v.internal_energy.to_bits(),
                        v.internal_energy_remainder_j.to_bits(),
                    )
                })
                .collect()
        };
        let before = ledger(&scene);
        let fraction = |scene: &Scene, voxel: usize, name: &str| {
            let component = id(&scene.materials, name);
            scene.voxels[voxel]
                .composition
                .components
                .iter()
                .find(|&&(c, _)| c == component)
                .map_or(0.0, |&(_, f)| f as f64)
        };
        let (hide0, leather0) = (fraction(&scene, 0, "hide"), fraction(&scene, 1, "leather"));
        let mut deferred = 0;
        let mut totals: Vec<RecipeTotal> = Vec::new();
        for _ in 0..48 {
            // Half-hour chemistry cadence for one simulated day.
            let r = scene.react(1800.0).unwrap();
            deferred += r.deferred_revs;
            for t in r.recipes {
                match totals.iter_mut().find(|x| x.name == t.name) {
                    Some(x) => {
                        x.extent_kg += t.extent_kg;
                        x.heat_j += t.heat_j;
                    }
                    None => totals.push(t),
                }
            }
        }
        // Mass and stored energy are untouched, bit for bit, in every REV.
        assert_eq!(before, ledger(&scene));
        for (i, voxel) in scene.voxels.iter().enumerate() {
            voxel.composition.validate(scene.materials.len()).unwrap();
            let mixed = Material::mixed(&voxel.composition, &scene.materials);
            assert_eq!(scene.properties[i].heat_capacity.to_bits(), mixed.heat_capacity.to_bits());
        }
        assert!(fraction(&scene, 0, "hide") < hide0, "hide did not rot");
        assert!(fraction(&scene, 0, "humus") > 0.0);
        assert!(fraction(&scene, 1, "leather") > 0.99 * leather0);
        assert!(fraction(&scene, 2, "leather") > 0.1, "no leather made");
        assert!(fraction(&scene, 2, "hide") < 0.5 * (1.0 / 2.4), "hide left {}", fraction(&scene, 2, "hide"));
        assert_eq!(fraction(&scene, 3, "leather"), 0.0);
        assert_eq!(fraction(&scene, 3, "humus"), 0.0);
        assert_eq!(scene.voxels[4].composition.components.len(), 1);
        let extent_of = |part: &str| {
            totals
                .iter()
                .filter(|t| t.name.contains(part))
                .map(|t| t.extent_kg)
                .sum::<f64>()
        };
        assert!(extent_of("tanning") > 0.0);
        assert!(extent_of("decay of hide") > 0.0);
        assert!(extent_of("decay of leather") < 0.05 * extent_of("decay of hide"), "leather {} hide {}", extent_of("decay of leather"), extent_of("decay of hide"));
        assert_eq!(deferred, 0);
        // The rotting REV warmed (closed, no loss) but stayed bounded; the
        // tanning REV did not change temperature (zero heat).
        let t_rot = scene.temperature_kelvin(0).unwrap();
        assert!(t_rot > 300.5 && t_rot < 345.0, "rot REV at {t_rot} K");
        let t_tan = scene.temperature_kelvin(2).unwrap();
        assert!((t_tan - 305.0).abs() < 0.2, "tanning REV at {t_tan} K");
        assert_eq!(scene.temperature_kelvin(3).unwrap().round(), 265.0);
    }

    #[test]
    fn steps_below_f32_resolution_are_deferred_and_reported_not_applied() {
        let table = Material::table();
        let mut spawn = VoxelSpawn::at_cell([0, 0, 0], id(&table, "hide"));
        spawn.composition = Composition {
            components: smallvec::smallvec![
                (id(&table, "hide"), 0.25),
                (id(&table, "oxygen"), 0.5),
                (id(&table, "water"), 0.25)
            ],
        };
        spawn.temperature_kelvin = 300.0;
        let mut scene = Scene::from_voxels(SceneConfig::default(), vec![spawn]).unwrap();
        let before = scene.voxels[0].composition.components.clone();
        let report = scene.react(0.001).unwrap();
        assert_eq!(report.reacting_revs, 0);
        assert_eq!(report.deferred_revs, 1);
        assert_eq!(scene.voxels[0].composition.components, before);
        // At a chemistry cadence it resolves.
        let report = scene.react(3600.0).unwrap();
        assert_eq!(report.reacting_revs, 1);
        assert_eq!(report.deferred_revs, 0);
        assert!(!report.recipes.is_empty());
    }

    fn chemical_energy_j(table: &[Material], masses: &[(u16, f64)]) -> f64 {
        masses
            .iter()
            .map(|&(id, m)| m * table[id as usize].formation_energy_j_per_kg as f64)
            .sum()
    }

    #[test]
    fn released_heat_equals_the_drop_in_chemical_energy_and_total_energy_is_unchanged() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let cases: Vec<(&str, Vec<(&str, f64)>, f64, f64)> = vec![
            ("dry_wood", vec![("dry_wood", 1.0e-3), ("oxygen", 1.5e-3)], 700.0, 2.0),
            ("tanning", vec![("hide", 1.0e-3), ("tannin", 4.0e-4), ("water", 1.0e-3)], 303.0, DAY_S),
            (
                "decay of hide",
                vec![("hide", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                2.0 * DAY_S,
            ),
            (
                "decay of muscle",
                vec![("muscle", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)],
                300.0,
                2.0 * DAY_S,
            ),
        ];
        for (name, parts, kelvin, dt) in cases {
            let (index, _) = recipe(&registry, name);
            let state0 = masses(&table, &parts);
            let energy = energy_at(&table, &state0, kelvin);
            // Reactions never add or remove energy: the REV-level step returns
            // the stored total as given.
            let composition = composition_from_masses(&state0);
            let outcome = registry
                .step(&table, &composition, total(&state0) as f32, energy, dt)
                .unwrap();
            assert_eq!(outcome.internal_energy_j.to_bits(), energy.to_bits());
            // Chemical energy before and after, from the f64 masses.
            let mut state1 = state0.clone();
            let extent = registry.step_masses(&table, &mut state1, energy, dt).unwrap();
            assert!(extent.recipes[index].extent_kg > 0.0, "{name} did not react");
            let drop = chemical_energy_j(&table, &state0) - chemical_energy_j(&table, &state1);
            let released = extent.recipes[index].heat_j;
            assert!(
                (drop - released).abs() <= 1e-9 * released.abs() + 1e-9,
                "{name}: chemical energy dropped {drop} J, released {released} J"
            );
            // Sensible (thermal) enthalpy rose by exactly that amount.
            let sensible_before = energy - chemical_energy_j(&table, &state0);
            let sensible_after = energy - chemical_energy_j(&table, &state1);
            assert!(
                (sensible_after - sensible_before - released).abs() <= 1e-9 * released.abs() + 1e-6,
                "{name}"
            );
            // Exothermic recipes (combustion, decay) really released heat.
            assert_eq!(released > 0.0, name != "tanning", "{name}: {released}");
        }
    }

    #[test]
    fn mass_is_conserved_over_thousands_of_steps_and_all_recipes() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        // Hide, tannin, flesh, leather, wood fuel, oxygen and water in one REV:
        // tanning, three kinds of rot and combustion compete for reactants.
        let mut state = masses(
            &table,
            &[
                ("hide", 1.0e-3),
                ("tannin", 3.0e-4),
                ("flesh", 5.0e-4),
                ("leather", 2.0e-4),
                ("dry_wood", 4.0e-4),
                ("oxygen", 4.0e-3),
                ("water", 1.5e-3),
            ],
        );
        let before = total(&state);
        let mut moved = 0.0;
        for step in 0..3000 {
            // Alternate warm and hot so that several recipes run in turn.
            let kelvin = if step % 3 == 0 { 700.0 } else { 303.0 };
            let energy = energy_at(&table, &state, kelvin);
            let extent = registry.step_masses(&table, &mut state, energy, 600.0).unwrap();
            moved += extent.moved_kg();
            assert!(state.iter().all(|&(_, m)| m >= 0.0 && m.is_finite()));
            assert!((total(&state) - before).abs() <= 1e-13 * before, "step {step}");
        }
        assert!(moved > 1.0e-3, "recipes barely ran: {moved}");
    }

    #[test]
    fn tanning_raises_strength_and_lowers_the_rot_rate() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let wet_hide = [("hide", 1.0e-3), ("tannin", 4.0e-4), ("oxygen", 3.0e-3), ("water", 1.0e-3)];
        let kelvin = 303.0;
        // Specific rot rate of a REV, kg of tissue per second per kg of REV,
        // summed over every decay recipe at the given state.
        let rot_rate = |state: &[(u16, f64)]| -> f64 {
            let sum = total(state);
            registry
                .recipes()
                .iter()
                .filter(|r| r.name.starts_with("decay of"))
                .map(|r| r.rate_per_s(kelvin, state, sum) * mass_of(state, r.key) / sum)
                .sum()
        };
        let mut state = masses(&table, &wet_hide);
        // Strength and stiffness of the hide/leather solid only (water, tannin,
        // humus and gases excluded), so the comparison is not diluted by mass shares.
        let strength = |state: &[(u16, f64)]| {
            let organic: Vec<(u16, f64)> = state
                .iter()
                .copied()
                .filter(|&(c, _)| ["hide", "leather"].iter().any(|n| id(&table, n) == c))
                .collect();
            let mixed = Material::mixed(&composition_from_masses(&organic), &table);
            (mixed.tensile_strength, mixed.young)
        };
        let whole = |state: &[(u16, f64)]| {
            Material::mixed(&composition_from_masses(state), &table).tensile_strength
        };
        let (tensile_before, young_before) = strength(&state);
        let (rot_before, whole_before) = (rot_rate(&state), whole(&state));
        assert!(rot_before > 0.0);
        isothermal(&table, &registry, &mut state, kelvin, 3.0, 3600.0);
        let (tensile_after, young_after) = strength(&state);
        let (rot_after, whole_after) = (rot_rate(&state), whole(&state));
        let (leather, hide) = (mass_of(&state, id(&table, "leather")), mass_of(&state, id(&table, "hide")));
        assert!(leather / (leather + hide) > 0.9, "tanned only {}", leather / (leather + hide));
        assert!(tensile_after > 1.5 * tensile_before, "{tensile_before} -> {tensile_after}");
        assert!(young_after > 10.0 * young_before, "{young_before} -> {young_after}");
        // Even the whole REV, with its water, tannin and gas, got stronger.
        assert!(whole_after > whole_before, "{whole_before} -> {whole_after}");
        assert!(rot_after < 0.05 * rot_before, "rot rate {rot_before} -> {rot_after}");
        // Material-level cross-check: the same moist aerated REV made of
        // leather instead of hide rots at least 100 times slower.
        let leather_rev = masses(&table, &[("leather", 1.0e-3), ("oxygen", 3.0e-3), ("water", 1.0e-3)]);
        let hide_rev = masses(&table, &[("hide", 1.0e-3), ("oxygen", 3.0e-3), ("water", 1.0e-3)]);
        assert!(rot_rate(&hide_rev) > 100.0 * rot_rate(&leather_rev));
    }

    #[test]
    fn rot_rate_rises_with_temperature_and_with_moisture() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        for tissue in ["hide", "flesh", "muscle"] {
            let (_, decay) = recipe(&registry, &format!("decay of {tissue}"));
            let reference = masses(&table, &[(tissue, 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
            let sum = total(&reference);
            // Temperature: strictly rising from near freezing to the mild
            // optimum (every step of 1 K between 275 and 305 K).
            let mut previous = 0.0;
            for kelvin in 275..=305 {
                let rate = decay.rate_per_s(kelvin as f64, &reference, sum);
                assert!(rate > previous, "{tissue}: rate at {kelvin} K is {rate}, below {previous}");
                previous = rate;
            }
            // Moisture: zero when dry, strictly rising while the tissue wets
            // up (water-to-tissue mass ratio 0.2 .. 0.55), flat once saturated.
            let rate_with_water = |ratio: f64| {
                let state =
                    masses(&table, &[(tissue, 1.0e-3), ("oxygen", 2.0e-3), ("water", ratio * 1.0e-3)]);
                decay.rate_per_s(300.0, &state, total(&state))
            };
            assert_eq!(rate_with_water(0.1), 0.0);
            let mut previous = 0.0;
            for ratio in [0.2, 0.3, 0.4, 0.5, 0.55] {
                let rate = rate_with_water(ratio);
                assert!(rate > previous, "{tissue}: rate at water ratio {ratio} is {rate}");
                previous = rate;
            }
            // Past the reference ratio the moisture factor is flat; the oxygen
            // factor is capped at the air fraction, so only the oxygen share
            // (not the water) could move the rate and it does not.
            assert!(rate_with_water(1.0) >= rate_with_water(0.6));
            assert!((rate_with_water(1.0) / rate_with_water(2.0) - 1.0).abs() < 1e-12);
            // Integrated over 2.4 hours the same orderings hold in consumed tissue
            // (short enough that the water made by the rot does not saturate the
            // moisture factor of the drier REV).
            // At a held temperature: a closed REV left to itself would warm up
            // as it rots (see decay_self_heating_is_bounded_by_sterilisation).
            let consumed = |kelvin: f64, ratio: f64| {
                let mut state =
                    masses(&table, &[(tissue, 1.0e-3), ("oxygen", 2.0e-3), ("water", ratio * 1.0e-3)]);
                isothermal(&table, &registry, &mut state, kelvin, 0.1, 300.0);
                1.0e-3 - mass_of(&state, id(&table, tissue))
            };
            assert!(consumed(280.0, 1.0) < consumed(290.0, 1.0));
            assert!(consumed(290.0, 1.0) < consumed(300.0, 1.0));
            assert!(consumed(300.0, 0.25) < consumed(300.0, 0.4));
            assert!(consumed(300.0, 0.4) < consumed(300.0, 1.0));
            assert_eq!(consumed(300.0, 0.1), 0.0);
        }
    }

    #[test]
    fn reaction_clock_accumulates_fixed_steps() {
        let mut clock = ReactionClock::new(1.0).unwrap();
        let mut fired = 0.0;
        let mut calls = 0;
        for _ in 0..2500 {
            if let Some(dt) = clock.advance(0.001) {
                fired += dt;
                calls += 1;
            }
        }
        assert_eq!(calls, 2);
        assert!((fired + clock.pending_s() - 2.5).abs() < 1e-9);
        assert!(ReactionClock::new(0.0).is_err());
        assert_eq!(clock.advance(f64::NAN), None);
    }
}
