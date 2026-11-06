//! Resource-based, event-driven plant growth (pure model, no rendering, no physics).
//!
//! This module implements the model required by `INTEGRATION_PLAN_2026-11-05.md`
//! section 4 and 8.4 and by `TREE_GROWTH_MODEL.md`. The older generator in
//! `world.rs` rebuilds a tree from genome and age; here a plant is a *state*:
//!
//! * stores of carbohydrate (carbon and chemical energy), water and minerals (kg);
//! * biomass pools roots, trunk, branches, leaves, fruit (dry kg);
//! * age, damage and the time of the next event.
//!
//! Dormancy, maturity, death, height, wood density and stiffness are never stored
//! as truth. They are functions of that state (and of the conditions) and are
//! recomputed whenever asked.
//!
//! # Conservation
//! Every kilogram and every joule moves between the plant, an explicit
//! [`Environment`] ledger and (for energy) the sun/heat ledger. Photosynthesis uses
//! the stoichiometry `CO2 + H2O -> CH2O + O2` (44 + 18 = 30 + 32), respiration is
//! its reverse. All transfers go through [`Plant::apply`], which moves each
//! amount out of one account and into another (clamped by what the source holds),
//! so `plant mass + environment mass - imported mass` is invariant to f64 rounding
//! whatever the step size. Energy: `radiation_in - heat_out` equals the change of
//! chemical energy held by plants and litter.
//!
//! # Time
//! The unit of time is the day (f64). [`Plant::advance`] splits the requested span
//! at events (season change, maturation, branching of the first orders,
//! senescence, death) and integrates the continuous model between them with an
//! adaptive explicit midpoint scheme. Slow store dynamics use exponential or
//! closed-form updates, so a long step over constant conditions costs a number of
//! substeps set by how fast the plant changes, not by the span length. A long step
//! is an integration, never a stopped clock.
//!
//! # Provisional coefficients
//! Biology constants below (efficiencies, maintenance rates, `wood E(rho)`
//! exponent, resorption fractions) are plausible orders of magnitude, not
//! calibrated data. They are collected in this file so they can be calibrated.

use crate::material::Material;

pub const YEAR_DAYS: f64 = 365.0;
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Chemical energy of carbohydrate (CH2O basis), J per kg (about glucose).
pub const SUGAR_ENERGY_J_PER_KG: f64 = 15.6e6;
/// Chemical energy of organic structural tissue, J per kg of organic mass.
pub const TISSUE_ENERGY_J_PER_KG: f64 = 17.5e6;
/// Kilograms of structural organic mass built per kilogram of sugar spent; the
/// remainder is respired (construction cost). `SUGAR_ENERGY / YIELD >= TISSUE_ENERGY`
/// guarantees the construction heat is non-negative.
pub const GROWTH_YIELD: f64 = 0.8;
const CO2_PER_SUGAR: f64 = 44.0 / 30.0;
const H2O_PER_SUGAR: f64 = 18.0 / 30.0;
const O2_PER_SUGAR: f64 = 32.0 / 30.0;

const STANDARD_GRAVITY: f64 = 9.80665;
/// Reference wood (European oak starter profile in `material.rs`): density and
/// longitudinal Young modulus at about 12% moisture.
const OAK_DENSITY_KG_M3: f64 = 625.0;
const OAK_YOUNG_PA: f64 = 12.088e9;
/// Provisional exponent of `E ~ rho^n` across wood and herbaceous tissue.
const STIFFNESS_DENSITY_EXPONENT: f64 = 1.5;

/// Photosynthetic conversion of absorbed PAR to chemical energy at low light.
const LIGHT_USE_EFFICIENCY: f64 = 0.05;
/// Beer-Lambert extinction coefficient of the canopy.
const CANOPY_EXTINCTION: f64 = 0.6;
/// Absorbed flux per unit leaf area where light use efficiency halves, W/m2.
const LIGHT_SATURATION_W_M2: f64 = 120.0;
/// Densest leaf packing: leaf area per crown ground area.
const MAX_LEAF_AREA_INDEX: f64 = 4.0;
/// Cold threshold below which growth is dormant, K.
const COLD_THRESHOLD_K: f64 = 278.0;

/// Relative change of any pool allowed in one substep.
const STEP_RELATIVE_CHANGE: f64 = 0.1;
const STEP_MIN_DAYS: f64 = 0.01;
const STEP_MAX_DAYS: f64 = 30.0;
/// Ring width at which new wood has its base density (typical temperate ring), m/year.
const REFERENCE_RING_WIDTH_M_PER_YEAR: f64 = 0.002;
/// Rate at which the sugar store is converted to growth, 1/day.
const GROWTH_CONVERSION_RATE: f64 = 0.15;

// ---------------------------------------------------------------------------------
// Environment ledger
// ---------------------------------------------------------------------------------

/// The patch of world a plant draws from and returns mass and heat to.
///
/// Soil water and minerals are finite and shared by every plant stepped against the
/// same `Environment`. The atmosphere reservoirs are large but finite so that the
/// ledger is closed. Mass and energy that enter from outside the ledger (rain, seeds,
/// litter placed by a scene) are recorded in `imported_*`.
#[derive(Clone, Debug)]
pub struct Environment {
    pub co2_kg: f64,
    pub o2_kg: f64,
    pub vapor_kg: f64,
    pub soil_water_kg: f64,
    /// Field capacity used to turn soil water into a relative saturation.
    pub soil_water_capacity_kg: f64,
    pub soil_minerals_kg: f64,
    /// Soil mineral mass at which root uptake reaches half its capacity.
    pub soil_mineral_half_saturation_kg: f64,
    pub litter_organic_kg: f64,
    pub litter_mineral_kg: f64,
    pub litter_energy_j: f64,
    /// Sun energy absorbed by plant surfaces (input to the energy ledger).
    pub radiation_in_j: f64,
    /// Heat returned to the surroundings.
    pub heat_out_j: f64,
    pub imported_mass_kg: f64,
    pub imported_energy_j: f64,
}

impl Environment {
    pub fn new(soil_water_kg: f64, soil_minerals_kg: f64) -> Self {
        assert!(soil_water_kg >= 0.0 && soil_minerals_kg >= 0.0);
        Self {
            co2_kg: 1.0e5,
            o2_kg: 3.0e5,
            vapor_kg: 1.0e3,
            soil_water_kg,
            soil_water_capacity_kg: soil_water_kg.max(1.0),
            soil_minerals_kg,
            soil_mineral_half_saturation_kg: (0.25 * soil_minerals_kg).max(1e-9),
            litter_organic_kg: 0.0,
            litter_mineral_kg: 0.0,
            litter_energy_j: 0.0,
            radiation_in_j: 0.0,
            heat_out_j: 0.0,
            imported_mass_kg: 0.0,
            imported_energy_j: 0.0,
        }
    }

    /// Mass held by all environment accounts (kg).
    pub fn total_mass(&self) -> f64 {
        self.co2_kg
            + self.o2_kg
            + self.vapor_kg
            + self.soil_water_kg
            + self.soil_minerals_kg
            + self.litter_organic_kg
            + self.litter_mineral_kg
    }

    /// Mass invariant: `plants + environment - imported`. Constant in time.
    pub fn closed_mass(&self, plant_mass_kg: f64) -> f64 {
        plant_mass_kg + self.total_mass() - self.imported_mass_kg
    }

    /// Energy residual (J): chemical energy of plants and litter minus
    /// `radiation_in - heat_out + imported`. Zero up to rounding.
    pub fn energy_residual(&self, plant_chemical_energy_j: f64) -> f64 {
        plant_chemical_energy_j + self.litter_energy_j
            - (self.radiation_in_j - self.heat_out_j + self.imported_energy_j)
    }

    /// Add rainfall to the soil. Recorded as an import.
    pub fn add_rain(&mut self, kg: f64) {
        assert!(kg >= 0.0);
        self.soil_water_kg += kg;
        self.imported_mass_kg += kg;
    }

    /// Add dead organic matter (for example a log a scene placed) to the litter.
    pub fn add_litter(&mut self, organic_kg: f64, mineral_kg: f64) {
        assert!(organic_kg >= 0.0 && mineral_kg >= 0.0);
        self.litter_organic_kg += organic_kg;
        self.litter_mineral_kg += mineral_kg;
        self.litter_energy_j += organic_kg * TISSUE_ENERGY_J_PER_KG;
        self.imported_mass_kg += organic_kg + mineral_kg;
        self.imported_energy_j += organic_kg * TISSUE_ENERGY_J_PER_KG;
    }

    /// Remove soil water (drainage, a scene pumping it out). Recorded as an export so the
    /// mass invariant still holds. Returns the amount actually removed.
    pub fn drain_soil_water(&mut self, kg: f64) -> f64 {
        let removed = kg.max(0.0).min(self.soil_water_kg);
        self.soil_water_kg -= removed;
        self.imported_mass_kg -= removed;
        removed
    }

    fn soil_saturation(&self) -> f64 {
        (self.soil_water_kg / self.soil_water_capacity_kg).clamp(0.0, 1.0)
    }

    /// Fraction of root capacity that can reach water (0 below the wilting point).
    fn water_availability(&self) -> f64 {
        let s = (self.soil_saturation() - 0.05).max(0.0);
        (s / (s + 0.1) * 1.1).min(1.0)
    }

    fn mineral_availability(&self) -> f64 {
        let m = self.soil_minerals_kg.max(0.0);
        m / (m + self.soil_mineral_half_saturation_kg)
    }

    fn litter_availability(&self) -> f64 {
        let l = self.litter_organic_kg.max(0.0);
        l / (l + 0.1)
    }

    /// Respire `sugar_kg` of plant sugar: O2 in, CO2 and water vapour out.
    /// Mass change of the environment is exactly `+sugar_kg`.
    fn respire_mass(&mut self, sugar_kg: f64) {
        self.o2_kg -= O2_PER_SUGAR * sugar_kg;
        self.co2_kg += CO2_PER_SUGAR * sugar_kg;
        self.vapor_kg += H2O_PER_SUGAR * sugar_kg;
    }
}

// ---------------------------------------------------------------------------------
// Genome
// ---------------------------------------------------------------------------------

pub use crate::world::SeedGenome;

/// Growth-habit genes. Together with the allocation and size genes of
/// [`SeedGenome`] they span one continuous space in which tree, shrub, grass, vine
/// and mushroom-like forms are regions, not separate code paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HabitGenes {
    /// 0 herbaceous tissue, 1 fully lignified wood: sets tissue density and the
    /// share of stem that is dead structural wood.
    pub lignification: f32,
    /// Target height over diameter of a stem. Stems are limited by Greenhill
    /// buckling from the stiffness and density the plant actually built.
    pub slenderness: f32,
    /// 0 self-supporting, 1 relies on external support (relaxes the buckling limit).
    pub climbing: f32,
    /// Target growth share of leaves, roots (or mycelium) and stem.
    pub leaf_fraction: f32,
    pub root_fraction: f32,
    pub stem_fraction: f32,
    /// 1 photosynthetic, 0 takes its carbon from dead organic matter in the soil.
    pub autotrophy: f32,
    /// Leaf area per leaf mass, m2/kg.
    pub specific_leaf_area: f32,
    /// Share of growth sent to fruit once mature.
    pub reproductive_fraction: f32,
    /// Age at maturity as a fraction of longevity.
    pub maturity_fraction: f32,
    /// Sign and size of the ring-density response to growth rate: positive means
    /// fast growth lays denser wood (ring-porous), negative lighter (conifer-like).
    pub ring_response: f32,
}

/// Complete heritable description of a plant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlantGenome {
    pub seed: SeedGenome,
    pub habit: HabitGenes,
}

/// Named regions of the parameter space, used for presets and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HabitPreset {
    Tree,
    Shrub,
    Grass,
    Vine,
    Mushroom,
}

/// Inclusive bounds of one gene.
#[derive(Clone, Copy, Debug)]
pub struct GeneSpec {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
}

const fn g(name: &'static str, min: f64, max: f64) -> GeneSpec {
    GeneSpec { name, min, max }
}

pub const GENE_COUNT: usize = 33;

/// Explicit bounds of every gene. Mutation and crossover never leave them.
pub const GENE_SPECS: [GeneSpec; GENE_COUNT] = [
    g("maximum_height_m", 0.02, 60.0),
    g("crown_spread_m", 0.05, 28.0),
    g("mature_trunk_radius_m", 0.002, 1.4),
    g("shade_tolerance", 0.0, 1.0),
    g("water_demand", 0.0, 1.0),
    g("nutrient_demand", 0.0, 1.0),
    g("longevity_years", 0.05, 500.0),
    g("moisture_optimum", 0.0, 1.0),
    g("temperature_optimum_kelvin", 265.0, 315.0),
    g("root_count", 1.0, 16.0),
    g("root_extent_m", 0.05, 6.0),
    g("root_depth_factor", 0.05, 2.0),
    g("trunk_height_fraction", 0.05, 1.0),
    g("branching_rate", 0.0, 1.0),
    g("apical_dominance", 0.0, 1.0),
    g("branch_spacing_m", 0.01, 3.0),
    g("branch_slope", 0.0, 1.5),
    g("branch_width", 0.01, 1.0),
    g("leaf_retention", 0.0, 1.0),
    g("mycorrhizal_affinity", 0.0, 1.0),
    g("germination_delay_years", 0.0, 5.0),
    g("mutation_probability", 0.0, 0.01),
    g("lignification", 0.0, 1.0),
    g("slenderness", 2.0, 400.0),
    g("climbing", 0.0, 1.0),
    g("leaf_fraction", 0.0, 0.9),
    g("root_fraction", 0.02, 0.8),
    g("stem_fraction", 0.0, 0.8),
    g("autotrophy", 0.0, 1.0),
    g("specific_leaf_area", 2.0, 60.0),
    g("reproductive_fraction", 0.0, 0.6),
    g("maturity_fraction", 0.01, 0.6),
    g("ring_response", -0.3, 0.3),
];

/// Probability that a mutation of a locus is an extreme jump anywhere in the
/// gene's full range instead of a small step.
pub const EXTREME_MUTATION_PROBABILITY: f64 = 0.02;
/// Standard deviation of an ordinary mutation step as a fraction of the gene range.
pub const MUTATION_STEP_RANGE_FRACTION: f64 = 0.05;

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn hash_words(words: &[u64]) -> u64 {
    let mut h = 0xA076_1D64_78BD_642F_u64;
    for &w in words {
        h = splitmix(h ^ splitmix(w));
    }
    h
}

fn unit(h: u64) -> f64 {
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal from two hashed uniforms (Box-Muller); deterministic.
fn hashed_normal(h1: u64, h2: u64) -> f64 {
    let u1 = unit(h1).max(1e-12);
    let u2 = unit(h2);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl PlantGenome {
    /// Gene values in [`GENE_SPECS`] order.
    pub fn genes(&self) -> [f64; GENE_COUNT] {
        let s = &self.seed;
        let h = &self.habit;
        [
            s.maximum_height_m as f64,
            s.crown_spread_m as f64,
            s.mature_trunk_radius_m as f64,
            s.shade_tolerance as f64,
            s.water_demand as f64,
            s.nutrient_demand as f64,
            s.longevity_years as f64,
            s.moisture_optimum as f64,
            s.temperature_optimum_kelvin as f64,
            s.root_count as f64,
            s.root_extent_m as f64,
            s.root_depth_factor as f64,
            s.trunk_height_fraction as f64,
            s.branching_rate as f64,
            s.apical_dominance as f64,
            s.branch_spacing_m as f64,
            s.branch_slope as f64,
            s.branch_width as f64,
            s.leaf_retention as f64,
            s.mycorrhizal_affinity as f64,
            s.germination_delay_years as f64,
            s.mutation_probability as f64,
            h.lignification as f64,
            h.slenderness as f64,
            h.climbing as f64,
            h.leaf_fraction as f64,
            h.root_fraction as f64,
            h.stem_fraction as f64,
            h.autotrophy as f64,
            h.specific_leaf_area as f64,
            h.reproductive_fraction as f64,
            h.maturity_fraction as f64,
            h.ring_response as f64,
        ]
    }

    /// Rebuild a genome from gene values, clamping every one into its bounds.
    /// Identity fields (`individual_id`, `morphotype_id`) come from `self`.
    pub fn with_genes(&self, values: &[f64; GENE_COUNT]) -> PlantGenome {
        let v: Vec<f64> = values
            .iter()
            .zip(GENE_SPECS.iter())
            .map(|(x, spec)| if x.is_finite() { x.clamp(spec.min, spec.max) } else { spec.min })
            .collect();
        PlantGenome {
            seed: SeedGenome {
                individual_id: self.seed.individual_id,
                morphotype_id: self.seed.morphotype_id,
                maximum_height_m: v[0] as f32,
                crown_spread_m: v[1] as f32,
                mature_trunk_radius_m: v[2] as f32,
                shade_tolerance: v[3] as f32,
                water_demand: v[4] as f32,
                nutrient_demand: v[5] as f32,
                longevity_years: v[6] as f32,
                moisture_optimum: v[7] as f32,
                temperature_optimum_kelvin: v[8] as f32,
                root_count: v[9].round().clamp(1.0, 16.0) as u8,
                root_extent_m: v[10] as f32,
                root_depth_factor: v[11] as f32,
                trunk_height_fraction: v[12] as f32,
                branching_rate: v[13] as f32,
                apical_dominance: v[14] as f32,
                branch_spacing_m: v[15] as f32,
                branch_slope: v[16] as f32,
                branch_width: v[17] as f32,
                leaf_retention: v[18] as f32,
                mycorrhizal_affinity: v[19] as f32,
                germination_delay_years: v[20] as f32,
                mutation_probability: v[21] as f32,
            },
            habit: HabitGenes {
                lignification: v[22] as f32,
                slenderness: v[23] as f32,
                climbing: v[24] as f32,
                leaf_fraction: v[25] as f32,
                root_fraction: v[26] as f32,
                stem_fraction: v[27] as f32,
                autotrophy: v[28] as f32,
                specific_leaf_area: v[29] as f32,
                reproductive_fraction: v[30] as f32,
                maturity_fraction: v[31] as f32,
                ring_response: v[32] as f32,
            },
        }
    }

    /// True when every gene lies inside [`GENE_SPECS`].
    pub fn in_bounds(&self) -> bool {
        self.genes()
            .iter()
            .zip(GENE_SPECS.iter())
            .all(|(x, spec)| x.is_finite() && *x >= spec.min && *x <= spec.max)
    }

    /// Copy of the genome with every gene forced into bounds.
    pub fn clamped(&self) -> PlantGenome {
        self.with_genes(&self.genes())
    }

    /// Offspring genome by deterministic mutation. Each locus mutates with
    /// `per_locus_probability` (default: the parent's own `mutation_probability`).
    /// The effect is a small normal step, or with probability
    /// [`EXTREME_MUTATION_PROBABILITY`] a jump to anywhere in the gene's range, so rare
    /// extremes occur but the value never leaves its bounds. All randomness is a hash
    /// of (world seed, parent, offspring, generation, locus).
    pub fn mutate(
        &self,
        world_seed: u64,
        offspring_id: u64,
        generation: u32,
        per_locus_probability: Option<f64>,
    ) -> PlantGenome {
        let probability = per_locus_probability
            .unwrap_or(self.seed.mutation_probability as f64)
            .clamp(0.0, 1.0);
        let mut genes = self.genes();
        for (locus, value) in genes.iter_mut().enumerate() {
            let base = [
                world_seed,
                self.seed.individual_id,
                offspring_id,
                generation as u64,
                locus as u64,
            ];
            let mut words = base.to_vec();
            words.push(1);
            if unit(hash_words(&words)) >= probability {
                continue;
            }
            let spec = GENE_SPECS[locus];
            let range = spec.max - spec.min;
            words[5] = 2;
            let extreme = unit(hash_words(&words)) < EXTREME_MUTATION_PROBABILITY;
            words[5] = 3;
            let h3 = hash_words(&words);
            words[5] = 4;
            let h4 = hash_words(&words);
            *value = if extreme {
                spec.min + range * unit(h3)
            } else {
                *value + hashed_normal(h3, h4) * range * MUTATION_STEP_RANGE_FRACTION
            };
        }
        let mut child = self.with_genes(&genes);
        child.seed.individual_id = hash_words(&[world_seed, self.seed.individual_id, offspring_id, generation as u64]);
        child
    }

    /// Offspring of two parents by deterministic per-locus crossover: take either
    /// parent's value or a blend of the two. Convex combinations stay in bounds.
    pub fn crossover(
        a: &PlantGenome,
        b: &PlantGenome,
        world_seed: u64,
        offspring_id: u64,
        generation: u32,
    ) -> PlantGenome {
        let ga = a.genes();
        let gb = b.genes();
        let mut genes = ga;
        for locus in 0..GENE_COUNT {
            let words = [
                world_seed,
                a.seed.individual_id,
                b.seed.individual_id,
                offspring_id,
                generation as u64,
                locus as u64,
            ];
            let h = hash_words(&words);
            let r = unit(h);
            genes[locus] = if r < 0.4 {
                ga[locus]
            } else if r < 0.8 {
                gb[locus]
            } else {
                let w = unit(splitmix(h));
                ga[locus] * w + gb[locus] * (1.0 - w)
            };
        }
        let mut child = a.with_genes(&genes);
        child.seed.individual_id = hash_words(&[
            world_seed,
            a.seed.individual_id,
            b.seed.individual_id,
            offspring_id,
            generation as u64,
        ]);
        child
    }

    /// A genome in the named region of the parameter space.
    pub fn preset(habit: HabitPreset, individual_id: u64) -> PlantGenome {
        // (max height, spread, trunk radius, shade tol, water demand, nutrient demand,
        //  longevity, leaf retention, apical dominance, branching rate)
        let (hmax, spread, radius, shade, water, nutrient, longevity, retention, apical, branching) =
            match habit {
                HabitPreset::Tree => (20.0, 6.0, 0.35, 0.4, 0.5, 0.4, 150.0, 0.2, 0.8, 0.6),
                HabitPreset::Shrub => (3.0, 2.5, 0.05, 0.5, 0.4, 0.4, 30.0, 0.5, 0.3, 0.8),
                HabitPreset::Grass => (0.6, 1.2, 0.004, 0.2, 0.5, 0.5, 3.0, 0.1, 0.5, 0.2),
                HabitPreset::Vine => (12.0, 3.0, 0.03, 0.4, 0.5, 0.4, 25.0, 0.3, 0.9, 0.3),
                HabitPreset::Mushroom => (0.15, 0.12, 0.01, 0.8, 0.2, 0.3, 5.0, 0.0, 0.5, 0.0),
            };
        let h = match habit {
            //            lig   slend climb leaf  root  stem  auto  sla   repro mature ring
            HabitPreset::Tree => (0.85, 45.0, 0.0, 0.25, 0.20, 0.55, 1.0, 14.0, 0.10, 0.12, 0.15),
            HabitPreset::Shrub => (0.60, 25.0, 0.0, 0.35, 0.25, 0.40, 1.0, 16.0, 0.10, 0.15, 0.10),
            HabitPreset::Grass => (0.05, 120.0, 0.0, 0.60, 0.35, 0.05, 1.0, 25.0, 0.15, 0.20, 0.0),
            HabitPreset::Vine => (0.30, 250.0, 1.0, 0.35, 0.20, 0.45, 1.0, 20.0, 0.10, 0.15, 0.05),
            HabitPreset::Mushroom => (0.0, 4.0, 0.0, 0.0, 0.55, 0.15, 0.0, 2.0, 0.50, 0.02, 0.0),
        };
        PlantGenome {
            seed: SeedGenome {
                individual_id,
                morphotype_id: 0,
                maximum_height_m: hmax,
                crown_spread_m: spread,
                mature_trunk_radius_m: radius,
                shade_tolerance: shade,
                water_demand: water,
                nutrient_demand: nutrient,
                longevity_years: longevity,
                moisture_optimum: 0.5,
                temperature_optimum_kelvin: 288.0,
                root_count: 6,
                root_extent_m: 1.0,
                root_depth_factor: 0.6,
                trunk_height_fraction: 0.7,
                branching_rate: branching,
                apical_dominance: apical,
                branch_spacing_m: 0.8,
                branch_slope: 0.4,
                branch_width: 0.3,
                leaf_retention: retention,
                mycorrhizal_affinity: 0.3,
                germination_delay_years: 0.1,
                mutation_probability: 0.0001,
            },
            habit: HabitGenes {
                lignification: h.0,
                slenderness: h.1,
                climbing: h.2,
                leaf_fraction: h.3,
                root_fraction: h.4,
                stem_fraction: h.5,
                autotrophy: h.6,
                specific_leaf_area: h.7,
                reproductive_fraction: h.8,
                maturity_fraction: h.9,
                ring_response: h.10,
            },
        }
    }
}

/// Values derived once from the genome (pure function of the genes).
#[derive(Clone, Copy, Debug)]
struct Params {
    /// Mineral share of dry tissue mass.
    mineral_fraction: f64,
    /// kg of water transpired per kg of sugar assimilated.
    water_per_sugar: f64,
    specific_leaf_area: f64,
    lignification: f64,
    /// Density of newly laid structural tissue before the ring response, kg/m3.
    base_density: f64,
    maturity_age_days: f64,
    longevity_days: f64,
    max_height: f64,
    temperature_optimum: f64,
    autotrophy: f64,
    uptake_scale: f64,
}

impl Params {
    fn from_genome(genome: &PlantGenome) -> Self {
        let s = &genome.seed;
        let h = &genome.habit;
        let longevity_days = (s.longevity_years as f64 * YEAR_DAYS).max(1.0);
        Self {
            mineral_fraction: 0.01 + 0.04 * (s.nutrient_demand as f64).clamp(0.0, 1.0),
            water_per_sugar: 120.0 + 400.0 * (s.water_demand as f64).clamp(0.0, 1.0),
            specific_leaf_area: (h.specific_leaf_area as f64).max(1.0),
            lignification: (h.lignification as f64).clamp(0.0, 1.0),
            base_density: 150.0 + 550.0 * (h.lignification as f64).clamp(0.0, 1.0),
            maturity_age_days: h.maturity_fraction as f64 * longevity_days,
            longevity_days,
            max_height: (s.maximum_height_m as f64).max(0.01),
            temperature_optimum: s.temperature_optimum_kelvin as f64,
            autotrophy: (h.autotrophy as f64).clamp(0.0, 1.0),
            uptake_scale: (s.root_extent_m as f64 / 1.0).clamp(0.2, 3.0),
        }
    }
}

// ---------------------------------------------------------------------------------
// Conditions and climate
// ---------------------------------------------------------------------------------

/// Conditions at the plant. Shade from neighbours enters as `light_fraction`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditions {
    /// Share of full-sun irradiance reaching the top of the canopy, 0..1.
    pub light_fraction: f64,
    pub temperature_k: f64,
    /// Photosynthetically active irradiance in full sun, 24 h mean, W/m2.
    pub full_sun_par_w_m2: f64,
}

impl Conditions {
    pub const SUMMER: Conditions = Conditions { light_fraction: 1.0, temperature_k: 291.0, full_sun_par_w_m2: 90.0 };
    pub const WINTER: Conditions = Conditions { light_fraction: 1.0, temperature_k: 270.0, full_sun_par_w_m2: 25.0 };

    pub fn with_light_fraction(mut self, light_fraction: f64) -> Self {
        self.light_fraction = light_fraction.clamp(0.0, 1.0);
        self
    }
}

/// Piecewise-constant climate: constant, or a warm and a cold season per year.
#[derive(Clone, Copy, Debug)]
pub struct Climate {
    warm: Conditions,
    cold: Conditions,
    seasonal: bool,
    year_days: f64,
    warm_start_day: f64,
    warm_days: f64,
}

impl Climate {
    pub fn constant(conditions: Conditions) -> Self {
        Self { warm: conditions, cold: conditions, seasonal: false, year_days: YEAR_DAYS, warm_start_day: 0.0, warm_days: YEAR_DAYS }
    }

    /// Warm season of `warm_days` starting `warm_start_day` into each year, cold otherwise.
    pub fn seasonal(warm: Conditions, cold: Conditions, warm_start_day: f64, warm_days: f64) -> Self {
        assert!(warm_start_day >= 0.0 && warm_days > 0.0 && warm_start_day + warm_days <= YEAR_DAYS);
        Self { warm, cold, seasonal: true, year_days: YEAR_DAYS, warm_start_day, warm_days }
    }

    pub fn at(&self, time_days: f64) -> Conditions {
        if !self.seasonal {
            return self.warm;
        }
        let phase = time_days.rem_euclid(self.year_days);
        if phase >= self.warm_start_day && phase < self.warm_start_day + self.warm_days {
            self.warm
        } else {
            self.cold
        }
    }

    /// Next time strictly after `time_days` at which the conditions change.
    pub fn next_change_after(&self, time_days: f64) -> f64 {
        if !self.seasonal {
            return f64::INFINITY;
        }
        let year_start = (time_days / self.year_days).floor() * self.year_days;
        let candidates = [
            year_start + self.warm_start_day,
            year_start + self.warm_start_day + self.warm_days,
            year_start + self.year_days + self.warm_start_day,
        ];
        candidates.into_iter().find(|c| *c > time_days + 1e-9).unwrap_or(f64::INFINITY)
    }
}

// ---------------------------------------------------------------------------------
// Plant state
// ---------------------------------------------------------------------------------

/// Dry biomass pools, kg.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pools {
    pub roots: f64,
    pub trunk: f64,
    pub branches: f64,
    pub leaves: f64,
    pub fruit: f64,
}

const R: usize = 0;
const T: usize = 1;
const B: usize = 2;
const L: usize = 3;
const F: usize = 4;

impl Pools {
    pub fn total(&self) -> f64 {
        self.roots + self.trunk + self.branches + self.leaves + self.fruit
    }
    fn to_array(self) -> [f64; 5] {
        [self.roots, self.trunk, self.branches, self.leaves, self.fruit]
    }
    fn from_array(a: [f64; 5]) -> Self {
        Self { roots: a[R], trunk: a[T], branches: a[B], leaves: a[L], fruit: a[F] }
    }
}

/// Non-structural stores, kg.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stores {
    /// Carbohydrate: the plant's carbon and chemical-energy reserve.
    pub sugar: f64,
    pub water: f64,
    pub minerals: f64,
}

impl Stores {
    pub fn total(&self) -> f64 {
        self.sugar + self.water + self.minerals
    }
}

/// One annual layer of structural wood (trunk and branches together).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RingLayer {
    pub mass_kg: f64,
    pub volume_m3: f64,
}

impl RingLayer {
    pub fn density(&self) -> f64 {
        if self.volume_m3 > 0.0 { self.mass_kg / self.volume_m3 } else { 0.0 }
    }
}

/// Kinds of events that bound an integration segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    SeasonChange,
    Branching,
    Maturation,
    Senescence,
    Death,
}

/// Result of one [`Plant::advance`] call.
#[derive(Clone, Debug, Default)]
pub struct AdvanceReport {
    /// Integration substeps taken.
    pub substeps: usize,
    /// Events the plant reached, with their times (days).
    pub events: Vec<(f64, EventKind)>,
    /// New branch orders produced inside the span.
    pub new_branches: u32,
}

/// Wood properties derived from the state of the rings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WoodProperties {
    /// Mean dry density of structural wood, kg/m3.
    pub density_kg_m3: f64,
    /// Young modulus from density, Pa.
    pub young_modulus_pa: f64,
    pub ring_count: usize,
    /// Bending stiffness EI of the trunk with its radial density profile, N m2.
    pub bending_stiffness_ei: f64,
    pub trunk_diameter_m: f64,
    pub trunk_height_m: f64,
}

impl WoodProperties {
    /// A copy of `base` (for example the `oak` entry of `Material::table()`) with
    /// density and Young modulus replaced by this plant's wood.
    pub fn to_material(&self, base: &Material) -> Material {
        let mut material = base.clone();
        material.density = self.density_kg_m3 as f32;
        material.young = self.young_modulus_pa as f32;
        material
    }
}

/// Form derived from the state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Morphology {
    pub height_m: f64,
    pub trunk_diameter_m: f64,
    pub crown_area_m2: f64,
    pub leaf_area_m2: f64,
    pub leaf_area_index: f64,
    pub branch_count: u32,
}

/// Young modulus (Pa) of tissue with dry density `density_kg_m3`.
pub fn young_modulus_from_density(density_kg_m3: f64) -> f64 {
    if density_kg_m3 <= 0.0 {
        return 0.0;
    }
    OAK_YOUNG_PA * (density_kg_m3 / OAK_DENSITY_KG_M3).powf(STIFFNESS_DENSITY_EXPONENT)
}

/// Stem height, diameter for a volume `volume_m3` of stem wood: the gene slenderness
/// unless Greenhill buckling of this wood forbids it.
fn stem_geometry(volume_m3: f64, density: f64, slenderness: f64, climbing: f64) -> (f64, f64) {
    if volume_m3 <= 0.0 || density <= 0.0 {
        return (0.0, 0.0);
    }
    let pi = std::f64::consts::PI;
    let diameter = (4.0 * volume_m3 / (pi * slenderness)).cbrt();
    let height = slenderness * diameter;
    let e = young_modulus_from_density(density);
    let c = 0.792 * (e / (density * STANDARD_GRAVITY)).cbrt() * (1.0 + 30.0 * climbing);
    let critical = c * diameter.powf(2.0 / 3.0);
    if height <= critical {
        (height, diameter)
    } else {
        let d = (4.0 * volume_m3 / (pi * c)).powf(3.0 / 8.0);
        (c * d.powf(2.0 / 3.0), d)
    }
}

/// A plant as resource state. See the module documentation.
#[derive(Clone, Debug)]
pub struct Plant {
    genome: PlantGenome,
    params: Params,
    pub stores: Stores,
    pub pools: Pools,
    /// Accumulated damage 0..1 (starvation, dehydration, wounds, senescence).
    pub damage: f64,
    /// Absolute simulation time, days.
    pub time_days: f64,
    pub birth_days: f64,
    /// Absolute time of the next event, days (refreshed by [`Plant::advance`]).
    pub next_event_day: f64,
    /// Volume of structural wood (trunk and branches), m3. Density is mass over this.
    wood_volume_m3: f64,
    /// Annual layers; `sum(mass) == trunk + branches`.
    rings: Vec<RingLayer>,
}

/// Rates and amounts over one substep. Everything is an amount over `h` days.
#[derive(Clone, Copy, Debug, Default)]
struct Flows {
    absorbed_j: f64,
    sugar_made: f64,
    transpired: f64,
    cuticular: f64,
    uptake_water: f64,
    uptake_minerals: f64,
    substrate: f64,
    maintenance: f64,
    growth_sugar: f64,
    /// Fraction of new tissue going to [roots, trunk, branches, leaves, fruit].
    shares: [f64; 5],
    ring_density: f64,
    loss: [f64; 5],
    damage: f64,
    heal: f64,
}

impl Plant {
    /// Plant sprouted from a seed of `seed_mass_kg`. The seed's mass and energy are
    /// registered as imports of `env`: 40% sugar, 25% water, 5% minerals, 30% tissue
    /// split by the genome's growth targets.
    pub fn from_seed(genome: PlantGenome, seed_mass_kg: f64, start_day: f64, env: &mut Environment) -> Plant {
        assert!(seed_mass_kg > 0.0 && seed_mass_kg.is_finite());
        let genome = genome.clamped();
        let params = Params::from_genome(&genome);
        let tissue_total = 0.30 * seed_mass_kg;
        let h = &genome.habit;
        let weights = [
            h.root_fraction as f64,
            (h.stem_fraction as f64 * 0.5).max(1e-6),
            0.0,
            h.leaf_fraction as f64,
            0.0,
        ];
        let sum: f64 = weights.iter().sum();
        let pools = Pools::from_array(weights.map(|w| tissue_total * w / sum));
        let mut plant = Plant {
            genome,
            params,
            stores: Stores { sugar: 0.40 * seed_mass_kg, water: 0.25 * seed_mass_kg, minerals: 0.05 * seed_mass_kg },
            pools,
            damage: 0.0,
            time_days: start_day,
            birth_days: start_day,
            next_event_day: start_day,
            wood_volume_m3: 0.0,
            rings: Vec::new(),
        };
        let wood = plant.pools.trunk + plant.pools.branches;
        if wood > 0.0 {
            plant.wood_volume_m3 = wood / params.base_density;
            plant.rings.push(RingLayer { mass_kg: wood, volume_m3: plant.wood_volume_m3 });
        }
        env.imported_mass_kg += seed_mass_kg;
        env.imported_energy_j += plant.chemical_energy_j();
        plant
    }

    pub fn genome(&self) -> &PlantGenome {
        &self.genome
    }

    // ----- derived state ------------------------------------------------------------

    pub fn age_days(&self) -> f64 {
        self.time_days - self.birth_days
    }

    /// Total mass held by the plant, kg.
    pub fn total_mass_kg(&self) -> f64 {
        self.stores.total() + self.pools.total()
    }

    /// Dry structural biomass, kg.
    pub fn biomass_kg(&self) -> f64 {
        self.pools.total()
    }

    /// Chemical energy of sugar and tissue, J.
    pub fn chemical_energy_j(&self) -> f64 {
        self.stores.sugar * SUGAR_ENERGY_J_PER_KG
            + self.pools.total() * (1.0 - self.params.mineral_fraction) * TISSUE_ENERGY_J_PER_KG
    }

    /// Dead when damage is complete or nothing living is left. Derived, not stored.
    pub fn is_dead(&self) -> bool {
        self.damage >= 0.999 || self.pools.total() <= 1e-12
    }

    /// 1 for a healthy plant, 0 for a dead one.
    pub fn vitality(&self) -> f64 {
        if self.is_dead() { 0.0 } else { (1.0 - self.damage).clamp(0.0, 1.0) }
    }

    /// Dormant when it is too cold to grow. A function of conditions, not a flag.
    pub fn is_dormant(&self, conditions: &Conditions) -> bool {
        conditions.temperature_k < COLD_THRESHOLD_K
    }

    pub fn is_mature(&self) -> bool {
        self.age_days() >= self.params.maturity_age_days
    }

    pub fn is_senescent(&self) -> bool {
        self.age_days() >= self.params.longevity_days
    }

    /// Mass of living tissue: all of roots, leaves and fruit and the living (sapwood)
    /// part of the stems.
    pub fn live_mass_kg(&self) -> f64 {
        let living_wood = 1.0 - 0.9 * self.params.lignification;
        self.pools.roots + self.pools.leaves + self.pools.fruit + living_wood * (self.pools.trunk + self.pools.branches)
    }

    fn water_capacity(&self) -> f64 {
        0.6 * self.live_mass_kg()
    }

    fn mineral_capacity(&self) -> f64 {
        0.6 * self.params.mineral_fraction * self.pools.total()
    }

    /// How far the canopy is below what the genome calls full: 0 when leafy, 1 when bare.
    /// A bare autotroph (deciduous in spring, after drought or a cut) may spend its
    /// reserve to refoliate; without this a deciduous seedling could never leaf out.
    fn leaf_deficit(&self) -> f64 {
        let target = self.genome.habit.leaf_fraction as f64 * self.pools.total();
        if self.params.autotrophy < 0.5 || target <= 1e-12 {
            return 0.0;
        }
        (1.0 - self.pools.leaves / (0.8 * target)).clamp(0.0, 1.0)
    }

    fn sugar_reserve(&self) -> f64 {
        0.12 * self.live_mass_kg() * (1.0 - self.leaf_deficit())
    }

    fn sugar_capacity(&self) -> f64 {
        0.40 * self.live_mass_kg() + 1e-12
    }

    /// Mean dry density of structural wood, kg/m3 (0 when there is none).
    pub fn wood_density(&self) -> f64 {
        let wood = self.pools.trunk + self.pools.branches;
        if self.wood_volume_m3 > 0.0 && wood > 0.0 { wood / self.wood_volume_m3 } else { 0.0 }
    }

    fn branch_count_for(&self, branch_mass: f64) -> u32 {
        let k = 1.5 * (0.2 + self.genome.seed.branching_rate as f64);
        (k * (branch_mass.max(0.0) / 0.01).sqrt()).floor() as u32
    }

    fn branch_mass_for_count(&self, count: u32) -> f64 {
        let k = 1.5 * (0.2 + self.genome.seed.branching_rate as f64);
        0.01 * (count as f64 / k).powi(2)
    }

    /// Form derived from pools and wood volume.
    pub fn morphology(&self) -> Morphology {
        let wood = self.pools.trunk + self.pools.branches;
        let trunk_volume = if wood > 0.0 { self.wood_volume_m3 * self.pools.trunk / wood } else { 0.0 };
        let (height, diameter) = stem_geometry(
            trunk_volume,
            self.wood_density(),
            self.genome.habit.slenderness as f64,
            self.genome.habit.climbing as f64,
        );
        let leaf_area = self.params.specific_leaf_area * self.pools.leaves;
        let relative_height = (height / self.params.max_height).clamp(0.0, 1.0);
        let radius = 0.5 * self.genome.seed.crown_spread_m as f64 * relative_height.powf(0.7);
        let height_crown = std::f64::consts::PI * radius * radius;
        let crown_area = height_crown.max(leaf_area / MAX_LEAF_AREA_INDEX);
        Morphology {
            height_m: height,
            trunk_diameter_m: diameter,
            crown_area_m2: crown_area,
            leaf_area_m2: leaf_area,
            leaf_area_index: if crown_area > 0.0 { leaf_area / crown_area } else { 0.0 },
            branch_count: self.branch_count_for(self.pools.branches),
        }
    }

    /// Wood properties derived from the rings (density from growth rate at the time
    /// each ring formed; stiffness from density).
    pub fn wood_properties(&self) -> WoodProperties {
        let wood = self.pools.trunk + self.pools.branches;
        let morph = self.morphology();
        let density = self.wood_density();
        let mut ei = 0.0;
        if wood > 0.0 && morph.height_m > 0.0 && morph.trunk_diameter_m > 0.0 {
            let trunk_share = self.pools.trunk / wood;
            let pi_h = std::f64::consts::PI * morph.height_m;
            let mut inner = 0.0_f64;
            let mut cumulative = 0.0;
            for ring in &self.rings {
                if ring.volume_m3 <= 0.0 {
                    continue;
                }
                cumulative += ring.volume_m3 * trunk_share;
                let outer = (cumulative / pi_h).sqrt();
                ei += young_modulus_from_density(ring.density()) * std::f64::consts::FRAC_PI_4 * (outer.powi(4) - inner.powi(4));
                inner = outer;
            }
        }
        WoodProperties {
            density_kg_m3: density,
            young_modulus_pa: young_modulus_from_density(density),
            ring_count: self.rings.iter().filter(|r| r.mass_kg > 0.0).count(),
            bending_stiffness_ei: ei,
            trunk_diameter_m: morph.trunk_diameter_m,
            trunk_height_m: morph.height_m,
        }
    }

    /// Annual wood layers, oldest first.
    pub fn rings(&self) -> &[RingLayer] {
        &self.rings
    }

    // ----- continuous model ---------------------------------------------------------

    /// Amounts transferred over `h` days given the current state. Pure.
    fn flows(&self, env: &Environment, cond: &Conditions, h: f64) -> Flows {
        let p = &self.params;
        let genome = &self.genome;
        let mut f = Flows::default();
        let vitality = self.vitality();
        let dormant = self.is_dormant(cond);
        let active = if dormant { 0.0 } else { 1.0 };
        let live = self.live_mass_kg();
        let morph = self.morphology();
        let q10 = 2.0_f64.powf((cond.temperature_k - 293.0) / 10.0);
        let tf = {
            let x = (cond.temperature_k - p.temperature_optimum) / 12.0;
            (-x * x).exp() * if cond.temperature_k < 273.15 { 0.0 } else { 1.0 }
        };

        // ---- light and potential photosynthesis
        let area = morph.leaf_area_m2;
        let mut potential_sugar = 0.0;
        if area > 0.0 && morph.crown_area_m2 > 0.0 {
            let intercepted = 1.0 - (-CANOPY_EXTINCTION * morph.leaf_area_index).exp();
            let absorbed_w = cond.full_sun_par_w_m2 * cond.light_fraction * morph.crown_area_m2 * intercepted;
            f.absorbed_j = absorbed_w * SECONDS_PER_DAY * h;
            let per_area = absorbed_w / area;
            let efficiency = LIGHT_USE_EFFICIENCY * tf / (1.0 + per_area / LIGHT_SATURATION_W_M2);
            let sink_feedback = (1.0 - self.stores.sugar / self.sugar_capacity()).clamp(0.0, 1.0);
            potential_sugar = f.absorbed_j * efficiency * p.autotrophy * vitality * sink_feedback / SUGAR_ENERGY_J_PER_KG;
        }

        // ---- water: uptake, stomatal regulation, cuticular loss
        let water_capacity = self.water_capacity();
        let water_setpoint = 0.6 * water_capacity;
        let water_use_available = (self.stores.water - water_setpoint).max(0.0);
        let root_capacity_water = 40.0 * self.pools.roots * env.water_availability() * p.uptake_scale * vitality;
        let uptake_cap = (root_capacity_water * h).min(env.soil_water_kg).max(0.0);
        let cuticular_demand = 0.1 * area * h;
        // Stomata close first: the cuticular loss is paid before any water goes to assimilation.
        let sugar_by_water = (uptake_cap + water_use_available - cuticular_demand).max(0.0) / (p.water_per_sugar + H2O_PER_SUGAR);
        f.sugar_made = potential_sugar.min(sugar_by_water).max(0.0);
        f.transpired = p.water_per_sugar * f.sugar_made;
        let stoich_water = H2O_PER_SUGAR * f.sugar_made;
        let demand = f.transpired + stoich_water + cuticular_demand;
        let refill = (water_capacity - self.stores.water).max(0.0) * (1.0 - (-h / 1.0).exp());
        f.uptake_water = uptake_cap.min(demand + refill);
        f.cuticular = cuticular_demand.min((self.stores.water + f.uptake_water - f.transpired - stoich_water).max(0.0));

        // ---- minerals: uptake into the mineral store
        let root_capacity_minerals =
            0.05 * self.pools.roots * env.mineral_availability() * (1.0 + genome.seed.mycorrhizal_affinity as f64) * p.uptake_scale * vitality;
        let mineral_refill = (self.mineral_capacity() - self.stores.minerals).max(0.0) * (1.0 - (-h / 2.0).exp());
        f.uptake_minerals = (root_capacity_minerals * h).min(env.soil_minerals_kg).min(mineral_refill).max(0.0);

        // ---- heterotrophic carbon from soil organic matter
        f.substrate = 0.3 * (1.0 - p.autotrophy) * self.pools.roots * env.litter_availability() * vitality * h;
        f.substrate = f.substrate.min(env.litter_organic_kg);

        // ---- maintenance respiration (sugar kg), temperature dependent
        let shade_tolerance = genome.seed.shade_tolerance as f64;
        let leaf_rate = 0.02 * (1.3 - 0.6 * shade_tolerance);
        let living_wood = 1.0 - 0.9 * p.lignification;
        let maintenance_rate = leaf_rate * self.pools.leaves
            + 0.006 * self.pools.roots
            + 0.004 * self.pools.fruit
            + 0.0008 * living_wood * (self.pools.trunk + self.pools.branches);
        f.maintenance = maintenance_rate * q10 * h * vitality;
        let sugar_income = f.sugar_made + f.substrate;
        let paid = f.maintenance.min(self.stores.sugar + sugar_income);
        let unpaid_fraction = if f.maintenance > 1e-18 { (f.maintenance - paid) / f.maintenance } else { 0.0 };
        f.maintenance = paid;

        // ---- growth: convert the store to tissue, share by genome allocation
        let reserve = self.sugar_reserve();
        let net_rate = if h > 0.0 { (sugar_income - paid) / h } else { 0.0 };
        let k = GROWTH_CONVERSION_RATE;
        let e = (-k * h).exp();
        let above = self.stores.sugar - reserve;
        let converted = (above * (1.0 - e) + net_rate * (h - (1.0 - e) / k)).clamp(0.0, (self.stores.sugar + sugar_income - paid).max(0.0));
        let c_min = p.mineral_fraction / (1.0 - p.mineral_fraction);
        let mineral_limit = (self.stores.minerals + f.uptake_minerals) / (c_min * GROWTH_YIELD).max(1e-30);
        let water_status = if water_capacity > 0.0 { self.stores.water / water_capacity } else { 0.0 };
        let water_gate = ((water_status - 0.25) / 0.35).clamp(0.0, 1.0);
        let shares = self.allocation_shares(cond, &morph);
        let share_sum: f64 = shares.iter().sum();
        if share_sum > 0.0 {
            f.growth_sugar = converted.min(mineral_limit) * water_gate * active * vitality;
            f.shares = shares.map(|s| s / share_sum);
        }
        let new_dry = f.growth_sugar * GROWTH_YIELD / (1.0 - p.mineral_fraction);
        // Ring width of the growth step (m per year) decides the density of the new
        // wood: wide rings from fast growth, narrow rings from slow growth.
        let trunk_surface = std::f64::consts::PI * morph.trunk_diameter_m * morph.height_m;
        let new_volume = new_dry * f.shares[T] / p.base_density;
        let width_per_year = if h > 0.0 { new_volume / trunk_surface.max(1e-9) * YEAR_DAYS / h } else { 0.0 };
        let ring_response = genome.habit.ring_response as f64;
        f.ring_density = if width_per_year > 0.0 {
            p.base_density * (1.0 + ring_response * (width_per_year / REFERENCE_RING_WIDTH_M_PER_YEAR).ln().tanh())
        } else {
            p.base_density
        };

        // ---- turnover to litter
        let retention = genome.seed.leaf_retention as f64;
        let mut leaf_rate_loss = 1.0 / (30.0 + 1800.0 * retention);
        if dormant {
            leaf_rate_loss += (1.0 - retention) * 0.15;
        }
        if water_status < 0.4 {
            leaf_rate_loss += 0.05 * (0.4 - water_status) / 0.4;
        }
        f.loss[L] = self.pools.leaves * (1.0 - (-leaf_rate_loss * h).exp());
        f.loss[R] = self.pools.roots * (1.0 - (-0.0006 * h).exp());
        let herb = (1.0 - p.lignification).powi(2);
        let stem_loss = 1.0 - (-(0.003 * herb) * h).exp();
        f.loss[T] = self.pools.trunk * stem_loss;
        f.loss[B] = self.pools.branches * stem_loss;
        f.loss[F] = self.pools.fruit * (1.0 - (-h / 90.0).exp());

        // ---- damage
        let mut damage_rate = 0.0;
        damage_rate += unpaid_fraction / 60.0;
        if water_status < 0.25 && water_capacity > 0.0 {
            damage_rate += (0.25 - water_status) / 0.25 / 40.0;
        }
        let age_ratio = self.age_days() / p.longevity_days;
        if age_ratio > 1.0 {
            damage_rate += 1.0 / (0.25 * p.longevity_days);
        }
        f.damage = damage_rate * h;
        f.heal = if damage_rate == 0.0 && live > 0.0 { 0.001 * h } else { 0.0 };
        f
    }

    /// Unnormalized growth shares of [roots, trunk, branches, leaves, fruit]. Genome
    /// targets (apical dominance splits stem between trunk and branches), shifted
    /// toward roots when water or minerals are scarce and toward leaves and stem when
    /// light is scarce, and closed by size gates.
    fn allocation_shares(&self, cond: &Conditions, morph: &Morphology) -> [f64; 5] {
        let h = &self.genome.habit;
        let s = &self.genome.seed;
        let total = self.pools.total().max(1e-12);
        let apical = s.apical_dominance as f64;
        let branching = s.branching_rate as f64;
        let stem = h.stem_fraction as f64;
        let mut target = [
            h.root_fraction as f64,
            stem * (0.25 + 0.75 * apical),
            stem * (0.15 + 0.85 * (1.0 - apical)) * (0.5 + branching),
            h.leaf_fraction as f64,
            if self.is_mature() && self.stores.sugar > 1.5 * self.sugar_reserve() { h.reproductive_fraction as f64 } else { 0.0 },
        ];
        let water_status = if self.water_capacity() > 0.0 { self.stores.water / self.water_capacity() } else { 0.0 };
        let mineral_status = if self.mineral_capacity() > 0.0 { self.stores.minerals / self.mineral_capacity() } else { 0.0 };
        let water_scarcity = ((0.9 - water_status) / 0.5).clamp(0.0, 1.0);
        let mineral_scarcity = ((0.6 - mineral_status) / 0.5).clamp(0.0, 1.0);
        let scarcity = water_scarcity.max(mineral_scarcity);
        let light_scarcity = (1.0 - cond.light_fraction).clamp(0.0, 1.0) * (1.0 - 0.5 * s.shade_tolerance as f64);
        let deficit = self.leaf_deficit();
        target[R] *= 1.0 + 1.5 * scarcity;
        target[L] *= (1.0 + 0.8 * light_scarcity) * (1.0 + 4.0 * deficit);
        target[T] *= (1.0 + 0.5 * light_scarcity) * (1.0 - 0.8 * deficit);
        target[B] *= 1.0 - 0.8 * deficit;

        let height_gate = (1.0 - (morph.height_m / self.params.max_height).powi(6)).clamp(0.0, 1.0);
        let leaf_gate = ((MAX_LEAF_AREA_INDEX - morph.leaf_area_index) / 1.5).clamp(0.0, 1.0);
        let root_share = self.pools.roots / total;
        let root_cap = (h.root_fraction as f64 * (1.0 + 1.5 * scarcity) + 0.05).min(0.9);
        let root_gate = ((root_cap - root_share) / 0.1).clamp(0.0, 1.0);
        // Carbon-heterotrophs have no leaves to gate; all gates stay as above.
        target[R] *= root_gate;
        target[T] *= height_gate;
        target[B] *= height_gate;
        target[L] *= leaf_gate;
        target
    }

    /// Apply `flows` atomically: every amount leaves one account and enters another,
    /// clamped by what the source holds. Conservation does not depend on `h`.
    fn apply(&mut self, env: &mut Environment, f: &Flows, h: f64, track_rings: bool) {
        let p = self.params;
        let fm = p.mineral_fraction;

        // Soil uptake.
        let uw = f.uptake_water.min(env.soil_water_kg).max(0.0);
        env.soil_water_kg -= uw;
        self.stores.water += uw;
        let um = f.uptake_minerals.min(env.soil_minerals_kg).max(0.0);
        env.soil_minerals_kg -= um;
        self.stores.minerals += um;

        // Photosynthesis: CO2 + H2O -> CH2O + O2, energy from absorbed light.
        env.radiation_in_j += f.absorbed_j;
        let s = f
            .sugar_made
            .min(self.stores.water / H2O_PER_SUGAR)
            .min(env.co2_kg / CO2_PER_SUGAR)
            .max(0.0);
        self.stores.water -= H2O_PER_SUGAR * s;
        env.co2_kg -= CO2_PER_SUGAR * s;
        env.o2_kg += O2_PER_SUGAR * s;
        self.stores.sugar += s;
        env.heat_out_j += f.absorbed_j - s * SUGAR_ENERGY_J_PER_KG;

        // Transpiration and cuticular loss to vapour.
        let lost = (f.transpired + f.cuticular).min(self.stores.water).max(0.0);
        self.stores.water -= lost;
        env.vapor_kg += lost;

        // Heterotrophic uptake of litter carbon.
        let litter = f.substrate.min(env.litter_organic_kg).max(0.0);
        if litter > 0.0 {
            let specific = env.litter_energy_j / env.litter_organic_kg;
            let energy_out = litter * specific;
            env.litter_organic_kg -= litter;
            env.litter_energy_j -= energy_out;
            self.stores.sugar += litter;
            env.heat_out_j += energy_out - litter * SUGAR_ENERGY_J_PER_KG;
        }

        // Maintenance respiration.
        let m = f.maintenance.min(self.stores.sugar).min(env.o2_kg / O2_PER_SUGAR).max(0.0);
        self.stores.sugar -= m;
        env.respire_mass(m);
        env.heat_out_j += m * SUGAR_ENERGY_J_PER_KG;

        // Growth: sugar -> structural organic mass (yield) + respired remainder.
        let c_min = fm / (1.0 - fm);
        let mut u = f.growth_sugar.min(self.stores.sugar).min(env.o2_kg / O2_PER_SUGAR).max(0.0);
        u = u.min(self.stores.minerals / (c_min * GROWTH_YIELD).max(1e-30));
        if u > 0.0 {
            let organic = GROWTH_YIELD * u;
            let minerals = c_min * organic;
            let dry = organic + minerals;
            self.stores.sugar -= u;
            self.stores.minerals -= minerals;
            env.respire_mass(u - organic);
            env.heat_out_j += u * SUGAR_ENERGY_J_PER_KG - organic * TISSUE_ENERGY_J_PER_KG;
            let mut pools = self.pools.to_array();
            for i in 0..5 {
                pools[i] += dry * f.shares[i];
            }
            self.pools = Pools::from_array(pools);
            let wood = dry * (f.shares[T] + f.shares[B]);
            if wood > 0.0 {
                let volume = wood / f.ring_density.max(1.0);
                self.wood_volume_m3 += volume;
                if track_rings {
                    let year = (self.age_days() / YEAR_DAYS).floor().max(0.0) as usize;
                    if self.rings.len() <= year {
                        self.rings.resize(year + 1, RingLayer::default());
                    }
                    self.rings[year].mass_kg += wood;
                    self.rings[year].volume_m3 += volume;
                }
            }
        }

        // Turnover: tissue goes to litter, part of the minerals is resorbed.
        let resorption = [0.2, 0.3, 0.3, 0.5, 0.0];
        let wood_before = self.pools.trunk + self.pools.branches;
        let mut pools = self.pools.to_array();
        for i in 0..5 {
            let dm = f.loss[i].min(pools[i]).max(0.0);
            if dm <= 0.0 {
                continue;
            }
            pools[i] -= dm;
            let organic = dm * (1.0 - fm);
            let minerals = dm * fm;
            env.litter_organic_kg += organic;
            env.litter_energy_j += organic * TISSUE_ENERGY_J_PER_KG;
            let back = minerals * resorption[i];
            self.stores.minerals += back;
            env.litter_mineral_kg += minerals - back;
        }
        self.pools = Pools::from_array(pools);
        let wood_after = self.pools.trunk + self.pools.branches;
        if wood_before > 0.0 && wood_after < wood_before {
            let k = wood_after / wood_before;
            self.wood_volume_m3 *= k;
            for ring in &mut self.rings {
                ring.mass_kg *= k;
                ring.volume_m3 *= k;
            }
        }

        self.damage = (self.damage + f.damage - f.heal).clamp(0.0, 1.0);
        self.time_days += h;
    }

    /// Per-day net change of [sugar, roots, trunk, branches, leaves, fruit].
    fn rates(&self, f: &Flows) -> [f64; 6] {
        let dry_per_sugar = GROWTH_YIELD / (1.0 - self.params.mineral_fraction);
        let new_dry = f.growth_sugar * dry_per_sugar;
        let mut out = [0.0; 6];
        out[0] = f.sugar_made + f.substrate - f.maintenance - f.growth_sugar;
        for i in 0..5 {
            out[i + 1] = new_dry * f.shares[i] - f.loss[i];
        }
        out
    }

    fn pick_step(&self, per_day: &Flows, remaining: f64) -> f64 {
        let floor = 1e-3 * self.total_mass_kg() + 1e-12;
        let values = [
            self.stores.sugar,
            self.pools.roots,
            self.pools.trunk,
            self.pools.branches,
            self.pools.leaves,
            self.pools.fruit,
        ];
        let rates = self.rates(per_day);
        let mut h = remaining.min(STEP_MAX_DAYS);
        for (x, r) in values.iter().zip(rates.iter()) {
            if r.abs() > 0.0 {
                h = h.min(STEP_RELATIVE_CHANGE * x.max(floor) / r.abs());
            }
        }
        // End a substep at the predicted crossing of the next branch order, so the
        // branching event is timed by the integration and not by a coarse step.
        let count = self.branch_count_for(self.pools.branches);
        if count < 16 && rates[3] > 0.0 {
            let needed = self.branch_mass_for_count(count + 1) - self.pools.branches;
            if needed > 0.0 {
                h = h.min((needed / rates[3]).max(STEP_MIN_DAYS));
            }
        }
        h.max(STEP_MIN_DAYS.min(remaining)).min(remaining)
    }

    fn scratch(&self) -> Plant {
        Plant {
            genome: self.genome,
            params: self.params,
            stores: self.stores,
            pools: self.pools,
            damage: self.damage,
            time_days: self.time_days,
            birth_days: self.birth_days,
            next_event_day: self.next_event_day,
            wood_volume_m3: self.wood_volume_m3,
            rings: Vec::new(),
        }
    }

    /// Integrate under constant conditions up to `end` (days).
    fn integrate(&mut self, env: &mut Environment, cond: &Conditions, end: f64, report: &mut AdvanceReport) {
        while end - self.time_days > 1e-12 {
            let remaining = end - self.time_days;
            let per_day = self.flows(env, cond, 1.0);
            let h = self.pick_step(&per_day, remaining);
            // Explicit midpoint: amounts for the whole step from the half-step state.
            let half = self.flows(env, cond, 0.5 * h);
            let mut mid = self.scratch();
            let mut mid_env = env.clone();
            mid.apply(&mut mid_env, &half, 0.5 * h, false);
            let full = mid.flows(&mid_env, cond, h);
            let branches_before = self.branch_count_for(self.pools.branches);
            self.apply(env, &full, h, true);
            if remaining - h < 1e-12 {
                self.time_days = end;
            }
            let branches_after = self.branch_count_for(self.pools.branches);
            if branches_after > branches_before {
                report.new_branches += branches_after - branches_before;
                report.events.push((self.time_days, EventKind::Branching));
            }
            report.substeps += 1;
        }
    }

    /// Earliest upcoming event and its absolute time.
    pub fn next_event(&self, env: &Environment, climate: &Climate, include_branching: bool) -> (f64, EventKind) {
        let t = self.time_days;
        let mut best = (climate.next_change_after(t), EventKind::SeasonChange);
        let mut consider = |time: f64, kind: EventKind| {
            if time < best.0 {
                best = (time, kind);
            }
        };
        let age = self.age_days();
        if age < self.params.maturity_age_days {
            consider(t + self.params.maturity_age_days - age, EventKind::Maturation);
        }
        if age < self.params.longevity_days {
            consider(t + self.params.longevity_days - age, EventKind::Senescence);
        }
        if !self.is_dead() {
            let cond = climate.at(t);
            let f = self.flows(env, &cond, 1.0);
            if f.damage > f.heal {
                consider(t + (1.0 - self.damage) / (f.damage - f.heal), EventKind::Death);
            }
            let count = self.branch_count_for(self.pools.branches);
            if include_branching && count < 16 {
                let rate = self.rates(&f)[3];
                if rate > 0.0 {
                    let needed = self.branch_mass_for_count(count + 1) - self.pools.branches;
                    if needed > 0.0 {
                        consider(t + needed / rate, EventKind::Branching);
                    }
                }
            }
        }
        best
    }

    /// Advance the plant by `dt_days` against `env` under `climate`. The span is split
    /// at events; between events the continuous model is integrated adaptively.
    pub fn advance(&mut self, env: &mut Environment, climate: &Climate, dt_days: f64) -> AdvanceReport {
        assert!(dt_days >= 0.0 && dt_days.is_finite());
        let mut report = AdvanceReport::default();
        let end = self.time_days + dt_days;
        while end - self.time_days > 1e-12 {
            let (event_time, kind) = self.next_event(env, climate, false);
            self.next_event_day = event_time;
            let segment_end = end.min(event_time).max(self.time_days + 1e-9);
            let cond = climate.at(self.time_days);
            self.integrate(env, &cond, segment_end, &mut report);
            if kind != EventKind::Branching && event_time <= self.time_days + 1e-9 && event_time <= end {
                report.events.push((self.time_days, kind));
            }
        }
        self.next_event_day = self.next_event(env, climate, true).0;
        report
    }

    // ----- cutting -----------------------------------------------------------------

    /// Cut off `mass_fraction` (0..1] of the above-ground tissue (trunk, branches,
    /// leaves, fruit). Pools, wood volume, rings and the stores (by tissue share) are
    /// split; roots stay. The cut costs the remaining plant a little damage. Mass and
    /// chemical energy are conserved exactly: they only change owner.
    pub fn cut_above(&mut self, mass_fraction: f64) -> SeveredPart {
        assert!(mass_fraction > 0.0 && mass_fraction <= 1.0);
        let k = mass_fraction;
        let before_total = self.total_mass_kg();
        let above = self.pools.trunk + self.pools.branches + self.pools.leaves + self.pools.fruit;
        let share = if before_total > 0.0 { (k * above / (self.pools.total() + 1e-300)).clamp(0.0, 1.0) } else { 0.0 };
        let removed = Pools { roots: 0.0, trunk: self.pools.trunk * k, branches: self.pools.branches * k, leaves: self.pools.leaves * k, fruit: self.pools.fruit * k };
        let removed_stores = Stores {
            sugar: self.stores.sugar * share,
            water: self.stores.water * share,
            minerals: self.stores.minerals * share,
        };
        let removed_volume = self.wood_volume_m3 * k;
        let removed_rings: Vec<RingLayer> = self
            .rings
            .iter()
            .map(|r| RingLayer { mass_kg: r.mass_kg * k, volume_m3: r.volume_m3 * k })
            .collect();
        self.pools.trunk -= removed.trunk;
        self.pools.branches -= removed.branches;
        self.pools.leaves -= removed.leaves;
        self.pools.fruit -= removed.fruit;
        self.stores.sugar -= removed_stores.sugar;
        self.stores.water -= removed_stores.water;
        self.stores.minerals -= removed_stores.minerals;
        self.wood_volume_m3 -= removed_volume;
        for ring in &mut self.rings {
            ring.mass_kg *= 1.0 - k;
            ring.volume_m3 *= 1.0 - k;
        }
        self.damage = (self.damage + 0.05 * k).min(1.0);
        SeveredPart {
            genome: self.genome,
            params: self.params,
            stores: removed_stores,
            pools: removed,
            wood_volume_m3: removed_volume,
            rings: removed_rings,
            age_days: self.age_days(),
        }
    }
}

/// The part of a plant that was cut off. It has no roots and gets no light, so it
/// takes up no water and makes no sugar; it keeps its mass and only respires some
/// of its sugar. Whether it can receive water or light is not a flag: both need
/// roots and an exposed canopy, which a severed part does not have.
#[derive(Clone, Debug)]
pub struct SeveredPart {
    genome: PlantGenome,
    params: Params,
    pub stores: Stores,
    pub pools: Pools,
    wood_volume_m3: f64,
    rings: Vec<RingLayer>,
    age_days: f64,
}

impl SeveredPart {
    pub fn total_mass_kg(&self) -> f64 {
        self.stores.total() + self.pools.total()
    }

    pub fn chemical_energy_j(&self) -> f64 {
        self.stores.sugar * SUGAR_ENERGY_J_PER_KG
            + self.pools.total() * (1.0 - self.params.mineral_fraction) * TISSUE_ENERGY_J_PER_KG
    }

    /// True only if the part has roots to take water from soil. Never, for a cut part.
    pub fn can_take_up_water(&self) -> bool {
        self.pools.roots > 0.0
    }

    pub fn genome(&self) -> &PlantGenome {
        &self.genome
    }

    pub fn age_days(&self) -> f64 {
        self.age_days
    }

    pub fn wood_density(&self) -> f64 {
        let wood = self.pools.trunk + self.pools.branches;
        if self.wood_volume_m3 > 0.0 && wood > 0.0 { wood / self.wood_volume_m3 } else { 0.0 }
    }

    pub fn rings(&self) -> &[RingLayer] {
        &self.rings
    }

    /// Density and stiffness of the severed wood (same derivation as for a plant).
    pub fn wood_properties(&self) -> WoodProperties {
        let density = self.wood_density();
        let (height, diameter) = stem_geometry(
            self.wood_volume_m3 * if self.pools.trunk + self.pools.branches > 0.0 { self.pools.trunk / (self.pools.trunk + self.pools.branches) } else { 0.0 },
            density,
            self.genome.habit.slenderness as f64,
            self.genome.habit.climbing as f64,
        );
        WoodProperties {
            density_kg_m3: density,
            young_modulus_pa: young_modulus_from_density(density),
            ring_count: self.rings.iter().filter(|r| r.mass_kg > 0.0).count(),
            bending_stiffness_ei: 0.0,
            trunk_diameter_m: diameter,
            trunk_height_m: height,
        }
    }

    /// Let `dt_days` pass: maintenance respiration of the stored sugar only (leaves
    /// and living wood), returned to the environment. No uptake, no photosynthesis.
    pub fn advance(&mut self, env: &mut Environment, cond: &Conditions, dt_days: f64) {
        assert!(dt_days >= 0.0);
        let q10 = 2.0_f64.powf((cond.temperature_k - 293.0) / 10.0);
        let living_wood = 1.0 - 0.9 * self.params.lignification;
        let rate = 0.01 * self.pools.leaves + 0.0008 * living_wood * (self.pools.trunk + self.pools.branches) + 0.004 * self.pools.fruit;
        let demand = rate * q10 * dt_days;
        let m = demand.min(self.stores.sugar).min(env.o2_kg / O2_PER_SUGAR).max(0.0);
        self.stores.sugar -= m;
        env.respire_mass(m);
        env.heat_out_j += m * SUGAR_ENERGY_J_PER_KG;
    }
}

#[cfg(test)]
mod tests;
