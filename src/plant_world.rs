//! Plant population of a generated world: one resource-based [`Plant`] per
//! vegetation individual, stepped by events against a shared, finite environment.
//!
//! This is the bridge between the pure plant model (`plant.rs`) and the generated
//! world (`world.rs`). It owns everything a *population* needs that a single plant
//! does not:
//!
//! * **Soil cells.** Plants rooted in the same soil cell draw from one
//!   [`Environment`] (one finite water and mineral pool). Rain and litter
//!   decomposition are the only inflows and are recorded as imports or moved
//!   between ledger accounts, so the world ledger stays closed.
//! * **A coarse shade grid** (default 4 m cells) rebuilt rarely (once per
//!   `chunk_days`, a quarter year by default). Neighbours shade each other by crown
//!   height and spread through the leaf area they put over a cell.
//! * **Deterministic setup.** A plant's seed mass is a hash of the world seed and its
//!   individual id; nothing uses a random number generator.
//! * **Cutting.** [`PlantPopulation::cut`] splits a plant with `Plant::cut_above`; the
//!   severed part keeps its mass, stops growing and is carried in the ledger.
//! * **Geometry from state.** [`PlantGeometry`] reports trunk height and radius,
//!   crown radius, leaf area and branch wood volume computed from pools and rings.
//!   Nothing here builds voxels; the world's generator takes these dimensions.
//!
//! # Time
//! The clock is in plant days (`plant::YEAR_DAYS` = 365 per year). The population is
//! advanced lazily: [`PlantPopulation::advance_to_days`] splits the span at chunk
//! boundaries (absolute multiples of `chunk_days`), where rain falls, litter
//! decomposes and shade is refreshed once per chunk. Splitting a span differently
//! changes only where `Plant::advance` places its substeps, never the chunk
//! boundaries, so one long call and many short calls agree to within the plant
//! model's own tolerance.
//!
//! # Provisional
//! Soil depth, rain, mineral stocks, decomposition rate and the seasonal offsets are
//! plausible orders of magnitude, not calibrated data (see `TREE_GROWTH_MODEL.md`).
//! Soil evaporation, runoff and lateral flow are not modelled, and every soil cell
//! has its own atmosphere reservoirs.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::plant::{
    Climate, Conditions, Environment, HabitPreset, Plant, PlantGenome, SeveredPart, YEAR_DAYS,
};
use crate::world::SeedGenome;

/// Extinction coefficient of the shade grid (matches the canopy used by `plant.rs`).
const SHADE_EXTINCTION: f64 = 0.6;
/// Volumetric water at field capacity, kg per m3 of soil.
const SOIL_WATER_CAPACITY_KG_M3: f64 = 300.0;
/// Warm-season temperature offset from the site's annual mean, K.
const WARM_OFFSET_K: f64 = 3.0;
/// Cold-season temperature offset from the site's annual mean, K.
const COLD_OFFSET_K: f64 = 5.0;
const WARM_SEASON_DAYS: f64 = 240.0;
const WARM_PAR_W_M2: f64 = 90.0;
const COLD_PAR_W_M2: f64 = 25.0;
/// Litter decomposition, 1/day (about 30% per year).
const LITTER_DECAY_PER_DAY: f64 = 0.001;

/// Tunable parameters of a population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationConfig {
    /// Side of a soil cell (m). Plants rooted in one cell share its water and minerals.
    pub soil_cell_m: f64,
    /// Ground area one soil cell stands for (m2); `None` is `soil_cell_m` squared.
    /// It sets the cell's water capacity, rain and mineral stock, so a scene can use
    /// fine cells that still hold a whole tree's rooting zone.
    pub soil_area_m2: Option<f64>,
    /// Side of a shade-grid cell (m).
    pub shade_cell_m: f64,
    /// Interval of rain, decomposition and shade updates (days).
    pub chunk_days: f64,
    /// Typical seed mass (kg); each individual varies it by a hash.
    pub seed_mass_kg: f64,
    /// Rooting depth used for the water capacity of a cell (m).
    pub soil_depth_m: f64,
    /// Plants of one soil cell are advanced in rounds of this length so that none
    /// is always first at the shared water (days).
    pub competition_round_days: f64,
    /// Step soil cells on the rayon pool (results are identical either way).
    pub parallel: bool,
    /// 0..1: where in the year the warm season starts (seed-derived calendar epoch).
    pub season_offset: f64,
}

impl Default for PopulationConfig {
    fn default() -> Self {
        Self {
            soil_cell_m: 8.0,
            soil_area_m2: None,
            shade_cell_m: 4.0,
            chunk_days: YEAR_DAYS / 4.0,
            seed_mass_kg: 0.05,
            soil_depth_m: 1.0,
            competition_round_days: 15.0,
            parallel: false,
            season_offset: 0.0,
        }
    }
}

/// Where and what one individual is.
#[derive(Clone, Copy, Debug)]
pub struct SiteInput {
    pub genome: SeedGenome,
    /// World position (m).
    pub x_m: f64,
    pub y_m: f64,
    /// Soil moisture and fertility fields at the site, 0..1.
    pub moisture: f64,
    pub fertility: f64,
    /// Annual mean temperature at the site (K).
    pub temperature_k: f64,
    /// Time of germination after world start (years).
    pub germination_years: f64,
}

/// Map a world seed genome to the plant model's genome. The genes both models share
/// are copied; the habit genes (lignification, slenderness, allocation targets, ring
/// response, ...) that the world genome does not have take the tree defaults.
///
/// One gene changes meaning: the world generator reads `crown_spread_m` as the crown
/// *radius*, the plant model as its *diameter*, so the plant gets twice the value
/// (clamped to the gene bound).
pub fn plant_genome_from_seed(seed: &SeedGenome) -> PlantGenome {
    let mut genome = PlantGenome::preset(HabitPreset::Tree, seed.individual_id);
    genome.seed = *seed;
    genome.seed.crown_spread_m = 2.0 * seed.crown_spread_m;
    genome.clamped()
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn unit_hash(world_seed: u64, id: u64, salt: u64) -> f64 {
    let h = splitmix(world_seed ^ splitmix(id ^ splitmix(salt)));
    (h >> 11) as f64 / (1u64 << 53) as f64
}

/// One soil cell: the environment its plants share.
#[derive(Clone, Debug)]
struct SoilCell {
    env: Environment,
    rain_kg_per_day: f64,
}

impl SoilCell {
    fn new(config: &PopulationConfig, moisture: f64, fertility: f64) -> Self {
        let area = config.soil_area_m2.unwrap_or(config.soil_cell_m * config.soil_cell_m);
        let capacity = area * config.soil_depth_m * SOIL_WATER_CAPACITY_KG_M3;
        let water = capacity * (0.35 + 0.5 * moisture.clamp(0.0, 1.0));
        let minerals = area * (0.1 + 0.6 * fertility.clamp(0.0, 1.0));
        let mut env = Environment::new(water, minerals);
        env.soil_water_capacity_kg = capacity;
        // 1 mm of rain is 1 kg per m2.
        let annual_mm = 350.0 + 900.0 * moisture.clamp(0.0, 1.0);
        Self { env, rain_kg_per_day: annual_mm * area / YEAR_DAYS }
    }

    /// Rain for `days` (limited to the room below field capacity; the rest runs off
    /// and is never imported) and litter decomposition over `days`.
    fn prepare(&mut self, days: f64) {
        let env = &mut self.env;
        let room = (env.soil_water_capacity_kg - env.soil_water_kg).max(0.0);
        let rain = (self.rain_kg_per_day * days).min(room);
        if rain > 0.0 {
            env.add_rain(rain);
        }
        let k = 1.0 - (-LITTER_DECAY_PER_DAY * days).exp();
        let organic = env.litter_organic_kg * k;
        if organic > 0.0 {
            // Litter is oxidised as CH2O: O2 in, CO2 and vapour out; mass change +organic.
            let energy = env.litter_energy_j / env.litter_organic_kg * organic;
            env.litter_organic_kg -= organic;
            env.litter_energy_j -= energy;
            env.o2_kg -= 32.0 / 30.0 * organic;
            env.co2_kg += 44.0 / 30.0 * organic;
            env.vapor_kg += 18.0 / 30.0 * organic;
            env.heat_out_j += energy;
        }
        let minerals = env.litter_mineral_kg * k;
        env.litter_mineral_kg -= minerals;
        env.soil_minerals_kg += minerals;
    }
}

#[derive(Clone, Debug)]
struct Member {
    plant: Plant,
    id: u64,
    x: f64,
    y: f64,
    soil: usize,
    temperature_k: f64,
    germination_day: f64,
    climate: Climate,
    /// Light fraction from the last shade update.
    light: f64,
    /// Highest light fraction seen (structural light for the generator).
    light_memory: f64,
    /// Sum and count of the light fractions of the chunks since germination.
    light_sum: f64,
    light_chunks: u32,
    /// Crown shape at the last shade update: height, crown radius, leaf area.
    shape: (f64, f64, f64),
    /// Bumped whenever the plant's state changes (advance or cut).
    revision: u64,
}

impl Member {
    fn active(&self, until_day: f64) -> bool {
        self.germination_day < until_day && !self.plant.is_dead()
    }
}

#[derive(Clone, Debug)]
struct SeveredRecord {
    owner: usize,
    soil: usize,
    part: SeveredPart,
}

#[derive(Clone, Copy, Debug)]
struct ShadeEntry {
    owner: u32,
    top_m: f32,
    lai: f32,
}

/// Dimensions the world's generator takes, all derived from the plant's state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PlantGeometry {
    pub germinated: bool,
    pub dead: bool,
    pub senescent: bool,
    pub age_years: f32,
    pub height_m: f32,
    /// Bole height: crown base, from the height and the genome's trunk fraction.
    pub trunk_height_m: f32,
    pub trunk_radius_m: f32,
    pub crown_radius_m: f32,
    pub leaf_area_m2: f32,
    /// Volume of branch wood: branch pool over the plant's actual wood density.
    pub branch_volume_m3: f32,
    pub branch_count: u32,
    pub biomass_kg: f32,
    /// Height relative to the genome's maximum height.
    pub relative_size: f32,
    pub light: f32,
    pub structural_light: f32,
    /// Soil saturation and mineral availability of the plant's cell, 0..1.
    pub water: f32,
    pub nutrients: f32,
    pub vigor: f32,
}

/// Totals of the world ledger.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PopulationLedger {
    pub plant_mass_kg: f64,
    pub severed_mass_kg: f64,
    pub environment_mass_kg: f64,
    pub imported_mass_kg: f64,
    /// `plants + severed + environment - imported`: constant in time.
    pub closed_mass_kg: f64,
    pub radiation_in_j: f64,
    pub heat_out_j: f64,
    pub imported_energy_j: f64,
    /// Chemical energy of plants, severed parts and litter minus
    /// `radiation - heat + imports`: zero up to rounding.
    pub energy_residual_j: f64,
}

/// All plants of a world with their shared soil, shade and ledger.
#[derive(Clone, Debug)]
pub struct PlantPopulation {
    config: PopulationConfig,
    members: Vec<Member>,
    soils: Vec<SoilCell>,
    severed: Vec<SeveredRecord>,
    shade: HashMap<(i32, i32), Vec<ShadeEntry>>,
    time_days: f64,
    prepared_chunk: i64,
    mass_reference_kg: f64,
}

impl PlantPopulation {
    /// Create the population: a seed of each individual lands in the soil cell of its
    /// site. Seed mass and energy are imports of that cell. Individuals germinate at
    /// `germination_years` and are not stepped before.
    pub fn new(world_seed: u64, sites: &[SiteInput], config: PopulationConfig) -> Self {
        assert!(config.soil_cell_m > 0.0 && config.shade_cell_m > 0.0 && config.chunk_days > 0.0);
        let mut soils: Vec<SoilCell> = Vec::new();
        let mut index: HashMap<(i32, i32), usize> = HashMap::new();
        let mut members = Vec::with_capacity(sites.len());
        for site in sites {
            let key = (
                (site.x_m / config.soil_cell_m).floor() as i32,
                (site.y_m / config.soil_cell_m).floor() as i32,
            );
            let soil = *index.entry(key).or_insert_with(|| {
                soils.push(SoilCell::new(&config, site.moisture, site.fertility));
                soils.len() - 1
            });
            let id = site.genome.individual_id;
            let seed_mass = config.seed_mass_kg * (0.7 + 0.6 * unit_hash(world_seed, id, 0x5EED));
            let germination_day = site.germination_years.max(0.0) * YEAR_DAYS;
            let plant = Plant::from_seed(
                plant_genome_from_seed(&site.genome),
                seed_mass,
                germination_day,
                &mut soils[soil].env,
            );
            members.push(Member {
                plant,
                id,
                x: site.x_m,
                y: site.y_m,
                soil,
                temperature_k: site.temperature_k,
                germination_day,
                climate: Climate::constant(Conditions::SUMMER),
                light: 1.0,
                light_memory: 0.0,
                light_sum: 0.0,
                light_chunks: 0,
                shape: (0.0, 0.0, 0.0),
                revision: 0,
            });
        }
        let mut population = Self {
            config,
            members,
            soils,
            severed: Vec::new(),
            shade: HashMap::new(),
            time_days: 0.0,
            prepared_chunk: -1,
            mass_reference_kg: 0.0,
        };
        population.mass_reference_kg = population.ledger().closed_mass_kg;
        population
    }

    // ----- accessors ------------------------------------------------------------------

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    pub fn config(&self) -> &PopulationConfig {
        &self.config
    }

    /// Population clock (plant days since world start).
    pub fn time_days(&self) -> f64 {
        self.time_days
    }

    pub fn time_years(&self) -> f64 {
        self.time_days / YEAR_DAYS
    }

    pub fn plant(&self, index: usize) -> &Plant {
        &self.members[index].plant
    }

    pub fn position(&self, index: usize) -> (f64, f64) {
        (self.members[index].x, self.members[index].y)
    }

    pub fn individual_id(&self, index: usize) -> u64 {
        self.members[index].id
    }

    /// Light fraction at the plant's crown top from the last shade update.
    pub fn light_fraction(&self, index: usize) -> f64 {
        self.members[index].light
    }

    /// Mean light fraction over all chunks since germination (1 before the first).
    /// Unlike [`PlantPopulation::light_fraction`] it does not drop out when the
    /// neighbours happen to be leafless at the last update.
    pub fn mean_light_fraction(&self, index: usize) -> f64 {
        let m = &self.members[index];
        if m.light_chunks == 0 { 1.0 } else { m.light_sum / m.light_chunks as f64 }
    }

    /// Index of the soil cell a plant is rooted in; plants with equal indices share it.
    pub fn soil_index(&self, index: usize) -> usize {
        self.members[index].soil
    }

    pub fn soil_count(&self) -> usize {
        self.soils.len()
    }

    pub fn soil_environment(&self, soil: usize) -> &Environment {
        &self.soils[soil].env
    }

    /// Mutable access for scenes (rain, drainage, litter). Changes go through the
    /// environment's own import/export bookkeeping, so the ledger stays closed.
    pub fn soil_environment_mut(&mut self, soil: usize) -> &mut Environment {
        &mut self.soils[soil].env
    }

    /// State revision of a plant; changes whenever the plant is advanced or cut.
    pub fn revision(&self, index: usize) -> u64 {
        self.members[index].revision
    }

    pub fn living_count(&self) -> usize {
        self.members.iter().filter(|m| !m.plant.is_dead() && self.time_days >= m.germination_day).count()
    }

    pub fn total_biomass_kg(&self) -> f64 {
        self.members.iter().map(|m| m.plant.biomass_kg()).sum()
    }

    pub fn severed_count(&self) -> usize {
        self.severed.len()
    }

    /// Severed part `n` and the plant index it was cut from.
    pub fn severed_part(&self, n: usize) -> (usize, &SeveredPart) {
        (self.severed[n].owner, &self.severed[n].part)
    }

    // ----- ledger ---------------------------------------------------------------------

    pub fn ledger(&self) -> PopulationLedger {
        let mut l = PopulationLedger::default();
        let mut chemical = 0.0;
        let mut litter_energy = 0.0;
        for m in &self.members {
            l.plant_mass_kg += m.plant.total_mass_kg();
            chemical += m.plant.chemical_energy_j();
        }
        for r in &self.severed {
            l.severed_mass_kg += r.part.total_mass_kg();
            chemical += r.part.chemical_energy_j();
        }
        for s in &self.soils {
            l.environment_mass_kg += s.env.total_mass();
            l.imported_mass_kg += s.env.imported_mass_kg;
            l.radiation_in_j += s.env.radiation_in_j;
            l.heat_out_j += s.env.heat_out_j;
            l.imported_energy_j += s.env.imported_energy_j;
            litter_energy += s.env.litter_energy_j;
        }
        l.closed_mass_kg = l.plant_mass_kg + l.severed_mass_kg + l.environment_mass_kg - l.imported_mass_kg;
        l.energy_residual_j = chemical + litter_energy - (l.radiation_in_j - l.heat_out_j + l.imported_energy_j);
        l
    }

    /// Drift of the mass invariant since creation (kg). Zero up to rounding.
    pub fn mass_drift_kg(&self) -> f64 {
        self.ledger().closed_mass_kg - self.mass_reference_kg
    }

    // ----- time -----------------------------------------------------------------------

    pub fn advance_years(&mut self, years: f64) {
        self.advance_to_days(self.time_days + years * YEAR_DAYS);
    }

    /// Advance every plant to the absolute plant time `target_days`. Does nothing if
    /// the population is already there.
    pub fn advance_to_days(&mut self, target_days: f64) {
        assert!(target_days.is_finite());
        while self.time_days < target_days - 1e-9 {
            let chunk = (self.time_days / self.config.chunk_days + 1e-9).floor() as i64;
            if chunk > self.prepared_chunk {
                self.prepare_chunk(chunk);
            }
            let chunk_end = (chunk + 1) as f64 * self.config.chunk_days;
            let segment_end = target_days.min(chunk_end).max(self.time_days + 1e-9);
            self.advance_members(segment_end);
            self.advance_severed(segment_end);
            self.time_days = segment_end;
        }
    }

    /// Rain, decomposition and shade for the whole chunk, once.
    fn prepare_chunk(&mut self, chunk: i64) {
        let days = self.config.chunk_days;
        for soil in &mut self.soils {
            soil.prepare(days);
        }
        self.update_shade();
        let warm_start = self.config.season_offset.clamp(0.0, 1.0) * (YEAR_DAYS - WARM_SEASON_DAYS);
        for m in &mut self.members {
            let light = m.light;
            let warm = Conditions { light_fraction: light, temperature_k: m.temperature_k + WARM_OFFSET_K, full_sun_par_w_m2: WARM_PAR_W_M2 };
            let cold = Conditions { light_fraction: light, temperature_k: m.temperature_k - COLD_OFFSET_K, full_sun_par_w_m2: COLD_PAR_W_M2 };
            m.climate = Climate::seasonal(warm, cold, warm_start, WARM_SEASON_DAYS);
            m.light_memory = m.light_memory.max(light);
            if m.shape != (0.0, 0.0, 0.0) {
                m.light_sum += light;
                m.light_chunks += 1;
            }
        }
        self.prepared_chunk = chunk;
    }

    fn advance_members(&mut self, segment_end: f64) {
        let start = self.time_days;
        let round = self.config.competition_round_days.max(1.0);
        let mut groups: Vec<Vec<&mut Member>> = (0..self.soils.len()).map(|_| Vec::new()).collect();
        for m in self.members.iter_mut() {
            if m.active(segment_end) {
                groups[m.soil].push(m);
            }
        }
        let job = |soil: &mut SoilCell, group: &mut Vec<&mut Member>| {
            advance_group(&mut soil.env, group, start, segment_end, round);
        };
        if self.config.parallel && self.soils.len() > 32 {
            self.soils.par_iter_mut().zip(groups.par_iter_mut()).for_each(|(s, g)| job(s, g));
        } else {
            for (s, g) in self.soils.iter_mut().zip(groups.iter_mut()) {
                job(s, g);
            }
        }
    }

    fn advance_severed(&mut self, segment_end: f64) {
        let dt = segment_end - self.time_days;
        for record in &mut self.severed {
            let condition = self.members[record.owner].climate.at(self.time_days);
            record.part.advance(&mut self.soils[record.soil].env, &condition, dt);
        }
    }

    // ----- shade ----------------------------------------------------------------------

    fn update_shade(&mut self) {
        let cell = self.config.shade_cell_m;
        let cell_area = cell * cell;
        self.shade.clear();
        for (i, m) in self.members.iter_mut().enumerate() {
            m.shape = (0.0, 0.0, 0.0);
            if !m.active(self.time_days + 1e-9) || self.time_days < m.germination_day {
                continue;
            }
            let morph = m.plant.morphology();
            let radius = (morph.crown_area_m2 / std::f64::consts::PI).sqrt();
            m.shape = (morph.height_m, radius, morph.leaf_area_m2);
            let cells = covered_cells(m.x, m.y, radius, cell);
            let lai = morph.leaf_area_m2 / (cells.len() as f64 * cell_area);
            for &key in &cells {
                self.shade.entry(key).or_default().push(ShadeEntry {
                    owner: i as u32,
                    top_m: morph.height_m as f32,
                    lai: lai as f32,
                });
            }
        }
        for (i, m) in self.members.iter_mut().enumerate() {
            if m.shape.0 <= 0.0 && m.shape.1 <= 0.0 && m.shape.2 <= 0.0 {
                m.light = 1.0;
                continue;
            }
            let height = m.shape.0.max(0.05);
            let cells = covered_cells(m.x, m.y, m.shape.1, cell);
            let mut transmitted = 0.0;
            let mut n = 0.0;
            for &key in &cells {
                let mut optical = 0.0;
                if let Some(entries) = self.shade.get(&key) {
                    for e in entries {
                        if e.owner as usize == i {
                            continue;
                        }
                        // A neighbour shades fully when its crown top is well above this
                        // plant's top, not at all when well below.
                        let t = ((e.top_m as f64 - 0.8 * height) / (0.4 * height)).clamp(0.0, 1.0);
                        optical += t * t * (3.0 - 2.0 * t) * e.lai as f64;
                    }
                }
                transmitted += (-SHADE_EXTINCTION * optical).exp();
                n += 1.0;
            }
            m.light = (transmitted / n).clamp(0.0, 1.0);
        }
    }

    // ----- geometry -------------------------------------------------------------------

    /// Dimensions derived from the plant's state (pools, rings, wood density).
    pub fn geometry(&self, index: usize) -> PlantGeometry {
        let m = &self.members[index];
        let germinated = self.time_days >= m.germination_day;
        if !germinated {
            return PlantGeometry::default();
        }
        let plant = &m.plant;
        let morph = plant.morphology();
        let dead = plant.is_dead();
        let density = plant.wood_density();
        let max_height = (plant.genome().seed.maximum_height_m as f64).max(0.01);
        let relative_size = (morph.height_m / max_height).clamp(0.0, 1.5);
        let soil = &self.soils[m.soil].env;
        let nutrients = soil.soil_minerals_kg / (soil.soil_minerals_kg + soil.soil_mineral_half_saturation_kg).max(1e-12);
        let trunk_fraction = plant.genome().seed.trunk_height_fraction as f64;
        PlantGeometry {
            germinated,
            dead,
            senescent: plant.is_senescent(),
            age_years: (plant.age_days() / YEAR_DAYS) as f32,
            height_m: morph.height_m as f32,
            trunk_height_m: (morph.height_m * trunk_fraction) as f32,
            trunk_radius_m: (0.5 * morph.trunk_diameter_m) as f32,
            crown_radius_m: (morph.crown_area_m2 / std::f64::consts::PI).sqrt() as f32,
            leaf_area_m2: if dead { 0.0 } else { morph.leaf_area_m2 as f32 },
            branch_volume_m3: if density > 0.0 { (plant.pools.branches / density) as f32 } else { 0.0 },
            branch_count: morph.branch_count,
            biomass_kg: plant.biomass_kg() as f32,
            relative_size: relative_size as f32,
            light: m.light as f32,
            structural_light: m.light_memory.max(m.light) as f32,
            water: (soil.soil_water_kg / soil.soil_water_capacity_kg.max(1e-9)).clamp(0.0, 1.0) as f32,
            nutrients: nutrients.clamp(0.0, 1.0) as f32,
            vigor: (plant.vitality() * (0.6 + 0.4 * m.light)) as f32,
        }
    }

    // ----- cutting --------------------------------------------------------------------

    /// Cut `mass_fraction` (0..1] of the plant's above-ground tissue off with
    /// `Plant::cut_above`. The rooted remainder keeps living; the severed part keeps
    /// its mass and stops growing. Returns the index of the severed part, or `None`
    /// if the plant has nothing above ground to cut.
    pub fn cut(&mut self, index: usize, mass_fraction: f64) -> Option<usize> {
        if index >= self.members.len() || !(mass_fraction > 0.0 && mass_fraction <= 1.0) {
            return None;
        }
        let m = &mut self.members[index];
        let above = m.plant.pools.trunk + m.plant.pools.branches + m.plant.pools.leaves + m.plant.pools.fruit;
        if self.time_days < m.germination_day || above <= 0.0 {
            return None;
        }
        let part = m.plant.cut_above(mass_fraction);
        m.revision += 1;
        let soil = m.soil;
        self.severed.push(SeveredRecord { owner: index, soil, part });
        Some(self.severed.len() - 1)
    }
}

/// Cells of side `cell` whose square meets the disc of `radius` at (x, y); the
/// centre cell at least.
fn covered_cells(x: f64, y: f64, radius: f64, cell: f64) -> Vec<(i32, i32)> {
    let x0 = ((x - radius) / cell).floor() as i32;
    let x1 = ((x + radius) / cell).floor() as i32;
    let y0 = ((y - radius) / cell).floor() as i32;
    let y1 = ((y + radius) / cell).floor() as i32;
    let mut keys = Vec::new();
    for cy in y0..=y1 {
        for cx in x0..=x1 {
            let nx = x.clamp(cx as f64 * cell, (cx + 1) as f64 * cell);
            let ny = y.clamp(cy as f64 * cell, (cy + 1) as f64 * cell);
            if (nx - x).powi(2) + (ny - y).powi(2) <= radius * radius {
                keys.push((cx, cy));
            }
        }
    }
    if keys.is_empty() {
        keys.push(((x / cell).floor() as i32, (y / cell).floor() as i32));
    }
    keys
}

/// Advance the plants of one soil cell from `start` to `end`. A lone plant takes the
/// whole span in one call; several take turns in rounds, starting with a different
/// plant each round, so the shared water is not given to the first one always.
fn advance_group(env: &mut Environment, group: &mut [&mut Member], start: f64, end: f64, round: f64) {
    if group.is_empty() {
        return;
    }
    let advance = |env: &mut Environment, m: &mut Member, to: f64| {
        let dt = to - m.plant.time_days;
        if dt > 1e-9 && !m.plant.is_dead() {
            m.plant.advance(env, &m.climate, dt);
            m.revision += 1;
        }
    };
    if group.len() == 1 {
        advance(env, group[0], end);
        return;
    }
    let n = group.len();
    let mut round_start = start;
    let mut round_index = 0usize;
    while round_start < end - 1e-9 {
        let round_end = (round_start + round).min(end);
        for k in 0..n {
            let member = &mut *group[(k + round_index) % n];
            advance(env, member, round_end);
        }
        round_start = round_end;
        round_index += 1;
    }
}

#[cfg(test)]
mod tests;
