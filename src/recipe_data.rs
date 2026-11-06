//! Reaction recipes as versioned data (DECISIONS_2026-11-05.md item 18).
//!
//! A recipe file is JSON that names components instead of numeric ids, so it
//! stays valid when the material table grows, can be edited and added without
//! recompiling, and is resolved against the table of the scene that loads it:
//!
//! ```json
//! {
//!   "version": 1,
//!   "evaporation": { "water": "water", "vapor": "water_vapor" },
//!   "recipes": [{
//!     "name": "wood rot",
//!     "key": "wood",
//!     "terms": [
//!       { "component": "wood", "nu": -1.0 },
//!       { "component": "oxygen", "nu": -0.5 },
//!       { "component": "humus", "nu": 1.5 }
//!     ],
//!     "rate": {
//!       "reference_rate_per_s": 1.0e-8, "reference_temperature_k": 300.0,
//!       "activation_energy_j_mol": 60000.0
//!     }
//!   }]
//! }
//! ```
//!
//! `terms` are signed mass coefficients per kilogram of the `key` reactant
//! (reactants negative, products positive, key exactly `-1`, sum exactly zero:
//! mass is conserved by construction and a file that does not balance is
//! rejected). The heat of the reaction is not written in the file: it follows
//! from the formation energies of the components in the material table, so a
//! recipe cannot release energy from nothing. The rate law is the one of
//! [`crate::recipes::RateLaw`]; optional fields default to "no such factor",
//! unknown fields are an error (a misspelt threshold must not be ignored), and
//! JSON has no infinity, so an absent `maximum_temperature_k` or `cap` means
//! unbounded.
//!
//! `version` is [`RECIPE_FORMAT_VERSION`]. A file with a higher version is
//! refused with a message instead of being half understood; lower, zero and
//! missing versions are refused too. When the format changes, the loader keeps
//! reading older versions and converts them here.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::material::Material;
use crate::recipes::{
    component_by_name, Deactivation, Dependence, Evaporation, Moisture, RateLaw, Recipe,
    RecipeRegistry, WATER_BOILING_K,
};

/// Version written by [`RecipeSet::to_json`] and the newest one understood.
pub const RECIPE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeSet {
    pub version: u32,
    /// Liquid water -> vapour process shared by all recipes (see
    /// [`crate::recipes::Evaporation`]); absent: none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaporation: Option<EvaporationData>,
    #[serde(default)]
    pub recipes: Vec<RecipeData>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaporationData {
    pub water: String,
    pub vapor: String,
    /// Boiling temperature, K; default 373.15.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub boiling_k: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeData {
    pub name: String,
    /// Component whose mass the rate is proportional to; coefficient `-1`.
    pub key: String,
    pub terms: Vec<TermData>,
    /// Counted in the legacy `fuel_burned_kg` / `combustion_heat_j` totals.
    #[serde(default, skip_serializing_if = "is_false")]
    pub combustion: bool,
    pub rate: RateData,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TermData {
    pub component: String,
    /// kg of the component per kg of key reactant; negative for reactants.
    pub nu: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateData {
    /// Arrhenius pre-exponential factor, 1/s. Give this, or the pair
    /// `reference_rate_per_s` at `reference_temperature_k`, not both.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_exponential_per_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_rate_per_s: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_temperature_k: Option<f64>,
    pub activation_energy_j_mol: f64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub minimum_temperature_k: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_temperature_k: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deactivation: Option<DeactivationData>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<DependenceData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moisture: Option<MoistureData>,
    #[serde(default = "default_max_fraction")]
    pub max_fraction_per_substep: f64,
    #[serde(default = "default_max_heating")]
    pub max_heating_per_substep_k: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeactivationData {
    pub midpoint_k: f64,
    pub width_k: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependenceData {
    pub component: String,
    pub reference_mass_fraction: f64,
    /// Upper bound of the factor; absent: unbounded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoistureData {
    pub water: String,
    pub minimum_ratio: f64,
    pub reference_ratio: f64,
}

fn default_max_fraction() -> f64 {
    0.25
}

fn default_max_heating() -> f64 {
    25.0
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_zero(value: &f64) -> bool {
    *value == 0.0
}

impl RecipeSet {
    /// Parse a recipe file and check its version. Components are resolved
    /// later, against a material table, by [`RecipeSet::resolve`].
    pub fn from_json(text: &str) -> Result<Self, String> {
        // Read the version first so that a file from a future format reports
        // that, not a confusing complaint about its unknown fields.
        let probe: serde_json::Value =
            serde_json::from_str(text).map_err(|e| format!("recipe file is not valid JSON: {e}"))?;
        match probe.get("version").and_then(serde_json::Value::as_u64) {
            None => return Err("recipe file needs an integer \"version\"".into()),
            Some(v) if v == 0 || v > RECIPE_FORMAT_VERSION as u64 => {
                return Err(format!(
                    "recipe file version {v} is not supported (this build reads 1 to {RECIPE_FORMAT_VERSION})"
                ))
            }
            Some(_) => {}
        }
        let set: Self =
            serde_json::from_value(probe).map_err(|e| format!("invalid recipe file: {e}"))?;
        Ok(set)
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("cannot write recipes: {e}"))
    }

    /// Resolve component names against `materials` and validate every recipe.
    /// The registry derives each heat of reaction from the table's formation
    /// energies.
    pub fn resolve(&self, materials: &[Material]) -> Result<RecipeRegistry, String> {
        let (recipes, evaporation) = self.resolve_parts(materials)?;
        RecipeRegistry::new(materials, recipes, evaporation).map_err(str::to_owned)
    }

    pub(crate) fn resolve_parts(
        &self,
        materials: &[Material],
    ) -> Result<(Vec<Recipe>, Option<Evaporation>), String> {
        let id = |name: &str, what: &str| {
            component_by_name(materials, name)
                .ok_or_else(|| format!("{what}: component \"{name}\" is not in the material table"))
        };
        let evaporation = match &self.evaporation {
            None => None,
            Some(e) => {
                let mut evaporation = Evaporation::for_components(
                    materials,
                    id(&e.water, "evaporation")?,
                    id(&e.vapor, "evaporation")?,
                );
                evaporation.boiling_k = e.boiling_k.unwrap_or(WATER_BOILING_K);
                Some(evaporation)
            }
        };
        let mut recipes = Vec::with_capacity(self.recipes.len());
        for data in &self.recipes {
            let what = format!("recipe \"{}\"", data.name);
            let mut terms = Vec::with_capacity(data.terms.len());
            for term in &data.terms {
                terms.push((id(&term.component, &what)?, term.nu));
            }
            let r = &data.rate;
            let mut rate = match (r.pre_exponential_per_s, r.reference_rate_per_s, r.reference_temperature_k) {
                (Some(pre), None, None) => RateLaw::arrhenius(pre, r.activation_energy_j_mol),
                (None, Some(rate), Some(kelvin)) if rate > 0.0 && kelvin > 0.0 => {
                    RateLaw::arrhenius_from_reference(rate, kelvin, r.activation_energy_j_mol)
                }
                _ => {
                    return Err(format!(
                        "{what}: give either pre_exponential_per_s or a positive reference_rate_per_s with reference_temperature_k"
                    ))
                }
            };
            rate.minimum_temperature_k = r.minimum_temperature_k;
            rate.maximum_temperature_k = r.maximum_temperature_k.unwrap_or(f64::INFINITY);
            rate.deactivation = r.deactivation.as_ref().map(|d| Deactivation {
                midpoint_k: d.midpoint_k,
                width_k: d.width_k,
            });
            for d in &r.dependencies {
                rate.dependencies.push(Dependence {
                    component: id(&d.component, &what)?,
                    reference_mass_fraction: d.reference_mass_fraction,
                    cap: d.cap.unwrap_or(f64::INFINITY),
                });
            }
            if let Some(m) = &r.moisture {
                rate.moisture = Some(Moisture {
                    water: id(&m.water, &what)?,
                    minimum_ratio: m.minimum_ratio,
                    reference_ratio: m.reference_ratio,
                });
            }
            rate.max_fraction_per_substep = r.max_fraction_per_substep;
            rate.max_heating_per_substep_k = r.max_heating_per_substep_k;
            let recipe = Recipe {
                name: data.name.clone(),
                key: id(&data.key, &what)?,
                terms,
                rate,
                is_combustion: data.combustion,
            };
            recipe.validate().map_err(|e| format!("{what}: {e}"))?;
            recipes.push(recipe);
        }
        Ok((recipes, evaporation))
    }
}

impl RecipeRegistry {
    /// The registry as data (component ids written as names of `materials`).
    /// Reading it back with [`RecipeSet::resolve`] gives the same recipes up to
    /// the last digit of the decimal rendering.
    pub fn to_set(&self, materials: &[Material]) -> RecipeSet {
        let name = |id: u16| materials[id as usize].name.clone();
        RecipeSet {
            version: RECIPE_FORMAT_VERSION,
            evaporation: self.evaporation().map(|e| EvaporationData {
                water: name(e.water),
                vapor: name(e.vapor),
                boiling_k: (e.boiling_k != WATER_BOILING_K).then_some(e.boiling_k),
            }),
            recipes: self
                .recipes()
                .iter()
                .map(|r| RecipeData {
                    name: r.name.clone(),
                    key: name(r.key),
                    terms: r
                        .terms
                        .iter()
                        .map(|&(id, nu)| TermData {
                            component: name(id),
                            nu,
                        })
                        .collect(),
                    combustion: r.is_combustion,
                    rate: RateData {
                        pre_exponential_per_s: Some(r.rate.pre_exponential_per_s),
                        reference_rate_per_s: None,
                        reference_temperature_k: None,
                        activation_energy_j_mol: r.rate.activation_energy_j_mol,
                        minimum_temperature_k: r.rate.minimum_temperature_k,
                        maximum_temperature_k: r
                            .rate
                            .maximum_temperature_k
                            .is_finite()
                            .then_some(r.rate.maximum_temperature_k),
                        deactivation: r.rate.deactivation.map(|d| DeactivationData {
                            midpoint_k: d.midpoint_k,
                            width_k: d.width_k,
                        }),
                        dependencies: r
                            .rate
                            .dependencies
                            .iter()
                            .map(|d| DependenceData {
                                component: name(d.component),
                                reference_mass_fraction: d.reference_mass_fraction,
                                cap: d.cap.is_finite().then_some(d.cap),
                            })
                            .collect(),
                        moisture: r.rate.moisture.map(|m| MoistureData {
                            water: name(m.water),
                            minimum_ratio: m.minimum_ratio,
                            reference_ratio: m.reference_ratio,
                        }),
                        max_fraction_per_substep: r.rate.max_fraction_per_substep,
                        max_heating_per_substep_k: r.rate.max_heating_per_substep_k,
                    },
                })
                .collect(),
        }
    }

    /// A registry made only from a recipe file.
    pub fn from_json(materials: &[Material], text: &str) -> Result<Self, String> {
        RecipeSet::from_json(text)?.resolve(materials)
    }

    pub fn from_file(materials: &[Material], path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read recipe file {}: {e}", path.display()))?;
        Self::from_json(materials, &text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// This registry plus the recipes of a data set (appended in file order,
    /// so earlier recipes keep priority over a shared reactant). Names must
    /// stay unique. The set's evaporation, when present, replaces the current
    /// one.
    pub fn extended_with(&self, materials: &[Material], set: &RecipeSet) -> Result<Self, String> {
        let (added, evaporation) = set.resolve_parts(materials)?;
        let mut recipes = self.recipes().to_vec();
        recipes.extend(added);
        RecipeRegistry::new(materials, recipes, evaporation.or(self.evaporation().copied()))
            .map_err(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::{composition_from_masses, mass_of};
    use crate::scene::{Scene, SceneConfig, VoxelSpawn};
    use crate::voxel::Composition;

    const DAY_S: f64 = 86_400.0;

    fn wood_rot_text() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join("wood_rot.json");
        std::fs::read_to_string(path).unwrap()
    }

    fn id(table: &[Material], name: &str) -> u16 {
        component_by_name(table, name).unwrap()
    }

    fn masses(table: &[Material], parts: &[(&str, f64)]) -> Vec<(u16, f64)> {
        parts.iter().map(|&(n, m)| (id(table, n), m)).collect()
    }

    fn energy_at(table: &[Material], state: &[(u16, f64)], kelvin: f64) -> f64 {
        let total: f64 = state.iter().map(|m| m.1).sum();
        Material::mixed(&composition_from_masses(state), table)
            .energy_at_temperature(total as f32, kelvin as f32) as f64
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
    }

    fn wood_composition(table: &[Material]) -> Composition {
        Composition {
            components: smallvec::smallvec![
                (id(table, "wood"), 0.25),
                (id(table, "oxygen"), 0.5),
                (id(table, "water"), 0.25)
            ],
        }
    }

    fn wood_scene(table: &[Material]) -> Scene {
        let mut spawn = VoxelSpawn::at_cell([0, 0, 0], id(table, "wood"));
        spawn.composition = wood_composition(table);
        spawn.temperature_kelvin = 300.0;
        Scene::from_voxels(SceneConfig::default(), vec![spawn]).unwrap()
    }

    #[test]
    fn builtin_registry_round_trips_through_versioned_json() {
        let table = Material::table();
        let registry = RecipeRegistry::builtin(&table);
        let text = registry.to_set(&table).to_json().unwrap();
        assert!(text.contains("\"version\": 1"));
        // Components are written by name, never by numeric id.
        assert!(text.contains("\"component\": \"hide\"") && text.contains("\"key\": \"dry_wood\""));
        let back = RecipeRegistry::from_json(&table, &text).unwrap();
        assert_eq!(back.recipes().len(), registry.recipes().len());
        assert_eq!(back.evaporation(), registry.evaporation());
        for (i, (a, b)) in registry.recipes().iter().zip(back.recipes()).enumerate() {
            assert_eq!((&a.name, a.key, a.is_combustion), (&b.name, b.key, b.is_combustion));
            assert_eq!(a.terms.len(), b.terms.len());
            for (x, y) in a.terms.iter().zip(&b.terms) {
                assert!(x.0 == y.0 && close(x.1, y.1), "{}: {x:?} {y:?}", a.name);
            }
            let (ra, rb) = (&a.rate, &b.rate);
            assert!(close(ra.pre_exponential_per_s, rb.pre_exponential_per_s), "{}", a.name);
            assert!(close(ra.activation_energy_j_mol, rb.activation_energy_j_mol));
            assert_eq!(ra.minimum_temperature_k, rb.minimum_temperature_k);
            assert_eq!(ra.maximum_temperature_k, rb.maximum_temperature_k);
            assert_eq!(ra.deactivation, rb.deactivation);
            assert_eq!(ra.dependencies.len(), rb.dependencies.len());
            assert_eq!(ra.moisture, rb.moisture);
            assert_eq!(ra.max_fraction_per_substep, rb.max_fraction_per_substep);
            assert_eq!(ra.max_heating_per_substep_k, rb.max_heating_per_substep_k);
            // The heat of reaction is not stored: both derive the same value
            // from the formation energies.
            assert!(close(
                registry.heat_of_reaction_j_per_kg(i) + 1.0,
                back.heat_of_reaction_j_per_kg(i) + 1.0
            ));
        }
        // And they behave the same.
        let start =
            masses(&table, &[("hide", 1.0e-3), ("tannin", 4.0e-4), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let energy = energy_at(&table, &start, 303.0);
        let (mut a, mut b) = (start.clone(), start);
        registry.step_masses(&table, &mut a, energy, DAY_S).unwrap();
        back.step_masses(&table, &mut b, energy, DAY_S).unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x.1 - y.1).abs() <= 1e-9 * 1.0e-3, "{x:?} {y:?}");
        }
    }

    #[test]
    fn a_recipe_file_adds_a_reaction_without_any_code_change() {
        let table = Material::table();
        let base = RecipeRegistry::builtin(&table);
        assert!(base.recipe_index("decay of wood").is_none());
        let wood_rev = masses(&table, &[("wood", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-3)]);
        let energy = energy_at(&table, &wood_rev, 300.0);
        let mut untouched = wood_rev.clone();
        let extent = base.step_masses(&table, &mut untouched, energy, 100.0 * DAY_S).unwrap();
        assert!(extent.is_empty() && untouched == wood_rev, "wood must not rot before the file is loaded");

        // The file alone is a registry; extended, it joins the built-ins.
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("recipes").join("wood_rot.json");
        let only = RecipeRegistry::from_file(&table, &path).unwrap();
        assert_eq!(only.recipes().len(), 1);
        let set = RecipeSet::from_json(&wood_rot_text()).unwrap();
        let registry = base.extended_with(&table, &set).unwrap();
        assert_eq!(registry.recipes().len(), base.recipes().len() + 1);
        assert!(registry.evaporation().is_some(), "the extension keeps the evaporation process");
        let index = registry.recipe_index("decay of wood").unwrap();
        // Heat of reaction derived from the formation energies of O2 and CO2.
        let q = registry.heat_of_reaction_j_per_kg(index);
        let expected = 0.75 * 15_678_100.0 - 0.875 * 6_736_800.0;
        assert!((q - expected).abs() < 2.0e3, "heat {q}");

        let mut state = wood_rev.clone();
        let before: f64 = state.iter().map(|m| m.1).sum();
        let extent = registry.step_masses(&table, &mut state, energy, 100.0 * DAY_S).unwrap();
        let xi = extent.recipes[index].extent_kg;
        assert!(xi > 1.0e-5 && xi < 5.0e-4, "wood rotted: {xi} kg in 100 days");
        assert!((state.iter().map(|m| m.1).sum::<f64>() - before).abs() <= 1e-13 * before);
        assert!((mass_of(&state, id(&table, "wood")) - (1.0e-3 - xi)).abs() < 1e-15);
        assert!((mass_of(&state, id(&table, "humus")) - 0.5 * xi).abs() < 1e-15);
        assert!((mass_of(&state, id(&table, "oxygen")) - (2.0e-3 - 0.75 * xi)).abs() < 1e-15);
        assert!((extent.recipes[index].heat_j - xi * q).abs() <= 1e-9 * xi * q);
        // Dry wood does not rot, and neither does wood in the cold.
        let mut dry = masses(&table, &[("wood", 1.0e-3), ("oxygen", 2.0e-3), ("water", 1.0e-4)]);
        let e = energy_at(&table, &dry, 300.0);
        assert!(registry.step_masses(&table, &mut dry, e, 100.0 * DAY_S).unwrap().is_empty());
        let mut cold = wood_rev.clone();
        let e = energy_at(&table, &cold, 260.0);
        assert!(registry.step_masses(&table, &mut cold, e, 100.0 * DAY_S).unwrap().is_empty());

        // In a scene the registry is passed in; mass and stored energy do not move.
        let mut scene = wood_scene(&table);
        let (mass, stored) = (scene.voxels[0].mass, scene.voxels[0].total_internal_energy_j());
        let report = scene.react_with(&registry, DAY_S as f32).unwrap();
        assert_eq!(report.reacting_revs, 1, "{report:?}");
        assert_eq!(report.recipes[0].name, "decay of wood");
        assert_eq!(scene.voxels[0].mass.to_bits(), mass.to_bits());
        assert_eq!(scene.voxels[0].total_internal_energy_j().to_bits(), stored.to_bits());
        let humus = id(&table, "humus");
        assert!(scene.voxels[0].composition.components.iter().any(|&(c, _)| c == humus));
        // The built-in registry alone leaves the same scene untouched.
        let mut other = wood_scene(&table);
        assert_eq!(other.react(DAY_S as f32).unwrap().reacting_revs, 0);
    }

    #[test]
    fn bad_recipe_files_are_refused_with_a_reason() {
        let table = Material::table();
        let good: serde_json::Value = serde_json::from_str(&wood_rot_text()).unwrap();
        let load = |value: &serde_json::Value| {
            RecipeRegistry::from_json(&table, &value.to_string()).map(|r| r.recipes().len())
        };
        assert_eq!(load(&good), Ok(1));
        let with = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut v = good.clone();
            edit(&mut v);
            load(&v).unwrap_err()
        };
        assert!(with(&|v| v["version"] = 2.into()).contains("not supported"));
        assert!(with(&|v| v["version"] = 0.into()).contains("not supported"));
        assert!(with(&|v| {
            v.as_object_mut().unwrap().remove("version");
        })
        .contains("version"));
        assert!(with(&|v| v["recipes"][0]["terms"][2]["component"] = "mithril".into())
            .contains("mithril"));
        assert!(with(&|v| v["recipes"][0]["terms"][2]["nu"] = 0.6.into()).contains("sum to zero"));
        assert!(with(&|v| v["recipes"][0]["terms"][0]["nu"] = (-2.0).into()).contains("-1"));
        assert!(with(&|v| v["recipes"][0]["rate"]["maximum_temperatur_k"] = 300.0.into())
            .contains("unknown field"));
        assert!(with(&|v| v["recipes"][0]["rate"]["pre_exponential_per_s"] = 1.0.into())
            .contains("pre_exponential_per_s"));
        assert!(with(&|v| v["recipes"][0]["rate"]["activation_energy_j_mol"] = (-1.0).into())
            .contains("activation"));
        assert!(with(&|v| v["recipes"][0]["rate"]["maximum_temperature_k"] = 100.0.into())
            .contains("thresholds"));
        assert!(RecipeRegistry::from_json(&table, "{ not json").unwrap_err().contains("JSON"));
        // A later format version is named, not half understood, even when its
        // other fields are unknown to this build.
        let future = r#"{ "version": 7, "layers": [] }"#;
        assert!(RecipeSet::from_json(future).unwrap_err().contains("version 7"));
        // The same recipe twice is refused when extending a registry.
        let set = RecipeSet::from_json(&wood_rot_text()).unwrap();
        let once = RecipeRegistry::builtin(&table).extended_with(&table, &set).unwrap();
        assert!(once.extended_with(&table, &set).unwrap_err().contains("unique"));
        // A table without the named components (an old scene) refuses cleanly.
        let old: Vec<Material> = table[..21].to_vec();
        assert!(set.resolve(&old).unwrap_err().contains("humus"));
    }
}
