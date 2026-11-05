//! Bounded, JSON-friendly snapshots of conserved and derived scene state.
//!
//! This is diagnostic aggregation only. It does not infer an energy residual
//! because external work, boundary exchange, and prescribed motion may be
//! incomplete in the available ledger.

use crate::{bond::EMPTY, scene::Scene};
use glam::DVec3;
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};

const MAX_COMPONENT_SUMMARIES: usize = 64;
const MAX_EXACT_RUNTIME_REVS: usize = 2_048;
const MAX_RUNTIME_SAMPLE_REVS: usize = 128;
const PERFORMANCE_WINDOW_CAPACITY: usize = 256;
// User revised the first hardware target on 2026-11-05 to 50k active REVs.
// A larger measured workload is eligible too; this is not a world-size cap.
const TARGET_ACTIVE_REVS_MIN: usize = 50_000;
const TARGET_WALL_MS_PER_10MS: f64 = 5.0;

#[derive(Clone, Copy, Debug)]
struct PerformanceSample {
    active_dynamic_fine_revs: usize,
    simulated_seconds: f64,
    solver_wall_ms: f64,
    full_worker_wall_ms: f64,
}

/// Bounded rolling measurements of actual worker batches. Percentiles are
/// computed only from observed batches; this helper never extrapolates by REV
/// count or marks the overall goal achieved.
#[derive(Clone, Debug, Default)]
pub struct PerformanceWindow {
    samples: VecDeque<PerformanceSample>,
}

impl PerformanceWindow {
    /// Add one completed batch. `full_worker_wall_ms` includes metrics publish
    /// and other observer cost, while `solver_wall_ms` is the solver portion.
    /// Durations must be finite, nonnegative, and consistent with that scope.
    /// Returns false without mutation for invalid data.
    pub fn push_batch(
        &mut self,
        active_dynamic_fine_revs: usize,
        completed_simulated_seconds: f64,
        solver_wall_ms: f64,
        full_worker_wall_ms: f64,
    ) -> bool {
        if !completed_simulated_seconds.is_finite()
            || completed_simulated_seconds <= 0.0
            || !solver_wall_ms.is_finite()
            || solver_wall_ms < 0.0
            || !full_worker_wall_ms.is_finite()
            || full_worker_wall_ms < solver_wall_ms
            || !normalized(solver_wall_ms, completed_simulated_seconds).is_finite()
            || !normalized(full_worker_wall_ms, completed_simulated_seconds).is_finite()
        {
            return false;
        }
        if self.samples.len() == PERFORMANCE_WINDOW_CAPACITY {
            self.samples.pop_front();
        }
        self.samples.push_back(PerformanceSample {
            active_dynamic_fine_revs,
            simulated_seconds: completed_simulated_seconds,
            solver_wall_ms,
            full_worker_wall_ms,
        });
        true
    }

    /// Return bounded statistics and explicit target status for real samples.
    /// `budget_pass` and active-REV eligibility are reported independently;
    /// neither establishes correctness or represents a goal-achieved claim.
    pub fn to_json(&self) -> Value {
        let count = self.samples.len();
        let simulated_seconds: f64 = self.samples.iter().map(|s| s.simulated_seconds).sum();
        let solver_wall_ms: f64 = self.samples.iter().map(|s| s.solver_wall_ms).sum();
        let full_worker_wall_ms: f64 = self.samples.iter().map(|s| s.full_worker_wall_ms).sum();
        let solver_costs = normalized_costs(&self.samples, true);
        let worker_costs = normalized_costs(&self.samples, false);
        let min_revs = self
            .samples
            .iter()
            .map(|s| s.active_dynamic_fine_revs)
            .min();
        let max_revs = self
            .samples
            .iter()
            .map(|s| s.active_dynamic_fine_revs)
            .max();
        let latest = self.samples.back();
        let eligible = min_revs.is_some_and(|n| n >= TARGET_ACTIVE_REVS_MIN);
        let observed_max_worker_cost = worker_costs.iter().copied().reduce(f64::max);
        let budget_pass =
            observed_max_worker_cost.is_some_and(|max_cost| max_cost <= TARGET_WALL_MS_PER_10MS);
        json!({
            "sample_count": count,
            "capacity": PERFORMANCE_WINDOW_CAPACITY,
            "measured_span": {
                "simulated_seconds": simulated_seconds,
                "solver_wall_ms": solver_wall_ms,
                "full_worker_wall_ms_including_metrics_publish": full_worker_wall_ms
            },
            "active_dynamic_fine_revs": {
                "min": min_revs,
                "max": max_revs,
                "latest": latest.map(|s| s.active_dynamic_fine_revs)
            },
            "target": {
                "active_dynamic_fine_revs_min": TARGET_ACTIVE_REVS_MIN,
                "active_dynamic_fine_revs_max": Value::Null,
                "active_count_semantics": "minimum measured load, not a runtime/world cap",
                "wall_ms": TARGET_WALL_MS_PER_10MS,
                "simulated_ms": 10.0,
                "normalized_to_simulated_ms": 10.0
            },
            "eligibility": {
                "in_active_rev_range": eligible,
                "measured_count_range_only": true
            },
            "solver_ms_per_10ms_simulated": distribution_json(&solver_costs),
            "full_worker_ms_per_10ms_simulated": distribution_json(&worker_costs),
            "latest_cost": latest.map(|sample| json!({
                "active_dynamic_fine_revs": sample.active_dynamic_fine_revs,
                "simulated_ms": sample.simulated_seconds * 1000.0,
                "solver_wall_ms": sample.solver_wall_ms,
                "solver_ms_per_10ms_simulated": normalized(sample.solver_wall_ms, sample.simulated_seconds),
                "full_worker_wall_ms_including_metrics_publish": sample.full_worker_wall_ms,
                "full_worker_ms_per_10ms_simulated": normalized(sample.full_worker_wall_ms, sample.simulated_seconds)
            })),
            "budget_pass": if count == 0 { Value::Null } else { json!(budget_pass) },
            "goal_status": "not_assessed; requires correctness validation and a representative workload",
            "extrapolation": "none; all statistics use observed batch samples"
        })
    }
}

fn normalized_costs(samples: &VecDeque<PerformanceSample>, solver: bool) -> Vec<f64> {
    samples
        .iter()
        .map(|sample| {
            normalized(
                if solver {
                    sample.solver_wall_ms
                } else {
                    sample.full_worker_wall_ms
                },
                sample.simulated_seconds,
            )
        })
        .collect()
}

fn normalized(wall_ms: f64, simulated_seconds: f64) -> f64 {
    wall_ms * 0.01 / simulated_seconds
}

fn distribution_json(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({
        "p50": percentile(&sorted, 0.50),
        "p95": percentile(&sorted, 0.95),
        "p99": percentile(&sorted, 0.99),
        "max": sorted[sorted.len() - 1]
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (fraction * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[derive(Default)]
struct ComponentMetrics {
    cluster_id: u32,
    identity: String,
    fixed: bool,
    count: usize,
    mass: f64,
    position_mass_sum: DVec3,
    velocity_mass_sum: DVec3,
    kinetic_j: f64,
    internal_j: f64,
    temperature_min_k: f64,
    temperature_max_k: f64,
    temperature_mass_sum: f64,
    temperature_mass: f64,
    solid_mass_kg: f64,
    liquid_mass_kg: f64,
    gas_mass_kg: f64,
    max_speed_m_s: f64,
    bbox_min_m: DVec3,
    bbox_max_m: DVec3,
    has_position: bool,
    has_temperature: bool,
}

impl ComponentMetrics {
    fn new(cluster_id: u32, identity: String, fixed: bool) -> Self {
        Self {
            cluster_id,
            identity,
            fixed,
            temperature_min_k: f64::INFINITY,
            temperature_max_k: f64::NEG_INFINITY,
            bbox_min_m: DVec3::splat(f64::INFINITY),
            bbox_max_m: DVec3::splat(f64::NEG_INFINITY),
            ..Self::default()
        }
    }

    fn json(&self, sampled: bool) -> Value {
        let center = if self.mass > 0.0 {
            self.position_mass_sum / self.mass
        } else {
            DVec3::ZERO
        };
        let mean_velocity = if self.mass > 0.0 {
            self.velocity_mass_sum / self.mass
        } else {
            DVec3::ZERO
        };
        json!({
            "identity": self.identity,
            "cluster_id": self.cluster_id,
            "fixed": self.fixed,
            "count": self.count,
            "sampled_member_count": if sampled { json!(self.count) } else { Value::Null },
            "population_count": if sampled { Value::Null } else { json!(self.count) },
            "scope": if sampled { "sampled members only; not the whole component" } else { "complete listed membership for this summary row" },
            "mass_kg": self.mass,
            "center_of_mass_m": vec3_json(center),
            "mean_velocity_m_s": vec3_json(mean_velocity),
            "max_speed_m_s": self.max_speed_m_s,
            "bounding_box_m": if self.has_position {
                json!({"min": vec3_json(self.bbox_min_m), "max": vec3_json(self.bbox_max_m)})
            } else { Value::Null },
            "temperature_range_k": if self.has_temperature {
                json!({"min": self.temperature_min_k, "max": self.temperature_max_k})
            } else { Value::Null },
            "mean_temperature_mass_weighted_k": if self.temperature_mass > 0.0 {
                json!(self.temperature_mass_sum / self.temperature_mass)
            } else { Value::Null },
            "phase_mass_kg": {
                "solid": self.solid_mass_kg,
                "liquid": self.liquid_mass_kg,
                "gas": self.gas_mass_kg
            },
            "kinetic_energy_j": self.kinetic_j,
            "internal_energy_j": self.internal_j
        })
    }
}

/// Collect one bounded diagnostic snapshot. `max_components` is additionally
/// capped at 64 so callers cannot accidentally request unbounded JSON output.
/// Kinematic REVs use the `u32::MAX` cluster sentinel and are combined into a
/// `fixed_supports` diagnostic group. Their authored model/material identity is
/// intentionally not reported.
pub fn collect_scene_metrics(scene: &Scene, max_components: usize) -> Value {
    collect_scene_metrics_impl(scene, max_components, 0..scene.voxels.len(), None)
}

/// Collect diagnostics suitable for runtime publication. Scenes up to 2,048
/// REVs receive the exact snapshot. Larger scenes inspect at most 128 evenly
/// spaced stored REV indices; global sums and topology counts remain unknown
/// rather than being estimated from that sample.
pub fn collect_runtime_metrics(scene: &Scene, max_components: usize) -> Value {
    if scene.voxels.len() <= MAX_EXACT_RUNTIME_REVS {
        return collect_scene_metrics(scene, max_components);
    }
    let indices = runtime_sample_indices(scene.voxels.len());
    collect_scene_metrics_impl(
        scene,
        max_components,
        indices.iter().copied(),
        Some(&indices),
    )
}

fn runtime_sample_indices(total: usize) -> Vec<usize> {
    let count = total.min(MAX_RUNTIME_SAMPLE_REVS);
    (0..count).map(|sample| sample * total / count).collect()
}

fn collect_scene_metrics_impl<I>(
    scene: &Scene,
    max_components: usize,
    indices: I,
    sampled_indices: Option<&[usize]>,
) -> Value
where
    I: Iterator<Item = usize>,
{
    let exact = sampled_indices.is_none();
    let limit = max_components.min(MAX_COMPONENT_SUMMARIES);
    let mut dynamic_count = 0usize;
    let mut fine_dynamic_count = 0usize;
    let mut coarse_dynamic_count = 0usize;
    let fixed_count = if exact {
        scene.voxels.iter().filter(|voxel| voxel.kinematic).count()
    } else {
        sampled_indices
            .unwrap_or_default()
            .iter()
            .filter(|&&index| scene.voxels[index].kinematic)
            .count()
    };
    let mut coarse_count = 0usize;
    let mut non_finite_count = 0usize;
    let mut out_of_range_count = 0usize;
    let mut mass = 0.0f64;
    let mut momentum = DVec3::ZERO;
    let mut kinetic = 0.0f64;
    let mut internal = 0.0f64;
    let mut temperature_min = f64::INFINITY;
    let mut temperature_max = f64::NEG_INFINITY;
    let mut temperature_mass_sum = 0.0f64;
    let mut temperature_mass = 0.0f64;
    let mut solid_mass = 0.0f64;
    let mut liquid_mass = 0.0f64;
    let mut gas_mass = 0.0f64;
    let mut components = BTreeMap::<u64, ComponentMetrics>::new();

    for i in indices {
        let Some(voxel) = scene.voxels.get(i) else {
            continue;
        };
        if !voxel.kinematic {
            dynamic_count += 1;
        }
        let is_coarse = scene.aggregate_for_voxel(i).is_some();
        if is_coarse {
            coarse_count += 1;
        }
        if !voxel.kinematic {
            if is_coarse {
                coarse_dynamic_count += 1;
            } else {
                fine_dynamic_count += 1;
            }
        }

        let m = finite_nonnegative(
            voxel.mass as f64,
            &mut non_finite_count,
            &mut out_of_range_count,
        );
        mass += m;
        let velocity = voxel.velocity.as_dvec3();
        if velocity.is_finite() {
            momentum += velocity * m;
            kinetic += 0.5 * m * velocity.length_squared();
        } else {
            non_finite_count += 1;
        }
        if !voxel.has_finite_thermal_state() {
            non_finite_count += 1;
        }
        let energy_j = finite_nonnegative(
            voxel.total_internal_energy_j(),
            &mut non_finite_count,
            &mut out_of_range_count,
        );
        internal += energy_j;

        let phase = scene
            .properties
            .get(i)
            .map(|material| material.phase_state(voxel.mass, voxel.thermal_energy_f32()));
        let (temp, solid, liquid, gas) = if let Some(phase) = phase {
            let temp = finite_nonnegative(
                phase.temperature_kelvin as f64,
                &mut non_finite_count,
                &mut out_of_range_count,
            );
            let solid = finite_fraction(
                phase.solid_fraction as f64,
                &mut non_finite_count,
                &mut out_of_range_count,
            );
            let liquid = finite_fraction(
                phase.liquid_fraction as f64,
                &mut non_finite_count,
                &mut out_of_range_count,
            );
            let gas = finite_fraction(
                phase.gas_fraction as f64,
                &mut non_finite_count,
                &mut out_of_range_count,
            );
            (temp, solid, liquid, gas)
        } else {
            non_finite_count += 1;
            (0.0, 0.0, 0.0, 0.0)
        };
        solid_mass += m * solid;
        liquid_mass += m * liquid;
        gas_mass += m * gas;
        temperature_min = temperature_min.min(temp);
        temperature_max = temperature_max.max(temp);
        temperature_mass_sum += temp * m;
        temperature_mass += m;

        let fixed = voxel.kinematic;
        let cluster_id = scene.clusters.get(i).copied().unwrap_or(u32::MAX);
        let fixed_component = fixed || cluster_id == u32::MAX;
        let key = if fixed_component {
            u32::MAX as u64
        } else {
            u64::from(cluster_id)
        };
        // Keep allocation bounded even for a scene with hundreds of thousands
        // of isolated dynamic REVs. Cluster labels are deterministic; retain
        // only the first `limit` identities encountered and count the rest via
        // the cached component_count below.
        let component_limit = if fixed_component {
            limit
        } else {
            limit.saturating_sub(usize::from(fixed_count > 0))
        };
        // Validate sampled or exact state before any component-cap early exit,
        // so invalid positions are never hidden by the diagnostic bound.
        let position = scene.positions.get(i).copied();
        if position.is_none_or(|p| !p.is_finite()) {
            non_finite_count += 1;
        }
        if !components.contains_key(&key) && components.len() >= component_limit {
            continue;
        }
        let component = components.entry(key).or_insert_with(|| {
            let identity = if fixed_component {
                "fixed_supports".to_owned()
            } else {
                format!("cluster:{cluster_id}")
            };
            ComponentMetrics::new(cluster_id, identity, fixed_component)
        });
        component.count += 1;
        component.mass += m;
        component.internal_j += energy_j;
        if velocity.is_finite() {
            component.velocity_mass_sum += velocity * m;
            component.kinetic_j += 0.5 * m * velocity.length_squared();
            component.max_speed_m_s = component.max_speed_m_s.max(velocity.length() as f64);
        }
        if let Some(position) = position {
            if position.is_finite() {
                component.position_mass_sum += position * m;
                component.bbox_min_m = component.bbox_min_m.min(position);
                component.bbox_max_m = component.bbox_max_m.max(position);
                component.has_position = true;
            }
        }
        component.temperature_min_k = component.temperature_min_k.min(temp);
        component.temperature_max_k = component.temperature_max_k.max(temp);
        component.temperature_mass_sum += temp * m;
        component.temperature_mass += m;
        component.has_temperature = true;
        component.solid_mass_kg += m * solid;
        component.liquid_mass_kg += m * liquid;
        component.gas_mass_kg += m * gas;
    }

    let (intact_bonds, elastic, stored_plastic) = if exact {
        let intact = scene
            .bonds
            .iter()
            .flatten()
            .filter(|bond| bond.neighbor != EMPTY)
            .count();
        let elastic = finite_nonnegative(
            scene.elastic_bond_energy_j(),
            &mut non_finite_count,
            &mut out_of_range_count,
        );
        let stored_plastic = finite_nonnegative(
            scene.stored_plastic_bond_energy_j(),
            &mut non_finite_count,
            &mut out_of_range_count,
        );
        (Some(intact), Some(elastic), Some(stored_plastic))
    } else {
        (None, None, None)
    };
    let mut ledger = serde_json::Map::new();
    macro_rules! ledger_channel {
        ($field:ident) => {{
            let value = finite_nonnegative_signed(scene.energy.$field, &mut non_finite_count);
            ledger.insert(stringify!($field).to_owned(), json!(value));
        }};
    }
    ledger_channel!(plastic_heat);
    ledger_channel!(stored_plastic);
    ledger_channel!(surface_work);
    ledger_channel!(damage_heat);
    ledger_channel!(contact_friction_heat);
    ledger_channel!(contact_restitution_heat);
    ledger_channel!(liquid_viscous_heat);
    ledger_channel!(solid_viscoelastic_heat);
    ledger_channel!(implicit_solver_heat);
    ledger_channel!(fusion_heat);
    ledger_channel!(external_work);
    ledger_channel!(pending_release);
    ledger_channel!(kinetic_release);

    let summaries = components
        .values()
        .take(limit)
        .map(|component| component.json(!exact))
        .collect::<Vec<_>>();
    // BTreeMap order makes truncation deterministic across runs.
    let component_count = scene.cluster_count + usize::from(fixed_count > 0);
    let omitted = component_count.saturating_sub(summaries.len());
    let thermal = if temperature_mass > 0.0 {
        json!({
            "temperature_k": {
                "min": temperature_min,
                "max": temperature_max,
                "mean_mass_weighted": temperature_mass_sum / temperature_mass
            },
            "phase_mass_fraction": {
                "solid": solid_mass / mass.max(f64::MIN_POSITIVE),
                "liquid": liquid_mass / mass.max(f64::MIN_POSITIVE),
                "gas": gas_mass / mass.max(f64::MIN_POSITIVE)
            },
            "phase_mass_kg": {"solid": solid_mass, "liquid": liquid_mass, "gas": gas_mass}
        })
    } else {
        Value::Null
    };

    let exact_total = |value: Value| if exact { value } else { Value::Null };
    let components_scope = if exact {
        "complete current component summaries, subject to output cap"
    } else {
        "sampled members only; rows are not whole-component summaries"
    };
    let emitted_summaries = summaries.len();
    let result = json!({
        "schema_version": 1,
        "limitations": {
            "liquid_pressure_transport": "not_implemented",
            "liquid_surface_tension": "not_implemented",
            "energy_residual": "not_reported; work and exchange terms may be incomplete",
            "fixed_components": "fixed REVs are grouped as fixed_supports; the scene cluster cache uses u32::MAX for fixed REVs",
            "runtime_sampling": if exact { "exact bounded-scene collection" } else { "at most 128 deterministic REV samples; global values are unavailable, not extrapolated" }
        },
        "rev_counts": {
            "total": scene.voxels.len(),
            "dynamic": exact_total(json!(dynamic_count)),
            "fine_dynamic": exact_total(json!(fine_dynamic_count)),
            "fixed": exact_total(json!(fixed_count)),
            "coarse": exact_total(json!(coarse_count)),
            "coarse_dynamic": exact_total(json!(coarse_dynamic_count)),
            "active_dynamic_fine": exact_total(json!(fine_dynamic_count)),
            "last_solver_active_fine": scene.last_step_diagnostics.as_ref().map(|d| d.active_fine_revs_last_solve),
            "active": exact_total(json!(scene.voxels.len().saturating_sub(coarse_count)))
        },
        "active_rev_scope": "legacy active count includes fixed non-coarse REVs; active_dynamic_fine is non-kinematic fine REVs and is the performance eligibility count",
        "totals": {
            "mass_kg": exact_total(json!(mass)),
            "momentum_kg_m_s": exact_total(vec3_json(momentum)),
            "kinetic_energy_j": exact_total(json!(kinetic)),
            "internal_energy_j": exact_total(json!(internal)),
            "elastic_energy_j": elastic,
            "stored_plastic_energy_j": stored_plastic,
            "accounted_energy_j": if exact { json!(kinetic + internal + elastic.unwrap_or(0.0) + stored_plastic.unwrap_or(0.0)) } else { Value::Null }
        },
        "energy_ledger_j": Value::Object(ledger),
        "skeletal_actuators": {
            "count":scene.skeletal_actuators.len(),
            "limit":16,
            "items":scene.skeletal_actuators.iter().take(16).enumerate().map(|(index, actor)| json!({
                "index":index,
                "voxel_range":[actor.voxel_range().start,actor.voxel_range().end],
                "clip_index":actor.clip_index,
                "pose_time_s":actor.elapsed_seconds(),
                "enabled":actor.enabled,
                "stalled":actor.stalled,
                "last_load_l1_norm_n":actor.last_force_norm_n(),
                "drive":"material_capped_internal_forces; no kinematic pose writes"
            })).collect::<Vec<_>>()
        },
        "thermal": exact_total(thermal.clone()),
        "topology": {"intact_bonds": intact_bonds, "broken_cumulative": scene.broken, "component_count": exact_total(json!(component_count)), "cached_dynamic_component_count": scene.cluster_count},
        "components": {"limit": limit, "emitted": summaries.len(), "omitted": exact_total(json!(omitted)), "scope": components_scope, "items": summaries},
        "non_finite": {"count": non_finite_count, "warning": non_finite_count > 0, "scope": if exact { "whole scene" } else { "sampled REVs only" }},
        "out_of_range_state": {"count": out_of_range_count, "warning": out_of_range_count > 0, "scope": if exact { "whole scene" } else { "sampled REVs only" }}
    });
    let mut result = result;
    if let Some(indices) = sampled_indices {
        let stride = if indices.len() > 1 {
            indices[1] - indices[0]
        } else {
            0
        };
        result["components"]["scope"] = json!(components_scope);
        result["components"]["omitted"] = Value::Null;
        result["sampled_state"] = json!({
            "scope": "sampled REV values only; sums are not whole-world totals and are not extrapolated",
            "selection": "deterministic evenly spaced indices in stored scene order",
            "sample_count": indices.len(),
            "population_count": scene.voxels.len(),
            "index_first": indices.first(),
            "index_last": indices.last(),
            "index_span_inclusive": [indices.first(), indices.last()],
            "nominal_index_stride": stride,
            "sampled_mass_kg": mass,
            "sampled_momentum_kg_m_s": vec3_json(momentum),
            "sampled_kinetic_energy_j": kinetic,
            "sampled_internal_energy_j": internal,
            "sampled_temperature_k": thermal,
            "sampled_component_count_emitted": emitted_summaries,
            "components_are_sampled_members_only": true
        });
    }
    result
}

fn finite_nonnegative(
    value: f64,
    non_finite_count: &mut usize,
    out_of_range_count: &mut usize,
) -> f64 {
    if !value.is_finite() {
        *non_finite_count += 1;
        0.0
    } else if value < 0.0 {
        *out_of_range_count += 1;
        0.0
    } else {
        value
    }
}

fn finite_nonnegative_signed(value: f64, non_finite_count: &mut usize) -> f64 {
    if value.is_finite() {
        value
    } else {
        *non_finite_count += 1;
        0.0
    }
}

fn finite_fraction(
    value: f64,
    non_finite_count: &mut usize,
    out_of_range_count: &mut usize,
) -> f64 {
    if !value.is_finite() {
        *non_finite_count += 1;
        0.0
    } else if !(0.0..=1.0).contains(&value) {
        *out_of_range_count += 1;
        value.clamp(0.0, 1.0)
    } else {
        value
    }
}

fn vec3_json(value: DVec3) -> Value {
    if value.is_finite() {
        json!([value.x, value.y, value.z])
    } else {
        json!([0.0, 0.0, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        material::Material,
        scene::{SceneBuilder, VoxelSpawn},
    };
    use glam::Vec3;

    fn scene_with_temperatures() -> Scene {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([10, 0, 0], iron))
            .build()
            .unwrap();
        scene.voxels[0].velocity = Vec3::X;
        scene.voxels[1].velocity = Vec3::Y * 2.0;
        scene.voxels[2].velocity = Vec3::Z * 3.0;
        // Give the third REV a known internal energy corresponding to 1000 K.
        scene.voxels[2].internal_energy =
            scene.voxels[2].mass * scene.properties[2].heat_capacity * 1000.0;
        scene
    }

    fn isolated_iron_scene(count: i32) -> Scene {
        let iron = Material::component_id("iron").unwrap();
        let mut builder = SceneBuilder::standard();
        for i in 0..count {
            builder = builder.add_voxel(VoxelSpawn::at_cell([i * 3, 0, 0], iron));
        }
        builder.build().unwrap()
    }

    #[test]
    fn aggregates_mass_momentum_temperature_phase_and_energy() {
        let scene = scene_with_temperatures();
        let metrics = collect_scene_metrics(&scene, 64);
        assert_eq!(metrics["rev_counts"]["total"], 3);
        assert_eq!(
            metrics["rev_counts"]["fine_dynamic"],
            metrics["rev_counts"]["active_dynamic_fine"]
        );
        assert_eq!(
            metrics["totals"]["mass_kg"].as_f64().unwrap(),
            scene.voxels.iter().map(|v| v.mass as f64).sum::<f64>()
        );
        let p = &metrics["totals"]["momentum_kg_m_s"];
        assert!(
            p[0].as_f64().unwrap() > 0.0
                && p[1].as_f64().unwrap() > 0.0
                && p[2].as_f64().unwrap() > 0.0
        );
        assert!(metrics["thermal"]["temperature_k"]["max"].as_f64().unwrap() >= 999.0);
        assert!(
            metrics["thermal"]["phase_mass_fraction"]["solid"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        assert_eq!(
            metrics["totals"]["internal_energy_j"].as_f64().unwrap(),
            scene
                .voxels
                .iter()
                .map(|v| v.total_internal_energy_j())
                .sum::<f64>()
        );
    }

    #[test]
    fn component_summaries_are_bounded_and_report_omissions() {
        let iron = Material::component_id("iron").unwrap();
        let mut builder = SceneBuilder::standard();
        for i in 0..70 {
            builder = builder.add_voxel(VoxelSpawn::at_cell([i * 3, 0, 0], iron));
        }
        let metrics = collect_scene_metrics(&builder.build().unwrap(), 200);
        assert_eq!(metrics["components"]["limit"], 64);
        assert_eq!(metrics["components"]["emitted"], 64);
        assert_eq!(metrics["components"]["omitted"], 6);
        assert_eq!(metrics["topology"]["component_count"], 70);
    }

    #[test]
    fn fixed_revs_keep_the_fixed_cluster_sentinel_without_model_identity() {
        let iron = Material::component_id("iron").unwrap();
        let mut scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .build()
            .unwrap();
        scene.set_voxel_kinematic(0, true).unwrap();
        let metrics = collect_scene_metrics(&scene, 2);
        assert_eq!(metrics["components"]["items"][0]["cluster_id"], u32::MAX);
        assert_eq!(metrics["components"]["items"][0]["fixed"], true);
        assert_eq!(metrics["rev_counts"]["fixed"], 1);
        assert_eq!(metrics["rev_counts"]["active_dynamic_fine"], 0);
        assert!(metrics["components"]["items"][0]
            .get("authored_model_id")
            .is_none());
    }

    #[test]
    fn performance_window_normalizes_observed_batches_and_separates_eligibility() {
        let mut window = PerformanceWindow::default();
        assert!(window.push_batch(50_000, 0.01, 2.0, 8.0));
        let report = window.to_json();
        assert_eq!(report["sample_count"], 1);
        assert_eq!(report["eligibility"]["in_active_rev_range"], true);
        assert_eq!(report["budget_pass"], false);
        assert_eq!(report["full_worker_ms_per_10ms_simulated"]["max"], 8.0);
        assert_eq!(report["solver_ms_per_10ms_simulated"]["p50"], 2.0);
        assert_eq!(report["target"]["active_dynamic_fine_revs_min"], 50_000);
        assert!(report["target"]["active_dynamic_fine_revs_max"].is_null());
        assert!(report["goal_status"]
            .as_str()
            .unwrap()
            .contains("requires correctness"));

        let mut out_of_range = PerformanceWindow::default();
        assert!(out_of_range.push_batch(49_999, 0.01, 1.0, 4.0));
        let report = out_of_range.to_json();
        assert_eq!(report["eligibility"]["in_active_rev_range"], false);
        assert_eq!(report["budget_pass"], true);

        let mut above_minimum = PerformanceWindow::default();
        assert!(above_minimum.push_batch(54_000, 0.01, 1.0, 4.0));
        assert_eq!(
            above_minimum.to_json()["eligibility"]["in_active_rev_range"],
            true
        );
    }

    #[test]
    fn performance_window_rejects_invalid_samples_and_keeps_only_latest_256() {
        let mut window = PerformanceWindow::default();
        assert!(!window.push_batch(350_000, 0.0, 0.0, 0.0));
        assert!(!window.push_batch(350_000, 0.01, 5.0, 4.0));
        assert!(!window.push_batch(350_000, 1e-308, 0.0, 1e308));
        for _ in 0..300 {
            assert!(window.push_batch(350_000, 0.01, 1.0, 2.0));
        }
        let report = window.to_json();
        assert_eq!(report["sample_count"], 256);
        assert_eq!(report["capacity"], 256);
        assert!(
            (report["measured_span"]["simulated_seconds"]
                .as_f64()
                .unwrap()
                - 2.56)
                .abs()
                < 1e-9
        );
        assert_eq!(
            report["latest_cost"]["full_worker_wall_ms_including_metrics_publish"],
            2.0
        );
        assert_eq!(report["budget_pass"], true);
    }

    #[test]
    fn performance_budget_uses_window_max_not_latest_sample_order() {
        let mut slow_then_fast = PerformanceWindow::default();
        assert!(slow_then_fast.push_batch(350_000, 0.01, 1.0, 8.0));
        assert!(slow_then_fast.push_batch(350_000, 0.01, 1.0, 2.0));
        let report = slow_then_fast.to_json();
        assert_eq!(report["full_worker_ms_per_10ms_simulated"]["max"], 8.0);
        assert_eq!(
            report["latest_cost"]["full_worker_ms_per_10ms_simulated"],
            2.0
        );
        assert_eq!(report["budget_pass"], false);

        let mut fast_then_slow = PerformanceWindow::default();
        assert!(fast_then_slow.push_batch(350_000, 0.01, 1.0, 2.0));
        assert!(fast_then_slow.push_batch(350_000, 0.01, 1.0, 8.0));
        let report = fast_then_slow.to_json();
        assert_eq!(report["full_worker_ms_per_10ms_simulated"]["max"], 8.0);
        assert_eq!(
            report["latest_cost"]["full_worker_ms_per_10ms_simulated"],
            8.0
        );
        assert_eq!(report["budget_pass"], false);
    }

    #[test]
    fn runtime_collector_uses_exact_path_below_threshold() {
        let scene = isolated_iron_scene(3);
        assert_eq!(
            collect_runtime_metrics(&scene, 16),
            collect_scene_metrics(&scene, 16)
        );
    }

    #[test]
    fn runtime_collector_bounds_large_scenes_and_does_not_extrapolate() {
        let mut scene = isolated_iron_scene((MAX_EXACT_RUNTIME_REVS + 1) as i32);
        scene.energy.external_work = 12.5;
        let metrics = collect_runtime_metrics(&scene, 32);
        assert_eq!(metrics["rev_counts"]["total"], MAX_EXACT_RUNTIME_REVS + 1);
        assert_eq!(metrics["rev_counts"]["dynamic"], Value::Null);
        assert_eq!(metrics["rev_counts"]["active_dynamic_fine"], Value::Null);
        assert_eq!(
            metrics["rev_counts"]["last_solver_active_fine"],
            Value::Null
        );
        assert_eq!(metrics["totals"]["mass_kg"], Value::Null);
        assert_eq!(metrics["totals"]["accounted_energy_j"], Value::Null);
        assert_eq!(metrics["thermal"], Value::Null);
        assert_eq!(metrics["topology"]["intact_bonds"], Value::Null);
        assert_eq!(metrics["topology"]["component_count"], Value::Null);
        assert_eq!(
            metrics["topology"]["cached_dynamic_component_count"],
            scene.cluster_count
        );
        assert_eq!(metrics["topology"]["broken_cumulative"], scene.broken);
        assert_eq!(metrics["energy_ledger_j"]["external_work"], 12.5);
        assert_eq!(
            metrics["sampled_state"]["sample_count"],
            MAX_RUNTIME_SAMPLE_REVS
        );
        assert_eq!(
            metrics["sampled_state"]["population_count"],
            MAX_EXACT_RUNTIME_REVS + 1
        );
        assert!(
            metrics["sampled_state"]["sampled_mass_kg"]
                .as_f64()
                .unwrap()
                > 0.0
        );
        assert_eq!(metrics["components"]["items"].as_array().unwrap().len(), 32);
        assert_eq!(
            metrics["components"]["items"][0]["population_count"],
            Value::Null
        );
        assert_eq!(metrics["components"]["items"][0]["sampled_member_count"], 1);
        assert_eq!(
            metrics["components"]["items"][0]["scope"],
            "sampled members only; not the whole component"
        );
    }

    #[test]
    fn exact_collector_reports_invalid_state_even_beyond_component_cap() {
        let mut scene = isolated_iron_scene(2);
        scene.voxels[0].mass = -1.0;
        scene.positions[1] = DVec3::NAN;
        let metrics = collect_scene_metrics(&scene, 1);
        assert_eq!(metrics["components"]["emitted"], 1);
        assert!(metrics["out_of_range_state"]["count"].as_u64().unwrap() >= 1);
        assert!(metrics["non_finite"]["count"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn exact_collector_reports_nan_in_stored_heat_even_if_total_is_clamped() {
        let mut scene = isolated_iron_scene(1);
        scene.voxels[0].internal_energy = f32::NAN;
        assert!(scene.voxels[0].total_internal_energy_j().is_finite());
        let metrics = collect_scene_metrics(&scene, 1);
        assert_eq!(metrics["non_finite"]["scope"], "whole scene");
        assert_eq!(metrics["non_finite"]["warning"], true);
        assert!(metrics["non_finite"]["count"].as_u64().unwrap() >= 1);
    }

    #[test]
    #[ignore = "manual runtime collector overhead benchmark"]
    fn runtime_collector_perf_probe_343k() {
        let started = std::time::Instant::now();
        let iron = Material::component_id("iron").unwrap();
        let config = crate::scene::SceneConfig {
            dimensions: [70; 3],
            gravity: [0.0; 3],
            ..crate::scene::SceneConfig::default()
        };
        let mut builder = SceneBuilder::new(config);
        for z in 0..70 {
            for y in 0..70 {
                for x in 0..70 {
                    builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], iron));
                }
            }
        }
        let scene = builder.build().unwrap();
        let build_ms = started.elapsed().as_secs_f64() * 1000.0;
        for _ in 0..2 {
            let _ = collect_runtime_metrics(&scene, 32);
        }
        let mut samples = Vec::new();
        let mut bytes = 0;
        for _ in 0..10 {
            let start = std::time::Instant::now();
            let value = collect_runtime_metrics(&scene, 32);
            samples.push(start.elapsed().as_secs_f64() * 1000.0);
            bytes = serde_json::to_vec(&value).unwrap().len();
            assert_eq!(
                value["sampled_state"]["sample_count"],
                MAX_RUNTIME_SAMPLE_REVS
            );
        }
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "runtime_metrics_343k mode=debug_opt2 build_ms={build_ms:.3} median_ms={:.3} max_ms={:.3} json_bytes={bytes}",
            (samples[4] + samples[5]) * 0.5,
            samples[9]
        );
    }
}
