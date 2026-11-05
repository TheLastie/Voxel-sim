//! First implementation of the creature architecture of `WORLD_PARADIGM.md`
//! section 2.4: resource accounting, sensors, a small spiking controller,
//! learning kept apart from heredity, and one controller boundary.
//!
//! What this module is not: it does not make a creature walk. The network that
//! is trained here learns a trivial signal-following task in tests; learned
//! walking is not implemented. Digestion, transport and the brain live in this
//! module's plain data, not yet in dedicated organ REVs (see the TODO list in
//! the commit report).
//!
//! Units are SI throughout: joules, watts, seconds, newtons, metres, kelvin.
//! Controllers only produce a dimensionless strength in `0..=1` per actuator.
//! They never write positions, velocities or forces; the existing
//! [`SkeletalActuator`] turns that scale into loads for the ordinary solver.

use crate::{scene::Scene, skeletal_actuator::SkeletalActuator};
use glam::{DMat3, DQuat, DVec3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::ops::Range;

const CELL_M: f64 = crate::coordinates::CELL_M;
/// Largest network accepted by this module (specification limit).
pub const MAX_NEURONS: usize = 256;
/// Gap allowed between a foot REV and another REV for them to count as touching.
const CONTACT_TOLERANCE_M: f64 = 0.1 * CELL_M;

// ---------------------------------------------------------------------------
// Deterministic random numbers (explicit state, no globals)
// ---------------------------------------------------------------------------

/// SplitMix64. All randomness in this module goes through an explicit value
/// of this type so that a seed fully determines a result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `[0, 1)`.
    pub fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f32 {
        self.next_unit() * 2.0 - 1.0
    }
    /// Approximately standard normal (sum of six uniforms), bounded to +-4.25.
    pub fn next_gauss(&mut self) -> f32 {
        let sum: f32 = (0..6).map(|_| self.next_unit()).sum();
        (sum - 3.0) * std::f32::consts::SQRT_2
    }
    /// Uniform integer in `0..bound` (`bound > 0`).
    pub fn next_below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound.max(1) as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// 1. Resource ledger
// ---------------------------------------------------------------------------

/// Heritable organ parameters that govern energy handling.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrganParams {
    /// Largest chemical energy the body can store (J).
    pub store_capacity_j: f64,
    /// Fraction of the store present at birth, `0..=1`.
    pub birth_store_fraction: f64,
    /// Largest food chemical energy the stomach holds (J).
    pub stomach_capacity_j: f64,
    /// Fastest rate at which digestion moves stomach contents out (W).
    pub digestion_power_w: f64,
    /// Fraction of digested chemical energy that reaches the store, `(0, 1]`.
    /// The rest leaves as waste and is booked as `excreted_j`.
    pub absorption_efficiency: f64,
    /// Mechanical work delivered per chemical joule spent, `(0, 1]`.
    pub muscle_efficiency: f64,
    /// Chemical power spent whether or not the muscles act (W); becomes heat.
    pub basal_power_w: f64,
    /// Design mechanical power of the muscles (W); bounds what one step may
    /// be allowed to cost when the store runs low.
    pub peak_mechanical_power_w: f64,
    /// Below this store fraction muscles weaken linearly towards zero.
    pub weak_fraction: f64,
    /// Store fraction muscles cannot use (a vital reserve); at or below it the
    /// strength limit is zero. Must be below `weak_fraction`.
    pub exhaustion_fraction: f64,
}

impl OrganParams {
    /// Allometric defaults for a body of the given mass (kg): about 10 % of the
    /// mass as fat at 37 MJ/kg, a stomach for 5 % of the mass in 4 MJ/kg food
    /// digested in an hour, Kleiber basal power `3.4 m^0.75`, 25 % muscle
    /// efficiency.
    pub fn from_body_mass_kg(mass_kg: f64) -> Self {
        Self {
            store_capacity_j: 3.7e6 * mass_kg,
            birth_store_fraction: 0.5,
            stomach_capacity_j: 2.0e5 * mass_kg,
            digestion_power_w: 2.0e5 * mass_kg / 3600.0,
            absorption_efficiency: 0.8,
            muscle_efficiency: 0.25,
            basal_power_w: 3.4 * mass_kg.powf(0.75),
            peak_mechanical_power_w: 40.0 * mass_kg,
            weak_fraction: 0.1,
            exhaustion_fraction: 0.01,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if !positive(self.store_capacity_j)
            || !positive(self.stomach_capacity_j)
            || !positive(self.peak_mechanical_power_w)
            || !self.digestion_power_w.is_finite()
            || self.digestion_power_w < 0.0
            || !self.basal_power_w.is_finite()
            || self.basal_power_w < 0.0
            || !unit(self.birth_store_fraction)
            || !(positive(self.absorption_efficiency) && self.absorption_efficiency <= 1.0)
            || !(positive(self.muscle_efficiency) && self.muscle_efficiency <= 1.0)
            || !(positive(self.weak_fraction) && self.weak_fraction <= 1.0)
            || !(self.exhaustion_fraction.is_finite()
                && (0.0..self.weak_fraction).contains(&self.exhaustion_fraction))
        {
            return Err("organ parameters are outside their physical ranges".into());
        }
        Ok(())
    }
}

/// Energy of one physical step as seen by the muscles.
#[derive(Clone, Debug, PartialEq)]
pub struct MuscleStepOutcome {
    /// Sum over REVs of `F . v_mean dt` (J); negative when muscles are braked.
    pub net_mechanical_work_j: f64,
    /// Sum of the positive per-REV work (J); this is what muscles are paid for.
    pub positive_work_j: f64,
    /// Magnitude of the negative per-REV work (J); absorbed as heat.
    pub negative_work_j: f64,
    /// Chemical energy removed from the store this step, basal included (J).
    pub debited_j: f64,
    /// Heat to add to each muscle REV, in the order the REVs were given (J).
    pub heat_per_rev_j: Vec<f64>,
    /// Heat with no REV to receive it (only when no muscle REV was given) (J).
    pub unlocalised_heat_j: f64,
    /// Positive work done that the store could not pay for (J). The physical
    /// step has already happened; this is the visible shortfall and the
    /// strength limit is the guard that keeps it near zero.
    pub unfunded_work_j: f64,
    /// Strength scale in `0..=1` the muscles may use during the next step.
    pub strength_limit: f32,
}

/// Stored chemical energy, the food intake path and cumulative accounts.
///
/// There is no hunger or satiety flag: both are computed from `stored_j` and
/// `stomach_j` on demand.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceLedger {
    params: OrganParams,
    stored_j: f64,
    stomach_j: f64,
    initial_j: f64,
    intake_j: f64,
    excreted_j: f64,
    debited_j: f64,
    heat_j: f64,
    mechanical_net_j: f64,
    unfunded_work_j: f64,
}

impl ResourceLedger {
    pub fn new(params: OrganParams) -> Result<Self, String> {
        params.validate()?;
        let stored = params.store_capacity_j * params.birth_store_fraction;
        Ok(Self {
            params,
            stored_j: stored,
            stomach_j: 0.0,
            initial_j: stored,
            intake_j: 0.0,
            excreted_j: 0.0,
            debited_j: 0.0,
            heat_j: 0.0,
            mechanical_net_j: 0.0,
            unfunded_work_j: 0.0,
        })
    }

    pub fn params(&self) -> &OrganParams {
        &self.params
    }
    pub fn stored_j(&self) -> f64 {
        self.stored_j
    }
    pub fn stomach_j(&self) -> f64 {
        self.stomach_j
    }
    pub fn stored_fraction(&self) -> f64 {
        (self.stored_j / self.params.store_capacity_j).clamp(0.0, 1.0)
    }
    pub fn debited_total_j(&self) -> f64 {
        self.debited_j
    }
    pub fn heat_total_j(&self) -> f64 {
        self.heat_j
    }
    pub fn mechanical_net_total_j(&self) -> f64 {
        self.mechanical_net_j
    }
    pub fn unfunded_work_total_j(&self) -> f64 {
        self.unfunded_work_j
    }

    /// Hunger in `0..=1`, derived from the energy in the store plus the part of
    /// the stomach contents that digestion will still deliver.
    pub fn hunger(&self) -> f64 {
        let expected = self.stored_j + self.params.absorption_efficiency * self.stomach_j;
        1.0 - (expected / self.params.store_capacity_j).clamp(0.0, 1.0)
    }

    /// Satiety in `0..=1`: how full the stomach is.
    pub fn satiety(&self) -> f64 {
        (self.stomach_j / self.params.stomach_capacity_j).clamp(0.0, 1.0)
    }

    /// Put food of the given chemical energy (J) into the stomach. Returns the
    /// energy accepted; the rest stays outside the body.
    pub fn ingest_food(&mut self, chemical_j: f64) -> Result<f64, String> {
        if !chemical_j.is_finite() || chemical_j < 0.0 {
            return Err("food energy must be finite and nonnegative".into());
        }
        let accepted = chemical_j.min(self.params.stomach_capacity_j - self.stomach_j);
        self.stomach_j += accepted;
        self.intake_j += accepted;
        Ok(accepted)
    }

    /// Digest for `dt_s`: moves stomach contents out at most at
    /// `digestion_power_w`; the absorbed fraction enters the store (capped by
    /// its capacity, the surplus is excreted) and the rest is excreted.
    pub fn advance_digestion(&mut self, dt_s: f64) {
        if !(dt_s.is_finite() && dt_s > 0.0) || self.stomach_j <= 0.0 {
            return;
        }
        let digested = self
            .stomach_j
            .min(self.params.digestion_power_w * dt_s);
        let absorbed = digested * self.params.absorption_efficiency;
        let room = (self.params.store_capacity_j - self.stored_j).max(0.0);
        let kept = absorbed.min(room);
        self.stomach_j -= digested;
        self.stored_j += kept;
        self.excreted_j += digested - kept;
    }

    /// Largest strength scale (`0..=1`) affordable for a step of `dt_s`: it
    /// weakens below `weak_fraction` of the store, is zero at or below
    /// `exhaustion_fraction`, and cannot exceed what the usable store pays for
    /// at design power.
    pub fn strength_limit(&self, dt_s: f64) -> f32 {
        if !(dt_s.is_finite() && dt_s > 0.0) || self.stored_j <= 0.0 {
            return 0.0;
        }
        let exhausted = self.params.exhaustion_fraction;
        let ramp = (self.stored_fraction() - exhausted) / (self.params.weak_fraction - exhausted);
        let step_cost = (self.params.peak_mechanical_power_w / self.params.muscle_efficiency
            + self.params.basal_power_w)
            * dt_s;
        let affordable = (self.stored_j - exhausted * self.params.store_capacity_j) / step_cost;
        ramp.min(affordable).clamp(0.0, 1.0) as f32
    }

    /// Settle one physical step. The three slices describe the same muscle
    /// REVs: the loads the actuator applied (N) and the REV velocities at the
    /// start and the end of the step (m/s). Work is `F . (v0 + v1)/2 * dt`.
    ///
    /// Positive work costs `work / muscle_efficiency` chemical joules. The loss
    /// part, the basal cost and the energy of braked (negative) work become heat
    /// in the same REVs, so `debited = net work + heat` while the store paid in
    /// full. Nothing here can create energy: the store only decreases except by
    /// digestion of ingested food.
    pub fn settle_muscle_step(
        &mut self,
        dt_s: f64,
        forces_n: &[Vec3],
        velocity_start_m_s: &[Vec3],
        velocity_end_m_s: &[Vec3],
    ) -> Result<MuscleStepOutcome, String> {
        let n = forces_n.len();
        if velocity_start_m_s.len() != n || velocity_end_m_s.len() != n {
            return Err("muscle step slices must have equal length".into());
        }
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return Err("dt must be finite and positive".into());
        }
        if forces_n
            .iter()
            .chain(velocity_start_m_s)
            .chain(velocity_end_m_s)
            .any(|v| !v.is_finite())
        {
            return Err("muscle step inputs must be finite".into());
        }
        self.advance_digestion(dt_s);
        let eta = self.params.muscle_efficiency;
        let work: Vec<f64> = (0..n)
            .map(|i| {
                let mean_v = (velocity_start_m_s[i].as_dvec3() + velocity_end_m_s[i].as_dvec3())
                    * 0.5;
                forces_n[i].as_dvec3().dot(mean_v) * dt_s
            })
            .collect();
        let positive: f64 = work.iter().map(|w| w.max(0.0)).sum();
        let negative: f64 = work.iter().map(|w| (-w).max(0.0)).sum();
        let net: f64 = work.iter().sum();

        let basal_paid = (self.params.basal_power_w * dt_s).min(self.stored_j);
        self.stored_j -= basal_paid;
        let wanted = positive / eta;
        let work_paid = wanted.min(self.stored_j);
        self.stored_j -= work_paid;
        let funded = if wanted > 0.0 { work_paid / wanted } else { 1.0 };
        let unfunded = positive * (1.0 - funded);
        let debited = basal_paid + work_paid;

        let (basal_each, unlocalised) = if n == 0 {
            (0.0, basal_paid)
        } else {
            (basal_paid / n as f64, 0.0)
        };
        let heat: Vec<f64> = work
            .iter()
            .map(|&w| (funded * w.max(0.0) / eta - w).max(0.0) + basal_each)
            .collect();

        self.debited_j += debited;
        self.heat_j += heat.iter().sum::<f64>() + unlocalised;
        self.mechanical_net_j += net;
        self.unfunded_work_j += unfunded;
        Ok(MuscleStepOutcome {
            net_mechanical_work_j: net,
            positive_work_j: positive,
            negative_work_j: negative,
            debited_j: debited,
            heat_per_rev_j: heat,
            unlocalised_heat_j: unlocalised,
            unfunded_work_j: unfunded,
            strength_limit: self.strength_limit(dt_s),
        })
    }

    /// `initial + intake - (stored + stomach + excreted + debited)`; zero up to
    /// rounding for a ledger that was only changed through its own methods.
    pub fn conservation_residual_j(&self) -> f64 {
        self.initial_j + self.intake_j
            - (self.stored_j + self.stomach_j + self.excreted_j + self.debited_j)
    }
}

// ---------------------------------------------------------------------------
// 2. Sensors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BoneGroup {
    bone_index: usize,
    /// Index into `SensorLayout::groups` of the nearest observable ancestor.
    reference: Option<usize>,
    /// Creature-local REV indices whose dominant skinning weight is this bone.
    voxels: Vec<usize>,
}

/// Where a creature's sensors read, fixed when the creature is built.
#[derive(Clone, Debug)]
pub struct SensorLayout {
    range: Range<usize>,
    total_scene_voxels: usize,
    foot_local: Vec<usize>,
    /// Model rest positions (m), model-local, one per creature REV.
    rest_m: Vec<DVec3>,
    groups: Vec<BoneGroup>,
    root_group: Option<usize>,
}

/// Rotation of one sensed bone relative to its nearest sensed ancestor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointReading {
    pub bone_index: usize,
    pub reference_bone_index: usize,
    /// Rotation vector (axis * angle, rad) in the reference bone's rest frame;
    /// `None` when the fitted REV set was degenerate at this instant.
    pub rotation_vector_rad: Option<Vec3>,
}

impl JointReading {
    pub fn angle_rad(&self) -> Option<f32> {
        self.rotation_vector_rad.map(|v| v.length())
    }
}

/// What a creature can sense of the real engine state in one instant.
///
/// Not available without changing the solver, and therefore absent: contact
/// normal force, friction state, light, and any exteroception beyond touch.
#[derive(Clone, Debug, PartialEq)]
pub struct SensorReading {
    /// For each foot REV: is another REV outside the creature touching it?
    pub foot_contact: Vec<bool>,
    /// Body orientation relative to the rest placement (rigid fit of the root
    /// bone's REVs, or of all REVs when the model has no usable root group).
    pub body_orientation: Option<Quat>,
    /// World "up" expressed in the body frame (from the scene's gravity).
    pub up_in_body: Option<Vec3>,
    /// Angular velocity of the whole body (rad/s, world axes).
    pub angular_velocity_rad_s: Vec3,
    /// Velocity of the centre of mass (m/s, world axes).
    pub com_velocity_m_s: Vec3,
    pub joints: Vec<JointReading>,
    /// Stored chemical energy over capacity, `0..=1`.
    pub stored_energy_fraction: f32,
    /// Derived from the ledger, `0..=1`.
    pub hunger: f32,
}

impl SensorLayout {
    /// Build the layout for `actuator`'s REV range. `foot_local` lists
    /// creature-local REV indices that act as feet.
    pub fn new(actuator: &SkeletalActuator, foot_local: Vec<usize>) -> Result<Self, String> {
        let model = actuator.model();
        let range = actuator.voxel_range();
        if foot_local.iter().any(|&f| f >= range.len()) {
            return Err("foot REV index outside the creature range".into());
        }
        let rest_m: Vec<DVec3> = model.voxels.iter().map(|v| v.rest_position_m).collect();
        let skeleton = model.skeleton.as_ref().ok_or("model has no skeleton")?;
        let mut dominant: Vec<Option<(usize, f32)>> = vec![None; model.voxels.len()];
        for binding in &skeleton.voxel_weights {
            let slot = &mut dominant[binding.voxel_index as usize];
            let candidate = (binding.bone_index as usize, binding.weight);
            *slot = match *slot {
                Some((bone, weight))
                    if weight > candidate.1 || (weight == candidate.1 && bone < candidate.0) =>
                {
                    Some((bone, weight))
                }
                _ => Some(candidate),
            };
        }
        let mut per_bone: Vec<Vec<usize>> = vec![Vec::new(); skeleton.bones.len()];
        for (local, d) in dominant.iter().enumerate() {
            if let Some((bone, _)) = d {
                per_bone[*bone].push(local);
            }
        }
        let observable: Vec<bool> = per_bone
            .iter()
            .map(|voxels| {
                let points: Vec<DVec3> = voxels.iter().map(|&i| rest_m[i]).collect();
                has_rank_two(&points)
            })
            .collect();
        let mut group_of_bone = vec![None; skeleton.bones.len()];
        let mut groups = Vec::new();
        for (bone, voxels) in per_bone.iter().enumerate() {
            if observable[bone] {
                group_of_bone[bone] = Some(groups.len());
                groups.push(BoneGroup {
                    bone_index: bone,
                    reference: None,
                    voxels: voxels.clone(),
                });
            }
        }
        for group in &mut groups {
            let mut parent = skeleton.bones[group.bone_index].parent.map(|p| p as usize);
            while let Some(p) = parent {
                if let Some(g) = group_of_bone[p] {
                    group.reference = Some(g);
                    break;
                }
                parent = skeleton.bones[p].parent.map(|p| p as usize);
            }
        }
        let root_group = group_of_bone.first().copied().flatten();
        Ok(Self {
            total_scene_voxels: actuator.total_scene_voxels(),
            range,
            foot_local,
            rest_m,
            groups,
            root_group,
        })
    }

    /// Creature-local REVs lowest along the scene's gravity, within `band_m` of
    /// the lowest one: a geometric foot choice from the real initial pose.
    pub fn lowest_revs(scene: &Scene, range: Range<usize>, band_m: f64) -> Result<Vec<usize>, String> {
        let g = Vec3::from_array(scene.config.gravity).as_dvec3();
        if g.length() < 1e-9 {
            return Err("feet cannot be chosen without gravity".into());
        }
        let down = g.normalize();
        let depth: Vec<f64> = range.clone().map(|i| scene.positions[i].dot(down)).collect();
        let deepest = depth.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(depth
            .iter()
            .enumerate()
            .filter(|(_, d)| deepest - **d <= band_m)
            .map(|(i, _)| i)
            .collect())
    }

    pub fn range(&self) -> Range<usize> {
        self.range.clone()
    }
    pub fn foot_count(&self) -> usize {
        self.foot_local.len()
    }
    pub fn joint_count(&self) -> usize {
        self.groups.iter().filter(|g| g.reference.is_some()).count()
    }
    /// Length of the vector produced by [`SensorReading::encode`].
    pub fn input_count(&self) -> usize {
        self.foot_count() + 3 + 3 + self.joint_count() + 2
    }
}

impl SensorReading {
    /// Read the real engine state of one creature. Does not mutate the scene.
    pub fn read(
        scene: &Scene,
        layout: &SensorLayout,
        ledger: &ResourceLedger,
    ) -> Result<Self, String> {
        let range = layout.range.clone();
        if scene.positions.len() != layout.total_scene_voxels
            || scene.voxels.len() != layout.total_scene_voxels
        {
            return Err("scene topology changed; sensor layout is stale".into());
        }
        let pos = &scene.positions;
        let usable = |i: usize| {
            let v = &scene.voxels[i];
            !v.kinematic && v.mass.is_finite() && v.mass > 0.0 && pos[i].is_finite()
        };

        let foot_contact = layout
            .foot_local
            .iter()
            .map(|&f| {
                let p = pos[range.start + f];
                (0..pos.len()).any(|j| {
                    if range.contains(&j) {
                        return false;
                    }
                    let d = p - pos[j];
                    d.is_finite() && d.abs().max_element() <= CELL_M + CONTACT_TOLERANCE_M
                })
            })
            .collect();

        let dynamic: Vec<usize> = range.clone().filter(|&i| usable(i)).collect();
        let total_mass: f64 = dynamic.iter().map(|&i| f64::from(scene.voxels[i].mass)).sum();
        let (com_velocity, omega) = if total_mass > 0.0 {
            let com = dynamic.iter().fold(DVec3::ZERO, |s, &i| {
                s + pos[i] * f64::from(scene.voxels[i].mass)
            }) / total_mass;
            let vcom = dynamic.iter().fold(DVec3::ZERO, |s, &i| {
                s + scene.voxels[i].velocity.as_dvec3() * f64::from(scene.voxels[i].mass)
            }) / total_mass;
            let (_, omega) =
                crate::skeletal_actuator::rigid_kinematics(&dynamic, pos, &scene.voxels, com, vcom);
            (vcom, omega)
        } else {
            (DVec3::ZERO, DVec3::ZERO)
        };

        let fit = |locals: &[usize]| -> Option<DQuat> {
            let mut rest = Vec::new();
            let mut cur = Vec::new();
            let mut w = Vec::new();
            for &l in locals {
                let i = range.start + l;
                if usable(i) {
                    rest.push(layout.rest_m[l]);
                    cur.push(pos[i]);
                    w.push(f64::from(scene.voxels[i].mass));
                }
            }
            fit_rotation(&rest, &cur, &w)
        };

        let all_local: Vec<usize> = (0..range.len()).collect();
        let body = layout
            .root_group
            .and_then(|g| fit(&layout.groups[g].voxels))
            .or_else(|| fit(&all_local));
        let up_in_body = body.and_then(|q| {
            let g = Vec3::from_array(scene.config.gravity).as_dvec3();
            (g.length() > 1e-9).then(|| (q.inverse() * (-g.normalize())).as_vec3())
        });

        let fits: Vec<Option<DQuat>> = layout.groups.iter().map(|g| fit(&g.voxels)).collect();
        let joints = layout
            .groups
            .iter()
            .enumerate()
            .filter_map(|(gi, group)| {
                let reference = group.reference?;
                let rotation_vector_rad = match (fits[reference], fits[gi]) {
                    (Some(parent), Some(child)) => {
                        Some(rotation_vector(parent.inverse() * child).as_vec3())
                    }
                    _ => None,
                };
                Some(JointReading {
                    bone_index: group.bone_index,
                    reference_bone_index: layout.groups[reference].bone_index,
                    rotation_vector_rad,
                })
            })
            .collect();

        let reading = Self {
            foot_contact,
            body_orientation: body.map(|q| q.as_quat()),
            up_in_body,
            angular_velocity_rad_s: omega.as_vec3(),
            com_velocity_m_s: com_velocity.as_vec3(),
            joints,
            stored_energy_fraction: ledger.stored_fraction() as f32,
            hunger: ledger.hunger() as f32,
        };
        Ok(reading)
    }

    /// True when every number in the reading is finite.
    pub fn is_finite(&self) -> bool {
        self.body_orientation.map_or(true, |q| q.is_finite())
            && self.up_in_body.map_or(true, |v| v.is_finite())
            && self.angular_velocity_rad_s.is_finite()
            && self.com_velocity_m_s.is_finite()
            && self
                .joints
                .iter()
                .all(|j| j.rotation_vector_rad.map_or(true, |v| v.is_finite()))
            && self.stored_energy_fraction.is_finite()
            && self.hunger.is_finite()
    }

    /// Fixed-length, bounded encoding for the network: foot contacts (0/1),
    /// up-in-body (3), tanh(omega / 2 rad/s) (3), joint angles / pi (one per
    /// joint), stored energy fraction, hunger. Unavailable values encode as 0.
    pub fn encode(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.foot_contact.iter().map(|&c| if c { 1.0 } else { 0.0 }));
        let up = self.up_in_body.unwrap_or(Vec3::ZERO);
        out.extend_from_slice(&[up.x, up.y, up.z]);
        let w = self.angular_velocity_rad_s;
        out.extend_from_slice(&[(w.x * 0.5).tanh(), (w.y * 0.5).tanh(), (w.z * 0.5).tanh()]);
        out.extend(self.joints.iter().map(|j| {
            (j.angle_rad().unwrap_or(0.0) / std::f32::consts::PI).clamp(0.0, 1.0)
        }));
        out.push(self.stored_energy_fraction.clamp(0.0, 1.0));
        out.push(self.hunger.clamp(0.0, 1.0));
    }
}

fn rotation_vector(q: DQuat) -> DVec3 {
    let q = if q.w < 0.0 { -q } else { q }.normalize();
    let s = DVec3::new(q.x, q.y, q.z);
    let len = s.length();
    if len < 1e-12 {
        DVec3::ZERO
    } else {
        s / len * (2.0 * len.atan2(q.w))
    }
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix. Returns
/// eigenvalues and eigenvectors (as columns of `v[row][col]`). Deterministic.
fn jacobi_eigen<const N: usize>(mut a: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..60 {
        let mut off = 0.0;
        for p in 0..N {
            for q in p + 1..N {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut values = [0.0; N];
    for (i, value) in values.iter_mut().enumerate() {
        *value = a[i][i];
    }
    (values, v)
}

/// True when the points span at least a plane (second principal variance
/// above 1e-8 m^2), so a rotation can be fitted.
fn has_rank_two(points: &[DVec3]) -> bool {
    if points.len() < 3 {
        return false;
    }
    let c = points.iter().fold(DVec3::ZERO, |s, p| s + *p) / points.len() as f64;
    let mut cov = [[0.0; 3]; 3];
    for p in points {
        let d = (*p - c).to_array();
        for (a, row) in cov.iter_mut().enumerate() {
            for (b, cell) in row.iter_mut().enumerate() {
                *cell += d[a] * d[b] / points.len() as f64;
            }
        }
    }
    let (mut values, _) = jacobi_eigen(cov);
    values.sort_by(f64::total_cmp);
    values[1] > 1e-8
}

/// Mass-weighted best rotation (Horn's quaternion method) taking the `rest`
/// points to the `current` points about their centroids; `None` when the
/// inputs are degenerate (fewer than three points or collinear).
pub fn fit_rotation(rest: &[DVec3], current: &[DVec3], weights: &[f64]) -> Option<DQuat> {
    if rest.len() != current.len() || rest.len() != weights.len() || !has_rank_two(rest) {
        return None;
    }
    let total: f64 = weights.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let cr = rest.iter().zip(weights).fold(DVec3::ZERO, |s, (p, w)| s + *p * *w) / total;
    let cc = current.iter().zip(weights).fold(DVec3::ZERO, |s, (p, w)| s + *p * *w) / total;
    let mut s = DMat3::ZERO.to_cols_array_2d();
    for ((p, q), w) in rest.iter().zip(current).zip(weights) {
        let (p, q) = ((*p - cr).to_array(), (*q - cc).to_array());
        for a in 0..3 {
            for b in 0..3 {
                s[a][b] += w * p[a] * q[b] / total;
            }
        }
    }
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let (values, vectors) = jacobi_eigen(n);
    let best = (0..4).fold(0, |b, i| if values[i] > values[b] { i } else { b });
    let q = DQuat::from_xyzw(
        vectors[1][best],
        vectors[2][best],
        vectors[3][best],
        vectors[0][best],
    );
    (q.is_finite() && q.length_squared() > 0.5).then(|| q.normalize())
}

// ---------------------------------------------------------------------------
// 3. Spiking network
// ---------------------------------------------------------------------------

/// Leaky integrate-and-fire parameters of one neuron.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuronParams {
    /// Membrane leak rate (1/s): `dv/dt = -leak * v + drive`.
    pub leak_per_s: f32,
    /// Spike when `v >= threshold` (membrane units).
    pub threshold: f32,
    /// Value of `v` after a spike and during refraction.
    pub reset: f32,
    /// Time after a spike during which the neuron ignores input (s).
    pub refractory_s: f32,
}

/// A recurrent connection; its weight lives in [`NetworkWeights`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynapseSpec {
    pub pre: u16,
    pub post: u16,
    /// Axonal delay in network steps (`dt_s`), at least 1.
    pub delay_steps: u16,
}

/// A connection from an analog input channel to a neuron.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputLink {
    pub input: u16,
    pub post: u16,
}

/// Structure of a network: everything except the learned numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkLayout {
    /// Internal integration step (s). `step(dt, ..)` runs `dt / dt_s` of them.
    pub dt_s: f32,
    pub input_count: u16,
    pub neurons: Vec<NeuronParams>,
    pub synapses: Vec<SynapseSpec>,
    pub input_links: Vec<InputLink>,
    /// Neurons whose spike rate is decoded as muscle activations.
    pub output_neurons: Vec<u16>,
    /// Time constant of the output rate estimate (s).
    pub rate_window_s: f32,
    /// Spike rate (Hz) that decodes to activation 1.
    pub max_rate_hz: f32,
    /// Bounds of learned numbers: recurrent weight (membrane units per spike),
    /// input weight (membrane units per second per unit input), bias (same).
    pub synaptic_weight_limit: f32,
    pub input_weight_limit: f32,
    pub bias_limit: f32,
    /// Initial weights are uniform within this fraction of each limit.
    pub init_weight_fraction: f32,
    /// Tonic drive every neuron starts with, as a fraction of `bias_limit`.
    pub init_bias_fraction: f32,
    pub init_seed: u64,
}

impl NetworkLayout {
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.neurons.len();
        let finite_pos = |v: f32| v.is_finite() && v > 0.0;
        if n == 0 || n > MAX_NEURONS {
            return Err(format!("neuron count must be 1..={MAX_NEURONS}"));
        }
        if !finite_pos(self.dt_s)
            || !finite_pos(self.rate_window_s)
            || !finite_pos(self.max_rate_hz)
            || !finite_pos(self.synaptic_weight_limit)
            || !finite_pos(self.input_weight_limit)
            || !self.bias_limit.is_finite()
            || self.bias_limit < 0.0
            || !(0.0..=1.0).contains(&self.init_weight_fraction)
            || !(-1.0..=1.0).contains(&self.init_bias_fraction)
        {
            return Err("network scalar parameters are invalid".into());
        }
        for p in &self.neurons {
            if !p.leak_per_s.is_finite()
                || p.leak_per_s < 0.0
                || !p.threshold.is_finite()
                || !p.reset.is_finite()
                || p.threshold <= p.reset
                || !p.refractory_s.is_finite()
                || p.refractory_s < 0.0
            {
                return Err("neuron parameters are invalid".into());
            }
        }
        if self
            .synapses
            .iter()
            .any(|s| s.pre as usize >= n || s.post as usize >= n || s.delay_steps == 0)
            || self.synapses.iter().any(|s| s.delay_steps > 1024)
        {
            return Err("synapse endpoint or delay is invalid".into());
        }
        if self
            .input_links
            .iter()
            .any(|l| l.input >= self.input_count || l.post as usize >= n)
        {
            return Err("input link is invalid".into());
        }
        if self.output_neurons.is_empty()
            || self.output_neurons.iter().any(|&o| o as usize >= n)
            || (1..self.output_neurons.len())
                .any(|i| self.output_neurons[..i].contains(&self.output_neurons[i]))
        {
            return Err("output neurons must be unique and in range".into());
        }
        Ok(())
    }

    /// Seeded random sparse recurrent layout: neurons `0..hidden` are hidden,
    /// the last `outputs` are outputs. Each input connects to `input_fanout`
    /// hidden neurons, each hidden neuron to `recurrent_fanout` others, and
    /// every hidden neuron to every output.
    pub fn random_recurrent(
        seed: u64,
        input_count: usize,
        hidden: usize,
        outputs: usize,
        input_fanout: usize,
        recurrent_fanout: usize,
        max_delay_steps: u16,
    ) -> Result<Self, String> {
        if hidden == 0 || outputs == 0 || hidden + outputs > MAX_NEURONS || max_delay_steps == 0 {
            return Err("invalid network size".into());
        }
        let mut rng = SplitMix64::new(seed);
        let delay = |rng: &mut SplitMix64| 1 + rng.next_below(max_delay_steps as usize) as u16;
        let mut input_links = Vec::new();
        for input in 0..input_count {
            for post in distinct(&mut rng, hidden, input_fanout.min(hidden), None) {
                input_links.push(InputLink {
                    input: input as u16,
                    post: post as u16,
                });
            }
        }
        let mut synapses = Vec::new();
        for pre in 0..hidden {
            for post in distinct(&mut rng, hidden, recurrent_fanout.min(hidden - 1), Some(pre)) {
                synapses.push(SynapseSpec {
                    pre: pre as u16,
                    post: post as u16,
                    delay_steps: delay(&mut rng),
                });
            }
            for o in 0..outputs {
                synapses.push(SynapseSpec {
                    pre: pre as u16,
                    post: (hidden + o) as u16,
                    delay_steps: delay(&mut rng),
                });
            }
        }
        let layout = Self {
            dt_s: 0.001,
            input_count: input_count as u16,
            neurons: vec![
                NeuronParams {
                    leak_per_s: 20.0,
                    threshold: 1.0,
                    reset: 0.0,
                    refractory_s: 0.002,
                };
                hidden + outputs
            ],
            synapses,
            input_links,
            output_neurons: (hidden..hidden + outputs).map(|o| o as u16).collect(),
            rate_window_s: 0.05,
            max_rate_hz: 100.0,
            synaptic_weight_limit: 1.5,
            input_weight_limit: 100.0,
            bias_limit: 40.0,
            init_weight_fraction: 0.5,
            init_bias_fraction: 0.5,
            init_seed: rng.next_u64(),
        };
        layout.validate()?;
        Ok(layout)
    }
}

/// `count` distinct values in `0..bound`, none equal to `skip`.
fn distinct(rng: &mut SplitMix64, bound: usize, count: usize, skip: Option<usize>) -> Vec<usize> {
    let mut pool: Vec<usize> = (0..bound).filter(|&i| Some(i) != skip).collect();
    let count = count.min(pool.len());
    for i in 0..count {
        let j = i + rng.next_below(pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(count);
    pool
}

/// Learned synaptic numbers, parallel to the layout's `synapses` and
/// `input_links`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NetworkWeights {
    pub synaptic: Vec<f32>,
    pub input: Vec<f32>,
}

/// Counters that make the cost of the network visible.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkCost {
    pub steps: u64,
    pub neuron_updates: u64,
    pub spikes: u64,
    /// Weighted spike deliveries (one per synapse of each spiking neuron).
    pub synaptic_events: u64,
    /// Input-link multiply-adds.
    pub input_ops: u64,
}

impl NetworkCost {
    pub fn operations(&self) -> u64 {
        self.neuron_updates + self.synaptic_events + self.input_ops
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct NetworkState {
    v: Vec<f32>,
    refractory_left: Vec<u16>,
    /// Delay ring: `slots * neurons` pending synaptic jumps.
    pending: Vec<f32>,
    head: usize,
    rate_hz: Vec<f32>,
    time_residual_s: f64,
}

#[derive(Clone, Debug, Default)]
struct SynapseIndex {
    built: bool,
    start: Vec<u32>,
    /// (post, delay_steps, weight index), grouped by pre neuron.
    edges: Vec<(u16, u16, u32)>,
    slots: usize,
    refractory_steps: Vec<u16>,
    output_of: Vec<Option<u16>>,
}

/// Small deterministic LIF network. No randomness during `step`; a given
/// layout, weights and input sequence always give identical output.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpikingNetwork {
    layout: NetworkLayout,
    weights: NetworkWeights,
    bias: Vec<f32>,
    state: NetworkState,
    cost: NetworkCost,
    outputs: Vec<f32>,
    #[serde(skip)]
    index: SynapseIndex,
}

impl SpikingNetwork {
    pub fn new(layout: NetworkLayout, learned: &LearnedState) -> Result<Self, String> {
        layout.validate()?;
        learned.validate_against(&layout)?;
        let n = layout.neuron_count();
        let mut net = Self {
            weights: learned.weights.clone(),
            bias: learned.neuron_bias.clone(),
            state: NetworkState {
                v: vec![0.0; n],
                refractory_left: vec![0; n],
                pending: Vec::new(),
                head: 0,
                rate_hz: vec![0.0; layout.output_neurons.len()],
                time_residual_s: 0.0,
            },
            cost: NetworkCost::default(),
            outputs: vec![0.0; layout.output_neurons.len()],
            index: SynapseIndex::default(),
            layout,
        };
        net.build_index();
        Ok(net)
    }

    pub fn layout(&self) -> &NetworkLayout {
        &self.layout
    }
    pub fn weights(&self) -> &NetworkWeights {
        &self.weights
    }
    pub fn cost(&self) -> NetworkCost {
        self.cost
    }
    pub fn reset_cost(&mut self) {
        self.cost = NetworkCost::default();
    }
    pub fn outputs(&self) -> &[f32] {
        &self.outputs
    }

    /// Replace learned numbers (keeps membrane state).
    pub fn load_learned(&mut self, learned: &LearnedState) -> Result<(), String> {
        learned.validate_against(&self.layout)?;
        self.weights = learned.weights.clone();
        self.bias = learned.neuron_bias.clone();
        Ok(())
    }

    /// Clear membrane, delay and rate state; keeps weights and cost counters.
    pub fn reset_state(&mut self) {
        let n = self.layout.neuron_count();
        self.state = NetworkState {
            v: vec![0.0; n],
            refractory_left: vec![0; n],
            pending: vec![0.0; self.index.slots * n],
            head: 0,
            rate_hz: vec![0.0; self.layout.output_neurons.len()],
            time_residual_s: 0.0,
        };
        self.outputs.fill(0.0);
    }

    fn build_index(&mut self) {
        let n = self.layout.neuron_count();
        let mut counts = vec![0u32; n + 1];
        for s in &self.layout.synapses {
            counts[s.pre as usize + 1] += 1;
        }
        for i in 0..n {
            counts[i + 1] += counts[i];
        }
        let mut fill = counts.clone();
        let mut edges = vec![(0u16, 0u16, 0u32); self.layout.synapses.len()];
        for (k, s) in self.layout.synapses.iter().enumerate() {
            let slot = &mut fill[s.pre as usize];
            edges[*slot as usize] = (s.post, s.delay_steps, k as u32);
            *slot += 1;
        }
        let max_delay = self.layout.synapses.iter().map(|s| s.delay_steps).max().unwrap_or(1);
        let mut output_of = vec![None; n];
        for (o, &neuron) in self.layout.output_neurons.iter().enumerate() {
            output_of[neuron as usize] = Some(o as u16);
        }
        self.index = SynapseIndex {
            built: true,
            start: counts,
            edges,
            slots: max_delay as usize + 1,
            refractory_steps: self
                .layout
                .neurons
                .iter()
                .map(|p| (p.refractory_s / self.layout.dt_s).round() as u16)
                .collect(),
            output_of,
        };
        if self.state.pending.len() != self.index.slots * n {
            self.state.pending = vec![0.0; self.index.slots * n];
            self.state.head = 0;
        }
    }

    /// Advance by `dt_s` seconds with analog `inputs` (each clamped to
    /// `-1..=1`, held constant during the call) and return the muscle
    /// activations in `0..=1` decoded from output spike rates.
    pub fn step(&mut self, dt_s: f64, inputs: &[f32]) -> Result<&[f32], String> {
        if !(dt_s.is_finite() && dt_s >= 0.0) {
            return Err("dt must be finite and nonnegative".into());
        }
        if inputs.len() != self.layout.input_count as usize {
            return Err(format!(
                "network expects {} inputs, got {}",
                self.layout.input_count,
                inputs.len()
            ));
        }
        if inputs.iter().any(|x| !x.is_finite()) {
            return Err("network inputs must be finite".into());
        }
        if !self.index.built {
            self.build_index();
        }
        let n = self.layout.neuron_count();
        let mut drive = self.bias.clone();
        for (link, w) in self.layout.input_links.iter().zip(&self.weights.input) {
            drive[link.post as usize] += w * inputs[link.input as usize].clamp(-1.0, 1.0);
        }
        self.cost.input_ops += self.layout.input_links.len() as u64;

        // Round away the f32 representation error of decimal steps (0.001f32).
        let dt_net = (f64::from(self.layout.dt_s) * 1e9).round() / 1e9;
        self.state.time_residual_s += dt_s;
        let steps = ((self.state.time_residual_s / dt_net) + 1e-6).floor().max(0.0) as u64;
        self.state.time_residual_s -= steps as f64 * dt_net;
        let dt = self.layout.dt_s;
        let alpha = (dt / self.layout.rate_window_s).min(1.0);
        let slots = self.index.slots;
        for _ in 0..steps {
            let head = self.state.head;
            for i in 0..n {
                self.cost.neuron_updates += 1;
                let incoming = std::mem::take(&mut self.state.pending[head * n + i]);
                let p = self.layout.neurons[i];
                let mut spiked = false;
                if self.state.refractory_left[i] > 0 {
                    self.state.refractory_left[i] -= 1;
                    self.state.v[i] = p.reset;
                } else {
                    let v = &mut self.state.v[i];
                    *v += (-p.leak_per_s * *v + drive[i]) * dt + incoming;
                    if *v >= p.threshold {
                        spiked = true;
                        *v = p.reset;
                        self.state.refractory_left[i] = self.index.refractory_steps[i];
                    }
                }
                if spiked {
                    self.cost.spikes += 1;
                    let (a, b) = (self.index.start[i] as usize, self.index.start[i + 1] as usize);
                    for &(post, delay, w) in &self.index.edges[a..b] {
                        let slot = (head + delay as usize) % slots;
                        self.state.pending[slot * n + post as usize] +=
                            self.weights.synaptic[w as usize];
                        self.cost.synaptic_events += 1;
                    }
                }
                if let Some(o) = self.index.output_of[i] {
                    let rate = &mut self.state.rate_hz[o as usize];
                    let instant = if spiked { 1.0 / dt } else { 0.0 };
                    *rate += (instant - *rate) * alpha;
                }
            }
            self.state.head = (head + 1) % slots;
            self.cost.steps += 1;
        }
        for (out, rate) in self.outputs.iter_mut().zip(&self.state.rate_hz) {
            *out = (rate / self.layout.max_rate_hz).clamp(0.0, 1.0);
        }
        Ok(&self.outputs)
    }
}

// ---------------------------------------------------------------------------
// 4. Heredity and learning
// ---------------------------------------------------------------------------

/// Anatomical description inherited by offspring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyStructure {
    pub model_name: String,
    /// Creature-local REV indices that act as feet.
    pub foot_voxels: Vec<u32>,
}

/// How an offspring genome may differ from its parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MutationRules {
    /// Relative standard deviation applied to each organ parameter.
    pub organ_relative_sigma: f64,
    /// Relative standard deviation applied to each neuron threshold.
    pub threshold_relative_sigma: f32,
    /// Probability that a synapse's delay changes by one step.
    pub delay_change_probability: f32,
    pub max_delay_steps: u16,
}

impl Default for MutationRules {
    fn default() -> Self {
        Self {
            organ_relative_sigma: 0.05,
            threshold_relative_sigma: 0.05,
            delay_change_probability: 0.05,
            max_delay_steps: 8,
        }
    }
}

/// Everything inherited: structure, organ parameters, initial network layout,
/// mutation rules. It contains no weight that life can change; weights the
/// network starts with are regenerated from `network.init_seed`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub structure: BodyStructure,
    pub organs: OrganParams,
    pub network: NetworkLayout,
    pub mutation: MutationRules,
}

impl Genome {
    pub fn validate(&self) -> Result<(), String> {
        self.organs.validate()?;
        self.network.validate()
    }

    /// A child genome: parameters jittered by the mutation rules, a fresh
    /// initial-weight seed. It does not carry the parent's learned state.
    pub fn offspring(&self, seed: u64) -> Result<Genome, String> {
        let mut rng = SplitMix64::new(seed);
        let mut child = self.clone();
        let m = self.mutation.clone();
        let jitter = |rng: &mut SplitMix64, value: f64| {
            value * (1.0 + m.organ_relative_sigma * f64::from(rng.next_gauss())).max(0.1)
        };
        let o = &mut child.organs;
        o.store_capacity_j = jitter(&mut rng, o.store_capacity_j);
        o.stomach_capacity_j = jitter(&mut rng, o.stomach_capacity_j);
        o.digestion_power_w = jitter(&mut rng, o.digestion_power_w);
        o.basal_power_w = jitter(&mut rng, o.basal_power_w);
        o.peak_mechanical_power_w = jitter(&mut rng, o.peak_mechanical_power_w);
        o.absorption_efficiency = jitter(&mut rng, o.absorption_efficiency).clamp(0.05, 1.0);
        o.muscle_efficiency = jitter(&mut rng, o.muscle_efficiency).clamp(0.05, 1.0);
        for p in &mut child.network.neurons {
            let f = 1.0 + m.threshold_relative_sigma * rng.next_gauss();
            p.threshold = (p.threshold * f.max(0.1)).max(p.reset + 1e-3);
        }
        for s in &mut child.network.synapses {
            if rng.next_unit() < m.delay_change_probability {
                let up = rng.next_unit() < 0.5;
                s.delay_steps = if up {
                    (s.delay_steps + 1).min(m.max_delay_steps.max(1))
                } else {
                    s.delay_steps.saturating_sub(1).max(1)
                };
            }
        }
        child.network.init_seed = rng.next_u64();
        child.validate()?;
        Ok(child)
    }
}

/// What a creature acquires in life: kept separate from the [`Genome`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearnedState {
    pub weights: NetworkWeights,
    /// Learned per-neuron tonic drive (the network's long-term memory of its
    /// operating point), membrane units per second.
    pub neuron_bias: Vec<f32>,
    pub lifetime_s: f64,
    pub trials: u64,
    pub accepted_updates: u64,
}

impl LearnedState {
    /// The state of a creature that knows nothing: weights drawn from the
    /// layout's seed.
    pub fn from_scratch(layout: &NetworkLayout) -> Self {
        let mut rng = SplitMix64::new(layout.init_seed);
        let sw = layout.synaptic_weight_limit * layout.init_weight_fraction;
        let iw = layout.input_weight_limit * layout.init_weight_fraction;
        Self {
            weights: NetworkWeights {
                synaptic: layout.synapses.iter().map(|_| rng.next_signed() * sw).collect(),
                input: layout.input_links.iter().map(|_| rng.next_signed() * iw).collect(),
            },
            neuron_bias: layout
                .neurons
                .iter()
                .map(|_| layout.init_bias_fraction * layout.bias_limit * (0.5 + rng.next_unit()))
                .map(|b| b.clamp(-layout.bias_limit, layout.bias_limit))
                .collect(),
            lifetime_s: 0.0,
            trials: 0,
            accepted_updates: 0,
        }
    }

    pub fn validate_against(&self, layout: &NetworkLayout) -> Result<(), String> {
        if self.weights.synaptic.len() != layout.synapses.len()
            || self.weights.input.len() != layout.input_links.len()
            || self.neuron_bias.len() != layout.neuron_count()
        {
            return Err("learned state does not match the network layout".into());
        }
        let ok = |values: &[f32], limit: f32| {
            values.iter().all(|v| v.is_finite() && v.abs() <= limit * 1.000_001)
        };
        if !ok(&self.weights.synaptic, layout.synaptic_weight_limit)
            || !ok(&self.weights.input, layout.input_weight_limit)
            || !ok(&self.neuron_bias, layout.bias_limit)
        {
            return Err("learned values are non-finite or exceed the layout limits".into());
        }
        Ok(())
    }
}

/// Reward-gated seeded hill climb ((1+1) evolution strategy) over the learned
/// numbers. A trial perturbs a random subset of weights and biases by at most
/// `sigma` times their limits (hard-clamped to the limits); the caller runs
/// it for an episode of real experience and reports a reward. The change is
/// kept only if the reward exceeds the best so far, so no update can make the
/// stored behaviour worse. Step size adapts (1/5-rule flavoured) within bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HillClimber {
    rng: SplitMix64,
    sigma: f32,
    sigma_min: f32,
    sigma_max: f32,
    mutate_fraction: f32,
    best_reward: Option<f64>,
}

impl HillClimber {
    pub fn new(seed: u64, sigma: f32) -> Self {
        Self {
            rng: SplitMix64::new(seed),
            sigma,
            sigma_min: 0.005,
            sigma_max: 0.5,
            mutate_fraction: 0.25,
            best_reward: None,
        }
    }
    pub fn sigma(&self) -> f32 {
        self.sigma
    }
    pub fn best_reward(&self) -> Option<f64> {
        self.best_reward
    }
    /// Record the reward of the current state, measured with the same episode
    /// that trials use.
    pub fn set_baseline(&mut self, reward: f64) {
        self.best_reward = Some(reward);
    }

    pub fn propose(&mut self, current: &LearnedState, layout: &NetworkLayout) -> LearnedState {
        let mut candidate = current.clone();
        let (sigma, fraction) = (self.sigma, self.mutate_fraction);
        let rng = &mut self.rng;
        let mut touched = false;
        let mut perturb = |values: &mut [f32], limit: f32, rng: &mut SplitMix64| {
            for v in values.iter_mut() {
                if rng.next_unit() < fraction {
                    *v = (*v + rng.next_gauss() * sigma * limit).clamp(-limit, limit);
                    touched = true;
                }
            }
        };
        perturb(&mut candidate.weights.synaptic, layout.synaptic_weight_limit, rng);
        perturb(&mut candidate.weights.input, layout.input_weight_limit, rng);
        perturb(&mut candidate.neuron_bias, layout.bias_limit, rng);
        if !touched && !candidate.weights.input.is_empty() {
            let k = rng.next_below(candidate.weights.input.len());
            let limit = layout.input_weight_limit;
            candidate.weights.input[k] =
                (candidate.weights.input[k] + rng.next_gauss() * sigma * limit).clamp(-limit, limit);
        }
        candidate
    }

    /// Accept `candidate` into `learned` iff `reward` beats the best so far.
    /// Returns whether it was accepted. Non-finite rewards are rejected.
    pub fn judge(&mut self, learned: &mut LearnedState, candidate: LearnedState, reward: f64) -> bool {
        learned.trials += 1;
        let better = reward.is_finite() && self.best_reward.map_or(true, |best| reward > best);
        if better {
            let (trials, lifetime) = (learned.trials, learned.lifetime_s);
            *learned = candidate;
            learned.trials = trials;
            learned.lifetime_s = lifetime;
            learned.accepted_updates += 1;
            self.best_reward = Some(reward);
            self.sigma = (self.sigma * 1.3).min(self.sigma_max);
        } else {
            self.sigma = (self.sigma * 0.93).max(self.sigma_min);
        }
        better
    }
}

// ---------------------------------------------------------------------------
// 5. Controller boundary
// ---------------------------------------------------------------------------

/// Per-actuator strength scale in `0..=1` produced by any controller.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlCommand {
    pub actuator_strength: Vec<f32>,
}

/// The one interface of recorded animation, hand-written and learned control.
/// A controller sees only sensor readings and answers with strengths; it has
/// no access to positions, velocities or forces.
pub trait Controller {
    fn name(&self) -> &str;
    fn control(
        &mut self,
        dt_s: f64,
        sensors: &SensorReading,
        actuator_count: usize,
    ) -> Result<ControlCommand, String>;
}

/// Plays the recorded clip at full commanded strength. The clip itself is
/// advanced by the actuator, so this is the baseline that is not learned.
#[derive(Clone, Debug, Default)]
pub struct RecordedAnimationController;

impl Controller for RecordedAnimationController {
    fn name(&self) -> &str {
        "recorded-animation"
    }
    fn control(&mut self, _: f64, _: &SensorReading, actuator_count: usize) -> Result<ControlCommand, String> {
        Ok(ControlCommand {
            actuator_strength: vec![1.0; actuator_count],
        })
    }
}

/// A fixed rule: hold `standing_strength` in the air and rise towards full
/// strength as more feet touch.
#[derive(Clone, Debug)]
pub struct HandWrittenController {
    pub standing_strength: f32,
}

impl Controller for HandWrittenController {
    fn name(&self) -> &str {
        "hand-written"
    }
    fn control(&mut self, _: f64, sensors: &SensorReading, actuator_count: usize) -> Result<ControlCommand, String> {
        let feet = sensors.foot_contact.len().max(1) as f32;
        let touching = sensors.foot_contact.iter().filter(|&&c| c).count() as f32;
        let s = self.standing_strength.clamp(0.0, 1.0);
        Ok(ControlCommand {
            actuator_strength: vec![s + (1.0 - s) * touching / feet; actuator_count],
        })
    }
}

/// Spiking-network controller: encodes the sensors, runs the network, and
/// averages the output activations over equal groups, one group per actuator.
#[derive(Clone, Debug)]
pub struct SpikingController {
    network: SpikingNetwork,
    inputs: Vec<f32>,
}

impl SpikingController {
    pub fn new(network: SpikingNetwork) -> Self {
        Self {
            network,
            inputs: Vec::new(),
        }
    }
    pub fn network(&self) -> &SpikingNetwork {
        &self.network
    }
    pub fn network_mut(&mut self) -> &mut SpikingNetwork {
        &mut self.network
    }
}

impl Controller for SpikingController {
    fn name(&self) -> &str {
        "spiking-network"
    }
    fn control(&mut self, dt_s: f64, sensors: &SensorReading, actuator_count: usize) -> Result<ControlCommand, String> {
        sensors.encode(&mut self.inputs);
        let outputs = self.network.step(dt_s, &self.inputs)?;
        if actuator_count == 0 || outputs.len() % actuator_count != 0 {
            return Err("output count must be a multiple of the actuator count".into());
        }
        let group = outputs.len() / actuator_count;
        Ok(ControlCommand {
            actuator_strength: outputs
                .chunks(group)
                .map(|c| c.iter().sum::<f32>() / group as f32)
                .collect(),
        })
    }
}

/// Write a command into actuators through their existing `strength_scale` and
/// `enabled` fields only. The applied scale is `min(command, limit)` with both
/// forced into `0..=1`; a non-finite command counts as zero. A zero scale also
/// disables the actuator so an exhausted creature does not move.
pub fn apply_command(command: &ControlCommand, limit: f32, actuators: &mut [SkeletalActuator]) -> Result<(), String> {
    if command.actuator_strength.len() != actuators.len() {
        return Err("command length differs from the actuator count".into());
    }
    let limit = if limit.is_finite() { limit.clamp(0.0, 1.0) } else { 0.0 };
    for (a, &c) in actuators.iter_mut().zip(&command.actuator_strength) {
        let c = if c.is_finite() { c.clamp(0.0, 1.0) } else { 0.0 };
        a.strength_scale = c.min(limit);
        a.enabled = a.strength_scale > 0.0;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The creature: genome + learned state + ledger + the glue of one step
// ---------------------------------------------------------------------------

/// What one driven step did.
#[derive(Clone, Debug)]
pub struct StepReport {
    pub commanded: f32,
    pub applied: f32,
    pub outcome: MuscleStepOutcome,
}

pub struct Creature {
    genome: Genome,
    learned: LearnedState,
    ledger: ResourceLedger,
    sensors: SensorLayout,
    actuator_index: usize,
}

impl Creature {
    /// Build from a genome. With `initial_gait == None` the creature starts
    /// from scratch (weights regenerated from the genome); with a gait it starts
    /// from that learned state and keeps learning from it.
    pub fn new(
        genome: Genome,
        initial_gait: Option<LearnedState>,
        scene: &Scene,
        actuator_index: usize,
    ) -> Result<Self, String> {
        genome.validate()?;
        let actuator = scene
            .skeletal_actuators
            .get(actuator_index)
            .ok_or("scene has no such actuator")?;
        let feet = genome.structure.foot_voxels.iter().map(|&f| f as usize).collect();
        let sensors = SensorLayout::new(actuator, feet)?;
        if genome.network.input_count as usize != sensors.input_count() {
            return Err(format!(
                "network expects {} inputs but the body senses {}",
                genome.network.input_count,
                sensors.input_count()
            ));
        }
        let learned = match initial_gait {
            Some(g) => {
                g.validate_against(&genome.network)?;
                g
            }
            None => LearnedState::from_scratch(&genome.network),
        };
        Ok(Self {
            ledger: ResourceLedger::new(genome.organs.clone())?,
            genome,
            learned,
            sensors,
            actuator_index,
        })
    }

    pub fn genome(&self) -> &Genome {
        &self.genome
    }
    pub fn learned(&self) -> &LearnedState {
        &self.learned
    }
    pub fn learned_mut(&mut self) -> &mut LearnedState {
        &mut self.learned
    }
    pub fn ledger(&self) -> &ResourceLedger {
        &self.ledger
    }
    pub fn ledger_mut(&mut self) -> &mut ResourceLedger {
        &mut self.ledger
    }
    pub fn sensor_layout(&self) -> &SensorLayout {
        &self.sensors
    }

    /// A fresh network built from the genome's layout and the current learned
    /// state.
    pub fn spiking_controller(&self) -> Result<SpikingController, String> {
        Ok(SpikingController::new(SpikingNetwork::new(
            self.genome.network.clone(),
            &self.learned,
        )?))
    }

    /// One physical step with sensing, control, actuation, accounting and
    /// heat. `advance` must perform exactly one fixed solver step on the scene
    /// (for example `|s| solver::step(s, &mut None)`).
    pub fn drive_step(
        &mut self,
        scene: &mut Scene,
        controller: &mut dyn Controller,
        advance: &mut dyn FnMut(&mut Scene),
    ) -> Result<StepReport, String> {
        let dt = f64::from(scene.config.fixed_dt);
        let reading = SensorReading::read(scene, &self.sensors, &self.ledger)?;
        let command = controller.control(dt, &reading, 1)?;
        let limit = self.ledger.strength_limit(dt);
        let idx = self.actuator_index;
        let actuators = scene
            .skeletal_actuators
            .get_mut(idx..=idx)
            .ok_or("creature actuator is missing")?;
        apply_command(&command, limit, actuators)?;
        let applied = scene.skeletal_actuators[idx].strength_scale;
        let range = scene.skeletal_actuators[idx].voxel_range();

        let muscle: Vec<usize> = range
            .clone()
            .filter(|&i| !scene.voxels[i].kinematic && scene.voxels[i].mass > 0.0)
            .collect();
        let v0: Vec<Vec3> = muscle.iter().map(|&i| scene.voxels[i].velocity).collect();
        advance(scene);
        if scene.voxels.len() != self.sensors.total_scene_voxels {
            return Err("scene topology changed during the step".into());
        }
        let loads = scene.skeletal_actuators[idx].last_forces_n();
        let forces: Vec<Vec3> = muscle
            .iter()
            .map(|&i| loads.get(i - range.start).copied().unwrap_or(Vec3::ZERO))
            .collect();
        let v1: Vec<Vec3> = muscle.iter().map(|&i| scene.voxels[i].velocity).collect();
        let outcome = self.ledger.settle_muscle_step(dt, &forces, &v0, &v1)?;
        for (&i, &heat) in muscle.iter().zip(&outcome.heat_per_rev_j) {
            scene.voxels[i].add_internal_energy_j(heat);
        }
        self.learned.lifetime_s += dt;
        Ok(StepReport {
            commanded: command.actuator_strength.first().copied().unwrap_or(0.0),
            applied,
            outcome,
        })
    }
}

/// Convenience for building a genome whose network matches a body: random
/// sparse recurrent layout with `outputs` output neurons.
pub fn genome_for_body(
    model_name: &str,
    sensors: &SensorLayout,
    organs: OrganParams,
    seed: u64,
    hidden: usize,
    outputs: usize,
) -> Result<Genome, String> {
    let genome = Genome {
        structure: BodyStructure {
            model_name: model_name.to_owned(),
            foot_voxels: sensors.foot_local.iter().map(|&f| f as u32).collect(),
        },
        organs,
        network: NetworkLayout::random_recurrent(seed, sensors.input_count(), hidden, outputs, 6, 6, 8)?,
        mutation: MutationRules::default(),
    };
    genome.validate()?;
    Ok(genome)
}

#[cfg(test)]
mod tests;
