//! Converts a material-limited voxel skeleton pose into ordinary solver loads.
//!
//! The actuator never writes positions or velocities. Its forces are transient
//! and must be added to the scene's applied-load array for one physical step.

use crate::{
    documents::{PreparedSkeletonAnimation, SkeletonAnimator, VoxelModelDocument},
    material::Material,
    voxel::Voxel,
};
use glam::{DMat3, DQuat, DVec3, Quat, Vec3};
use std::ops::Range;

const CELL_M: f64 = crate::coordinates::CELL_M;
const FORCE_FRACTION_OF_STRENGTH: f64 = 0.005;

/// A spring-damper motor that pulls a scene instance toward its skeleton pose.
///
/// `voxel_range` identifies the contiguous scene REV range occupied by this
/// model. The range and total count are checked before every load generation;
/// topology changes therefore pause actuation rather than retarget shifted REV
/// indices. Units: playback speed is dimensionless, strength scale is
/// dimensionless, and emitted loads are newtons.
pub struct SkeletalActuator {
    model: VoxelModelDocument,
    prepared_animation: PreparedSkeletonAnimation,
    voxel_range: Range<usize>,
    total_scene_voxels: usize,
    instance_rotation: DQuat,
    root_bone_weights: Vec<f64>,
    elapsed_s: f64,
    previous_elapsed_s: f64,
    last_force_norm_n: f64,
    /// Loads emitted by the latest call, indexed from the start of `voxel_range`.
    last_forces_n: Vec<Vec3>,
    pub(crate) stalled: bool,
    invalidated: bool,
    editor_bond_protection: bool,
    pub clip_index: usize,
    pub playback_speed: f64,
    pub strength_scale: f32,
    pub enabled: bool,
}

/// Changing part of an actuator, stored in a checkpoint.
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug)]
pub struct ActuatorRuntime {
    elapsed_s: f64,
    previous_elapsed_s: f64,
    stalled: bool,
    invalidated: bool,
    editor_bond_protection: bool,
    clip_index: usize,
    playback_speed: f64,
    strength_scale: f32,
    enabled: bool,
}

impl SkeletalActuator {
    pub(crate) fn model(&self) -> &VoxelModelDocument {
        &self.model
    }

    pub(crate) fn instance_rotation_f32(&self) -> Quat {
        self.instance_rotation.as_quat()
    }

    pub(crate) fn runtime(&self) -> ActuatorRuntime {
        ActuatorRuntime {
            elapsed_s: self.elapsed_s,
            previous_elapsed_s: self.previous_elapsed_s,
            stalled: self.stalled,
            invalidated: self.invalidated,
            editor_bond_protection: self.editor_bond_protection,
            clip_index: self.clip_index,
            playback_speed: self.playback_speed,
            strength_scale: self.strength_scale,
            enabled: self.enabled,
        }
    }

    pub(crate) fn restore_runtime(&mut self, runtime: ActuatorRuntime) {
        self.elapsed_s = runtime.elapsed_s;
        self.previous_elapsed_s = runtime.previous_elapsed_s;
        self.stalled = runtime.stalled;
        self.invalidated = runtime.invalidated;
        self.editor_bond_protection = runtime.editor_bond_protection;
        self.clip_index = runtime.clip_index;
        self.playback_speed = runtime.playback_speed;
        self.strength_scale = runtime.strength_scale;
        self.enabled = runtime.enabled;
    }

    pub fn voxel_range(&self) -> Range<usize> {
        self.voxel_range.clone()
    }
    pub fn total_scene_voxels(&self) -> usize {
        self.total_scene_voxels
    }
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_s
    }
    pub fn last_force_norm_n(&self) -> f64 {
        self.last_force_norm_n
    }
    /// Per-REV loads (N) that the latest `accumulate_forces` call added to the
    /// scene, indexed from `voxel_range().start`; zero for fixed REVs and when
    /// the call produced no load. Used for resource accounting of motor work.
    pub fn last_forces_n(&self) -> &[Vec3] {
        &self.last_forces_n
    }

    /// Distance (m) of each REV from the pose the clip commands at the
    /// actuator's current clip time, in the same mass-comoving, body-rotated
    /// frame the force law uses (rigid translation and rotation of the whole
    /// body do not count as error). Indexed from `voxel_range().start`; zero for
    /// fixed REVs. Read-only: it changes neither the clip time nor any load.
    /// `None` when the actuator is disabled by a topology change or the inputs
    /// do not match its range.
    pub fn pose_error_m(
        &self,
        positions: &[DVec3],
        voxels: &[Voxel],
        rotations: &[Quat],
    ) -> Option<Vec<f32>> {
        let count = positions.len();
        if self.invalidated
            || count != self.total_scene_voxels
            || voxels.len() != count
            || rotations.len() != count
            || self.voxel_range.end > count
        {
            return None;
        }
        let clip = self.model.skeleton.as_ref()?.clips.get(self.clip_index)?;
        let time = self.elapsed_s.rem_euclid(clip.duration_s.max(1e-6));
        let poses = self
            .prepared_animation
            .evaluate(&self.model, self.clip_index, time)
            .ok()
            .filter(|poses| poses.len() == self.voxel_range.len())?;
        let dynamic: Vec<usize> = self
            .voxel_range
            .clone()
            .filter(|&i| !voxels[i].kinematic && voxels[i].mass.is_finite() && voxels[i].mass > 0.0)
            .collect();
        let total_mass: f64 = dynamic.iter().map(|&i| f64::from(voxels[i].mass)).sum();
        if dynamic.is_empty() || !(total_mass.is_finite() && total_mass > 0.0) {
            return None;
        }
        let mut current_com = DVec3::ZERO;
        let mut target_com_local = DVec3::ZERO;
        for &i in &dynamic {
            let mass = f64::from(voxels[i].mass);
            current_com += positions[i] * mass;
            target_com_local += poses[i - self.voxel_range.start].position_m * mass;
        }
        current_com /= total_mass;
        target_com_local /= total_mass;
        let rotation = self.body_rotation(dynamic.iter().copied(), voxels, rotations);
        let mut errors = vec![0.0f32; self.voxel_range.len()];
        for &i in &dynamic {
            let local = poses[i - self.voxel_range.start].position_m - target_com_local;
            let target = current_com + rotation * local;
            let error = (target - positions[i]).length();
            errors[i - self.voxel_range.start] = if error.is_finite() { error as f32 } else { 0.0 };
        }
        Some(errors)
    }

    /// Enables the editor preview's protection from material bond damage.
    /// Gameplay actors leave this disabled so ordinary world damage applies.
    pub fn set_editor_bond_protection(&mut self, enabled: bool) {
        self.editor_bond_protection = enabled;
    }

    pub fn new(
        model: VoxelModelDocument,
        voxel_range: Range<usize>,
        instance_rotation: Quat,
        total_scene_voxels: usize,
    ) -> Result<Self, String> {
        let animator = SkeletonAnimator::new(&model)?;
        let skeleton = model
            .skeleton
            .as_ref()
            .ok_or("voxel model has no skeleton")?;
        if skeleton.clips.is_empty() {
            return Err("voxel model skeleton has no animation clips".into());
        }
        if voxel_range.start > voxel_range.end
            || voxel_range.end > total_scene_voxels
            || voxel_range.len() != model.voxels.len()
        {
            return Err("skeletal actuator voxel range does not match model/scene topology".into());
        }
        if !instance_rotation.is_finite() || instance_rotation.length_squared() < 1e-8 {
            return Err("skeletal actuator instance rotation must be finite and nonzero".into());
        }
        // Force evaluation now so malformed clips cannot produce a partially
        // registered actor. The animator validates clips and material limits.
        animator.evaluate(0, 0.0)?;
        let prepared_animation = animator.into_prepared();
        let mut root_bone_weights = vec![0.0; model.voxels.len()];
        if let Some(skeleton) = &model.skeleton {
            for binding in &skeleton.voxel_weights {
                if binding.bone_index == 0 {
                    root_bone_weights[binding.voxel_index as usize] += f64::from(binding.weight);
                }
            }
        }
        Ok(Self {
            model,
            prepared_animation,
            voxel_range,
            total_scene_voxels,
            instance_rotation: DQuat::from_xyzw(
                f64::from(instance_rotation.x),
                f64::from(instance_rotation.y),
                f64::from(instance_rotation.z),
                f64::from(instance_rotation.w),
            )
            .normalize(),
            root_bone_weights,
            elapsed_s: 0.0,
            previous_elapsed_s: 0.0,
            last_force_norm_n: 0.0,
            last_forces_n: Vec::new(),
            stalled: false,
            invalidated: false,
            editor_bond_protection: false,
            clip_index: 0,
            playback_speed: 1.0,
            strength_scale: 1.0,
            enabled: true,
        })
    }

    /// Generate transient physical loads for the next fixed step.
    ///
    /// The target is expressed in the actor's mass-comoving frame, so animation
    /// cannot translate the instance through the world. Rigid translation and
    /// rotation are projected out of the force field, leaving zero resultant
    /// force and torque. Fixed REVs receive no motor force.
    pub fn accumulate_forces(
        &mut self,
        positions: &[DVec3],
        voxels: &[Voxel],
        rotations: &[Quat],
        properties: &[Material],
        dt_s: f64,
        out: &mut [Vec3],
    ) -> bool {
        self.last_force_norm_n = 0.0;
        self.last_forces_n.clear();
        let count = positions.len();
        if count != self.total_scene_voxels {
            self.invalidated = true;
            return false;
        }
        if !self.enabled
            || self.invalidated
            || voxels.len() != count
            || rotations.len() != count
            || properties.len() != count
            || out.len() != count
            || self.voxel_range.end > count
            || positions[self.voxel_range.clone()]
                .iter()
                .any(|position| !position.is_finite())
            || voxels[self.voxel_range.clone()]
                .iter()
                .any(|voxel| !voxel.velocity.is_finite())
            || rotations[self.voxel_range.clone()]
                .iter()
                .any(|rotation| !rotation.is_finite() || rotation.length_squared() < 1e-8)
            || !dt_s.is_finite()
            || dt_s <= 0.0
            || !self.playback_speed.is_finite()
            || !self.strength_scale.is_finite()
            || self.strength_scale < 0.0
        {
            return false;
        }
        let Some(skeleton) = self.model.skeleton.as_ref() else {
            return false;
        };
        let Some(clip) = skeleton.clips.get(self.clip_index) else {
            return false;
        };
        let next_time = if self.stalled {
            self.elapsed_s
        } else {
            self.elapsed_s + dt_s * self.playback_speed
        };
        if !next_time.is_finite() {
            return false;
        }
        let time = next_time.rem_euclid(clip.duration_s.max(dt_s));
        let poses = match self
            .prepared_animation
            .evaluate(&self.model, self.clip_index, time)
        {
            Ok(poses) if poses.len() == self.voxel_range.len() => poses,
            _ => return false,
        };

        let dynamic: Vec<_> = self
            .voxel_range
            .clone()
            .filter(|&i| !voxels[i].kinematic && voxels[i].mass.is_finite() && voxels[i].mass > 0.0)
            .collect();
        let total_mass: f64 = dynamic.iter().map(|&i| f64::from(voxels[i].mass)).sum();
        if dynamic.is_empty() || !total_mass.is_finite() || total_mass <= 0.0 {
            return false;
        }

        let current_com = dynamic.iter().fold(DVec3::ZERO, |sum, &i| {
            sum + positions[i] * f64::from(voxels[i].mass)
        }) / total_mass;
        let center_velocity = dynamic.iter().fold(DVec3::ZERO, |sum, &i| {
            sum + voxels[i].velocity.as_dvec3() * f64::from(voxels[i].mass)
        }) / total_mass;
        let target_com_local = dynamic.iter().fold(DVec3::ZERO, |sum, &i| {
            sum + poses[i - self.voxel_range.start].position_m * f64::from(voxels[i].mass)
        }) / total_mass;
        let body_rotation = self.body_rotation(dynamic.iter().copied(), voxels, rotations);
        let target_positions: Vec<_> = dynamic
            .iter()
            .map(|&i| {
                let local = poses[i - self.voxel_range.start].position_m - target_com_local;
                current_com + body_rotation * local
            })
            .collect();

        let (inertia, omega) =
            rigid_kinematics(&dynamic, positions, voxels, current_com, center_velocity);
        let mut forces = Vec::with_capacity(dynamic.len());
        let mut resultant = DVec3::ZERO;
        let mut torque = DVec3::ZERO;
        for (slot, &i) in dynamic.iter().enumerate() {
            let mass = f64::from(voxels[i].mass);
            let r = positions[i] - current_com;
            let target = target_positions[slot];
            let error = target - positions[i];
            let relative_velocity =
                voxels[i].velocity.as_dvec3() - center_velocity - omega.cross(r);
            let material = &properties[i];
            let stiffness = f64::from(material.young.max(0.0)) * CELL_M;
            let damping = 0.8 * (stiffness * mass).sqrt();
            let mut force =
                (error * stiffness - relative_velocity * damping) * f64::from(self.strength_scale);
            if !force.is_finite() {
                force = DVec3::ZERO;
            }
            let tensile = material
                .tensile_strength
                .min(material.tensile_strength_perp)
                .max(0.0) as f64;
            let cap = tensile * CELL_M * CELL_M * FORCE_FRACTION_OF_STRENGTH;
            if force.length() > cap && cap.is_finite() {
                force *= cap / force.length();
            }
            forces.push(force);
            resultant += force;
            torque += r.cross(force);
        }
        // Remove the force and torque components that would propel/spin the
        // instance as a whole. This keeps actuation in its comoving frame.
        let alpha = inverse_inertia(inertia)
            .map(|inv| inv * torque)
            .unwrap_or(DVec3::ZERO);
        for (slot, &i) in dynamic.iter().enumerate() {
            let mass_fraction = f64::from(voxels[i].mass) / total_mass;
            let r = positions[i] - current_com;
            forces[slot] -= resultant * mass_fraction + alpha.cross(r) * f64::from(voxels[i].mass);
        }
        // Projection can increase an individual load. Preserve the same
        // conservative material limit after projection.
        let mut scale = 1.0f64;
        for (slot, &i) in dynamic.iter().enumerate() {
            let material = &properties[i];
            let cap = material
                .tensile_strength
                .min(material.tensile_strength_perp)
                .max(0.0) as f64
                * CELL_M
                * CELL_M
                * FORCE_FRACTION_OF_STRENGTH;
            if forces[slot].length() > cap && forces[slot].length() > 0.0 {
                scale = scale.min(cap / forces[slot].length());
            }
        }
        if !scale.is_finite() {
            return false;
        }
        if !self.stalled {
            self.previous_elapsed_s = self.elapsed_s;
            self.elapsed_s = next_time;
        }
        self.last_forces_n.resize(self.voxel_range.len(), Vec3::ZERO);
        for (slot, &i) in dynamic.iter().enumerate() {
            let f = forces[slot] * scale;
            if f.is_finite() {
                out[i] += f.as_vec3();
                self.last_forces_n[i - self.voxel_range.start] = f.as_vec3();
                self.last_force_norm_n += f.length();
            }
        }
        true
    }

    pub(crate) fn protects_bond(&self, a: usize, b: usize) -> bool {
        if !self.editor_bond_protection
            || self.invalidated
            || a < self.voxel_range.start
            || b < self.voxel_range.start
            || a >= self.voxel_range.end
            || b >= self.voxel_range.end
        {
            return false;
        }
        let pa = self.model.voxels[a - self.voxel_range.start].rest_position_m;
        let pb = self.model.voxels[b - self.voxel_range.start].rest_position_m;
        let delta = (pa - pb).abs();
        let axis_count = [delta.x, delta.y, delta.z]
            .iter()
            .filter(|&&component| (component - CELL_M).abs() < 1.0e-5)
            .count();
        axis_count == 1 && delta.max_element() <= CELL_M + 1.0e-5
    }

    pub(crate) fn stall(&mut self) {
        if !self.stalled {
            self.elapsed_s = self.previous_elapsed_s;
        }
        self.stalled = true;
    }

    pub(crate) fn matches_scene_topology(&self, scene_voxel_count: usize) -> bool {
        !self.invalidated && self.total_scene_voxels == scene_voxel_count
    }

    fn body_rotation(
        &self,
        indices: impl Iterator<Item = usize>,
        voxels: &[Voxel],
        rotations: &[Quat],
    ) -> DQuat {
        let has_root_weights = self.root_bone_weights.iter().any(|&weight| weight > 0.0);
        let mut sum = glam::DVec4::ZERO;
        let mut reference: Option<DQuat> = None;
        for index in indices {
            let local = index - self.voxel_range.start;
            let root_weight = if has_root_weights {
                self.root_bone_weights[local]
            } else {
                1.0
            };
            let weight = f64::from(voxels[index].mass) * root_weight;
            if weight <= 0.0 {
                continue;
            }
            let rotation = DQuat::from_xyzw(
                f64::from(rotations[index].x),
                f64::from(rotations[index].y),
                f64::from(rotations[index].z),
                f64::from(rotations[index].w),
            )
            .normalize();
            let reference = *reference.get_or_insert(rotation);
            let aligned = if reference.dot(rotation) < 0.0 {
                -rotation
            } else {
                rotation
            };
            sum += glam::DVec4::new(aligned.x, aligned.y, aligned.z, aligned.w) * weight;
        }
        let delta = if sum.length_squared() > 1e-24 {
            DQuat::from_xyzw(sum.x, sum.y, sum.z, sum.w).normalize()
        } else {
            DQuat::IDENTITY
        };
        delta * self.instance_rotation
    }
}

pub(crate) fn rigid_kinematics(
    indices: &[usize],
    positions: &[DVec3],
    voxels: &[Voxel],
    com: DVec3,
    center_velocity: DVec3,
) -> (DMat3, DVec3) {
    let mut inertia = DMat3::ZERO;
    let mut angular_momentum = DVec3::ZERO;
    if indices.is_empty() {
        return (inertia, DVec3::ZERO);
    }
    for &i in indices {
        let mass = f64::from(voxels[i].mass);
        let r = positions[i] - com;
        inertia += DMat3::from_diagonal(DVec3::splat(mass * r.length_squared()))
            - DMat3::from_cols(r * r.x * mass, r * r.y * mass, r * r.z * mass);
        angular_momentum += r.cross((voxels[i].velocity.as_dvec3() - center_velocity) * mass);
    }
    let omega = inverse_inertia(inertia)
        .map(|inverse| inverse * angular_momentum)
        .unwrap_or(DVec3::ZERO);
    (inertia, omega)
}

fn inverse_inertia(inertia: DMat3) -> Option<DMat3> {
    let scale = inertia
        .x_axis
        .length()
        .max(inertia.y_axis.length())
        .max(inertia.z_axis.length());
    if !scale.is_finite() || scale <= 1e-30 {
        return None;
    }
    let regularized = inertia + DMat3::from_diagonal(DVec3::splat(scale * 1e-12));
    let determinant = regularized.determinant();
    (determinant.is_finite() && determinant.abs() > scale.powi(3) * 1e-36)
        .then(|| regularized.inverse())
        .filter(|inverse| inverse.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        documents::{
            BoneKeyframeDocument, EditorDocument, SkeletonDocument, VoxelBoneWeightDocument,
        },
        scene::{SceneBuilder, VoxelSpawn},
    };

    fn fixture() -> (VoxelModelDocument, crate::scene::Scene) {
        let scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([0, 1, 0], 1))
            .build()
            .unwrap();
        let mut model = VoxelModelDocument::from_scene("actuator-fixture", &scene);
        let mut skeleton =
            SkeletonDocument::with_root_bone("root", DVec3::new(0.005, 0.005, 0.005));
        skeleton
            .add_bone("limb", Some(0), DVec3::new(0.01, 0.0, 0.0))
            .unwrap();
        skeleton.voxel_weights = vec![
            VoxelBoneWeightDocument {
                voxel_index: 0,
                bone_index: 0,
                weight: 1.0,
            },
            VoxelBoneWeightDocument {
                voxel_index: 1,
                bone_index: 1,
                weight: 1.0,
            },
            VoxelBoneWeightDocument {
                voxel_index: 2,
                bone_index: 1,
                weight: 1.0,
            },
        ];
        let clip = skeleton.add_animation_clip("bend", 1.0).unwrap();
        for (time_s, angle) in [(0.0, 0.0), (1.0, 0.35)] {
            skeleton
                .set_keyframe(
                    clip,
                    1,
                    BoneKeyframeDocument {
                        time_s,
                        translation_m: DVec3::new(0.01, 0.0, 0.0),
                        rotation: Quat::from_rotation_z(angle),
                    },
                )
                .unwrap();
        }
        model.skeleton = Some(skeleton);
        model.validate().unwrap();
        (model, scene)
    }

    #[test]
    fn owned_actor_reuses_preparation_with_identical_pose_results() {
        let (model, _) = fixture();
        let reference = SkeletonAnimator::new(&model).unwrap();
        let actor = SkeletalActuator::new(model.clone(), 0..3, Quat::IDENTITY, 3).unwrap();
        for time_s in [0.0, 0.01, 0.25, 0.5, 1.0] {
            let expected = reference.evaluate(0, time_s).unwrap();
            let actual = actor
                .prepared_animation
                .evaluate(&actor.model, 0, time_s)
                .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.position_m, expected.position_m);
                assert_eq!(actual.rotation, expected.rotation);
            }
        }
    }

    #[test]
    fn bond_protection_is_editor_only_and_requires_valid_topology_and_range() {
        let (model, _) = fixture();
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();

        assert!(
            !actor.protects_bond(0, 1),
            "gameplay damage must remain enabled by default"
        );
        actor.set_editor_bond_protection(true);
        assert!(
            actor.protects_bond(0, 1),
            "editor protection should cover adjacent model REVs"
        );
        assert!(
            !actor.protects_bond(0, 3),
            "protection must not extend outside the actor range"
        );
        assert!(
            !actor.protects_bond(1, 2),
            "protection must not cover nonadjacent REVs"
        );

        actor.invalidated = true;
        assert!(
            !actor.protects_bond(0, 1),
            "invalid topology must disable protection"
        );
    }

    #[test]
    fn rejects_invalid_range_and_clip_errors_without_advancing() {
        let (model, _) = fixture();
        assert!(SkeletalActuator::new(model.clone(), 0..4, Quat::IDENTITY, 3).is_err());
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();
        actor.clip_index = 9;
        let (_, scene) = fixture();
        let mut out = vec![Vec3::ZERO; 3];
        assert!(!actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.01,
            &mut out
        ));
        assert_eq!(actor.elapsed_seconds(), 0.0);
        assert!(
            out.iter().all(|force| *force == Vec3::ZERO),
            "rest forces={out:?}; positions={:?}; rotations={:?}",
            scene.positions,
            scene.rotations
        );
    }

    #[test]
    fn rest_pose_is_load_free_and_fixed_voxels_are_not_actuated() {
        let (model, mut scene) = fixture();
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();
        actor.playback_speed = 0.0;
        let mut out = vec![Vec3::ZERO; 3];
        assert!(actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.01,
            &mut out
        ));
        assert!(
            out.iter().all(|force| force.length() < 1e-5),
            "rest forces={out:?}"
        );

        scene.voxels[0].kinematic = true;
        actor.playback_speed = 1.0;
        out.fill(Vec3::ZERO);
        actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.5,
            &mut out,
        );
        assert_eq!(out[0], Vec3::ZERO);
    }

    #[test]
    fn free_actor_motor_load_has_zero_resultant_force_and_torque_and_respects_cap() {
        let (model, scene) = fixture();
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();
        let mut out = vec![Vec3::ZERO; 3];
        assert!(actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.5,
            &mut out
        ));
        let com = scene
            .positions
            .iter()
            .enumerate()
            .fold(DVec3::ZERO, |sum, (i, p)| {
                sum + *p * f64::from(scene.voxels[i].mass)
            })
            / scene.voxels.iter().map(|v| f64::from(v.mass)).sum::<f64>();
        let force_sum = out
            .iter()
            .fold(DVec3::ZERO, |sum, force| sum + force.as_dvec3());
        let torque = out.iter().enumerate().fold(DVec3::ZERO, |sum, (i, force)| {
            sum + (scene.positions[i] - com).cross(force.as_dvec3())
        });
        assert!(force_sum.length() < 1e-6, "resultant={force_sum:?}");
        assert!(torque.length() < 1e-8, "torque={torque:?}");
        for (i, force) in out.iter().enumerate() {
            let material = &scene.properties[i];
            let cap = material
                .tensile_strength
                .min(material.tensile_strength_perp)
                * (CELL_M * CELL_M) as f32
                * FORCE_FRACTION_OF_STRENGTH as f32;
            assert!(force.length() <= cap + 1e-6);
        }
    }

    #[test]
    fn solver_counts_motor_work_through_applied_load_ledger() {
        let (model, mut scene) = fixture();
        scene.config.gravity = [0.0; 3];
        scene.config.fixed_dt = 0.01;
        scene
            .skeletal_actuators
            .push(SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap());
        scene.tick();
        assert!(
            scene.energy.external_work.abs() > 1e-20,
            "motor work was not entered in the external work ledger"
        );
    }

    #[test]
    fn rigidly_rotated_rest_pose_tracks_physical_rotation_without_motor_load() {
        let (model, mut scene) = fixture();
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();
        actor.playback_speed = 0.0;
        let com = scene
            .positions
            .iter()
            .fold(DVec3::ZERO, |sum, position| sum + *position)
            / scene.positions.len() as f64;
        let rotation_f32 = Quat::from_rotation_z(0.73);
        let rotation = DQuat::from_xyzw(
            f64::from(rotation_f32.x),
            f64::from(rotation_f32.y),
            f64::from(rotation_f32.z),
            f64::from(rotation_f32.w),
        )
        .normalize();
        for i in 0..scene.positions.len() {
            scene.positions[i] = com + rotation * (scene.positions[i] - com);
            scene.rotations[i] = rotation_f32;
        }
        let mut out = vec![Vec3::ZERO; scene.voxels.len()];
        assert!(actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.01,
            &mut out,
        ));
        assert!(
            out.iter().all(|force| force.length() < 1e-5),
            "rotated-rest forces={out:?}; positions={:?}; rotations={:?}",
            scene.positions,
            scene.rotations
        );
    }

    #[test]
    fn stalled_actor_keeps_last_safe_pose_and_invalid_topology_stays_disabled() {
        let (model, scene) = fixture();
        let mut actor = SkeletalActuator::new(model, 0..3, Quat::IDENTITY, 3).unwrap();
        let mut out = vec![Vec3::ZERO; 3];
        for _ in 0..2 {
            assert!(actor.accumulate_forces(
                &scene.positions,
                &scene.voxels,
                &scene.rotations,
                &scene.properties,
                0.2,
                &mut out,
            ));
        }
        actor.stall();
        let safe_time = actor.elapsed_seconds();
        out.fill(Vec3::ZERO);
        assert!(actor.accumulate_forces(
            &scene.positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.1,
            &mut out,
        ));
        assert_eq!(actor.elapsed_seconds(), safe_time);
        assert!(out.iter().any(|force| force.length() > 0.0));

        let mut changed_positions = scene.positions[..2].to_vec();
        let mut changed_out = vec![Vec3::ZERO; 2];
        assert!(!actor.accumulate_forces(
            &changed_positions,
            &scene.voxels[..2],
            &scene.rotations[..2],
            &scene.properties[..2],
            0.1,
            &mut changed_out,
        ));
        changed_positions.extend_from_slice(&scene.positions[2..]);
        assert!(!actor.matches_scene_topology(3));
        let mut restored_out = vec![Vec3::ZERO; 3];
        assert!(!actor.accumulate_forces(
            &changed_positions,
            &scene.voxels,
            &scene.rotations,
            &scene.properties,
            0.1,
            &mut restored_out,
        ));
        assert!(restored_out.iter().all(|force| *force == Vec3::ZERO));
    }
}
