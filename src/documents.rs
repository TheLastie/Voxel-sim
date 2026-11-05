//! Versioned, compact files used by the desktop editor.
//!
//! `.vxm` stores a reusable voxel asset and the per-REV state it owns. A `.vxscene`
//! is currently an authored scene document; a lossless running-physics checkpoint
//! is deliberately a separate, not-yet-implemented extension of this schema.

use crate::{
    material::{FractureMode, Material},
    scene::{Scene, SceneConfig},
    skeletal_actuator::SkeletalActuator,
    scene_welding::SceneSimulationSettings,
    voxel::{Composition, Microstructure, Voxel},
};
use bincode::Options;
use glam::{DQuat, DVec3, DVec4, IVec3, Quat, Vec3};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &[u8; 4] = b"VXED";
const FORMAT_VERSION: u16 = 6;
const HEADER_BYTES: usize = 28;
const MAX_PAYLOAD_BYTES: u64 = 512 * 1024 * 1024;
const MAX_DOCUMENT_ITEMS: u64 = 8_000_000;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DocumentKind {
    Model = 1,
    Scene = 2,
    Project = 3,
}

impl DocumentKind {
    fn from_byte(value: u8) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Model),
            2 => Ok(Self::Scene),
            3 => Ok(Self::Project),
            _ => Err(format!("unknown voxel document type {value}")),
        }
    }

    fn extension(self) -> &'static str {
        match self {
            Self::Model => "vxm",
            Self::Scene => "vxscene",
            Self::Project => "vxproj",
        }
    }
}

/// A stable, versioned document family. Runtime caches never belong in these DTOs.
pub trait EditorDocument: Serialize + DeserializeOwned {
    const KIND: DocumentKind;

    /// Number of top-level records, checked before the payload is decoded.
    fn record_count(&self) -> usize;

    fn validate(&self) -> Result<(), String>;

    /// Decode the current schema or an explicitly supported earlier schema.
    fn decode_payload(version: u16, payload: &[u8]) -> Result<Self, String> {
        if !(1..=FORMAT_VERSION).contains(&version) {
            return Err(format!(
                "unsupported native document schema version {version}"
            ));
        }
        codec()
            .deserialize(payload)
            .map_err(|error| format!("decode native document: {error}"))
    }
}

/// One authored or saved REV. Its composition indices refer to the material
/// snapshot stored alongside the model, not to a process-global table order.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVoxel {
    pub position_m: DVec3,
    pub rest_position_m: DVec3,
    pub state: Voxel,
}

/// Reusable model asset. It preserves each REV's constitutive and thermal state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelModelDocument {
    pub name: String,
    pub cell_size_m: f64,
    pub materials: Vec<Material>,
    pub voxels: Vec<ModelVoxel>,
    #[serde(default)]
    pub groups: Vec<VoxelGroupDocument>,
    #[serde(default)]
    pub motion_constraints: Vec<MotionConstraintDocument>,
    #[serde(default)]
    pub skeleton: Option<SkeletonDocument>,
}

/// Named sets are an editing and constraint-selection aid; they do not merge
/// the REV state into rigid bodies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoxelGroupDocument {
    pub name: String,
    pub voxels: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConstraintPartDocument {
    Voxel(u32),
    Group(u32),
}

/// Endpoints refer to model-local REVs or named groups. Anchors are in model
/// local metres, so a scene instance transform is applied before solving.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotionConstraintDocument {
    pub name: String,
    pub a: ConstraintPartDocument,
    pub b: ConstraintPartDocument,
    pub anchor_a_m: DVec3,
    pub anchor_b_m: DVec3,
    pub kind: MotionConstraintKindDocument,
    pub linear_compliance_m_per_n: f64,
    pub angular_compliance_rad_per_nm: f64,
    pub linear_damping_n_s_per_m: f64,
    pub angular_damping_nm_s_per_rad: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum MotionConstraintKindDocument {
    Fixed,
    Hinge {
        axis_a_local: Vec3,
        axis_b_local: Vec3,
    },
    Ball,
    Slider {
        axis_a_local: Vec3,
        axis_b_local: Vec3,
    },
}

/// A skeleton is optional on any voxel model, including plants, fungi, and creatures.
/// Bones are topologically ordered: a parent always precedes its child.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkeletonDocument {
    pub bones: Vec<BoneDocument>,
    pub voxel_weights: Vec<VoxelBoneWeightDocument>,
    pub clips: Vec<AnimationClipDocument>,
}

/// A visual transform for one REV. It does not modify physical state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatedVoxelPose {
    pub position_m: DVec3,
    /// Weighted skinning rotation relative to the model's rest pose.
    pub rotation: DQuat,
}

/// GPU-friendly rest-to-pose transform for one bone. Applying it to a rest
/// position gives `translation_m + rotation * rest_position`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneSkinTransform {
    pub translation_m: DVec3,
    pub rotation: DQuat,
}

/// Material-derived limits for an animated child bone. Stretch limits combine
/// the material's elastic yield strain and plastic failure allowance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoneMaterialLimits {
    pub max_joint_rotation_rad: f64,
    pub max_extension_strain: f64,
    pub max_compression_strain: f64,
}

impl BoneMaterialLimits {
    fn unrestricted() -> Self {
        Self {
            max_joint_rotation_rad: std::f64::consts::PI,
            max_extension_strain: 0.5,
            max_compression_strain: 0.5,
        }
    }
}

#[derive(Clone, Copy)]
struct BoneTransform {
    translation_m: DVec3,
    rotation: DQuat,
}

/// Reusable evaluator for a universal voxel skeleton, including plants and fungi.
/// Prepare it once and evaluate poses without revalidating or regrouping weights
/// on every rendered frame.
pub struct SkeletonAnimator<'a> {
    model: &'a VoxelModelDocument,
    prepared: PreparedSkeletonAnimation,
}

/// Preparation for one immutable model. Kept separately so physical actors can
/// own their model and reuse the evaluator without a self-referencing borrow.
pub(crate) struct PreparedSkeletonAnimation {
    rest_bones_global: Vec<BoneTransform>,
    bone_material_limits: Vec<BoneMaterialLimits>,
    weight_offsets: Vec<usize>,
    voxel_weights: Vec<(usize, f64)>,
}

impl<'a> SkeletonAnimator<'a> {
    pub fn new(model: &'a VoxelModelDocument) -> Result<Self, String> {
        model.validate()?;
        let skeleton = model
            .skeleton
            .as_ref()
            .ok_or_else(|| "voxel model has no skeleton".to_owned())?;
        let rest_local: Vec<_> = skeleton
            .bones
            .iter()
            .map(|bone| BoneTransform {
                translation_m: bone.rest_translation_m,
                rotation: to_dquat(bone.rest_rotation),
            })
            .collect();
        let rest_bones_global = compose_bone_transforms(&skeleton.bones, &rest_local);
        let bone_material_limits = model
            .skeleton_material_limits()
            .expect("validated model has a skeleton");
        let mut voxel_weights: Vec<_> = skeleton
            .voxel_weights
            .iter()
            .map(|binding| {
                (
                    binding.voxel_index as usize,
                    binding.bone_index as usize,
                    f64::from(binding.weight),
                )
            })
            .collect();
        voxel_weights.sort_unstable_by_key(|binding| binding.0);
        let mut weight_offsets = vec![0; model.voxels.len() + 1];
        for &(voxel_index, _, _) in &voxel_weights {
            weight_offsets[voxel_index + 1] += 1;
        }
        for index in 1..weight_offsets.len() {
            weight_offsets[index] += weight_offsets[index - 1];
        }
        let voxel_weights = voxel_weights
            .into_iter()
            .map(|(_, bone_index, weight)| (bone_index, weight))
            .collect();

        Ok(Self {
            model,
            prepared: PreparedSkeletonAnimation {
                rest_bones_global,
                bone_material_limits,
                weight_offsets,
                voxel_weights,
            },
        })
    }

    pub(crate) fn into_prepared(self) -> PreparedSkeletonAnimation {
        self.prepared
    }

    /// Evaluate one clip at local time in seconds. Time is clamped to the clip
    /// interval; the caller controls playback and looping. Keyframe transforms
    /// are bone-local poses that replace the corresponding rest-local transform.
    pub fn evaluate(
        &self,
        clip_index: usize,
        time_s: f64,
    ) -> Result<Vec<AnimatedVoxelPose>, String> {
        self.prepared.evaluate(self.model, clip_index, time_s)
    }
}

impl PreparedSkeletonAnimation {
    /// Only use with the immutable model passed to `SkeletonAnimator::new`.
    pub(crate) fn evaluate(
        &self,
        model: &VoxelModelDocument,
        clip_index: usize,
        time_s: f64,
    ) -> Result<Vec<AnimatedVoxelPose>, String> {
        if !time_s.is_finite() {
            return Err("animation time must be finite".into());
        }
        let skeleton = model.skeleton.as_ref().expect("validated skeleton");
        let clip = skeleton
            .clips
            .get(clip_index)
            .ok_or_else(|| "animation clip index is out of range".to_owned())?;
        let time_s = time_s.clamp(0.0, clip.duration_s);
        let local_bones = sample_local_bones(
            &skeleton.bones,
            &skeleton.clips,
            clip_index,
            time_s,
            Some(&self.bone_material_limits),
        )?;
        let posed_bones_global = compose_bone_transforms(&skeleton.bones, &local_bones);

        let mut poses = Vec::with_capacity(model.voxels.len());
        for (voxel_index, voxel) in model.voxels.iter().enumerate() {
            let binding_start = self.weight_offsets[voxel_index];
            let binding_end = self.weight_offsets[voxel_index + 1];
            let bindings = &self.voxel_weights[binding_start..binding_end];
            if bindings.is_empty() {
                poses.push(AnimatedVoxelPose {
                    position_m: voxel.rest_position_m,
                    rotation: DQuat::IDENTITY,
                });
                continue;
            }

            let mut position_m = DVec3::ZERO;
            let mut rotation_sum = DVec4::ZERO;
            let mut reference_rotation = None;
            for &(bone_index, weight) in bindings {
                let rest = self.rest_bones_global[bone_index];
                let posed = posed_bones_global[bone_index];
                let delta_rotation = (posed.rotation * rest.rotation.conjugate()).normalize();
                let local_position =
                    rest.rotation.conjugate() * (voxel.rest_position_m - rest.translation_m);
                position_m += (posed.translation_m + posed.rotation * local_position) * weight;

                let reference = *reference_rotation.get_or_insert(delta_rotation);
                let aligned = if reference.dot(delta_rotation) < 0.0 {
                    -delta_rotation
                } else {
                    delta_rotation
                };
                rotation_sum += DVec4::new(aligned.x, aligned.y, aligned.z, aligned.w) * weight;
            }
            let rotation = if rotation_sum.length_squared() > 1e-24 {
                DQuat::from_xyzw(
                    rotation_sum.x,
                    rotation_sum.y,
                    rotation_sum.z,
                    rotation_sum.w,
                )
                .normalize()
            } else {
                DQuat::IDENTITY
            };
            poses.push(AnimatedVoxelPose {
                position_m,
                rotation,
            });
        }
        Ok(poses)
    }
}

fn to_dquat(rotation: Quat) -> DQuat {
    DQuat::from_xyzw(
        f64::from(rotation.x),
        f64::from(rotation.y),
        f64::from(rotation.z),
        f64::from(rotation.w),
    )
    .normalize()
}

fn compose_bone_transforms(bones: &[BoneDocument], local: &[BoneTransform]) -> Vec<BoneTransform> {
    let mut global: Vec<BoneTransform> = Vec::with_capacity(bones.len());
    for (index, bone) in bones.iter().enumerate() {
        let transform = local[index];
        let world = match bone.parent {
            Some(parent) => {
                let parent = global[parent as usize];
                BoneTransform {
                    translation_m: parent.translation_m + parent.rotation * transform.translation_m,
                    rotation: (parent.rotation * transform.rotation).normalize(),
                }
            }
            None => transform,
        };
        global.push(world);
    }
    global
}

fn clamp_bone_pose(
    bone: &BoneDocument,
    mut pose: BoneTransform,
    limits: BoneMaterialLimits,
) -> BoneTransform {
    if bone.parent.is_none() {
        return pose;
    }

    let rest_rotation = to_dquat(bone.rest_rotation);
    let relative_rotation = (rest_rotation.conjugate() * pose.rotation).normalize();
    let relative_rotation = if relative_rotation.w < 0.0 {
        -relative_rotation
    } else {
        relative_rotation
    };
    let angle = 2.0 * relative_rotation.w.clamp(-1.0, 1.0).acos();
    if angle > limits.max_joint_rotation_rad && angle > 1.0e-12 {
        let fraction = (limits.max_joint_rotation_rad / angle).clamp(0.0, 1.0);
        pose.rotation =
            (rest_rotation * DQuat::IDENTITY.slerp(relative_rotation, fraction)).normalize();
    }

    let rest_length = bone.rest_translation_m.length();
    if rest_length > 1.0e-12 {
        let pose_length = pose.translation_m.length();
        let direction = if pose_length > 1.0e-12 {
            pose.translation_m / pose_length
        } else {
            bone.rest_translation_m / rest_length
        };
        let min_length = rest_length * (1.0 - limits.max_compression_strain);
        let max_length = rest_length * (1.0 + limits.max_extension_strain);
        pose.translation_m = direction * pose_length.clamp(min_length, max_length);
    }
    pose
}

fn sample_local_bones(
    bones: &[BoneDocument],
    clips: &[AnimationClipDocument],
    clip_index: usize,
    time_s: f64,
    material_limits: Option<&[BoneMaterialLimits]>,
) -> Result<Vec<BoneTransform>, String> {
    if !time_s.is_finite() {
        return Err("animation time must be finite".into());
    }
    if material_limits.is_some_and(|limits| limits.len() != bones.len()) {
        return Err("bone material limits do not match the skeleton".into());
    }
    let clip = clips
        .get(clip_index)
        .ok_or_else(|| "animation clip index is out of range".to_owned())?;
    let time_s = time_s.clamp(0.0, clip.duration_s);
    let mut local: Vec<_> = bones
        .iter()
        .map(|bone| BoneTransform {
            translation_m: bone.rest_translation_m,
            rotation: to_dquat(bone.rest_rotation),
        })
        .collect();
    for track in &clip.tracks {
        if track.keyframes.is_empty() || track.bone_index as usize >= bones.len() {
            return Err("animation track has an invalid bone or no keyframes".into());
        }
        let bone_index = track.bone_index as usize;
        let pose = sample_bone_track(track, time_s);
        local[bone_index] = material_limits.map_or(pose, |limits| {
            clamp_bone_pose(&bones[bone_index], pose, limits[bone_index])
        });
    }
    Ok(local)
}

fn sample_bone_track(track: &BoneTrackDocument, time_s: f64) -> BoneTransform {
    let frames = &track.keyframes;
    let first = frames[0];
    if time_s <= first.time_s {
        return BoneTransform {
            translation_m: first.translation_m,
            rotation: to_dquat(first.rotation),
        };
    }
    let last = frames[frames.len() - 1];
    if time_s >= last.time_s {
        return BoneTransform {
            translation_m: last.translation_m,
            rotation: to_dquat(last.rotation),
        };
    }
    let upper = frames.partition_point(|frame| frame.time_s < time_s);
    let a = frames[upper - 1];
    let b = frames[upper];
    let blend = (time_s - a.time_s) / (b.time_s - a.time_s);
    let rotation_a = to_dquat(a.rotation);
    let rotation_b = to_dquat(b.rotation);
    BoneTransform {
        translation_m: a.translation_m.lerp(b.translation_m, blend),
        rotation: rotation_a.slerp(rotation_b, blend).normalize(),
    }
}

/// Generate normalized, sparse voxel-to-bone weights using shortest paths over
/// the occupied REV lattice. This is a fast, topology-aware approximation of
/// heat-based automatic weights: it follows connected voxel geometry instead
/// of letting a nearby bone influence REVs through empty space. It is not a
/// byte-for-byte reproduction of Blender's heat solver.
///
/// At most four influences are emitted per REV. The calculation is intended
/// for a background worker for large models; its temporary graph state is
/// linear in the number of REVs.
pub fn generate_auto_bone_weights(
    model: &VoxelModelDocument,
    max_influences: usize,
) -> Result<Vec<VoxelBoneWeightDocument>, String> {
    model.validate()?;
    let skeleton = model
        .skeleton
        .as_ref()
        .ok_or_else(|| "create at least one bone before generating weights".to_owned())?;
    let rest_positions: Vec<_> = model
        .voxels
        .iter()
        .map(|voxel| voxel.rest_position_m)
        .collect();
    generate_auto_bone_weights_from_rest_positions(
        &rest_positions,
        model.cell_size_m,
        skeleton,
        max_influences,
    )
}

/// Worker-friendly entry point: callers can copy only the compact rest-position
/// array and skeleton, without cloning every REV's thermal and material state.
pub fn generate_auto_bone_weights_from_rest_positions(
    rest_positions: &[DVec3],
    cell_size_m: f64,
    skeleton: &SkeletonDocument,
    max_influences: usize,
) -> Result<Vec<VoxelBoneWeightDocument>, String> {
    if !(1..=4).contains(&max_influences) {
        return Err("automatic weights support one to four bone influences per REV".into());
    }
    if rest_positions.is_empty() {
        return Err("cannot bind an empty voxel model".into());
    }
    if !cell_size_m.is_finite()
        || cell_size_m <= 0.0
        || rest_positions.iter().any(|p| !p.is_finite())
    {
        return Err("model lattice has an invalid cell size or REV position".into());
    }
    if skeleton.bones.is_empty()
        || skeleton.bones.iter().enumerate().any(|(index, bone)| {
            !bone.rest_translation_m.is_finite()
                || !valid_rotation(bone.rest_rotation)
                || match bone.parent {
                    None => index != 0,
                    Some(parent) => parent as usize >= index,
                }
        })
    {
        return Err("skeleton bones must be non-empty and topologically ordered".into());
    }

    let local_bones: Vec<_> = skeleton
        .bones
        .iter()
        .map(|bone| BoneTransform {
            translation_m: bone.rest_translation_m,
            rotation: to_dquat(bone.rest_rotation),
        })
        .collect();
    let global_bones = compose_bone_transforms(&skeleton.bones, &local_bones);
    let anchors: Vec<_> = skeleton
        .bones
        .iter()
        .enumerate()
        .map(|(index, bone)| {
            let endpoint = global_bones[index].translation_m;
            bone.parent.map_or(endpoint, |parent| {
                (global_bones[parent as usize].translation_m + endpoint) * 0.5
            })
        })
        .collect();

    let cell_size = cell_size_m;
    let coordinates: Vec<_> = rest_positions
        .iter()
        .map(|position| voxel_lattice_coordinate(*position, cell_size))
        .collect();
    let mut first_at_coordinate =
        FxHashMap::with_capacity_and_hasher(rest_positions.len(), Default::default());
    let mut duplicate_of = vec![None; rest_positions.len()];
    for (index, coordinate) in coordinates.iter().copied().enumerate() {
        if let Some(&first) = first_at_coordinate.get(&coordinate) {
            duplicate_of[index] = Some(first);
        } else {
            first_at_coordinate.insert(coordinate, index);
        }
    }

    let mut best = vec![[BoneInfluence::EMPTY; 4]; rest_positions.len()];
    let mut pending = VecDeque::new();
    for (bone_index, anchor) in anchors.iter().copied().enumerate() {
        let seed = closest_model_voxel(anchor, &first_at_coordinate, rest_positions, cell_size);
        if insert_bone_influence(&mut best[seed], 0, bone_index as u32, max_influences) {
            pending.push_back(BoneWaveEntry {
                voxel_index: seed as u32,
                bone_index: bone_index as u32,
                distance: 0,
            });
        }
    }

    const NEIGHBORS: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];
    while let Some(entry) = pending.pop_front() {
        let index = entry.voxel_index as usize;
        let bone_index = entry.bone_index;
        let distance = entry.distance;
        if !best[index][..max_influences]
            .iter()
            .any(|entry| entry.bone_index == bone_index && entry.distance == distance)
        {
            continue;
        }
        let coordinate = coordinates[index];
        let next_distance = distance.saturating_add(1);
        for offset in NEIGHBORS {
            let neighbor =
                std::array::from_fn(|axis| coordinate[axis].saturating_add(offset[axis]));
            let Some(&neighbor_index) = first_at_coordinate.get(&neighbor) else {
                continue;
            };
            if insert_bone_influence(
                &mut best[neighbor_index],
                next_distance,
                bone_index,
                max_influences,
            ) {
                pending.push_back(BoneWaveEntry {
                    voxel_index: neighbor_index as u32,
                    bone_index,
                    distance: next_distance,
                });
            }
        }
    }

    // Disconnected voxel islands cannot receive a graph wave from any bone.
    // Bind each such island's REV to its nearest bone anchor rather than leave
    // it undeformed or invent a connection through empty space.
    best.par_iter_mut()
        .enumerate()
        .for_each(|(index, influences)| {
            if duplicate_of[index].is_some() || influences[0].bone_index != u32::MAX {
                return;
            }
            let point = rest_positions[index];
            let nearest = anchors
                .iter()
                .enumerate()
                .min_by(|(a_index, a), (b_index, b)| {
                    point
                        .distance_squared(**a)
                        .total_cmp(&point.distance_squared(**b))
                        .then_with(|| a_index.cmp(b_index))
                })
                .map(|(index, _)| index as u32)
                .expect("validated skeleton has at least one bone");
            influences[0] = BoneInfluence {
                bone_index: nearest,
                distance: 0,
            };
        });

    for index in 0..rest_positions.len() {
        if let Some(first) = duplicate_of[index] {
            best[index] = best[first];
        }
    }

    let mut weights = Vec::with_capacity(rest_positions.len() * max_influences);
    for (voxel_index, influences) in best.iter().enumerate() {
        let active_count = influences[..max_influences]
            .iter()
            .take_while(|entry| entry.bone_index != u32::MAX)
            .count();
        if active_count == 0 {
            return Err(format!(
                "automatic weights did not reach REV {}",
                voxel_index + 1
            ));
        }
        let total: f64 = influences[..active_count]
            .iter()
            .map(|entry| 1.0 / (f64::from(entry.distance) + 0.5).powi(2))
            .sum();
        for entry in &influences[..active_count] {
            let raw_weight = 1.0 / (f64::from(entry.distance) + 0.5).powi(2);
            weights.push(VoxelBoneWeightDocument {
                voxel_index: voxel_index as u32,
                bone_index: entry.bone_index,
                weight: (raw_weight / total) as f32,
            });
        }
    }
    Ok(weights)
}

#[derive(Clone, Copy)]
struct BoneInfluence {
    bone_index: u32,
    distance: u32,
}

#[derive(Clone, Copy)]
struct BoneWaveEntry {
    voxel_index: u32,
    bone_index: u32,
    distance: u32,
}

impl BoneInfluence {
    const EMPTY: Self = Self {
        bone_index: u32::MAX,
        distance: u32::MAX,
    };
}

pub(crate) fn voxel_lattice_coordinate(position_m: DVec3, cell_size_m: f64) -> [i32; 3] {
    ((position_m / cell_size_m - DVec3::splat(0.5)).round())
        .as_ivec3()
        .to_array()
}

fn closest_model_voxel(
    point_m: DVec3,
    first_at_coordinate: &FxHashMap<[i32; 3], usize>,
    rest_positions: &[DVec3],
    cell_size_m: f64,
) -> usize {
    let center = voxel_lattice_coordinate(point_m, cell_size_m);
    for radius in 0..=3 {
        let mut closest = None;
        let mut closest_distance = f64::INFINITY;
        for x in -radius..=radius {
            for y in -radius..=radius {
                for z in -radius..=radius {
                    let coordinate = [center[0] + x, center[1] + y, center[2] + z];
                    let Some(&index) = first_at_coordinate.get(&coordinate) else {
                        continue;
                    };
                    let distance = rest_positions[index].distance_squared(point_m);
                    if distance < closest_distance {
                        closest = Some(index);
                        closest_distance = distance;
                    }
                }
            }
        }
        if let Some(index) = closest {
            return index;
        }
    }
    rest_positions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.distance_squared(point_m)
                .total_cmp(&b.distance_squared(point_m))
        })
        .map(|(index, _)| index)
        .expect("non-empty model checked by caller")
}

fn insert_bone_influence(
    influences: &mut [BoneInfluence; 4],
    distance: u32,
    bone_index: u32,
    max_influences: usize,
) -> bool {
    let mut count = influences[..max_influences]
        .iter()
        .take_while(|entry| entry.bone_index != u32::MAX)
        .count();
    if let Some(existing) = influences[..count]
        .iter()
        .position(|entry| entry.bone_index == bone_index)
    {
        if influences[existing].distance <= distance {
            return false;
        }
        for index in existing..count - 1 {
            influences[index] = influences[index + 1];
        }
        count -= 1;
        influences[count] = BoneInfluence::EMPTY;
    } else if count == max_influences
        && (distance, bone_index)
            >= (
                influences[count - 1].distance,
                influences[count - 1].bone_index,
            )
    {
        return false;
    }

    let insert_at = influences[..count]
        .iter()
        .position(|entry| (distance, bone_index) < (entry.distance, entry.bone_index))
        .unwrap_or(count);
    if insert_at >= max_influences {
        return false;
    }
    for index in (insert_at + 1..=count.min(max_influences - 1)).rev() {
        influences[index] = influences[index - 1];
    }
    influences[insert_at] = BoneInfluence {
        bone_index,
        distance,
    };
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoneDocument {
    pub name: String,
    pub parent: Option<u32>,
    pub rest_translation_m: DVec3,
    pub rest_rotation: Quat,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct VoxelBoneWeightDocument {
    pub voxel_index: u32,
    pub bone_index: u32,
    pub weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationClipDocument {
    pub name: String,
    pub duration_s: f64,
    pub tracks: Vec<BoneTrackDocument>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoneTrackDocument {
    pub bone_index: u32,
    pub keyframes: Vec<BoneKeyframeDocument>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct BoneKeyframeDocument {
    pub time_s: f64,
    pub translation_m: DVec3,
    pub rotation: Quat,
}

impl SkeletonDocument {
    /// Evaluate compact bone transforms for viewport skinning. Work is
    /// proportional to the number of bones and keyframes, not the REV count.
    pub fn bone_skin_transforms_at(
        &self,
        clip_index: usize,
        time_s: f64,
    ) -> Result<Vec<BoneSkinTransform>, String> {
        self.bone_skin_transforms_at_with_material_limits(clip_index, time_s, None)
    }

    /// Evaluate the clip after applying cached material-specific joint and
    /// segment-stretch limits. The limits must be precomputed when the model's
    /// material table or skin weights change, not inside the frame loop.
    pub fn bone_skin_transforms_at_with_material_limits(
        &self,
        clip_index: usize,
        time_s: f64,
        material_limits: Option<&[BoneMaterialLimits]>,
    ) -> Result<Vec<BoneSkinTransform>, String> {
        let rest_local: Vec<_> = self
            .bones
            .iter()
            .map(|bone| BoneTransform {
                translation_m: bone.rest_translation_m,
                rotation: to_dquat(bone.rest_rotation),
            })
            .collect();
        let rest_global = compose_bone_transforms(&self.bones, &rest_local);
        let posed_local = sample_local_bones(
            &self.bones,
            &self.clips,
            clip_index,
            time_s,
            material_limits,
        )?;
        let posed_global = compose_bone_transforms(&self.bones, &posed_local);
        Ok(rest_global
            .into_iter()
            .zip(posed_global)
            .map(|(rest, posed)| {
                let rotation = (posed.rotation * rest.rotation.conjugate()).normalize();
                BoneSkinTransform {
                    translation_m: posed.translation_m - rotation * rest.translation_m,
                    rotation,
                }
            })
            .collect())
    }

    pub fn bone_local_poses_at_with_material_limits(
        &self,
        clip_index: usize,
        time_s: f64,
        material_limits: &[BoneMaterialLimits],
    ) -> Result<Vec<BoneKeyframeDocument>, String> {
        let local = sample_local_bones(
            &self.bones,
            &self.clips,
            clip_index,
            time_s,
            Some(material_limits),
        )?;
        Ok(local
            .into_iter()
            .map(|pose| BoneKeyframeDocument {
                time_s,
                translation_m: pose.translation_m,
                rotation: Quat::from_xyzw(
                    pose.rotation.x as f32,
                    pose.rotation.y as f32,
                    pose.rotation.z as f32,
                    pose.rotation.w as f32,
                )
                .normalize(),
            })
            .collect())
    }

    pub fn with_root_bone(name: impl Into<String>, rest_translation_m: DVec3) -> Self {
        Self {
            bones: vec![BoneDocument {
                name: name.into(),
                parent: None,
                rest_translation_m,
                rest_rotation: Quat::IDENTITY,
            }],
            voxel_weights: Vec::new(),
            clips: Vec::new(),
        }
    }

    pub fn add_bone(
        &mut self,
        name: impl Into<String>,
        parent: Option<usize>,
        rest_translation_m: DVec3,
    ) -> Result<usize, String> {
        let name = name.into();
        validate_name(&name, "bone")?;
        if self.bones.iter().any(|bone| bone.name == name) {
            return Err(format!("a bone named {name:?} already exists"));
        }
        if parent.is_some_and(|parent| parent >= self.bones.len())
            || (self.bones.is_empty() && parent.is_some())
            || (!self.bones.is_empty() && parent.is_none())
            || self.bones.len() >= u32::MAX as usize
            || !rest_translation_m.is_finite()
        {
            return Err("new bone has an invalid parent or transform".into());
        }
        let index = self.bones.len();
        self.bones.push(BoneDocument {
            name,
            parent: parent.map(|parent| parent as u32),
            rest_translation_m,
            rest_rotation: Quat::IDENTITY,
        });
        Ok(index)
    }

    pub fn add_animation_clip(
        &mut self,
        name: impl Into<String>,
        duration_s: f64,
    ) -> Result<usize, String> {
        let name = name.into();
        validate_name(&name, "animation clip")?;
        if self.clips.iter().any(|clip| clip.name == name)
            || !duration_s.is_finite()
            || duration_s <= 0.0
        {
            return Err("animation clip has a duplicate name or invalid duration".into());
        }
        let index = self.clips.len();
        self.clips.push(AnimationClipDocument {
            name,
            duration_s,
            tracks: Vec::new(),
        });
        Ok(index)
    }

    /// Insert or replace a bone pose at a clip-local time, keeping keyframes
    /// strictly ordered for deterministic interpolation.
    pub fn set_keyframe(
        &mut self,
        clip_index: usize,
        bone_index: usize,
        keyframe: BoneKeyframeDocument,
    ) -> Result<(), String> {
        let clip = self
            .clips
            .get_mut(clip_index)
            .ok_or_else(|| "animation clip index is out of range".to_owned())?;
        if bone_index >= self.bones.len()
            || !keyframe.time_s.is_finite()
            || keyframe.time_s < 0.0
            || keyframe.time_s > clip.duration_s
            || !keyframe.translation_m.is_finite()
            || !valid_rotation(keyframe.rotation)
        {
            return Err("keyframe has an invalid bone, time, or transform".into());
        }
        let bone_index = bone_index as u32;
        let track_index = if let Some(index) = clip
            .tracks
            .iter()
            .position(|track| track.bone_index == bone_index)
        {
            index
        } else {
            clip.tracks.push(BoneTrackDocument {
                bone_index,
                keyframes: Vec::new(),
            });
            clip.tracks.len() - 1
        };
        let frames = &mut clip.tracks[track_index].keyframes;
        let index = frames.partition_point(|frame| frame.time_s < keyframe.time_s);
        if frames
            .get(index)
            .is_some_and(|frame| frame.time_s == keyframe.time_s)
        {
            frames[index] = keyframe;
        } else {
            frames.insert(index, keyframe);
        }
        Ok(())
    }

    /// Sample a bone's local pose at clip time, falling back to its rest pose
    /// when the track has no keys yet. This lets the editor prepare a pose
    /// without creating a hidden keyframe as a side effect.
    pub fn sample_bone_local_pose(
        &self,
        clip_index: usize,
        bone_index: usize,
        time_s: f64,
    ) -> Result<BoneKeyframeDocument, String> {
        if bone_index >= self.bones.len() || !time_s.is_finite() {
            return Err("bone pose sample has an invalid bone or time".into());
        }
        let clip = self
            .clips
            .get(clip_index)
            .ok_or_else(|| "animation clip index is out of range".to_owned())?;
        let time_s = time_s.clamp(0.0, clip.duration_s);
        let track = clip
            .tracks
            .iter()
            .find(|track| track.bone_index as usize == bone_index);
        let pose = if let Some(track) = track {
            if track.keyframes.is_empty() {
                return Err("animation track has no keyframes".into());
            }
            sample_bone_track(track, time_s)
        } else {
            let bone = &self.bones[bone_index];
            BoneTransform {
                translation_m: bone.rest_translation_m,
                rotation: to_dquat(bone.rest_rotation),
            }
        };
        Ok(BoneKeyframeDocument {
            time_s,
            translation_m: pose.translation_m,
            rotation: Quat::from_xyzw(
                pose.rotation.x as f32,
                pose.rotation.y as f32,
                pose.rotation.z as f32,
                pose.rotation.w as f32,
            )
            .normalize(),
        })
    }

    /// Return the current global joint positions for the viewport rig overlay.
    /// The work is proportional to bone count and independent of model REV count.
    pub fn bone_global_positions_at(
        &self,
        clip_index: usize,
        time_s: f64,
    ) -> Result<Vec<DVec3>, String> {
        self.bone_global_positions_at_with_material_limits(clip_index, time_s, None)
    }

    pub fn bone_global_positions_at_with_material_limits(
        &self,
        clip_index: usize,
        time_s: f64,
        material_limits: Option<&[BoneMaterialLimits]>,
    ) -> Result<Vec<DVec3>, String> {
        let local = sample_local_bones(
            &self.bones,
            &self.clips,
            clip_index,
            time_s,
            material_limits,
        )?;
        Ok(compose_bone_transforms(&self.bones, &local)
            .into_iter()
            .map(|transform| transform.translation_m)
            .collect())
    }

    /// Adjust one REV's influence for a selected bone and renormalize its other
    /// influences. Automatic bindings are stored in voxel order, so a paint
    /// stroke touches only this REV's small influence range.
    pub fn adjust_voxel_bone_weight(
        &mut self,
        voxel_count: usize,
        voxel_index: usize,
        bone_index: usize,
        delta: f32,
    ) -> Result<f32, String> {
        if voxel_index >= voxel_count
            || bone_index >= self.bones.len()
            || !delta.is_finite()
            || delta == 0.0
        {
            return Err("weight edit has an invalid REV, bone, or adjustment".into());
        }

        let bindings = &mut self.voxel_weights;
        // Older or hand-authored documents may list valid influences in any
        // order. Normalize once on the first edit so subsequent weight-brush
        // operations can find a REV's small range with binary searches.
        if bindings
            .windows(2)
            .any(|pair| pair[0].voxel_index > pair[1].voxel_index)
        {
            bindings.sort_unstable_by_key(|binding| binding.voxel_index);
        }
        let start = bindings.partition_point(|binding| binding.voxel_index < voxel_index as u32);
        let end = bindings.partition_point(|binding| binding.voxel_index <= voxel_index as u32);
        let selected = bindings[start..end]
            .iter()
            .position(|binding| binding.bone_index == bone_index as u32)
            .map(|offset| start + offset);

        if let Some(selected) = selected {
            if end - start < 2 {
                return Err(
                    "add another bone influence before adjusting this REV's only weight".into(),
                );
            }
            let other_sum: f64 = bindings[start..end]
                .iter()
                .enumerate()
                .filter(|(index, _)| start + index != selected)
                .map(|(_, binding)| f64::from(binding.weight))
                .sum();
            if !other_sum.is_finite() || other_sum <= 0.0 {
                return Err("REV has no other valid bone influences to receive the weight".into());
            }
            let new_weight = (bindings[selected].weight + delta).clamp(0.001, 0.999);
            let remaining = 1.0 - f64::from(new_weight);
            let other_indices: Vec<_> = (start..end).filter(|&index| index != selected).collect();
            let mut assigned = 0.0_f64;
            for (position, index) in other_indices.iter().copied().enumerate() {
                let value = if position + 1 == other_indices.len() {
                    remaining - assigned
                } else {
                    f64::from(bindings[index].weight) * remaining / other_sum
                };
                bindings[index].weight = value as f32;
                assigned += f64::from(bindings[index].weight);
            }
            bindings[selected].weight = new_weight;
            return Ok(new_weight);
        }

        if delta < 0.0 {
            return Err("selected bone has no influence on this REV".into());
        }
        if end - start >= 4 {
            return Err("a REV is limited to four bone influences".into());
        }
        if start == end {
            bindings.insert(
                start,
                VoxelBoneWeightDocument {
                    voxel_index: voxel_index as u32,
                    bone_index: bone_index as u32,
                    weight: 1.0,
                },
            );
            return Ok(1.0);
        }
        let existing_sum: f64 = bindings[start..end]
            .iter()
            .map(|binding| f64::from(binding.weight))
            .sum();
        if (existing_sum - 1.0).abs() > 1.0e-4 {
            return Err("existing REV weights are not normalized".into());
        }
        let new_weight = delta.clamp(0.001, 0.25);
        let remaining = 1.0 - f64::from(new_weight);
        let mut assigned = 0.0_f64;
        for index in start..end {
            let value = if index + 1 == end {
                remaining - assigned
            } else {
                f64::from(bindings[index].weight) * remaining / existing_sum
            };
            bindings[index].weight = value as f32;
            assigned += f64::from(bindings[index].weight);
        }
        bindings.insert(
            end,
            VoxelBoneWeightDocument {
                voxel_index: voxel_index as u32,
                bone_index: bone_index as u32,
                weight: new_weight,
            },
        );
        Ok(new_weight)
    }
}

#[derive(Serialize, Deserialize)]
struct V1VoxelModelDocument {
    name: String,
    cell_size_m: f64,
    materials: Vec<LegacyMaterialV2>,
    voxels: Vec<ModelVoxel>,
}

// Bincode encodes struct fields positionally, so serde(default) cannot migrate
// an appended field when reading an older payload. Keep this exact v2 material
// layout for both v1 and v2 documents; v1 differs only in the model fields.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyMaterialV2 {
    name: String,
    density: f32,
    young: f32,
    young_perp: f32,
    shear: f32,
    poisson: f32,
    tensile_strength: f32,
    tensile_strength_perp: f32,
    compressive_strength: f32,
    compressive_strength_perp: f32,
    fracture_mode: FractureMode,
    hardening_modulus: f32,
    failure_plastic_strain: f32,
    shear_strength: f32,
    shear_strength_perp: f32,
    yield_strength: f32,
    hardness: f32,
    heat_capacity: f32,
    thermal_cond: f32,
    dynamic_viscosity_pa_s: f32,
    surface_friction_coefficient: f32,
    viscosity_reference_temperature_kelvin: f32,
    viscosity_activation_energy_j_mol: f32,
    electrical_conductivity: f32,
    relative_permittivity: f32,
    thermal_expansion_parallel: f32,
    thermal_expansion_perpendicular: f32,
    melt_temp: f32,
    strength_softening_start_ratio: f32,
    boil_temp: f32,
    latent_fusion: f32,
    latent_vaporization: f32,
    fracture_energy: f32,
    fracture_energy_perp: f32,
    moisture_content_dry_basis: f32,
    solid_kelvin_voigt_time_s: f32,
}

// Version 3 is the complete material record before skeleton joint ranges were
// stored per material. Keep this exact positional layout for bincode migration.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyMaterialV3 {
    name: String,
    density: f32,
    young: f32,
    young_perp: f32,
    shear: f32,
    poisson: f32,
    tensile_strength: f32,
    tensile_strength_perp: f32,
    compressive_strength: f32,
    compressive_strength_perp: f32,
    fracture_mode: FractureMode,
    hardening_modulus: f32,
    failure_plastic_strain: f32,
    shear_strength: f32,
    shear_strength_perp: f32,
    yield_strength: f32,
    hardness: f32,
    heat_capacity: f32,
    thermal_cond: f32,
    dynamic_viscosity_pa_s: f32,
    surface_friction_coefficient: f32,
    viscosity_reference_temperature_kelvin: f32,
    viscosity_activation_energy_j_mol: f32,
    electrical_conductivity: f32,
    relative_permittivity: f32,
    thermal_expansion_parallel: f32,
    thermal_expansion_perpendicular: f32,
    melt_temp: f32,
    strength_softening_start_ratio: f32,
    boil_temp: f32,
    latent_fusion: f32,
    latent_vaporization: f32,
    fracture_energy: f32,
    fracture_energy_perp: f32,
    moisture_content_dry_basis: f32,
    solid_kelvin_voigt_time_s: f32,
    internal_friction_coefficient: f32,
}

impl From<LegacyMaterialV3> for Material {
    fn from(old: LegacyMaterialV3) -> Self {
        let mut material: Material = LegacyMaterialV2 {
            name: old.name,
            density: old.density,
            young: old.young,
            young_perp: old.young_perp,
            shear: old.shear,
            poisson: old.poisson,
            tensile_strength: old.tensile_strength,
            tensile_strength_perp: old.tensile_strength_perp,
            compressive_strength: old.compressive_strength,
            compressive_strength_perp: old.compressive_strength_perp,
            fracture_mode: old.fracture_mode,
            hardening_modulus: old.hardening_modulus,
            failure_plastic_strain: old.failure_plastic_strain,
            shear_strength: old.shear_strength,
            shear_strength_perp: old.shear_strength_perp,
            yield_strength: old.yield_strength,
            hardness: old.hardness,
            heat_capacity: old.heat_capacity,
            thermal_cond: old.thermal_cond,
            dynamic_viscosity_pa_s: old.dynamic_viscosity_pa_s,
            surface_friction_coefficient: old.surface_friction_coefficient,
            viscosity_reference_temperature_kelvin: old.viscosity_reference_temperature_kelvin,
            viscosity_activation_energy_j_mol: old.viscosity_activation_energy_j_mol,
            electrical_conductivity: old.electrical_conductivity,
            relative_permittivity: old.relative_permittivity,
            thermal_expansion_parallel: old.thermal_expansion_parallel,
            thermal_expansion_perpendicular: old.thermal_expansion_perpendicular,
            melt_temp: old.melt_temp,
            strength_softening_start_ratio: old.strength_softening_start_ratio,
            boil_temp: old.boil_temp,
            latent_fusion: old.latent_fusion,
            latent_vaporization: old.latent_vaporization,
            fracture_energy: old.fracture_energy,
            fracture_energy_perp: old.fracture_energy_perp,
            moisture_content_dry_basis: old.moisture_content_dry_basis,
            solid_kelvin_voigt_time_s: old.solid_kelvin_voigt_time_s,
        }
        .into();
        material.internal_friction_coefficient = old.internal_friction_coefficient;
        material
    }
}

impl From<&Material> for LegacyMaterialV3 {
    fn from(material: &Material) -> Self {
        Self {
            name: material.name.clone(),
            density: material.density,
            young: material.young,
            young_perp: material.young_perp,
            shear: material.shear,
            poisson: material.poisson,
            tensile_strength: material.tensile_strength,
            tensile_strength_perp: material.tensile_strength_perp,
            compressive_strength: material.compressive_strength,
            compressive_strength_perp: material.compressive_strength_perp,
            fracture_mode: material.fracture_mode,
            hardening_modulus: material.hardening_modulus,
            failure_plastic_strain: material.failure_plastic_strain,
            shear_strength: material.shear_strength,
            shear_strength_perp: material.shear_strength_perp,
            yield_strength: material.yield_strength,
            hardness: material.hardness,
            heat_capacity: material.heat_capacity,
            thermal_cond: material.thermal_cond,
            dynamic_viscosity_pa_s: material.dynamic_viscosity_pa_s,
            surface_friction_coefficient: material.surface_friction_coefficient,
            viscosity_reference_temperature_kelvin: material.viscosity_reference_temperature_kelvin,
            viscosity_activation_energy_j_mol: material.viscosity_activation_energy_j_mol,
            electrical_conductivity: material.electrical_conductivity,
            relative_permittivity: material.relative_permittivity,
            thermal_expansion_parallel: material.thermal_expansion_parallel,
            thermal_expansion_perpendicular: material.thermal_expansion_perpendicular,
            melt_temp: material.melt_temp,
            strength_softening_start_ratio: material.strength_softening_start_ratio,
            boil_temp: material.boil_temp,
            latent_fusion: material.latent_fusion,
            latent_vaporization: material.latent_vaporization,
            fracture_energy: material.fracture_energy,
            fracture_energy_perp: material.fracture_energy_perp,
            moisture_content_dry_basis: material.moisture_content_dry_basis,
            solid_kelvin_voigt_time_s: material.solid_kelvin_voigt_time_s,
            internal_friction_coefficient: material.internal_friction_coefficient,
        }
    }
}

impl From<LegacyMaterialV2> for Material {
    fn from(old: LegacyMaterialV2) -> Self {
        Self {
            name: old.name,
            density: old.density,
            young: old.young,
            young_perp: old.young_perp,
            shear: old.shear,
            poisson: old.poisson,
            tensile_strength: old.tensile_strength,
            tensile_strength_perp: old.tensile_strength_perp,
            compressive_strength: old.compressive_strength,
            compressive_strength_perp: old.compressive_strength_perp,
            fracture_mode: old.fracture_mode,
            hardening_modulus: old.hardening_modulus,
            failure_plastic_strain: old.failure_plastic_strain,
            shear_strength: old.shear_strength,
            shear_strength_perp: old.shear_strength_perp,
            yield_strength: old.yield_strength,
            hardness: old.hardness,
            heat_capacity: old.heat_capacity,
            thermal_cond: old.thermal_cond,
            dynamic_viscosity_pa_s: old.dynamic_viscosity_pa_s,
            surface_friction_coefficient: old.surface_friction_coefficient,
            viscosity_reference_temperature_kelvin: old.viscosity_reference_temperature_kelvin,
            viscosity_activation_energy_j_mol: old.viscosity_activation_energy_j_mol,
            electrical_conductivity: old.electrical_conductivity,
            relative_permittivity: old.relative_permittivity,
            thermal_expansion_parallel: old.thermal_expansion_parallel,
            thermal_expansion_perpendicular: old.thermal_expansion_perpendicular,
            melt_temp: old.melt_temp,
            strength_softening_start_ratio: old.strength_softening_start_ratio,
            boil_temp: old.boil_temp,
            latent_fusion: old.latent_fusion,
            latent_vaporization: old.latent_vaporization,
            fracture_energy: old.fracture_energy,
            fracture_energy_perp: old.fracture_energy_perp,
            moisture_content_dry_basis: old.moisture_content_dry_basis,
            solid_kelvin_voigt_time_s: old.solid_kelvin_voigt_time_s,
            internal_friction_coefficient: 0.0,
            // Older models had no joint material limit; preserve their poses.
            skeletal_joint_limit_rad: std::f32::consts::PI,
        }
    }
}

impl From<&Material> for LegacyMaterialV2 {
    fn from(material: &Material) -> Self {
        Self {
            name: material.name.clone(),
            density: material.density,
            young: material.young,
            young_perp: material.young_perp,
            shear: material.shear,
            poisson: material.poisson,
            tensile_strength: material.tensile_strength,
            tensile_strength_perp: material.tensile_strength_perp,
            compressive_strength: material.compressive_strength,
            compressive_strength_perp: material.compressive_strength_perp,
            fracture_mode: material.fracture_mode,
            hardening_modulus: material.hardening_modulus,
            failure_plastic_strain: material.failure_plastic_strain,
            shear_strength: material.shear_strength,
            shear_strength_perp: material.shear_strength_perp,
            yield_strength: material.yield_strength,
            hardness: material.hardness,
            heat_capacity: material.heat_capacity,
            thermal_cond: material.thermal_cond,
            dynamic_viscosity_pa_s: material.dynamic_viscosity_pa_s,
            surface_friction_coefficient: material.surface_friction_coefficient,
            viscosity_reference_temperature_kelvin: material.viscosity_reference_temperature_kelvin,
            viscosity_activation_energy_j_mol: material.viscosity_activation_energy_j_mol,
            electrical_conductivity: material.electrical_conductivity,
            relative_permittivity: material.relative_permittivity,
            thermal_expansion_parallel: material.thermal_expansion_parallel,
            thermal_expansion_perpendicular: material.thermal_expansion_perpendicular,
            melt_temp: material.melt_temp,
            strength_softening_start_ratio: material.strength_softening_start_ratio,
            boil_temp: material.boil_temp,
            latent_fusion: material.latent_fusion,
            latent_vaporization: material.latent_vaporization,
            fracture_energy: material.fracture_energy,
            fracture_energy_perp: material.fracture_energy_perp,
            moisture_content_dry_basis: material.moisture_content_dry_basis,
            solid_kelvin_voigt_time_s: material.solid_kelvin_voigt_time_s,
        }
    }
}

impl From<V2VoxelModelDocument> for VoxelModelDocument {
    fn from(old: V2VoxelModelDocument) -> Self {
        Self {
            name: old.name,
            cell_size_m: old.cell_size_m,
            materials: old.materials.into_iter().map(Material::from).collect(),
            voxels: old.voxels,
            groups: old.groups,
            motion_constraints: old.motion_constraints,
            skeleton: old.skeleton,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct V2VoxelModelDocument {
    name: String,
    cell_size_m: f64,
    materials: Vec<LegacyMaterialV2>,
    voxels: Vec<ModelVoxel>,
    groups: Vec<VoxelGroupDocument>,
    motion_constraints: Vec<MotionConstraintDocument>,
    skeleton: Option<SkeletonDocument>,
}

#[derive(Serialize, Deserialize)]
struct V3VoxelModelDocument {
    name: String,
    cell_size_m: f64,
    materials: Vec<LegacyMaterialV3>,
    voxels: Vec<ModelVoxel>,
    groups: Vec<VoxelGroupDocument>,
    motion_constraints: Vec<MotionConstraintDocument>,
    skeleton: Option<SkeletonDocument>,
}

impl From<V3VoxelModelDocument> for VoxelModelDocument {
    fn from(old: V3VoxelModelDocument) -> Self {
        Self {
            name: old.name,
            cell_size_m: old.cell_size_m,
            materials: old.materials.into_iter().map(Material::from).collect(),
            voxels: old.voxels,
            groups: old.groups,
            motion_constraints: old.motion_constraints,
            skeleton: old.skeleton,
        }
    }
}

impl VoxelModelDocument {
    /// Precompute per-bone material behavior once when a model or its skin
    /// weights change. This linear pass is deliberately kept out of animation
    /// frame evaluation so million-REV models do not rescan on every frame.
    pub fn skeleton_material_limits(&self) -> Option<Vec<BoneMaterialLimits>> {
        let skeleton = self.skeleton.as_ref()?;
        let mut sums = vec![[0.0_f64; 4]; skeleton.bones.len()];
        for binding in &skeleton.voxel_weights {
            let voxel = &self.voxels[binding.voxel_index as usize];
            let bone = binding.bone_index as usize;
            let binding_mass = f64::from(voxel.state.mass) * f64::from(binding.weight);
            for &(material_index, fraction) in &voxel.state.composition.components {
                let material = &self.materials[material_index as usize];
                let mass = binding_mass * f64::from(fraction);
                let axial_yield_strain = if material.young > 1.0 {
                    f64::from(material.yield_strength / material.young)
                } else {
                    0.5
                };
                let compressive_yield_strain = if material.young > 1.0 {
                    f64::from(material.compressive_strength / material.young)
                } else {
                    0.5
                };
                sums[bone][0] += f64::from(material.skeletal_joint_limit_rad) * mass;
                sums[bone][1] += (axial_yield_strain + f64::from(material.failure_plastic_strain))
                    .clamp(0.0, 0.5)
                    * mass;
                sums[bone][2] += (compressive_yield_strain
                    + f64::from(material.failure_plastic_strain))
                .clamp(0.0, 0.5)
                    * mass;
                sums[bone][3] += mass;
            }
        }
        Some(
            sums.into_iter()
                .map(|sum| {
                    if sum[3] <= f64::EPSILON {
                        BoneMaterialLimits::unrestricted()
                    } else {
                        BoneMaterialLimits {
                            max_joint_rotation_rad: (sum[0] / sum[3])
                                .clamp(0.0, std::f64::consts::PI),
                            max_extension_strain: (sum[1] / sum[3]).clamp(0.0, 0.5),
                            max_compression_strain: (sum[2] / sum[3]).clamp(0.0, 0.5),
                        }
                    }
                })
                .collect(),
        )
    }

    /// Snapshot the current REV values and positions as a reusable model asset.
    /// Inter-REV bond damage belongs to a scene checkpoint, not a `.vxm` asset.
    pub fn from_scene(name: impl Into<String>, scene: &Scene) -> Self {
        Self {
            name: name.into(),
            cell_size_m: crate::coordinates::CELL_M,
            materials: scene.materials.clone(),
            voxels: scene
                .voxels
                .iter()
                .cloned()
                .zip(scene.positions.iter().copied())
                .zip(scene.rest_positions.iter().copied())
                .map(|((state, position_m), rest_position_m)| ModelVoxel {
                    position_m,
                    rest_position_m,
                    state,
                })
                .collect(),
            groups: Vec::new(),
            motion_constraints: Vec::new(),
            skeleton: None,
        }
    }

    /// Rebuild the physical REV state and derived scene topology from an asset.
    /// Saved constraints and skeleton data are rejected until their runtime
    /// solver/animation integration is available, so loading cannot ignore them.
    pub fn build_scene(&self, config: SceneConfig) -> Result<Scene, String> {
        self.validate()?;
        if !self.groups.is_empty() {
            return Err("runtime scene does not preserve editor voxel groups yet".into());
        }
        if !self.motion_constraints.is_empty() {
            return Err("runtime motion-constraint restoration is not implemented yet".into());
        }
        if self.skeleton.is_some() {
            return Err("runtime skeleton restoration is not implemented yet".into());
        }
        self.build_voxel_scene(config)
    }

    /// Build a non-physical editor preview from an authored model. Groups,
    /// constraints, and the skeleton remain in this document and are not
    /// transferred to the temporary Scene until their runtime integration is
    /// implemented. This avoids cloning a million-REV document just to show it.
    pub fn build_editor_preview_scene(&self, config: SceneConfig) -> Result<Scene, String> {
        self.validate()?;
        self.build_voxel_scene(config)
    }

    fn build_voxel_scene(&self, config: SceneConfig) -> Result<Scene, String> {
        let snapshots = self
            .voxels
            .iter()
            .map(|voxel| (voxel.position_m, voxel.rest_position_m, voxel.state.clone()))
            .collect();
        Scene::from_saved_voxel_states(config, self.materials.clone(), snapshots)
    }
}

impl EditorDocument for VoxelModelDocument {
    const KIND: DocumentKind = DocumentKind::Model;

    fn record_count(&self) -> usize {
        let skeleton_records = self.skeleton.as_ref().map_or(0, |skeleton| {
            skeleton.bones.len()
                + skeleton.voxel_weights.len()
                + skeleton.clips.len()
                + skeleton
                    .clips
                    .iter()
                    .map(|clip| {
                        clip.tracks.len()
                            + clip
                                .tracks
                                .iter()
                                .map(|track| track.keyframes.len())
                                .sum::<usize>()
                    })
                    .sum::<usize>()
        });
        self.voxels.len() + self.groups.len() + self.motion_constraints.len() + skeleton_records
    }

    fn decode_payload(version: u16, payload: &[u8]) -> Result<Self, String> {
        match version {
            1 => {
                let old: V1VoxelModelDocument = codec()
                    .deserialize(payload)
                    .map_err(|error| format!("decode version 1 voxel model: {error}"))?;
                Ok(Self {
                    name: old.name,
                    cell_size_m: old.cell_size_m,
                    materials: old.materials.into_iter().map(Material::from).collect(),
                    voxels: old.voxels,
                    groups: Vec::new(),
                    motion_constraints: Vec::new(),
                    skeleton: None,
                })
            }
            2 => {
                let old: V2VoxelModelDocument = codec()
                    .deserialize(payload)
                    .map_err(|error| format!("decode version 2 voxel model: {error}"))?;
                Ok(old.into())
            }
            3 => {
                let old: V3VoxelModelDocument = codec()
                    .deserialize(payload)
                    .map_err(|error| format!("decode version 3 voxel model: {error}"))?;
                Ok(old.into())
            }
            4..=FORMAT_VERSION => codec()
                .deserialize(payload)
                .map_err(|error| format!("decode native document: {error}")),
            _ => Err(format!("unsupported .vxm schema version {version}")),
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.name.len() > 4096 {
            return Err("model name must contain 1..=4096 UTF-8 bytes".into());
        }
        if (self.cell_size_m - crate::coordinates::CELL_M).abs() > 1e-12 {
            return Err("this engine build supports 1 cm mechanical REVs".into());
        }
        if self.voxels.is_empty() {
            return Err("voxel model must contain at least one REV".into());
        }
        if self.voxels.len() as u64 > MAX_DOCUMENT_ITEMS {
            return Err("voxel model exceeds the supported REV count".into());
        }
        if self.materials.is_empty() || self.materials.len() > u16::MAX as usize + 1 {
            return Err("voxel model has an invalid material table".into());
        }
        for (index, material) in self.materials.iter().enumerate() {
            if material.name.trim().is_empty()
                || self.materials[..index]
                    .iter()
                    .any(|previous| previous.name == material.name)
                || !material.skeletal_joint_limit_rad.is_finite()
                || !(0.0..=std::f32::consts::PI).contains(&material.skeletal_joint_limit_rad)
            {
                return Err("material names and skeletal joint limits are invalid".into());
            }
        }
        for voxel in &self.voxels {
            if !voxel.position_m.is_finite()
                || !voxel.rest_position_m.is_finite()
                || !voxel.state.mass.is_finite()
                || voxel.state.mass <= 0.0
                || !voxel.state.velocity.is_finite()
                || !voxel.state.internal_energy.is_finite()
                || voxel.state.internal_energy < 0.0
                || !voxel.state.internal_energy_remainder_j.is_finite()
                || !voxel.state.microstructure.orientation.is_finite()
                || voxel.state.microstructure.orientation.length_squared() < 1e-8
                || !voxel.state.microstructure.anisotropy.is_finite()
                || !(0.0..=1.0).contains(&voxel.state.microstructure.anisotropy)
            {
                return Err("voxel model contains invalid physical state".into());
            }
            voxel
                .state
                .composition
                .validate(self.materials.len())
                .map_err(|error| format!("invalid voxel composition: {error}"))?;
            let mut thermal = [0.0; crate::voxel::THERMAL_CELL_COUNT];
            voxel.state.copy_thermal_cell_energies_j(&mut thermal);
            if thermal
                .iter()
                .any(|energy| !energy.is_finite() || *energy < 0.0)
            {
                return Err("voxel model contains invalid thermal-cell energy".into());
            }
            let thermal_total: f64 = thermal.iter().sum();
            if (thermal_total - voxel.state.total_internal_energy_j()).abs()
                > 1e-9 * thermal_total.abs().max(1.0)
            {
                return Err("thermal-cell energy does not match the REV energy".into());
            }
        }
        validate_groups(&self.groups, self.voxels.len())?;
        validate_motion_constraints(&self.motion_constraints, &self.groups, self.voxels.len())?;
        if let Some(skeleton) = &self.skeleton {
            validate_skeleton(skeleton, self.voxels.len())?;
        }
        Ok(())
    }
}

fn validate_groups(groups: &[VoxelGroupDocument], voxel_count: usize) -> Result<(), String> {
    if groups.len() as u64 > MAX_DOCUMENT_ITEMS {
        return Err("voxel model contains too many groups".into());
    }
    let mut names = HashSet::with_capacity(groups.len());
    for group in groups {
        validate_name(&group.name, "voxel group")?;
        if !names.insert(group.name.as_str()) {
            return Err("voxel group names must be unique".into());
        }
        if group.voxels.is_empty() || group.voxels.len() as u64 > MAX_DOCUMENT_ITEMS {
            return Err("voxel groups must contain at least one supported REV index".into());
        }
        let mut members = HashSet::with_capacity(group.voxels.len());
        for &index in &group.voxels {
            if index as usize >= voxel_count || !members.insert(index) {
                return Err("voxel group contains an invalid or duplicate REV index".into());
            }
        }
    }
    Ok(())
}

fn validate_motion_constraints(
    constraints: &[MotionConstraintDocument],
    groups: &[VoxelGroupDocument],
    voxel_count: usize,
) -> Result<(), String> {
    if constraints.len() as u64 > MAX_DOCUMENT_ITEMS {
        return Err("voxel model contains too many motion constraints".into());
    }
    let mut names = HashSet::with_capacity(constraints.len());
    for constraint in constraints {
        validate_name(&constraint.name, "motion constraint")?;
        if !names.insert(constraint.name.as_str()) {
            return Err("motion constraint names must be unique".into());
        }
        validate_constraint_part(constraint.a, groups.len(), voxel_count)?;
        validate_constraint_part(constraint.b, groups.len(), voxel_count)?;
        if constraint.a == constraint.b
            || !constraint.anchor_a_m.is_finite()
            || !constraint.anchor_b_m.is_finite()
            || !valid_nonnegative(constraint.linear_compliance_m_per_n)
            || !valid_nonnegative(constraint.angular_compliance_rad_per_nm)
            || !valid_nonnegative(constraint.linear_damping_n_s_per_m)
            || !valid_nonnegative(constraint.angular_damping_nm_s_per_rad)
        {
            return Err("motion constraint has invalid endpoints or physical parameters".into());
        }
        match constraint.kind {
            MotionConstraintKindDocument::Fixed | MotionConstraintKindDocument::Ball => {}
            MotionConstraintKindDocument::Hinge {
                axis_a_local,
                axis_b_local,
            }
            | MotionConstraintKindDocument::Slider {
                axis_a_local,
                axis_b_local,
            } => {
                if !valid_axis(axis_a_local) || !valid_axis(axis_b_local) {
                    return Err("hinge and slider axes must be finite and non-zero".into());
                }
            }
        }
    }
    Ok(())
}

fn validate_constraint_part(
    part: ConstraintPartDocument,
    group_count: usize,
    voxel_count: usize,
) -> Result<(), String> {
    match part {
        ConstraintPartDocument::Voxel(index) if index as usize >= voxel_count => {
            Err("motion constraint references an invalid REV index".into())
        }
        ConstraintPartDocument::Group(index) if index as usize >= group_count => {
            Err("motion constraint references an invalid group index".into())
        }
        _ => Ok(()),
    }
}

fn validate_skeleton(skeleton: &SkeletonDocument, voxel_count: usize) -> Result<(), String> {
    if skeleton.bones.is_empty() || skeleton.bones.len() as u64 > MAX_DOCUMENT_ITEMS {
        return Err("skeleton must contain a supported non-zero number of bones".into());
    }
    let mut bone_names = HashSet::with_capacity(skeleton.bones.len());
    for (index, bone) in skeleton.bones.iter().enumerate() {
        validate_name(&bone.name, "bone")?;
        if !bone_names.insert(bone.name.as_str())
            || !bone.rest_translation_m.is_finite()
            || !valid_rotation(bone.rest_rotation)
            || match bone.parent {
                None => index != 0,
                Some(parent) => parent as usize >= index,
            }
        {
            return Err("skeleton bones must be uniquely named and topologically ordered".into());
        }
    }

    if skeleton.voxel_weights.len() as u64 > MAX_DOCUMENT_ITEMS {
        return Err("skeleton contains too many voxel weights".into());
    }
    let mut weights_by_voxel = BTreeMap::<u32, f64>::new();
    let mut pairs = HashSet::with_capacity(skeleton.voxel_weights.len());
    for binding in &skeleton.voxel_weights {
        if binding.voxel_index as usize >= voxel_count
            || binding.bone_index as usize >= skeleton.bones.len()
            || !binding.weight.is_finite()
            || binding.weight <= 0.0
            || binding.weight > 1.0
            || !pairs.insert((binding.voxel_index, binding.bone_index))
        {
            return Err("skeleton contains an invalid or duplicate voxel-to-bone weight".into());
        }
        *weights_by_voxel.entry(binding.voxel_index).or_default() += f64::from(binding.weight);
    }
    if weights_by_voxel
        .values()
        .any(|sum| (*sum - 1.0).abs() > 1e-4)
    {
        return Err("weights for each bound REV must sum to 1 within 1e-4".into());
    }

    if skeleton.clips.len() as u64 > MAX_DOCUMENT_ITEMS {
        return Err("skeleton contains too many animation clips".into());
    }
    let mut clip_names = HashSet::with_capacity(skeleton.clips.len());
    for clip in &skeleton.clips {
        validate_name(&clip.name, "animation clip")?;
        if !clip_names.insert(clip.name.as_str())
            || !clip.duration_s.is_finite()
            || clip.duration_s <= 0.0
            || clip.tracks.len() as u64 > MAX_DOCUMENT_ITEMS
        {
            return Err("animation clip has an invalid name, duration, or track count".into());
        }
        let mut tracked_bones = HashSet::with_capacity(clip.tracks.len());
        for track in &clip.tracks {
            if track.bone_index as usize >= skeleton.bones.len()
                || !tracked_bones.insert(track.bone_index)
                || track.keyframes.is_empty()
                || track.keyframes.len() as u64 > MAX_DOCUMENT_ITEMS
            {
                return Err("animation track has an invalid bone or keyframe list".into());
            }
            let mut previous_time = -1.0;
            for keyframe in &track.keyframes {
                if !keyframe.time_s.is_finite()
                    || keyframe.time_s < 0.0
                    || keyframe.time_s > clip.duration_s
                    || keyframe.time_s <= previous_time
                    || !keyframe.translation_m.is_finite()
                    || !valid_rotation(keyframe.rotation)
                {
                    return Err("animation keyframes must have ordered finite transforms".into());
                }
                previous_time = keyframe.time_s;
            }
        }
    }
    Ok(())
}

fn valid_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn valid_axis(axis: Vec3) -> bool {
    axis.is_finite() && axis.length_squared() > 1e-8
}

fn valid_rotation(rotation: Quat) -> bool {
    rotation.is_finite() && rotation.length_squared() > 1e-8
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum InterfaceLanguage {
    #[default]
    Russian,
    English,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectDocument {
    pub name: String,
    pub scenes: Vec<PathBuf>,
    pub assets: Vec<PathBuf>,
    pub experiments: Vec<ExperimentDefinition>,
    pub language: InterfaceLanguage,
}

impl ProjectDocument {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            scenes: Vec::new(),
            assets: Vec::new(),
            experiments: Vec::new(),
            language: InterfaceLanguage::Russian,
        }
    }
}

impl EditorDocument for ProjectDocument {
    const KIND: DocumentKind = DocumentKind::Project;

    fn record_count(&self) -> usize {
        self.scenes.len() + self.assets.len() + self.experiments.len()
    }

    fn validate(&self) -> Result<(), String> {
        validate_name(&self.name, "project")?;
        if self.record_count() as u64 > MAX_DOCUMENT_ITEMS {
            return Err("project contains too many entries".into());
        }
        for experiment in &self.experiments {
            experiment.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneDocument {
    pub name: String,
    pub config: SceneConfig,
    pub models: Vec<ModelInstanceDocument>,
    /// Per-instance editor values, keyed by the stable index in `models`.
    /// Defaults are implicit so existing scenes need no placeholder records.
    pub scene_object_settings: Vec<SceneObjectSettings>,
    /// Runtime authoring settings; legacy scenes default to disabled welding.
    pub simulation_settings: SceneSimulationSettings,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SceneObjectSettings {
    pub model_index: usize,
    pub initial_velocity_m_s: Option<Vec3>,
    pub force_n: Vec3,
    pub material_asset: Option<PathBuf>,
    pub voxel_materials: Vec<SceneVoxelMaterialOverride>,
}

impl SceneObjectSettings {
    fn defaults_for(model_index: usize) -> Self {
        Self {
            model_index,
            initial_velocity_m_s: None,
            force_n: Vec3::ZERO,
            material_asset: None,
            voxel_materials: Vec::new(),
        }
    }

    fn is_default(&self) -> bool {
        self.initial_velocity_m_s.is_none()
            && self.force_n == Vec3::ZERO
            && self.material_asset.is_none()
            && self.voxel_materials.is_empty()
    }

    fn validate(&self) -> Result<(), String> {
        if self
            .initial_velocity_m_s
            .is_some_and(|velocity| !velocity.is_finite())
            || !self.force_n.is_finite()
            || self
                .material_asset
                .as_ref()
                .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(
                "scene object settings contain an invalid velocity, force, or material path".into(),
            );
        }
        if self.voxel_materials.len() as u64 > MAX_DOCUMENT_ITEMS {
            return Err("scene object has too many per-REV material overrides".into());
        }
        let mut voxel_indices = HashSet::with_capacity(self.voxel_materials.len());
        for override_ in &self.voxel_materials {
            if override_.material_asset.as_os_str().is_empty()
                || !voxel_indices.insert(override_.voxel_index)
            {
                return Err(
                    "per-REV material overrides need unique indices and non-empty paths".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SceneVoxelMaterialOverride {
    pub voxel_index: u32,
    pub material_asset: PathBuf,
}

/// The schema 1–4 scene payload did not contain authoring overrides. Keep an
/// explicit old shape because bincode cannot safely default fields at EOF.
#[derive(Serialize, Deserialize)]
struct LegacySceneDocument {
    name: String,
    config: SceneConfig,
    models: Vec<ModelInstanceDocument>,
}

/// Version 5 had per-instance editor settings but no simulation settings.
#[derive(Serialize, Deserialize)]
struct LegacySceneV5Document {
    name: String,
    config: SceneConfig,
    models: Vec<ModelInstanceDocument>,
    scene_object_settings: Vec<SceneObjectSettings>,
}

impl SceneDocument {
    pub fn new(name: impl Into<String>, config: SceneConfig) -> Result<Self, String> {
        config.validate().map_err(str::to_owned)?;
        let document = Self {
            name: name.into(),
            config,
            models: Vec::new(),
            scene_object_settings: Vec::new(),
            simulation_settings: SceneSimulationSettings::default(),
        };
        document.validate()?;
        Ok(document)
    }

    pub fn add_model(&mut self, model: ModelInstanceDocument) -> Result<(), String> {
        model.validate()?;
        self.models.push(model);
        Ok(())
    }

    /// Return saved settings for an instance, or its empty authoring defaults.
    pub fn object_settings(&self, index: usize) -> SceneObjectSettings {
        self.scene_object_settings
            .iter()
            .find(|settings| settings.model_index == index)
            .cloned()
            .unwrap_or_else(|| SceneObjectSettings::defaults_for(index))
    }

    pub fn set_simulation_settings(
        &mut self,
        settings: SceneSimulationSettings,
    ) -> Result<(), String> {
        settings.validate(self.models.len())?;
        self.simulation_settings = settings;
        Ok(())
    }

    pub fn set_object_initial_temperature(
        &mut self,
        model_index: usize,
        temperature_kelvin: Option<f32>,
    ) -> Result<(), String> {
        if model_index >= self.models.len() {
            return Err(format!("model instance index {model_index} is out of bounds"));
        }
        let mut settings = self.simulation_settings.clone();
        settings.set_object_temperature_kelvin(model_index, temperature_kelvin)?;
        self.set_simulation_settings(settings)
    }

    pub fn set_thermal_bonding(
        &mut self,
        settings: Option<crate::scene_welding::ThermalBondingSettings>,
    ) -> Result<(), String> {
        let mut simulation = self.simulation_settings.clone();
        simulation.thermal_bonding = settings;
        self.set_simulation_settings(simulation)
    }

    /// Keep relative asset references pointing at their existing files when a
    /// scene is moved to a different directory (including its first save).
    pub fn rebased_paths(
        &self,
        old_scene_path: &Path,
        new_scene_path: &Path,
    ) -> Result<Self, String> {
        let old_base = absolute_scene_directory(old_scene_path)?;
        let new_base = absolute_scene_directory(new_scene_path)?;
        if old_base == new_base {
            return Ok(self.clone());
        }

        let mut rebased = self.clone();
        for model in &mut rebased.models {
            make_asset_path_absolute(&mut model.asset_path, &old_base);
        }
        for settings in &mut rebased.scene_object_settings {
            if let Some(path) = &mut settings.material_asset {
                make_asset_path_absolute(path, &old_base);
            }
            for override_ in &mut settings.voxel_materials {
                make_asset_path_absolute(&mut override_.material_asset, &old_base);
            }
        }
        Ok(rebased)
    }

    /// Save validated editor settings for one existing model instance.
    pub fn set_object_settings(
        &mut self,
        index: usize,
        settings: SceneObjectSettings,
    ) -> Result<(), String> {
        self.validate()?;
        if index >= self.models.len() {
            return Err(format!("model instance index {index} is out of bounds"));
        }
        if settings.model_index != index {
            return Err("scene object settings index does not match the selected instance".into());
        }
        settings.validate()?;
        if let Some(existing) = self
            .scene_object_settings
            .iter_mut()
            .find(|existing| existing.model_index == index)
        {
            *existing = settings;
        } else if !settings.is_default() {
            self.scene_object_settings.push(settings);
            self.scene_object_settings
                .sort_unstable_by_key(|settings| settings.model_index);
        }
        Ok(())
    }

    /// Remove a model instance by its current index and return the removed instance.
    pub fn remove_model(&mut self, index: usize) -> Result<ModelInstanceDocument, String> {
        self.validate()?;
        if index >= self.models.len() {
            return Err(format!("model instance index {index} is out of bounds"));
        }
        let removed = self.models.remove(index);
        self.scene_object_settings.retain_mut(|settings| {
            if settings.model_index == index {
                false
            } else {
                if settings.model_index > index {
                    settings.model_index -= 1;
                }
                true
            }
        });
        self.simulation_settings.remove_object(index);
        Ok(removed)
    }

    /// Translate an instance by an integer number of 1 cm REV lattice cells.
    /// The operation is committed only if the resulting transform remains valid.
    pub fn translate_model_cells(
        &mut self,
        index: usize,
        delta_cells: IVec3,
    ) -> Result<(), String> {
        self.validate()?;
        let instance = self
            .models
            .get(index)
            .ok_or_else(|| format!("model instance index {index} is out of bounds"))?;
        validate_instance_lattice_transform(instance)?;

        let cell_size = crate::coordinates::CELL_M;
        let delta_m = DVec3::new(
            delta_cells.x as f64 * cell_size,
            delta_cells.y as f64 * cell_size,
            delta_cells.z as f64 * cell_size,
        );
        let mut translated = instance.clone();
        translated.translation_m += delta_m;
        translated.validate()?;
        validate_instance_lattice_transform(&translated)?;
        validate_instance_translation_lattice(&translated)?;

        self.models[index] = translated;
        Ok(())
    }

    /// Rotate an instance by +90 degrees around world Z, keeping its origin fixed.
    pub fn rotate_model_z_90(&mut self, index: usize) -> Result<(), String> {
        self.rotate_model_quarter_turn(index, IVec3::Z, 1)
    }

    /// Rotate an instance by a signed quarter turn around a world axis.
    /// Restricting turns to 90 degree increments preserves the 1 cm REV lattice.
    pub fn rotate_model_quarter_turn(
        &mut self,
        index: usize,
        axis: IVec3,
        direction: i8,
    ) -> Result<(), String> {
        self.validate()?;
        if axis.abs().element_sum() != 1 || !matches!(direction, -1 | 1) {
            return Err("rotation needs one axis and a direction of -1 or 1".into());
        }
        let instance = self
            .models
            .get(index)
            .ok_or_else(|| format!("model instance index {index} is out of bounds"))?;
        validate_instance_lattice_transform(instance)?;

        let quarter_turn = Quat::from_axis_angle(
            axis.as_vec3(),
            std::f32::consts::FRAC_PI_2 * f32::from(direction),
        );
        let mut rotated = instance.clone();
        rotated.rotation = (quarter_turn * instance.rotation.normalize()).normalize();
        rotated.validate()?;
        validate_instance_lattice_transform(&rotated)?;

        self.models[index] = rotated;
        Ok(())
    }

    /// Copy an instance and offset it by an integer number of REV cells.
    pub fn duplicate_model_cells(
        &mut self,
        index: usize,
        delta_cells: IVec3,
    ) -> Result<usize, String> {
        self.validate()?;
        let instance = self
            .models
            .get(index)
            .ok_or_else(|| format!("model instance index {index} is out of bounds"))?;
        validate_instance_lattice_transform(instance)?;

        let delta_m = DVec3::new(
            delta_cells.x as f64 * crate::coordinates::CELL_M,
            delta_cells.y as f64 * crate::coordinates::CELL_M,
            delta_cells.z as f64 * crate::coordinates::CELL_M,
        );
        let mut duplicate = instance.clone();
        duplicate.translation_m += delta_m;
        duplicate.validate()?;
        validate_instance_lattice_transform(&duplicate)?;
        validate_instance_translation_lattice(&duplicate)?;
        self.models.push(duplicate);
        let duplicate_index = self.models.len() - 1;
        let mut settings = self.object_settings(index);
        settings.model_index = duplicate_index;
        if !settings.is_default() {
            self.scene_object_settings.push(settings);
        }
        if let Some(temperature_kelvin) = self
            .simulation_settings
            .object_temperature_kelvin(index)
        {
            self.simulation_settings
                .set_object_temperature_kelvin(duplicate_index, Some(temperature_kelvin))?;
        }
        Ok(duplicate_index)
    }

    /// Load each referenced `.vxm` beside this `.vxscene` and assemble the
    /// authored instances into a physical scene. Instance transforms are
    /// accepted only when they preserve the 1 cm REV lattice exactly; geometry
    /// that needs resampling must go through the mesh voxelizer explicitly.
    pub fn build_scene(&self, scene_path: &Path) -> Result<Scene, String> {
        self.validate()?;
        if self.models.is_empty() {
            let mut scene = Scene::from_voxel_objects_with_materials(
                self.config.clone(),
                Material::table(),
                Vec::new(),
            )?;
            self.apply_simulation_settings(&mut scene)?;
            return Ok(scene);
        }

        let asset_directory = scene_path.parent().unwrap_or_else(|| Path::new("."));
        let mut materials = Vec::<Material>::new();
        let mut material_indices = BTreeMap::<Vec<u8>, u16>::new();
        let mut objects = Vec::with_capacity(self.models.len());
        let mut object_forces = Vec::with_capacity(self.models.len());
        let mut skeletal_models = Vec::new();
        let mut scene_voxel_count = 0usize;

        let mut loaded_materials = BTreeMap::<PathBuf, u16>::new();
        for (instance_index, instance) in self.models.iter().enumerate() {
            let (rotation, rotation_f32) = validate_instance_lattice_transform(instance)?;
            validate_instance_translation_lattice(instance)?;
            let settings = self.object_settings(instance_index);

            let model_path = if instance.asset_path.is_absolute() {
                instance.asset_path.clone()
            } else {
                asset_directory.join(&instance.asset_path)
            };
            let model: VoxelModelDocument = load_document(&model_path)
                .map_err(|error| format!("load model {}: {error}", model_path.display()))?;
            model.validate()?;
            if !model.groups.is_empty() || !model.motion_constraints.is_empty() {
                return Err(format!(
                    "model {} contains editor groups or constraints that the physical scene solver does not restore yet",
                    model_path.display()
                ));
            }
            let range_start = scene_voxel_count;
            scene_voxel_count = scene_voxel_count
                .checked_add(model.voxels.len())
                .ok_or_else(|| "scene voxel index range overflow".to_owned())?;
            if let Some(skeleton) = &model.skeleton {
                if skeleton.clips.is_empty() {
                    return Err(format!(
                        "model {} has a skeleton but no animation clips to drive its physical actuator",
                        model_path.display()
                    ));
                }
                skeletal_models.push((
                    model.clone(),
                    range_start..scene_voxel_count,
                    rotation_f32,
                ));
            }

            let mut component_remap = Vec::with_capacity(model.materials.len());
            for material in &model.materials {
                component_remap.push(insert_scene_material(
                    material,
                    &mut materials,
                    &mut material_indices,
                )?);
            }

            settings.validate()?;
            if let Some(override_) = settings
                .voxel_materials
                .iter()
                .find(|override_| override_.voxel_index as usize >= model.voxels.len())
            {
                return Err(format!(
                    "material override REV {} is out of bounds for model instance {instance_index}",
                    override_.voxel_index
                ));
            }
            let voxel_materials: BTreeMap<_, _> = settings
                .voxel_materials
                .iter()
                .map(|override_| (override_.voxel_index, &override_.material_asset))
                .collect();

            let mut object = Vec::with_capacity(model.voxels.len());
            for (voxel_index, voxel) in model.voxels.into_iter().enumerate() {
                let mut state = voxel.state;
                let selected_material_path = voxel_materials
                    .get(&(voxel_index as u32))
                    .copied()
                    .or(settings.material_asset.as_ref());
                if let Some(material_path) = selected_material_path {
                    let resolved_path = if material_path.is_absolute() {
                        material_path.clone()
                    } else {
                        asset_directory.join(material_path)
                    };
                    let material_index = if let Some(index) = loaded_materials.get(&resolved_path) {
                        *index
                    } else {
                        let material = crate::material_asset::load_material(&resolved_path)
                            .map_err(|error| {
                                format!(
                                    "load material {} for model instance {instance_index}: {error}",
                                    resolved_path.display()
                                )
                            })?;
                        let index = insert_scene_material(
                            &material,
                            &mut materials,
                            &mut material_indices,
                        )?;
                        loaded_materials.insert(resolved_path, index);
                        index
                    };
                    let new_material = &materials[material_index as usize];
                    let previous_material = Material::mixed(&state.composition, &model.materials);
                    let previous_cells = {
                        let mut cells = [0.0; crate::voxel::THERMAL_CELL_COUNT];
                        state.copy_thermal_cell_energies_j(&mut cells);
                        cells
                    };
                    let mass = new_material.density * (model.cell_size_m.powi(3) as f32);
                    let cell_mass = state.mass / crate::voxel::THERMAL_CELL_COUNT as f32;
                    let new_cell_mass = mass / crate::voxel::THERMAL_CELL_COUNT as f32;
                    let remapped_cells = previous_cells.map(|cell_energy| {
                        let temperature =
                            previous_material.temperature(cell_mass, cell_energy as f32);
                        f64::from(new_material.energy_at_temperature(new_cell_mass, temperature))
                    });
                    state.mass = mass;
                    state.composition = Composition::pure(material_index);
                    state.set_thermal_cell_energies_j(&remapped_cells);
                } else {
                    for (component, _) in &mut state.composition.components {
                        *component = *component_remap
                            .get(*component as usize)
                            .ok_or_else(|| "model voxel refers to a missing material".to_owned())?;
                    }
                }
                if let Some(temperature_kelvin) = self
                    .simulation_settings
                    .object_temperature_kelvin(instance_index)
                {
                    let resolved_material = Material::mixed(&state.composition, &materials);
                    let cell_mass = state.mass / crate::voxel::THERMAL_CELL_COUNT as f32;
                    let energy_per_cell_j = f64::from(
                        resolved_material.energy_at_temperature(cell_mass, temperature_kelvin),
                    );
                    if !energy_per_cell_j.is_finite() || energy_per_cell_j < 0.0 {
                        return Err(format!(
                            "initial temperature {temperature_kelvin} K is outside the supported EOS range for model instance {instance_index}"
                        ));
                    }
                    state.set_thermal_cell_energies_j(&[
                        energy_per_cell_j;
                        crate::voxel::THERMAL_CELL_COUNT
                    ]);
                }
                state.velocity = rotation_f32 * state.velocity;
                if let Some(velocity) = settings.initial_velocity_m_s {
                    state.velocity = velocity;
                }
                state.microstructure.orientation =
                    (rotation_f32 * state.microstructure.orientation).normalize();
                object.push((
                    rotation * voxel.position_m + instance.translation_m,
                    rotation * voxel.rest_position_m + instance.translation_m,
                    state,
                ));
            }
            let mobile_mass_kg: f64 = object
                .iter()
                .filter(|(_, _, state)| !state.kinematic)
                .map(|(_, _, state)| f64::from(state.mass))
                .sum();
            if settings.force_n != Vec3::ZERO && mobile_mass_kg <= 0.0 {
                return Err(format!(
                    "scene object {instance_index} has a nonzero force but no mobile mass"
                ));
            }
            let per_voxel_force: Vec<_> = object
                .iter()
                .map(|(_, _, state)| {
                    if state.kinematic {
                        Vec3::ZERO
                    } else if mobile_mass_kg > 0.0 {
                        settings.force_n * (f64::from(state.mass) / mobile_mass_kg) as f32
                    } else {
                        Vec3::ZERO
                    }
                })
                .collect();
            object_forces.push(per_voxel_force);
            objects.push(object);
        }

        let mut scene = Scene::from_saved_voxel_objects(self.config.clone(), materials, objects)?;
        let mut scene_voxel_index = 0;
        for forces in object_forces {
            for force in forces {
                scene.external_forces[scene_voxel_index] = force;
                scene_voxel_index += 1;
            }
        }
        let total_scene_voxels = scene.voxels.len();
        for (model, voxel_range, instance_rotation) in skeletal_models {
            let actuator = SkeletalActuator::new(
                model,
                voxel_range,
                instance_rotation,
                total_scene_voxels,
            )?;
            scene.skeletal_actuators.push(actuator);
        }
        self.apply_simulation_settings(&mut scene)?;
        Ok(scene)
    }

    fn apply_simulation_settings(&self, scene: &mut Scene) -> Result<(), String> {
        if let Some(rule) = &self.simulation_settings.thermal_bonding {
            rule.apply(scene)?;
        }
        Ok(())
    }
}

fn absolute_scene_directory(scene_path: &Path) -> Result<PathBuf, String> {
    let current_directory = std::env::current_dir()
        .map_err(|error| format!("resolve scene asset directory: {error}"))?;
    let parent = scene_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(if parent.is_absolute() {
        parent.to_path_buf()
    } else {
        current_directory.join(parent)
    })
}

fn make_asset_path_absolute(path: &mut PathBuf, old_base: &Path) {
    if !path.is_absolute() {
        *path = old_base.join(&*path);
    }
}

fn insert_scene_material(
    material: &Material,
    materials: &mut Vec<Material>,
    material_indices: &mut BTreeMap<Vec<u8>, u16>,
) -> Result<u16, String> {
    let encoded = codec()
        .serialize(material)
        .map_err(|error| format!("encode material {}: {error}", material.name))?;
    if let Some(index) = material_indices.get(&encoded) {
        return Ok(*index);
    }
    let index = u16::try_from(materials.len())
        .map_err(|_| "combined scene exceeds the 16-bit material table".to_owned())?;
    material_indices.insert(encoded, index);
    materials.push(material.clone());
    Ok(index)
}

fn validate_instance_translation_lattice(instance: &ModelInstanceDocument) -> Result<(), String> {
    let translation_cells = instance.translation_m / crate::coordinates::CELL_M;
    if (translation_cells - translation_cells.round())
        .abs()
        .max_element()
        > 1e-6
    {
        return Err(format!(
            "model instance {:?} translation must align to the 1 cm REV lattice",
            instance.asset_path
        ));
    }
    Ok(())
}

fn validate_instance_lattice_transform(
    instance: &ModelInstanceDocument,
) -> Result<(DQuat, Quat), String> {
    instance.validate()?;
    if (instance.scale - Vec3::ONE).abs().max_element() > 1e-6 {
        return Err(format!(
            "model instance {:?} scale must be 1 until transformed geometry can be revoxelized",
            instance.asset_path
        ));
    }

    let rotation_f32 = instance.rotation.normalize();
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let rotated = rotation_f32 * axis;
        let lattice_axis = rotated.round();
        let nonzero_axes = lattice_axis
            .to_array()
            .into_iter()
            .filter(|component| component.abs() > 0.5)
            .count();
        if nonzero_axes != 1 || (rotated - lattice_axis).abs().max_element() > 1e-5 {
            return Err(format!(
                "model instance {:?} rotation must map cube axes onto cube axes",
                instance.asset_path
            ));
        }
    }
    let rotation = DQuat::from_xyzw(
        rotation_f32.x as f64,
        rotation_f32.y as f64,
        rotation_f32.z as f64,
        rotation_f32.w as f64,
    ).normalize();
    Ok((rotation, rotation_f32))
}

impl EditorDocument for SceneDocument {
    const KIND: DocumentKind = DocumentKind::Scene;

    fn record_count(&self) -> usize {
        self.models.len()
            + self.scene_object_settings.len()
            + self.simulation_settings.object_initial_temperatures.len()
    }

    fn decode_payload(version: u16, payload: &[u8]) -> Result<Self, String> {
        match version {
            1..=4 => {
                let old: LegacySceneDocument = codec()
                    .deserialize(payload)
                    .map_err(|error| format!("decode version {version} scene: {error}"))?;
                Ok(Self {
                    name: old.name,
                    config: old.config,
                    models: old.models,
                    scene_object_settings: Vec::new(),
                    simulation_settings: SceneSimulationSettings::default(),
                })
            }
            5 => {
                let old: LegacySceneV5Document = codec()
                    .deserialize(payload)
                    .map_err(|error| format!("decode version 5 scene: {error}"))?;
                Ok(Self {
                    name: old.name,
                    config: old.config,
                    models: old.models,
                    scene_object_settings: old.scene_object_settings,
                    simulation_settings: SceneSimulationSettings::default(),
                })
            }
            FORMAT_VERSION => codec()
                .deserialize(payload)
                .map_err(|error| format!("decode native scene document: {error}")),
            _ => Err(format!("unsupported .vxscene schema version {version}")),
        }
    }

    fn validate(&self) -> Result<(), String> {
        validate_name(&self.name, "scene")?;
        self.config.validate().map_err(str::to_owned)?;
        if self.record_count() as u64 > MAX_DOCUMENT_ITEMS {
            return Err("scene contains too many model instances".into());
        }
        for model in &self.models {
            model.validate()?;
        }
        if self
            .scene_object_settings
            .windows(2)
            .any(|pair| pair[0].model_index >= pair[1].model_index)
        {
            return Err("scene object settings must have unique sorted model indices".into());
        }
        for settings in &self.scene_object_settings {
            if settings.model_index >= self.models.len() {
                return Err("scene object settings refer to a missing model instance".into());
            }
            settings.validate()?;
        }
        self.simulation_settings.validate(self.models.len())?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInstanceDocument {
    pub asset_path: PathBuf,
    pub translation_m: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl ModelInstanceDocument {
    pub fn validate(&self) -> Result<(), String> {
        if self.asset_path.as_os_str().is_empty()
            || !self.translation_m.is_finite()
            || !self.rotation.is_finite()
            || self.rotation.length_squared() < 1e-8
            || !self.scale.is_finite()
            || self.scale.min_element() <= 0.0
        {
            return Err("model instance has an invalid path or transform".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentDefinition {
    pub name: String,
    pub scene_path: PathBuf,
    pub random_seed: u64,
    pub duration_s: f64,
    pub time_scale: f64,
}

impl ExperimentDefinition {
    pub fn validate(&self) -> Result<(), String> {
        validate_name(&self.name, "experiment")?;
        if self.scene_path.as_os_str().is_empty()
            || !self.duration_s.is_finite()
            || self.duration_s < 0.0
            || !self.time_scale.is_finite()
            || self.time_scale < 0.0
        {
            return Err("experiment has an invalid scene path or time settings".into());
        }
        Ok(())
    }
}

pub fn save_document<T: EditorDocument>(path: &Path, document: &T) -> Result<(), String> {
    validate_extension(path, T::KIND)?;
    document.validate()?;
    let payload = codec()
        .serialize(document)
        .map_err(|error| format!("encode native document: {error}"))?;
    if payload.len() as u64 > MAX_PAYLOAD_BYTES {
        return Err("native document exceeds the 512 MiB payload limit".into());
    }
    let record_count = document.record_count() as u64;
    if record_count > MAX_DOCUMENT_ITEMS {
        return Err("native document contains too many records".into());
    }

    let mut bytes = Vec::with_capacity(HEADER_BYTES + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(T::KIND as u8);
    bytes.push(0); // reserved flags
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    bytes.extend_from_slice(&crc32(&payload).to_le_bytes());
    bytes.extend_from_slice(&payload);
    write_atomically(path, &bytes)
}

pub fn load_document<T: EditorDocument>(path: &Path) -> Result<T, String> {
    validate_extension(path, T::KIND)?;
    let mut file = File::open(path).map_err(|error| format!("open native document: {error}"))?;
    let file_size = file
        .metadata()
        .map_err(|error| format!("read native document metadata: {error}"))?
        .len();
    if file_size < HEADER_BYTES as u64 || file_size > MAX_PAYLOAD_BYTES + HEADER_BYTES as u64 {
        return Err("native document file size is outside supported limits".into());
    }
    let mut header = [0_u8; HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|error| format!("read native document header: {error}"))?;
    if &header[0..4] != MAGIC {
        return Err("invalid voxel document signature".into());
    }
    let kind = DocumentKind::from_byte(header[4])?;
    if kind != T::KIND {
        return Err(format!(
            "expected .{}, found .{} document",
            T::KIND.extension(),
            kind.extension()
        ));
    }
    if header[5] != 0 {
        return Err("unsupported native document flags".into());
    }
    let version = u16::from_le_bytes([header[6], header[7]]);
    if version == 0 || version > FORMAT_VERSION {
        return Err(format!(
            "unsupported {} schema version {version}",
            kind.extension()
        ));
    }
    let payload_len = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let record_count = u64::from_le_bytes(header[16..24].try_into().unwrap());
    let expected_crc = u32::from_le_bytes(header[24..28].try_into().unwrap());
    if payload_len > MAX_PAYLOAD_BYTES
        || payload_len + HEADER_BYTES as u64 != file_size
        || record_count > MAX_DOCUMENT_ITEMS
    {
        return Err("native document header lengths exceed supported limits".into());
    }
    let mut payload = vec![0_u8; payload_len as usize];
    file.read_exact(&mut payload)
        .map_err(|error| format!("read native document payload: {error}"))?;
    if crc32(&payload) != expected_crc {
        return Err("native document checksum mismatch".into());
    }
    let document = T::decode_payload(version, &payload)?;
    if document.record_count() as u64 != record_count {
        return Err("native document record count does not match its header".into());
    }
    document.validate()?;
    Ok(document)
}

fn codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(MAX_PAYLOAD_BYTES)
        .reject_trailing_bytes()
}

fn validate_extension(path: &Path, kind: DocumentKind) -> Result<(), String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if !extension.eq_ignore_ascii_case(kind.extension()) {
        return Err(format!(
            "{} documents must use the .{} extension",
            kind.extension(),
            kind.extension()
        ));
    }
    Ok(())
}

fn write_atomically(path: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| "native document path has no file name".to_string())?;
    let mut temporary = PathBuf::new();
    let mut temporary_file = None;
    for _ in 0..16 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut candidate_name = file_name.to_os_string();
        candidate_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        temporary = parent.join(candidate_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => {
                temporary_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create temporary native document: {error}")),
        }
    }
    let mut file =
        temporary_file.ok_or_else(|| "could not reserve a temporary file".to_string())?;
    let write_result = file
        .write_all(contents)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write temporary native document: {error}"));
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("replace native document atomically: {error}"));
    }
    Ok(())
}

fn validate_name(name: &str, kind: &str) -> Result<(), String> {
    if name.trim().is_empty() || name.len() > 4096 {
        Err(format!("{kind} name must contain 1..=4096 UTF-8 bytes"))
    } else {
        Ok(())
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

/// Convenience constructor for editor-authored initial voxel data.
pub fn authored_voxel(
    center_m: DVec3,
    components: Vec<(u16, f32)>,
    velocity_m_s: Vec3,
    microstructure: Microstructure,
    kinematic: bool,
    temperature_kelvin: f32,
) -> crate::scene::VoxelSpawn {
    crate::scene::VoxelSpawn {
        center: center_m,
        composition: Composition {
            components: components.into_iter().collect(),
        },
        velocity: velocity_m_s,
        microstructure,
        kinematic,
        temperature_kelvin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneBuilder, VoxelSpawn};

    fn sample_model() -> VoxelModelDocument {
        let scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1).at_temperature(315.0))
            .build()
            .unwrap();
        VoxelModelDocument::from_scene("test-oak", &scene)
    }

    #[test]
    fn voxel_model_round_trips_full_rev_state_and_material_table() {
        let mut original = sample_model();
        original.materials[3].internal_friction_coefficient = 0.577_350_26;
        let payload = codec().serialize(&original).unwrap();
        let decoded: VoxelModelDocument = codec().deserialize(&payload).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.name, original.name);
        assert_eq!(decoded.materials.len(), original.materials.len());
        assert_eq!(decoded.materials[1].name, original.materials[1].name);
        assert_eq!(
            decoded.materials[3].internal_friction_coefficient,
            original.materials[3].internal_friction_coefficient
        );
        assert_eq!(decoded.voxels.len(), 1);
        assert_eq!(decoded.voxels[0].position_m, original.voxels[0].position_m);
        assert_eq!(
            decoded.voxels[0].rest_position_m,
            original.voxels[0].rest_position_m
        );
        assert_eq!(decoded.voxels[0].state.mass, original.voxels[0].state.mass);
        assert_eq!(
            decoded.voxels[0].state.velocity,
            original.voxels[0].state.velocity
        );
        assert_eq!(
            decoded.voxels[0].state.total_internal_energy_j(),
            original.voxels[0].state.total_internal_energy_j()
        );
        assert_eq!(
            decoded.voxels[0].state.composition.components,
            original.voxels[0].state.composition.components
        );
    }

    #[test]
    fn version_two_model_migrates_appended_material_property() {
        let mut current = sample_model();
        current.materials[3].internal_friction_coefficient = 0.5;
        current.groups.push(VoxelGroupDocument {
            name: "trunk".into(),
            voxels: vec![0],
        });
        let old = V2VoxelModelDocument {
            name: current.name.clone(),
            cell_size_m: current.cell_size_m,
            materials: current
                .materials
                .iter()
                .map(LegacyMaterialV2::from)
                .collect(),
            voxels: current.voxels.clone(),
            groups: current.groups.clone(),
            motion_constraints: current.motion_constraints.clone(),
            skeleton: current.skeleton.clone(),
        };
        let old_payload = codec().serialize(&old).unwrap();
        let migrated = VoxelModelDocument::decode_payload(2, &old_payload).unwrap();

        assert_eq!(migrated.name, current.name);
        assert_eq!(migrated.materials[3].name, current.materials[3].name);
        assert_eq!(migrated.materials[3].internal_friction_coefficient, 0.0);
        assert_eq!(migrated.groups[0].name, "trunk");
        assert_eq!(migrated.voxels.len(), current.voxels.len());
    }

    #[test]
    fn version_three_model_migrates_joint_limits_without_changing_old_poses() {
        let mut current = sample_model();
        current.materials[3].internal_friction_coefficient = 0.5;
        current.skeleton = Some(SkeletonDocument::with_root_bone("Root", DVec3::ZERO));
        let old = V3VoxelModelDocument {
            name: current.name.clone(),
            cell_size_m: current.cell_size_m,
            materials: current
                .materials
                .iter()
                .map(LegacyMaterialV3::from)
                .collect(),
            voxels: current.voxels.clone(),
            groups: current.groups.clone(),
            motion_constraints: current.motion_constraints.clone(),
            skeleton: current.skeleton.clone(),
        };
        let payload = codec().serialize(&old).unwrap();
        let migrated = VoxelModelDocument::decode_payload(3, &payload).unwrap();

        assert_eq!(migrated.materials[3].internal_friction_coefficient, 0.5);
        assert_eq!(
            migrated.materials[3].skeletal_joint_limit_rad,
            std::f32::consts::PI
        );
        assert!(migrated.skeleton.is_some());
    }

    #[test]
    fn skeleton_pose_clamps_rotation_and_segment_stretch_from_bound_materials() {
        let mut model = sample_model();
        let flesh = Material::component_id("flesh").unwrap();
        model.voxels[0].state.composition.components = smallvec::smallvec![(flesh, 1.0)];
        let mut skeleton = SkeletonDocument::with_root_bone("Root", DVec3::ZERO);
        let child = skeleton
            .add_bone("limb", Some(0), DVec3::new(0.01, 0.0, 0.0))
            .unwrap();
        skeleton.voxel_weights.push(VoxelBoneWeightDocument {
            voxel_index: 0,
            bone_index: child as u32,
            weight: 1.0,
        });
        let clip = skeleton.add_animation_clip("test", 1.0).unwrap();
        skeleton
            .set_keyframe(
                clip,
                child,
                BoneKeyframeDocument {
                    time_s: 1.0,
                    translation_m: DVec3::new(0.03, 0.0, 0.0),
                    rotation: Quat::from_rotation_z(150.0_f32.to_radians()),
                },
            )
            .unwrap();
        model.skeleton = Some(skeleton);

        let limits = model.skeleton_material_limits().unwrap();
        let pose = model
            .skeleton
            .as_ref()
            .unwrap()
            .bone_local_poses_at_with_material_limits(clip, 1.0, &limits)
            .unwrap()[child];
        let rest_length = model.skeleton.as_ref().unwrap().bones[child]
            .rest_translation_m
            .length();
        let allowed_length = rest_length * (1.0 + limits[child].max_extension_strain);
        let requested_angle = 2.0 * f64::from(pose.rotation.w.abs()).acos();

        assert!((pose.translation_m.length() - allowed_length).abs() < 1e-8);
        assert!((requested_angle - limits[child].max_joint_rotation_rad).abs() < 1e-5);
        assert!(limits[child].max_extension_strain > 0.4);
    }

    #[test]
    fn version_one_model_migrates_appended_material_property() {
        let current = sample_model();
        let old = V1VoxelModelDocument {
            name: current.name.clone(),
            cell_size_m: current.cell_size_m,
            materials: current
                .materials
                .iter()
                .map(LegacyMaterialV2::from)
                .collect(),
            voxels: current.voxels.clone(),
        };
        let old_payload = codec().serialize(&old).unwrap();
        let migrated = VoxelModelDocument::decode_payload(1, &old_payload).unwrap();

        assert_eq!(migrated.materials.len(), current.materials.len());
        assert!(migrated
            .materials
            .iter()
            .all(|material| material.internal_friction_coefficient == 0.0));
        assert!(migrated.groups.is_empty());
        assert!(migrated.motion_constraints.is_empty());
        assert!(migrated.skeleton.is_none());
    }

    #[test]
    fn rejects_corrupt_payload_and_wrong_extension() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("corrupt-document-{}.vxm", std::process::id()));
        let original = sample_model();
        save_document(&path, &original).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(load_document::<VoxelModelDocument>(&path)
            .unwrap_err()
            .contains("checksum"));
        assert!(validate_extension(Path::new("model.vxscene"), DocumentKind::Model).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn scene_document_validates_runtime_config_and_model_transform() {
        let mut scene = SceneDocument::new("empty", SceneConfig::default()).unwrap();
        scene
            .add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("assets/tree.vxm"),
                translation_m: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .unwrap();
        scene.validate().unwrap();
        assert_eq!(scene.record_count(), 1);
    }

    #[test]
    fn scene_object_settings_round_trip_and_legacy_scenes_migrate() {
        let mut scene = SceneDocument::new("authored", SceneConfig::default()).unwrap();
        scene
            .add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("assets/cube.vxm"),
                translation_m: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .unwrap();
        let settings = SceneObjectSettings {
            model_index: 0,
            initial_velocity_m_s: Some(Vec3::new(1.0, -2.0, 0.5)),
            force_n: Vec3::new(0.0, 12.0, 0.0),
            material_asset: Some(PathBuf::from("materials/iron.vxmat")),
            voxel_materials: vec![SceneVoxelMaterialOverride {
                voxel_index: 7,
                material_asset: PathBuf::from("materials/steel.vxmat"),
            }],
        };
        scene.set_object_settings(0, settings.clone()).unwrap();
        scene.set_object_initial_temperature(0, Some(525.0)).unwrap();
        scene
            .set_thermal_bonding(Some(crate::scene_welding::ThermalBondingSettings {
                material_name: "iron".into(),
                minimum_mass_fraction: 0.8,
                minimum_temperature_kelvin: 900.0,
                minimum_impact_energy_j: 0.01,
                minimum_contact_pressure_pa: 2.0e6,
                minimum_dwell_seconds: 0.02,
            }))
            .unwrap();
        let payload = codec().serialize(&scene).unwrap();
        let decoded = SceneDocument::decode_payload(FORMAT_VERSION, &payload).unwrap();
        assert_eq!(decoded.object_settings(0), settings);
        assert_eq!(decoded.simulation_settings, scene.simulation_settings);

        let legacy_v5 = LegacySceneV5Document {
            name: scene.name.clone(),
            config: scene.config.clone(),
            models: scene.models.clone(),
            scene_object_settings: scene.scene_object_settings.clone(),
        };
        let old_v5_payload = codec().serialize(&legacy_v5).unwrap();
        let migrated_v5 = SceneDocument::decode_payload(5, &old_v5_payload).unwrap();
        assert_eq!(migrated_v5.object_settings(0), settings);
        assert_eq!(migrated_v5.simulation_settings, SceneSimulationSettings::default());

        let legacy = LegacySceneDocument {
            name: scene.name.clone(),
            config: scene.config.clone(),
            models: scene.models.clone(),
        };
        let old_payload = codec().serialize(&legacy).unwrap();
        for version in 1..=4 {
            let migrated = SceneDocument::decode_payload(version, &old_payload).unwrap();
            assert!(migrated.scene_object_settings.is_empty());
            assert_eq!(migrated.simulation_settings, SceneSimulationSettings::default());
            assert_eq!(migrated.models.len(), 1);
        }
    }

    #[test]
    fn saving_scene_to_another_directory_keeps_relative_asset_targets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join("tmpfixture")
            .join(format!("scene-rebase-{}", std::process::id()));
        let old_directory = root.join("original");
        let new_directory = root.join("moved");
        fs::create_dir_all(&old_directory).unwrap();
        fs::create_dir_all(&new_directory).unwrap();
        let model_path = old_directory.join("cube.vxm");
        let material_path = old_directory.join("iron.vxmat");
        save_document(&model_path, &sample_model()).unwrap();
        crate::material_asset::save_material(
            &material_path,
            &Material::table().into_iter().find(|material| material.name == "iron").unwrap(),
        ).unwrap();

        let mut scene = SceneDocument::new("rebase", SceneConfig::default()).unwrap();
        scene.add_model(ModelInstanceDocument {
            asset_path: PathBuf::from("cube.vxm"),
            translation_m: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }).unwrap();
        scene.set_object_settings(0, SceneObjectSettings {
            model_index: 0,
            material_asset: Some(PathBuf::from("iron.vxmat")),
            voxel_materials: vec![SceneVoxelMaterialOverride {
                voxel_index: 0,
                material_asset: PathBuf::from("iron.vxmat"),
            }],
            ..SceneObjectSettings::defaults_for(0)
        }).unwrap();

        let old_scene_path = old_directory.join("scene.vxscene");
        let new_scene_path = new_directory.join("scene.vxscene");
        let moved = scene.rebased_paths(&old_scene_path, &new_scene_path).unwrap();
        assert_eq!(moved.models[0].asset_path, model_path);
        let settings = moved.object_settings(0);
        assert_eq!(settings.material_asset.as_deref(), Some(material_path.as_path()));
        assert_eq!(settings.voxel_materials[0].material_asset, material_path);
        save_document(&new_scene_path, &moved).unwrap();
        let reloaded: SceneDocument = load_document(&new_scene_path).unwrap();
        assert_eq!(reloaded.models[0].asset_path, model_path);
        assert!(reloaded.models[0].asset_path.exists());
        assert!(settings.material_asset.unwrap().exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deleting_and_duplicating_instances_keeps_settings_aligned() {
        let mut scene = SceneDocument::new("objects", SceneConfig::default()).unwrap();
        for index in 0..3 {
            scene
                .add_model(ModelInstanceDocument {
                    asset_path: PathBuf::from(format!("{index}.vxm")),
                    translation_m: DVec3::ZERO,
                    rotation: Quat::IDENTITY,
                    scale: Vec3::ONE,
                })
                .unwrap();
        }
        for (index, force) in [(0, Vec3::X), (2, Vec3::Z)] {
            scene
                .set_object_settings(
                    index,
                    SceneObjectSettings {
                        model_index: index,
                        force_n: force,
                        ..SceneObjectSettings::defaults_for(index)
                    },
                )
                .unwrap();
        }
        let duplicate = scene.duplicate_model_cells(0, IVec3::X).unwrap();
        assert_eq!(scene.object_settings(duplicate).force_n, Vec3::X);
        scene.remove_model(1).unwrap();
        assert_eq!(scene.object_settings(0).force_n, Vec3::X);
        assert_eq!(scene.object_settings(1).force_n, Vec3::Z);
        assert_eq!(scene.object_settings(2).force_n, Vec3::X);
        scene.validate().unwrap();
    }

    #[test]
    fn scene_build_applies_asset_materials_world_velocity_and_object_force() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("scene-object-settings-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("two.vxm");
        let source_scene = SceneBuilder::standard()
            .add_voxel(
                VoxelSpawn::at_cell([0, 0, 0], Material::component_id("iron").unwrap())
                    .at_temperature(340.0),
            )
            .add_voxel(
                VoxelSpawn::at_cell([1, 0, 0], Material::component_id("iron").unwrap())
                    .at_temperature(355.0),
            )
            .build()
            .unwrap();
        let model = VoxelModelDocument::from_scene("two", &source_scene);
        save_document(&model_path, &model).unwrap();

        let mut first_material = Material::table()
            .into_iter()
            .find(|material| material.name == "iron")
            .unwrap();
        first_material.name = "scene test alloy A".into();
        first_material.density *= 1.5;
        first_material.heat_capacity *= 1.4;
        let first_material_path = directory.join("alloy-a.vxmat");
        crate::material_asset::save_material(&first_material_path, &first_material).unwrap();

        let mut second_material = first_material.clone();
        second_material.name = "scene test alloy B".into();
        second_material.density *= 1.7;
        second_material.heat_capacity *= 0.8;
        let second_material_path = directory.join("alloy-b.vxmat");
        crate::material_asset::save_material(&second_material_path, &second_material).unwrap();

        let mut scene_document = SceneDocument::new("materials", SceneConfig::default()).unwrap();
        scene_document
            .add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("two.vxm"),
                translation_m: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .unwrap();
        scene_document
            .set_object_settings(
                0,
                SceneObjectSettings {
                    model_index: 0,
                    initial_velocity_m_s: Some(Vec3::new(1.25, -0.5, 0.25)),
                    force_n: Vec3::new(0.0, 20.0, 0.0),
                    material_asset: Some(PathBuf::from("alloy-a.vxmat")),
                    voxel_materials: vec![SceneVoxelMaterialOverride {
                        voxel_index: 1,
                        material_asset: PathBuf::from("alloy-b.vxmat"),
                    }],
                },
            )
            .unwrap();
        let scene = scene_document
            .build_scene(&directory.join("materials.vxscene"))
            .unwrap();

        let first_index = scene.voxels[0].composition.components[0].0 as usize;
        let second_index = scene.voxels[1].composition.components[0].0 as usize;
        assert_eq!(scene.materials[first_index].name, first_material.name);
        assert_eq!(scene.materials[second_index].name, second_material.name);
        assert!((scene.voxels[0].mass - first_material.density * 1.0e-6).abs() < 1.0e-9);
        assert!((scene.voxels[1].mass - second_material.density * 1.0e-6).abs() < 1.0e-9);
        assert!((scene.temperature_kelvin(0).unwrap() - 340.0).abs() < 1.0e-3);
        assert!((scene.temperature_kelvin(1).unwrap() - 355.0).abs() < 1.0e-3);
        assert!(scene
            .voxels
            .iter()
            .all(|voxel| { voxel.velocity == Vec3::new(1.25, -0.5, 0.25) }));
        let applied_force = scene
            .external_forces
            .iter()
            .copied()
            .fold(Vec3::ZERO, |sum, force| sum + force);
        assert!((applied_force - Vec3::new(0.0, 20.0, 0.0)).length() < 1.0e-5);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn scene_build_restores_skeletal_actuators_with_model_order_ranges() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("skeletal-scene-restore-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();

        let iron = Material::component_id("iron").unwrap();
        let source = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        let mut rigged_model = VoxelModelDocument::from_scene("rigged", &source);
        let mut skeleton = SkeletonDocument::with_root_bone("root", DVec3::ZERO);
        let limb = skeleton
            .add_bone("limb", Some(0), DVec3::new(0.01, 0.0, 0.0))
            .unwrap();
        for voxel_index in 0..rigged_model.voxels.len() {
            skeleton.voxel_weights.push(VoxelBoneWeightDocument {
                voxel_index: voxel_index as u32,
                bone_index: limb as u32,
                weight: 1.0,
            });
        }
        let clip = skeleton.add_animation_clip("bend", 1.0).unwrap();
        skeleton
            .set_keyframe(
                clip,
                limb,
                BoneKeyframeDocument {
                    time_s: 0.0,
                    translation_m: DVec3::new(0.01, 0.0, 0.0),
                    rotation: Quat::IDENTITY,
                },
            )
            .unwrap();
        skeleton
            .set_keyframe(
                clip,
                limb,
                BoneKeyframeDocument {
                    time_s: 1.0,
                    translation_m: DVec3::new(0.01, 0.0, 0.01),
                    rotation: Quat::IDENTITY,
                },
            )
            .unwrap();
        rigged_model.skeleton = Some(skeleton);
        let rigged_path = directory.join("rigged.vxm");
        save_document(&rigged_path, &rigged_model).unwrap();

        let plain_source = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .build()
            .unwrap();
        let plain_model = VoxelModelDocument::from_scene("plain", &plain_source);
        let plain_path = directory.join("plain.vxm");
        save_document(&plain_path, &plain_model).unwrap();

        let mut document = SceneDocument::new("skeletal restore", SceneConfig::default()).unwrap();
        for (asset_path, x, rotation) in [
            ("rigged.vxm", 0.0, Quat::IDENTITY),
            ("plain.vxm", 0.05, Quat::IDENTITY),
            (
                "rigged.vxm",
                0.10,
                Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
            ),
        ] {
            document
                .add_model(ModelInstanceDocument {
                    asset_path: PathBuf::from(asset_path),
                    translation_m: DVec3::new(x, 0.0, 0.0),
                    rotation,
                    scale: Vec3::ONE,
                })
                .unwrap();
        }

        let scene = document
            .build_scene(&directory.join("skeletal.vxscene"))
            .unwrap();
        assert_eq!(scene.skeletal_actuators.len(), 2);
        assert_eq!(scene.skeletal_actuators[0].voxel_range(), 0..2);
        assert_eq!(scene.skeletal_actuators[1].voxel_range(), 3..5);
        assert_eq!(scene.skeletal_actuators[0].clip_index, 0);
        assert_eq!(scene.skeletal_actuators[1].clip_index, 0);
        assert!(scene
            .skeletal_actuators
            .iter()
            .all(|actor| actor.total_scene_voxels() == scene.voxels.len()));
        assert_eq!(scene.voxels.len(), 5);
        assert!((scene.positions[2]
            - (plain_model.voxels[0].position_m + DVec3::new(0.05, 0.0, 0.0)))
            .length()
            < 1.0e-12);
        assert_eq!(scene.voxels[2].mass, plain_model.voxels[0].state.mass);
        let rotated_instance = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2).normalize();
        let rotated_instance = glam::DQuat::from_xyzw(
            f64::from(rotated_instance.x),
            f64::from(rotated_instance.y),
            f64::from(rotated_instance.z),
            f64::from(rotated_instance.w),
        );
        let expected_position = rotated_instance.normalize() * rigged_model.voxels[0].position_m
            + DVec3::new(0.10, 0.0, 0.0);
        let transform_error_m = scene.positions[3].distance(expected_position);
        assert!(
            transform_error_m < 1.0e-12,
            "second rig instance position mismatch: actual={:?}, expected={expected_position:?}, error={transform_error_m:.3e} m",
            scene.positions[3]
        );
        let restored_model: VoxelModelDocument = load_document(&rigged_path).unwrap();
        let restored_skeleton = restored_model.skeleton.unwrap();
        assert_eq!(restored_skeleton.clips.len(), 1);
        assert_eq!(restored_skeleton.clips[0].name, "bend");
        assert_eq!(restored_skeleton.clips[0].tracks[0].keyframes.len(), 2);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn version_four_voxel_model_remains_readable_after_scene_schema_bump() {
        let current = sample_model();
        let old_payload = codec().serialize(&current).unwrap();
        let loaded = VoxelModelDocument::decode_payload(4, &old_payload).unwrap();
        assert_eq!(loaded.voxels.len(), current.voxels.len());
        assert_eq!(loaded.materials.len(), current.materials.len());
        assert_eq!(
            loaded.voxels[0].state.total_internal_energy_j(),
            current.voxels[0].state.total_internal_energy_j()
        );
    }

    #[test]
    fn scene_object_force_is_mass_weighted_with_zero_com_torque() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join("tmpfixture")
            .join(format!("object-force-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("unequal.vxm");
        let iron = Material::component_id("iron").unwrap();
        let model_scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        let mut model = VoxelModelDocument::from_scene("unequal", &model_scene);
        model.voxels[1].state.mass *= 3.0;
        save_document(&model_path, &model).unwrap();

        let mut document = SceneDocument::new("force", SceneConfig::default()).unwrap();
        document
            .add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("unequal.vxm"),
                translation_m: DVec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .unwrap();
        let total_force = Vec3::new(0.0, 12.0, 0.0);
        document
            .set_object_settings(
                0,
                SceneObjectSettings {
                    model_index: 0,
                    force_n: total_force,
                    ..SceneObjectSettings::defaults_for(0)
                },
            )
            .unwrap();
        let scene = document
            .build_scene(&directory.join("force.vxscene"))
            .unwrap();

        let force_sum = scene
            .external_forces
            .iter()
            .copied()
            .fold(Vec3::ZERO, |sum, force| sum + force);
        assert!((force_sum - total_force).length() < 1.0e-5);
        let acceleration_a = scene.external_forces[0] / scene.voxels[0].mass;
        let acceleration_b = scene.external_forces[1] / scene.voxels[1].mass;
        let acceleration_tolerance = f32::EPSILON
            * acceleration_a.length().max(acceleration_b.length()).max(1.0);
        assert!(
            (acceleration_a - acceleration_b).length() <= acceleration_tolerance,
            "mass-weighted force should yield equal acceleration within one f32 relative epsilon: mass=({:?}, {:?}) kg, force=({:?}, {:?}) N, acceleration=({:?}, {:?}) m/s^2, tolerance={acceleration_tolerance} m/s^2",
            scene.voxels[0].mass,
            scene.voxels[1].mass,
            scene.external_forces[0],
            scene.external_forces[1],
            acceleration_a,
            acceleration_b,
        );

        let total_mass: f64 = scene.voxels.iter().map(|voxel| f64::from(voxel.mass)).sum();
        let center = scene
            .positions
            .iter()
            .zip(&scene.voxels)
            .fold(DVec3::ZERO, |sum, (position, voxel)| {
                sum + *position * f64::from(voxel.mass)
            })
            / total_mass;
        let torque = scene
            .positions
            .iter()
            .zip(&scene.external_forces)
            .fold(DVec3::ZERO, |sum, (position, force)| {
                sum + (*position - center).cross(force.as_dvec3())
            });
        assert!(torque.length() < 1.0e-8);

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn scene_object_force_is_conserved_over_mobile_mass_and_rejects_all_fixed() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join("tmpfixture")
            .join(format!("mobile-object-force-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("mixed.vxm");
        let iron = Material::component_id("iron").unwrap();
        let model_scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], iron))
            .build()
            .unwrap();
        save_document(&model_path, &VoxelModelDocument::from_scene("mixed", &model_scene)).unwrap();

        let mut document = SceneDocument::new("force", SceneConfig::default()).unwrap();
        document.add_model(ModelInstanceDocument {
            asset_path: PathBuf::from("mixed.vxm"),
            translation_m: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }).unwrap();
        let force = Vec3::new(0.0, 18.0, 0.0);
        document.set_object_settings(0, SceneObjectSettings {
            model_index: 0,
            force_n: force,
            ..SceneObjectSettings::defaults_for(0)
        }).unwrap();
        let scene = document.build_scene(&directory.join("mixed.vxscene")).unwrap();
        assert!(scene.voxels[0].kinematic);
        assert_eq!(scene.external_forces[0], Vec3::ZERO);
        assert!((scene.external_forces[1] - force).length() < 1.0e-5);

        let fixed_path = directory.join("fixed.vxm");
        let fixed_scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], iron).fixed())
            .build().unwrap();
        save_document(&fixed_path, &VoxelModelDocument::from_scene("fixed", &fixed_scene)).unwrap();
        document.models[0].asset_path = PathBuf::from("fixed.vxm");
        let fixed_result = document.build_scene(&directory.join("fixed.vxscene"));
        assert!(matches!(fixed_result, Err(error) if error.contains("no mobile mass")));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn same_named_distinct_material_profiles_get_distinct_scene_ids() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join("tmpfixture")
            .join(format!("same-name-materials-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("one.vxm");
        let source = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], Material::component_id("iron").unwrap()))
            .build().unwrap();
        save_document(&model_path, &VoxelModelDocument::from_scene("one", &source)).unwrap();
        let first = Material::table().into_iter().find(|material| material.name == "iron").unwrap();
        let mut second = first.clone();
        second.density *= 2.0;
        crate::material_asset::save_material(&directory.join("first.vxmat"), &first).unwrap();
        crate::material_asset::save_material(&directory.join("second.vxmat"), &second).unwrap();

        let mut document = SceneDocument::new("same names", SceneConfig::default()).unwrap();
        for index in 0..2 {
            document.add_model(ModelInstanceDocument {
                asset_path: PathBuf::from("one.vxm"),
                translation_m: DVec3::new(index as f64 * 0.02, 0.0, 0.0),
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            }).unwrap();
            document.set_object_settings(index, SceneObjectSettings {
                model_index: index,
                material_asset: Some(PathBuf::from(if index == 0 { "first.vxmat" } else { "second.vxmat" })),
                ..SceneObjectSettings::defaults_for(index)
            }).unwrap();
        }
        let scene = document.build_scene(&directory.join("same.vxscene")).unwrap();
        let first_id = scene.voxels[0].composition.components[0].0;
        let second_id = scene.voxels[1].composition.components[0].0;
        assert_ne!(first_id, second_id);
        assert_eq!(scene.materials[first_id as usize].name, "iron");
        assert_eq!(scene.materials[second_id as usize].name, "iron");
        assert_eq!(scene.materials[first_id as usize].density, first.density);
        assert_eq!(scene.materials[second_id as usize].density, second.density);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn material_override_preserves_each_thermal_cell_temperature() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("validation")
            .join("tmpfixture")
            .join(format!("thermal-material-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let model_path = directory.join("thermal.vxm");
        let source = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], Material::component_id("iron").unwrap()).at_temperature(300.0))
            .build().unwrap();
        let mut model = VoxelModelDocument::from_scene("thermal", &source);
        let state = &mut model.voxels[0].state;
        let old_material = Material::mixed(&state.composition, &model.materials);
        let cell_mass = state.mass / crate::voxel::THERMAL_CELL_COUNT as f32;
        let energies: Vec<_> = (0..crate::voxel::THERMAL_CELL_COUNT)
            .map(|index| f64::from(old_material.energy_at_temperature(cell_mass, 280.0 + index as f32 * 11.0)))
            .collect();
        state.set_thermal_cell_energies_j(&energies);
        save_document(&model_path, &model).unwrap();
        let mut replacement = Material::table().into_iter().find(|material| material.name == "iron").unwrap();
        replacement.name = "thermal replacement".into();
        replacement.heat_capacity *= 1.7;
        replacement.latent_fusion *= 1.4;
        crate::material_asset::save_material(&directory.join("replacement.vxmat"), &replacement).unwrap();
        let mut document = SceneDocument::new("thermal", SceneConfig::default()).unwrap();
        document.add_model(ModelInstanceDocument {
            asset_path: PathBuf::from("thermal.vxm"),
            translation_m: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }).unwrap();
        document.set_object_settings(0, SceneObjectSettings {
            model_index: 0,
            material_asset: Some(PathBuf::from("replacement.vxmat")),
            ..SceneObjectSettings::defaults_for(0)
        }).unwrap();
        let scene = document.build_scene(&directory.join("thermal.vxscene")).unwrap();
        let new_material = &scene.materials[scene.voxels[0].composition.components[0].0 as usize];
        let new_cell_mass = scene.voxels[0].mass / crate::voxel::THERMAL_CELL_COUNT as f32;
        for (index, expected_temperature) in (0..crate::voxel::THERMAL_CELL_COUNT)
            .map(|index| 280.0 + index as f32 * 11.0).enumerate() {
            let new_energy = scene.voxels[0].thermal_cell_energy_at(index) as f32;
            assert!((new_material.temperature(new_cell_mass, new_energy) - expected_temperature).abs() < 0.02);
        }

        document.set_object_initial_temperature(0, Some(510.0)).unwrap();
        document
            .set_thermal_bonding(Some(crate::scene_welding::ThermalBondingSettings {
                material_name: "iron".into(),
                minimum_mass_fraction: 1.0,
                minimum_temperature_kelvin: 450.0,
                minimum_impact_energy_j: 0.0,
                minimum_contact_pressure_pa: 1.0,
                minimum_dwell_seconds: 0.01,
            }))
            .unwrap();
        let hot_scene = document.build_scene(&directory.join("thermal.vxscene")).unwrap();
        let hot_material =
            &hot_scene.materials[hot_scene.voxels[0].composition.components[0].0 as usize];
        let hot_cell_mass = hot_scene.voxels[0].mass / crate::voxel::THERMAL_CELL_COUNT as f32;
        for cell_index in 0..crate::voxel::THERMAL_CELL_COUNT {
            let cell_energy = hot_scene.voxels[0].thermal_cell_energy_at(cell_index) as f32;
            assert!((hot_material.temperature(hot_cell_mass, cell_energy) - 510.0).abs() < 0.02);
        }
        assert!(hot_scene.thermal_bonding_rule().is_some());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn all_native_extensions_round_trip_through_atomic_files() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("document-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();

        let model_path = directory.join("oak.vxm");
        let model = sample_model();
        save_document(&model_path, &model).unwrap();
        let loaded_model: VoxelModelDocument = load_document(&model_path).unwrap();
        assert_eq!(loaded_model.voxels.len(), model.voxels.len());
        assert_eq!(
            loaded_model.voxels[0].state.total_internal_energy_j(),
            model.voxels[0].state.total_internal_energy_j()
        );

        let scene_path = directory.join("test.vxscene");
        let scene = SceneDocument::new("test", SceneConfig::default()).unwrap();
        save_document(&scene_path, &scene).unwrap();
        let loaded_scene: SceneDocument = load_document(&scene_path).unwrap();
        assert_eq!(loaded_scene.name, scene.name);
        assert_eq!(loaded_scene.config.gravity, scene.config.gravity);

        let project_path = directory.join("test.vxproj");
        let mut project = ProjectDocument::new("test-project");
        project.scenes.push(PathBuf::from("test.vxscene"));
        project.language = InterfaceLanguage::English;
        save_document(&project_path, &project).unwrap();
        let loaded_project: ProjectDocument = load_document(&project_path).unwrap();
        assert_eq!(loaded_project.language, InterfaceLanguage::English);
        assert_eq!(loaded_project.scenes, project.scenes);

        // A second save replaces the existing file without exposing a partial
        // document if writing the replacement fails before the final rename.
        project.name = "updated-project".into();
        save_document(&project_path, &project).unwrap();
        let replaced: ProjectDocument = load_document(&project_path).unwrap();
        assert_eq!(replaced.name, "updated-project");

        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn automatic_weights_are_normalized_and_follow_voxel_connectivity() {
        let scene = SceneBuilder::standard()
            .add_voxel(VoxelSpawn::at_cell([0, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([1, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([2, 0, 0], 1))
            .add_voxel(VoxelSpawn::at_cell([10, 0, 0], 1))
            .build()
            .unwrap();
        let mut model = VoxelModelDocument::from_scene("weighted-line", &scene);
        model.skeleton = Some(SkeletonDocument {
            bones: vec![
                BoneDocument {
                    name: "root".into(),
                    parent: None,
                    rest_translation_m: DVec3::new(0.005, 0.005, 0.005),
                    rest_rotation: Quat::IDENTITY,
                },
                BoneDocument {
                    name: "child".into(),
                    parent: Some(0),
                    rest_translation_m: DVec3::new(0.02, 0.0, 0.0),
                    rest_rotation: Quat::IDENTITY,
                },
            ],
            voxel_weights: Vec::new(),
            clips: Vec::new(),
        });

        let weights = generate_auto_bone_weights(&model, 4).unwrap();
        let sum_for = |voxel_index: u32| -> f32 {
            weights
                .iter()
                .filter(|weight| weight.voxel_index == voxel_index)
                .map(|weight| weight.weight)
                .sum()
        };
        let weight_for = |voxel_index: u32, bone_index: u32| -> f32 {
            weights
                .iter()
                .find(|weight| weight.voxel_index == voxel_index && weight.bone_index == bone_index)
                .map_or(0.0, |weight| weight.weight)
        };

        for voxel_index in 0..model.voxels.len() as u32 {
            assert!((sum_for(voxel_index) - 1.0).abs() < 1.0e-5);
        }
        assert!(weight_for(0, 0) > weight_for(0, 1));
        assert!(weight_for(2, 1) > weight_for(2, 0));
        assert_eq!(
            weights
                .iter()
                .filter(|weight| weight.voxel_index == 3)
                .count(),
            1
        );
        assert!(generate_auto_bone_weights(&model, 0).is_err());
    }

    #[test]
    fn skeleton_authors_ordered_bones_clips_and_replaceable_keyframes() {
        let mut skeleton =
            SkeletonDocument::with_root_bone("root", DVec3::new(0.005, 0.005, 0.005));
        assert_eq!(
            skeleton
                .add_bone("branch", Some(0), DVec3::new(0.0, 0.0, 0.2))
                .unwrap(),
            1
        );
        let clip = skeleton.add_animation_clip("bend", 2.0).unwrap();
        skeleton
            .set_keyframe(
                clip,
                1,
                BoneKeyframeDocument {
                    time_s: 1.0,
                    translation_m: DVec3::new(0.0, 0.0, 0.2),
                    rotation: Quat::IDENTITY,
                },
            )
            .unwrap();
        skeleton
            .set_keyframe(
                clip,
                1,
                BoneKeyframeDocument {
                    time_s: 0.0,
                    translation_m: DVec3::new(0.0, 0.0, 0.2),
                    rotation: Quat::IDENTITY,
                },
            )
            .unwrap();
        skeleton
            .set_keyframe(
                clip,
                1,
                BoneKeyframeDocument {
                    time_s: 1.0,
                    translation_m: DVec3::new(0.0, 0.0, 0.3),
                    rotation: Quat::IDENTITY,
                },
            )
            .unwrap();

        let frames = &skeleton.clips[0].tracks[0].keyframes;
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].time_s, 0.0);
        assert_eq!(frames[1].time_s, 1.0);
        assert_eq!(frames[1].translation_m.z, 0.3);
        assert!(skeleton.add_bone("orphan", None, DVec3::ZERO).is_err());
    }
}
