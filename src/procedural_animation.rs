//! Procedural in-place walking clips. The generated poses are ordinary skeleton
//! targets and preserve the model's mechanical/material state. Preview skinning
//! uses one rigid influence per anatomical segment; this is not a physics solve.
use crate::documents::{
    voxel_lattice_coordinate, BoneKeyframeDocument, EditorDocument, SkeletonDocument,
    VoxelBoneWeightDocument, VoxelModelDocument,
};
use glam::{DQuat, DVec3, Quat};
use serde::Serialize;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Debug)]
pub struct WalkOptions {
    pub clip_name: String,
    /// None infers the axis from separated ground contacts below a raised body.
    pub up_axis: Option<usize>,
    pub forward_direction: Option<DVec3>,
    pub duration_s: f64,
    pub stride_m: f64,
    pub lift_m: f64,
    pub stance_fraction: f64,
    pub sample_fps: f64,
}
impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            clip_name: "Walk".into(),
            up_axis: None,
            forward_direction: None,
            duration_s: 1.8,
            stride_m: 0.025,
            lift_m: 0.014,
            stance_fraction: 0.80,
            sample_fps: 120.0,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WalkReport {
    pub limb_count: usize,
    pub body_voxel_count: usize,
    pub weighted_voxel_count: usize,
    pub bone_count: usize,
    pub keyframe_count: usize,
    pub duration_s: f64,
    pub up_axis: usize,
    pub ground_plane_m: f64,
    pub max_foot_target_error_m: f64,
    pub max_segment_length_error_m: f64,
    pub max_stance_height_error_m: f64,
    pub loop_joint_error_m: f64,
    pub min_grounded_legs: usize,
}

#[derive(Clone, Debug)]
struct Leg {
    voxels: Vec<usize>,
    hip: DVec3,
    knee: DVec3,
    foot: DVec3,
    phase: f64,
    upper: usize,
    lower: usize,
    tip: usize,
}
struct Anatomy {
    body: Vec<usize>,
    center: DVec3,
    legs: Vec<Leg>,
    up_axis: usize,
}

fn axis(index: usize) -> DVec3 {
    match index {
        0 => DVec3::X,
        1 => DVec3::Y,
        _ => DVec3::Z,
    }
}
fn planar(value: DVec3, up: DVec3) -> DVec3 {
    value - up * value.dot(up)
}

fn components(cells: &[[i32; 3]], included: &[bool]) -> Vec<Vec<usize>> {
    let index: HashMap<_, _> = cells
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| included[*i])
        .map(|(i, c)| (c, i))
        .collect();
    let mut seen = vec![false; cells.len()];
    let mut result = Vec::new();
    for start in 0..cells.len() {
        if !included[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::from([start]);
        let mut group = Vec::new();
        while let Some(i) = queue.pop_front() {
            group.push(i);
            for d in [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ] {
                let c = std::array::from_fn(|a| cells[i][a].saturating_add(d[a]));
                if let Some(&next) = index.get(&c) {
                    if !seen[next] {
                        seen[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }
        result.push(group);
    }
    result
}

fn infer_anatomy(model: &VoxelModelDocument, up_axis: usize) -> Result<Anatomy, String> {
    if up_axis > 2 || model.voxels.is_empty() {
        return Err("model has no usable ground contacts".into());
    }
    let size = model.cell_size_m;
    let positions: Vec<_> = model.voxels.iter().map(|v| v.rest_position_m).collect();
    let cells: Vec<_> = positions
        .iter()
        .map(|p| voxel_lattice_coordinate(*p, size))
        .collect();
    if cells.iter().copied().collect::<HashSet<_>>().len() != positions.len() {
        return Err("duplicate voxels prevent anatomical segmentation".into());
    }
    let bottom = cells.iter().map(|p| p[up_axis]).min().unwrap();
    let contacts = components(
        &cells,
        &cells
            .iter()
            .map(|p| p[up_axis] == bottom)
            .collect::<Vec<_>>(),
    );
    if !(2..=16).contains(&contacts.len()) {
        return Err(format!(
            "axis {up_axis} has {} separate ground contacts; expected 2 to 16",
            contacts.len()
        ));
    }
    let plane_axes: Vec<_> = (0..3).filter(|&a| a != up_axis).collect();
    let mut columns: HashMap<[i32; 2], Vec<usize>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        columns
            .entry([c[plane_axes[0]], c[plane_axes[1]]])
            .or_default()
            .push(i);
    }
    let mut feet = Vec::new();
    for contact in &contacts {
        let foot =
            contact.iter().fold(DVec3::ZERO, |s, &i| s + positions[i]) / contact.len() as f64;
        let footprint: HashSet<_> = contact
            .iter()
            .map(|&i| [cells[i][plane_axes[0]], cells[i][plane_axes[1]]])
            .collect();
        let column: Vec<_> = footprint
            .iter()
            .flat_map(|p| columns[p].iter().copied())
            .collect();
        let level = column.iter().map(|&i| cells[i][up_axis]).max().unwrap();
        let knee_points: Vec<_> = column
            .iter()
            .copied()
            .filter(|&i| cells[i][up_axis] == level)
            .collect();
        let knee = knee_points
            .iter()
            .fold(DVec3::ZERO, |s, &i| s + positions[i])
            / knee_points.len() as f64;
        if knee[up_axis] - foot[up_axis] < size * 1.5 {
            return Err("ground contacts have no raised shin segment".into());
        }
        feet.push((foot, knee));
    }
    let knee_height = feet
        .iter()
        .map(|(_, k)| k[up_axis])
        .fold(f64::NEG_INFINITY, f64::max);
    let bulk: Vec<_> = positions
        .iter()
        .enumerate()
        .filter(|(_, p)| p[up_axis] > knee_height + size * 0.25)
        .map(|(i, _)| i)
        .collect();
    if bulk.is_empty() {
        return Err("could not find a raised body above the leg joints".into());
    }
    let center = bulk.iter().fold(DVec3::ZERO, |s, &i| s + positions[i]) / bulk.len() as f64;
    let lo = bulk
        .iter()
        .fold(DVec3::splat(f64::INFINITY), |b, &i| b.min(positions[i]))
        - DVec3::splat(size * 0.5);
    let hi = bulk
        .iter()
        .fold(DVec3::splat(f64::NEG_INFINITY), |b, &i| b.max(positions[i]))
        + DVec3::splat(size * 0.5);
    let is_body: Vec<_> = positions
        .iter()
        .map(|p| {
            p[up_axis] >= knee_height - size * 0.25
                && plane_axes
                    .iter()
                    .all(|&a| p[a] >= lo[a] - 1e-9 && p[a] <= hi[a] + 1e-9)
        })
        .collect();
    let body: Vec<_> = (0..positions.len()).filter(|&i| is_body[i]).collect();
    let parts = components(&cells, &is_body.iter().map(|b| !b).collect::<Vec<_>>());
    if parts.len() != contacts.len() {
        return Err(format!(
            "found {} leg contacts but {} limb branches after separating the body",
            contacts.len(),
            parts.len()
        ));
    }
    let mut membership = vec![usize::MAX; positions.len()];
    for (part, ids) in parts.iter().enumerate() {
        for &i in ids {
            membership[i] = part;
        }
    }
    let mut used = HashSet::new();
    let mut legs = Vec::new();
    for ((foot, knee), contact) in feet.into_iter().zip(contacts) {
        let part = membership[contact[0]];
        if part == usize::MAX
            || !used.insert(part)
            || contact.iter().any(|&i| membership[i] != part)
        {
            return Err("ground contacts do not each belong to a separate limb branch".into());
        }
        let delta = planar(knee - center, axis(up_axis));
        let mut t = 1.0_f64;
        for &a in &plane_axes {
            if delta[a].abs() > 1e-10 {
                let edge = if delta[a] > 0.0 { hi[a] } else { lo[a] };
                t = t.min((edge - center[a]) / delta[a]);
            }
        }
        let mut hip = center + delta * t;
        hip[up_axis] = knee[up_axis];
        if hip.distance(knee) < size * 0.75 {
            return Err("limb's upper segment is too short to rig automatically".into());
        }
        legs.push(Leg {
            voxels: parts[part].clone(),
            hip,
            knee,
            foot,
            phase: 0.0,
            upper: 0,
            lower: 0,
            tip: 0,
        });
    }
    let right = axis(plane_axes[0]);
    let forward = axis(plane_axes[1]);
    legs.sort_by(|a, b| {
        let a = a.foot - center;
        let b = b.foot - center;
        a.dot(forward)
            .atan2(a.dot(right))
            .total_cmp(&b.dot(forward).atan2(b.dot(right)))
    });
    let count = legs.len();
    for (i, leg) in legs.iter_mut().enumerate() {
        leg.phase = i as f64 / count as f64;
    }
    Ok(Anatomy {
        body,
        center,
        legs,
        up_axis,
    })
}

/// Returns forward displacement and height of a foot. The stance portion is
/// exactly on the ground. Position, velocity and acceleration match at lift-off,
/// touchdown and the loop boundary.
fn foot_path(phase: f64, options: &WalkOptions) -> (f64, f64) {
    let p = phase.rem_euclid(1.0);
    let d = options.stance_fraction;
    if p < d {
        return (options.stride_m * (0.5 - p / d), 0.0);
    }
    let t = (p - d) / (1.0 - d);
    let q = t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
    let m = -options.stride_m * (1.0 - d) / d;
    (
        (options.stride_m - m) * q + m * t - options.stride_m * 0.5,
        options.lift_m * 64.0 * t.powi(3) * (1.0 - t).powi(3),
    )
}

/// Exact two-link IK with the authored knee as the bend-plane pole. Bone
/// lengths and parent-local translations remain fixed at every keyed pose.
fn solve_leg(
    hip: DVec3,
    rest_upper: DVec3,
    rest_lower: DVec3,
    target: DVec3,
) -> Result<(DQuat, DQuat), String> {
    let l1 = rest_upper.length();
    let l2 = rest_lower.length();
    let delta = target - hip;
    let d = delta.length();
    if d < 1e-10 || d > (l1 + l2) * (1.0 - 1e-8) || d < (l1 - l2).abs() + 1e-9 {
        return Err("foot target exceeds joint reach; reduce stride or lift".into());
    }
    let along = delta / d;
    let pole = (rest_upper - along * rest_upper.dot(along)).normalize_or_zero();
    if pole.length_squared() < 0.5 {
        return Err("joint bend plane is singular".into());
    }
    let a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d);
    let h = (l1 * l1 - a * a).max(0.0).sqrt();
    let knee = hip + along * a + pole * h;
    let upper = DQuat::from_rotation_arc(rest_upper / l1, (knee - hip) / l1).normalize();
    let lower_global = DQuat::from_rotation_arc(rest_lower / l2, (target - knee) / l2).normalize();
    Ok((upper, (upper.conjugate() * lower_global).normalize()))
}
fn as_quat(q: DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32).normalize()
}

/// Authors one ordinary skeleton clip, committing only after validation. Auto
/// inference requires a raised body and distinct shins ending on a common plane.
pub fn author_walk(
    model: &mut VoxelModelDocument,
    options: &WalkOptions,
) -> Result<WalkReport, String> {
    model.validate()?;
    if model.skeleton.is_some() {
        return Err("source already has a skeleton; choose the original rig-free model".into());
    }
    if !options.duration_s.is_finite()
        || !(0.1..=10.0).contains(&options.duration_s)
        || !options.stride_m.is_finite()
        || !(0.0..=0.5).contains(&options.stride_m)
        || !options.lift_m.is_finite()
        || !(0.0..=0.5).contains(&options.lift_m)
        || !options.stance_fraction.is_finite()
        || !(0.55..=0.95).contains(&options.stance_fraction)
        || !options.sample_fps.is_finite()
        || !(15.0..=120.0).contains(&options.sample_fps)
    {
        return Err("invalid walk duration, stride, lift, stance fraction or sample rate".into());
    }
    let mut anatomy = match options.up_axis {
        Some(a) => infer_anatomy(model, a)?,
        None => {
            let mut valid = (0..3).filter_map(|a| infer_anatomy(model, a).ok());
            let result = valid
                .next()
                .ok_or("could not infer upright legs; set up_axis or author explicit bones")?;
            if valid.next().is_some() {
                return Err("upright direction is ambiguous; specify up_axis".into());
            }
            result
        }
    };
    let up = axis(anatomy.up_axis);
    let fallback = if anatomy.up_axis == 2 {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let forward = planar(options.forward_direction.unwrap_or(fallback), up).normalize_or_zero();
    if !forward.is_finite() || forward.length_squared() < 0.5 {
        return Err("forward_direction must be finite and horizontal".into());
    }
    let mut skeleton = SkeletonDocument::with_root_bone("Body", anatomy.center);
    for (i, leg) in anatomy.legs.iter_mut().enumerate() {
        leg.upper = skeleton.add_bone(
            format!("Leg{:02}_Hip", i + 1),
            Some(0),
            leg.hip - anatomy.center,
        )?;
        leg.lower = skeleton.add_bone(
            format!("Leg{:02}_Knee", i + 1),
            Some(leg.upper),
            leg.knee - leg.hip,
        )?;
        leg.tip = skeleton.add_bone(
            format!("Leg{:02}_Foot", i + 1),
            Some(leg.lower),
            leg.foot - leg.knee,
        )?;
    }
    let clip = skeleton.add_animation_clip(&options.clip_name, options.duration_s)?;
    let samples = (options.duration_s * options.sample_fps).ceil() as usize;
    for sample in 0..=samples {
        let phase = if sample == samples {
            0.0
        } else {
            sample as f64 / samples as f64
        };
        let time_s = options.duration_s * sample as f64 / samples as f64;
        let paths: Vec<_> = anatomy
            .legs
            .iter()
            .map(|leg| foot_path(phase + leg.phase, options))
            .collect();
        let mut support = DVec3::ZERO;
        let mut total = 0.0;
        for (leg, &(_, height)) in anatomy.legs.iter().zip(&paths) {
            let weight = if options.lift_m > 1e-12 {
                1.0 - height / options.lift_m
            } else {
                1.0
            };
            support += planar(leg.foot - anatomy.center, up) * weight;
            total += weight;
        }
        let body_offset = support / total.max(1e-9) * 0.08
            + up * (options.lift_m * 0.045 * (phase * std::f64::consts::TAU * 2.0).sin());
        skeleton.set_keyframe(
            clip,
            0,
            BoneKeyframeDocument {
                time_s,
                translation_m: anatomy.center + body_offset,
                rotation: Quat::IDENTITY,
            },
        )?;
        for (leg, &(stride, height)) in anatomy.legs.iter().zip(&paths) {
            let target = leg.foot + forward * stride + up * height;
            let (upper, lower) = solve_leg(
                leg.hip + body_offset,
                leg.knee - leg.hip,
                leg.foot - leg.knee,
                target,
            )
            .map_err(|e| format!("{} at {time_s:.3} s: {e}", skeleton.bones[leg.upper].name))?;
            for (bone, rotation) in [
                (leg.upper, as_quat(upper)),
                (leg.lower, as_quat(lower)),
                (leg.tip, Quat::IDENTITY),
            ] {
                skeleton.set_keyframe(
                    clip,
                    bone,
                    BoneKeyframeDocument {
                        time_s,
                        translation_m: skeleton.bones[bone].rest_translation_m,
                        rotation,
                    },
                )?;
            }
        }
    }
    skeleton.voxel_weights = anatomy
        .body
        .iter()
        .map(|&i| VoxelBoneWeightDocument {
            voxel_index: i as u32,
            bone_index: 0,
            weight: 1.0,
        })
        .collect();
    for leg in &anatomy.legs {
        for &i in &leg.voxels {
            let below = model.voxels[i].rest_position_m.dot(up)
                < leg.knee.dot(up) - model.cell_size_m * 0.25;
            skeleton.voxel_weights.push(VoxelBoneWeightDocument {
                voxel_index: i as u32,
                bone_index: if below { leg.lower } else { leg.upper } as u32,
                weight: 1.0,
            });
        }
    }
    skeleton.voxel_weights.sort_by_key(|w| w.voxel_index);
    let mut report = WalkReport {
        limb_count: anatomy.legs.len(),
        body_voxel_count: anatomy.body.len(),
        weighted_voxel_count: skeleton.voxel_weights.len(),
        bone_count: skeleton.bones.len(),
        keyframe_count: skeleton.clips[clip]
            .tracks
            .iter()
            .map(|t| t.keyframes.len())
            .sum(),
        duration_s: options.duration_s,
        up_axis: anatomy.up_axis,
        ground_plane_m: anatomy.legs[0].foot.dot(up) - model.cell_size_m * 0.5,
        max_foot_target_error_m: 0.0,
        max_segment_length_error_m: 0.0,
        max_stance_height_error_m: 0.0,
        loop_joint_error_m: 0.0,
        min_grounded_legs: anatomy.legs.len(),
    };
    // Include between-key samples: successful keyed IK alone does not prove
    // that the interpolated viewport clip preserves the foot contacts.
    let check_samples = samples * 4;
    for sample in 0..=check_samples {
        let phase = sample as f64 / check_samples as f64;
        let joints = skeleton.bone_global_positions_at(clip, options.duration_s * phase)?;
        let mut grounded = 0;
        for leg in &anatomy.legs {
            let (stride, height) = foot_path(phase + leg.phase, options);
            let target = leg.foot + forward * stride + up * height;
            report.max_foot_target_error_m = report
                .max_foot_target_error_m
                .max(joints[leg.tip].distance(target));
            for (a, b, length) in [
                (leg.upper, leg.lower, leg.hip.distance(leg.knee)),
                (leg.lower, leg.tip, leg.knee.distance(leg.foot)),
            ] {
                report.max_segment_length_error_m = report
                    .max_segment_length_error_m
                    .max((joints[a].distance(joints[b]) - length).abs());
            }
            if height < 1e-10 {
                grounded += 1;
                report.max_stance_height_error_m = report
                    .max_stance_height_error_m
                    .max((joints[leg.tip].dot(up) - leg.foot.dot(up)).abs());
            }
        }
        report.min_grounded_legs = report.min_grounded_legs.min(grounded);
    }
    let start = skeleton.bone_global_positions_at(clip, 0.0)?;
    let end = skeleton.bone_global_positions_at(clip, options.duration_s)?;
    report.loop_joint_error_m = start
        .iter()
        .zip(end)
        .map(|(a, b)| a.distance(b))
        .fold(0.0, f64::max);
    if report.max_foot_target_error_m > model.cell_size_m * 0.01 {
        return Err(format!(
            "interpolated foot target error is {:.6} m; increase sample_fps or reduce stride/lift",
            report.max_foot_target_error_m
        ));
    }
    model.skeleton = Some(skeleton);
    if let Err(error) = model.validate() {
        model.skeleton = None;
        return Err(error);
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        material::Material,
        scene::{Scene, SceneConfig, VoxelSpawn},
    };
    fn creature() -> VoxelModelDocument {
        let mut cells = HashSet::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in 1..=5 {
                    cells.insert([x, y, z]);
                }
            }
        }
        for t in -8_i32..=8 {
            cells.insert([t, 0, 0]);
            cells.insert([0, t, 0]);
        }
        for z in -6..0 {
            for p in [[-8, 0, z], [8, 0, z], [0, -8, z], [0, 8, z]] {
                cells.insert(p);
            }
        }
        let material = Material::component_id("wood").unwrap();
        let mut ordered: Vec<_> = cells.into_iter().collect();
        ordered.sort();
        let scene = Scene::from_voxels(
            SceneConfig::default(),
            ordered
                .into_iter()
                .map(|p| VoxelSpawn::at_cell(p, material))
                .collect(),
        )
        .unwrap();
        VoxelModelDocument::from_scene("Four legs", &scene)
    }
    #[test]
    fn detects_real_connected_limbs_on_negative_half_cell_lattice() {
        let model = creature();
        let anatomy = infer_anatomy(&model, 2).unwrap();
        assert_eq!(anatomy.legs.len(), 4);
        assert_eq!(
            anatomy.body.len() + anatomy.legs.iter().map(|l| l.voxels.len()).sum::<usize>(),
            model.voxels.len()
        );
        assert!(infer_anatomy(&model, 0).is_err());
        assert!(infer_anatomy(&model, 1).is_err());
    }
    #[test]
    fn ik_preserves_lengths_and_hits_target() {
        let a = DVec3::new(0.06, 0.0, 0.0);
        let b = DVec3::new(0.0, 0.0, -0.07);
        let target = DVec3::new(0.045, 0.012, -0.06);
        let (upper, lower) = solve_leg(DVec3::ZERO, a, b, target).unwrap();
        assert!((upper * a + upper * lower * b).distance(target) < 1e-12);
        assert!(((upper * a).length() - a.length()).abs() < 1e-12);
        assert!(solve_leg(DVec3::ZERO, a, b, DVec3::X).is_err());
    }
    #[test]
    fn planted_contacts_loop_and_rigid_body_are_preserved() {
        let mut model = creature();
        let options = WalkOptions::default();
        let report = author_walk(&mut model, &options).unwrap();
        assert_eq!(report.limb_count, 4);
        assert_eq!(report.bone_count, 13);
        assert!(report.max_foot_target_error_m < 1e-4, "{report:?}");
        assert!(report.max_stance_height_error_m < 1e-5, "{report:?}");
        assert!(report.min_grounded_legs >= 3);
        assert!(report.max_segment_length_error_m < 1e-10);
        assert!(report.loop_joint_error_m < 1e-10);
        assert_eq!(report.weighted_voxel_count, model.voxels.len());
        assert!(model
            .skeleton
            .as_ref()
            .unwrap()
            .voxel_weights
            .iter()
            .all(|w| w.weight == 1.0));
    }
    #[test]
    fn invalid_stride_does_not_commit_partial_rig() {
        let mut model = creature();
        let options = WalkOptions {
            stride_m: 0.5,
            ..WalkOptions::default()
        };
        assert!(author_walk(&mut model, &options).is_err());
        assert!(model.skeleton.is_none());
    }
}
