use crate::{
    bond::EMPTY,
    scene::{Scene, CELL},
    tool::Tool,
    ui::Rect,
};
use bytemuck::{Pod, Zeroable};
use glam::{DQuat, DVec3, Mat4, Quat, Vec3};
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    sync::{Arc, OnceLock},
};
use voxel_physics_lab::documents::{
    BoneMaterialLimits, BoneSkinTransform, SkeletonDocument, VoxelBoneWeightDocument,
    VoxelModelDocument,
};
use voxel_physics_lab::voxel_surface::exterior_face_masks;
use voxel_physics_lab::world::{GeneratedCellContent, GeneratedWorld, WORLD_SIDE_M};
use wgpu::util::DeviceExt;
use winit::window::Window;

// wgpu 0.19.4 API checked with cargo check; GPU/window runtime validation remains.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    center: [f32; 3],
    size: [f32; 3],
    color: [f32; 4],
    rotation: [f32; 4],
    surface: f32,
    skin_index: u32,
    thermal_emission: [f32; 3],
}

/// Lightweight render metadata retained while the authored `Scene` is owned
/// by its physics worker. Build again after authored material edits.
pub struct AuthoredPhysicsRenderSource {
    dynamic_scene_indices: Vec<usize>,
    material_colors: Vec<[f32; 4]>,
    kinematic_poses: Vec<AuthoredKinematicPose>,
    cache_key: u64,
}

#[derive(Clone, Copy)]
struct AuthoredKinematicPose {
    scene_index: usize,
    position: DVec3,
    rotation: Quat,
    color: [f32; 4],
    thermal_emission: [f32; 3],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinInfluenceGpu {
    bone_indices: [u32; 4],
    weights: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoneSkinTransformGpu {
    translation: [f32; 4],
    rotation: [f32; 4],
}

fn storage_buffer<T: Pod>(device: &wgpu::Device, count: usize, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count.max(1) * std::mem::size_of::<T>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_skinning_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    influences: &wgpu::Buffer,
    bones: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("voxel skeleton skinning buffers"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: influences.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bones.as_entire_binding(),
            },
        ],
    })
}

fn pack_skin_influences(
    voxel_count: usize,
    weights: &[VoxelBoneWeightDocument],
) -> Vec<SkinInfluenceGpu> {
    let mut packed = vec![SkinInfluenceGpu::zeroed(); voxel_count];
    let mut counts = vec![0_u8; voxel_count];
    for binding in weights {
        let voxel_index = binding.voxel_index as usize;
        let Some(influence_count) = counts.get_mut(voxel_index) else {
            continue;
        };
        if *influence_count >= 4 {
            continue;
        }
        let slot = usize::from(*influence_count);
        let Some(influence) = packed.get_mut(voxel_index) else {
            continue;
        };
        influence.bone_indices[slot] = binding.bone_index;
        influence.weights[slot] = binding.weight.max(0.0);
        *influence_count += 1;
    }
    packed
}

fn pack_bone_transforms(transforms: &[BoneSkinTransform]) -> Vec<BoneSkinTransformGpu> {
    transforms
        .iter()
        .map(|transform| BoneSkinTransformGpu {
            translation: [
                transform.translation_m.x as f32,
                transform.translation_m.y as f32,
                transform.translation_m.z as f32,
                0.0,
            ],
            rotation: [
                transform.rotation.x as f32,
                transform.rotation.y as f32,
                transform.rotation.z as f32,
                transform.rotation.w as f32,
            ],
        })
        .collect()
}
fn hash_vec3<H: Hasher>(value: Vec3, hasher: &mut H) {
    for component in value.to_array() {
        component.to_bits().hash(hasher);
    }
}

fn bone_gizmo_key(
    skeleton: &SkeletonDocument,
    selected_bone: Option<usize>,
    cell_size: f32,
    posed_positions: Option<&[DVec3]>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    skeleton.bones.len().hash(&mut hasher);
    selected_bone.hash(&mut hasher);
    cell_size.to_bits().hash(&mut hasher);
    for bone in &skeleton.bones {
        bone.parent.hash(&mut hasher);
        for value in bone.rest_translation_m.to_array() {
            value.to_bits().hash(&mut hasher);
        }
        for value in bone.rest_rotation.to_array() {
            value.to_bits().hash(&mut hasher);
        }
    }
    if let Some(positions) = posed_positions {
        positions.len().hash(&mut hasher);
        for position in positions {
            for value in position.to_array() {
                value.to_bits().hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

fn build_bone_gizmos(
    skeleton: &SkeletonDocument,
    selected_bone: Option<usize>,
    cell_size: f32,
    posed_positions: Option<&[DVec3]>,
) -> Vec<Instance> {
    let mut global: Vec<(DVec3, DQuat)> = Vec::with_capacity(skeleton.bones.len());
    for bone in &skeleton.bones {
        let local_position = bone.rest_translation_m;
        let local_rotation = DQuat::from_xyzw(
            f64::from(bone.rest_rotation.x),
            f64::from(bone.rest_rotation.y),
            f64::from(bone.rest_rotation.z),
            f64::from(bone.rest_rotation.w),
        )
        .normalize();
        let world = match bone.parent {
            Some(parent) if (parent as usize) < global.len() => {
                let (parent_position, parent_rotation) = global[parent as usize];
                (
                    parent_position + parent_rotation * local_position,
                    (parent_rotation * local_rotation).normalize(),
                )
            }
            _ => (local_position, local_rotation),
        };
        global.push(world);
    }
    if let Some(positions) = posed_positions.filter(|positions| positions.len() == global.len()) {
        for (transform, position) in global.iter_mut().zip(positions.iter().copied()) {
            transform.0 = position;
        }
    }

    let mut instances = Vec::with_capacity(skeleton.bones.len().saturating_mul(2));
    let radius = (cell_size.max(0.01) * 0.8).clamp(0.006, 0.022);
    for (index, bone) in skeleton.bones.iter().enumerate() {
        let (position, _) = global[index];
        let selected = selected_bone == Some(index);
        let joint_color = if selected {
            [0.08, 1.0, 0.72, 1.0]
        } else if bone.parent.is_none() {
            [1.0, 0.72, 0.18, 1.0]
        } else {
            [0.18, 0.72, 1.0, 1.0]
        };
        let joint_radius = if bone.parent.is_none() {
            radius * 1.45
        } else {
            radius
        };
        instances.push(Instance {
            center: position.as_vec3().to_array(),
            size: [joint_radius * 2.0; 3],
            color: joint_color,
            rotation: Quat::IDENTITY.to_array(),
            surface: 0.0,
            skin_index: u32::MAX,
            thermal_emission: [0.0; 3],
        });
        if let Some(parent) = bone
            .parent
            .map(|parent| parent as usize)
            .filter(|&parent| parent < index)
        {
            let start = global[parent].0.as_vec3();
            let end = position.as_vec3();
            let direction = end - start;
            let length = direction.length();
            if length > 1e-6 {
                instances.push(Instance {
                    center: ((start + end) * 0.5).to_array(),
                    size: [radius * 0.82, radius * 0.82, length + radius],
                    color: if selected {
                        [0.12, 0.92, 0.68, 1.0]
                    } else {
                        [0.28, 0.62, 0.86, 1.0]
                    },
                    rotation: Quat::from_rotation_arc(Vec3::Z, direction / length).to_array(),
                    surface: 0.0,
                    skin_index: u32::MAX,
                    thermal_emission: [0.0; 3],
                });
            }
        }
    }
    instances
}
fn surface_kind(content: GeneratedCellContent) -> f32 {
    match content {
        GeneratedCellContent::Terrain => 1.0,
        GeneratedCellContent::PlantWood => 2.0,
        _ => 0.0,
    }
}

fn material_display_color(name: &str) -> [f32; 3] {
    let name = name.to_ascii_lowercase();
    if ["wood", "oak", "timber"]
        .iter()
        .any(|part| name.contains(part))
    {
        [0.55, 0.32, 0.14]
    } else if ["iron", "steel", "metal", "aluminum", "aluminium"]
        .iter()
        .any(|part| name.contains(part))
    {
        [0.58, 0.62, 0.66]
    } else if ["stone", "rock", "granite", "concrete"]
        .iter()
        .any(|part| name.contains(part))
    {
        [0.47, 0.49, 0.52]
    } else if ["water", "ice"].iter().any(|part| name.contains(part)) {
        [0.16, 0.48, 0.86]
    } else if ["flesh", "skin", "meat"]
        .iter()
        .any(|part| name.contains(part))
    {
        [0.86, 0.43, 0.43]
    } else if name.contains("glass") {
        [0.48, 0.78, 0.84]
    } else if ["sand", "soil", "dirt"]
        .iter()
        .any(|part| name.contains(part))
    {
        [0.66, 0.52, 0.32]
    } else {
        [0.68, 0.70, 0.73]
    }
}

fn voxel_display_color(scene: &Scene, voxel_index: usize) -> [f32; 4] {
    let Some(voxel) = scene.voxels.get(voxel_index) else {
        return [0.68, 0.70, 0.73, 1.0];
    };
    let mut color = [0.0; 3];
    let mut weight_sum = 0.0;
    for &(component, fraction) in &voxel.composition.components {
        let Some(material) = scene.materials.get(component as usize) else {
            continue;
        };
        let weight = fraction.max(0.0);
        let component_color = material_display_color(&material.name);
        for channel in 0..3 {
            color[channel] += component_color[channel] * weight;
        }
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        for channel in &mut color {
            *channel /= weight_sum;
        }
    } else {
        color = [0.68, 0.70, 0.73];
    }
    [color[0], color[1], color[2], 1.0]
}

/// Display-only blackbody color fit, based on the Planckian locus. The fit
/// approximates the chromaticity of a blackbody in sRGB without evaluating a
/// spectrum per voxel. Its output is a visible-light color, not calibrated
/// radiative power; `thermal_emission` applies the viewer exposure and a
/// generic greybody phase factor.
fn blackbody_srgb_sample(temperature_kelvin: f32) -> [f32; 3] {
    let temperature = (temperature_kelvin / 100.0).clamp(1.0, 400.0);
    if temperature <= 10.0 {
        return [1.0, 0.0, 0.0];
    }
    let red;
    let green;
    let blue;
    if temperature <= 66.0 {
        red = 255.0;
        green = (99.470_8 * temperature.ln() - 161.119_57).clamp(0.0, 255.0);
        blue = if temperature <= 19.0 {
            0.0
        } else {
            (138.517_73 * (temperature - 10.0).ln() - 305.044_8).clamp(0.0, 255.0)
        };
    } else {
        red = (329.698_73 * (temperature - 60.0).powf(-0.133_204_76)).clamp(0.0, 255.0);
        green = (288.122_16 * (temperature - 60.0).powf(-0.075_514_85)).clamp(0.0, 255.0);
        blue = 255.0;
    }
    [red / 255.0, green / 255.0, blue / 255.0]
}

/// A cached 100 K table keeps per-REV updates to one lookup and one linear
/// interpolation; the Planckian RGB fit is evaluated only once per process.
fn blackbody_linear(temperature_kelvin: f32) -> [f32; 3] {
    const LAST_INDEX: usize = 400;
    static LUT: OnceLock<Vec<[f32; 3]>> = OnceLock::new();
    let lut = LUT.get_or_init(|| {
        (0..=LAST_INDEX)
            .map(|index| {
                let temperature = index as f32 * 100.0;
                // Planck radiance at 650 nm relative to 1300 K sets a fixed
                // viewer exposure. Cache it with the chromaticity so a frame
                // update performs no logarithms or exponentials per voxel.
                const C2_OVER_WAVELENGTH: f32 = 14_387_769.0 / 650.0;
                let exposure = if temperature <= 650.0 { 0.0 } else {
                    (C2_OVER_WAVELENGTH / 1_300.0).exp_m1()
                        / (C2_OVER_WAVELENGTH / temperature).exp_m1()
                };
                blackbody_srgb_sample(temperature).map(|channel| {
                    let linear = if channel <= 0.04045 {
                        channel / 12.92
                    } else {
                        ((channel + 0.055) / 1.055).powf(2.4)
                    };
                    linear * exposure
                })
            })
            .collect()
    });
    let position = (temperature_kelvin / 100.0).clamp(0.0, LAST_INDEX as f32);
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(LAST_INDEX);
    let mix = position - lower as f32;
    std::array::from_fn(|channel| lut[lower][channel] * (1.0 - mix) + lut[upper][channel] * mix)
}

/// Convert the actual voxel temperature and phase fractions into linear
/// display emission. Cold albedo stays untouched; thermal light is additive.
/// Fixed exposure makes the first visible glow faint red, while the
/// Planckian color fit progresses through orange/yellow to blue-white.
fn thermal_emission(temperature_kelvin: f32, liquid_fraction: f32, gas_fraction: f32) -> [f32; 3] {
    if !temperature_kelvin.is_finite() || temperature_kelvin <= 650.0 {
        return [0.0; 3];
    }
    let liquid = if liquid_fraction.is_finite() {
        liquid_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let gas = if gas_fraction.is_finite() {
        gas_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let solid = (1.0 - liquid - gas).clamp(0.0, 1.0);
    // Generic, non-material-calibrated greybody emissivity. The vapor factor
    // only changes visible emission; it does not imply gas transport/pressure.
    let emissivity = (0.82 * solid + 0.96 * liquid + 0.48 * gas).clamp(0.0, 1.0);
    let color = blackbody_linear(temperature_kelvin);
    [
        color[0] * emissivity,
        color[1] * emissivity,
        color[2] * emissivity,
    ]
}

impl AuthoredPhysicsRenderSource {
    pub fn from_scene(scene: &Scene) -> Self {
        let mut dynamic_scene_indices = Vec::new();
        let mut material_colors = Vec::with_capacity(scene.voxels.len());
        let mut kinematic_poses = Vec::new();
        for (index, voxel) in scene.voxels.iter().enumerate() {
            let color = voxel_display_color(scene, index);
            let emission = scene
                .phase_state(index)
                .map(|phase| {
                    thermal_emission(
                        phase.temperature_kelvin,
                        phase.liquid_fraction,
                        phase.gas_fraction,
                    )
                })
                .unwrap_or([0.0; 3]);
            material_colors.push(color);
            if voxel.kinematic {
                kinematic_poses.push(AuthoredKinematicPose {
                    scene_index: index,
                    position: scene.positions[index],
                    rotation: scene.rotations[index],
                    color,
                    thermal_emission: emission,
                });
            } else {
                dynamic_scene_indices.push(index);
            }
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        dynamic_scene_indices.hash(&mut hasher);
        for color in &material_colors {
            for channel in color {
                channel.to_bits().hash(&mut hasher);
            }
        }
        for emission in kinematic_poses.iter().map(|pose| pose.thermal_emission) {
            for channel in emission {
                channel.to_bits().hash(&mut hasher);
            }
        }
        for pose in &kinematic_poses {
            for component in pose.position.to_array() {
                component.to_bits().hash(&mut hasher);
            }
            for component in pose.rotation.to_array() {
                component.to_bits().hash(&mut hasher);
            }
            for channel in pose.color {
                channel.to_bits().hash(&mut hasher);
            }
        }
        Self {
            dynamic_scene_indices,
            material_colors,
            kinematic_poses,
            cache_key: hasher.finish(),
        }
    }
}

fn authored_frame_indices_match(
    source: &AuthoredPhysicsRenderSource,
    cells: &[voxel_physics_lab::world_physics::PhysicalCellPose],
) -> bool {
    cells.len() == source.dynamic_scene_indices.len()
        && cells
            .iter()
            .zip(&source.dynamic_scene_indices)
            .all(|(cell, &expected_index)| cell.scene_index == expected_index)
}

fn has_axis_aligned_rotation(rotation: Quat) -> bool {
    const TOLERANCE: f32 = 1.0e-4;
    if !rotation.is_finite() || (rotation.length_squared() - 1.0).abs() > TOLERANCE * 4.0 {
        return false;
    }
    let mut used_axes = [false; 3];
    for basis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let rotated = (rotation * basis).abs();
        let axis = if rotated.x >= rotated.y && rotated.x >= rotated.z {
            0
        } else if rotated.y >= rotated.z {
            1
        } else {
            2
        };
        if used_axes[axis]
            || rotated[axis] < 1.0 - TOLERANCE
            || (0..3).any(|component| component != axis && rotated[component] > TOLERANCE)
        {
            return false;
        }
        used_axes[axis] = true;
    }
    true
}

/// Convert world-axis mask bits to the matching local cube-face bits. The
/// shader uses local vertex normals, while the surface classifier uses world axes.
fn local_face_mask(world_mask: u8, rotation: Quat) -> u8 {
    let mut local_mask = 0u8;
    for local_axis in 0..3 {
        for positive in [false, true] {
            let sign = if positive { 1.0 } else { -1.0 };
            let mut local_normal = Vec3::ZERO;
            local_normal[local_axis] = sign;
            let world_normal = rotation * local_normal;
            let world_axis = if world_normal.x.abs() >= world_normal.y.abs()
                && world_normal.x.abs() >= world_normal.z.abs()
            {
                0
            } else if world_normal.y.abs() >= world_normal.z.abs() {
                1
            } else {
                2
            };
            let world_positive = world_normal[world_axis] > 0.0;
            let world_bit = 1u8 << (world_axis * 2 + usize::from(world_positive));
            if world_mask & world_bit != 0 {
                local_mask |= 1u8 << (local_axis * 2 + usize::from(positive));
            }
        }
    }
    local_mask
}

fn apply_surface_culling(instances: &mut [Instance], pose_at: impl FnMut(usize) -> (DVec3, Quat)) {
    let poses: Vec<_> = (0..instances.len()).map(pose_at).collect();
    let masks = compute_surface_masks(&poses);
    apply_surface_masks(instances, &masks, |index| poses[index].1);
}

fn compute_surface_masks(poses: &[(DVec3, Quat)]) -> Vec<Option<u8>> {
    let mut result = vec![None; poses.len()];
    if poses.is_empty() {
        return result;
    }
    let eligible_indices: Vec<_> = poses
        .iter()
        .enumerate()
        .filter_map(|(index, (_, rotation))| has_axis_aligned_rotation(*rotation).then_some(index))
        .collect();
    let Some(&first_eligible) = eligible_indices.first() else {
        return result;
    };
    let anchor = poses[first_eligible].0;
    let positions: Vec<_> = eligible_indices
        .iter()
        .map(|&index| (poses[index].0 - anchor).as_vec3().to_array())
        .collect();
    let masks = exterior_face_masks(&positions, CELL);
    for (&index, &mask) in eligible_indices.iter().zip(&masks) {
        result[index] = Some(mask);
    }
    result
}

fn apply_surface_masks(
    instances: &mut [Instance],
    masks: &[Option<u8>],
    mut rotation_at: impl FnMut(usize) -> Quat,
) {
    for instance in instances.iter_mut() {
        // Preserve complete REV geometry; actual pose offsets carry gaps.
        instance.size = [CELL; 3];
        instance.surface = 0.0;
    }
    for (index, instance) in instances.iter_mut().enumerate() {
        if let Some(Some(world_mask)) = masks.get(index) {
            let local_mask = local_face_mask(*world_mask, rotation_at(index));
            instance.size = [CELL; 3];
            instance.surface = -(f32::from(local_mask) + 1.0);
        }
    }
}

/// Drop only cells whose six faces are hidden. `skin_index` remains the source
/// REV index, so compacting the draw list does not compact skinning data.
fn compact_fully_hidden(instances: &mut Vec<Instance>, voxel_count: usize) -> usize {
    let voxel_count = voxel_count.min(instances.len());
    let mut compacted: Vec<_> = instances
        .drain(..voxel_count)
        .filter(|instance| instance.surface != -1.0)
        .collect();
    let kept_voxels = compacted.len();
    compacted.append(instances);
    *instances = compacted;
    kept_voxels
}

impl Instance {
    fn cube(center: Vec3, rotation: Quat, color: [f32; 4]) -> Self {
        Self {
            center: center.to_array(),
            size: [CELL * 0.98; 3],
            color,
            rotation: rotation.to_array(),
            surface: 0.0,
            skin_index: u32::MAX,
            thermal_emission: [0.0; 3],
        }
    }
}

fn push_box_outline(instances: &mut Vec<Instance>, bounds: (Vec3, Vec3), color: [f32; 4]) {
    let padding = 0.0008;
    let minimum = bounds.0.min(bounds.1) - Vec3::splat(padding);
    let maximum = bounds.0.max(bounds.1) + Vec3::splat(padding);
    let thickness = ((maximum - minimum).min_element() * 0.07).clamp(0.00045, 0.0025);
    for axis in 0..3 {
        let side_axes = (0..3)
            .filter(|&candidate| candidate != axis)
            .collect::<Vec<_>>();
        for side in 0..4 {
            let mut start = minimum;
            for (bit, &side_axis) in side_axes.iter().enumerate() {
                if side & (1 << bit) != 0 {
                    start[side_axis] = maximum[side_axis];
                }
            }
            let mut end = start;
            end[axis] = maximum[axis];
            let mut size = Vec3::splat(thickness);
            size[axis] = (end[axis] - start[axis]).max(thickness);
            instances.push(Instance {
                center: ((start + end) * 0.5).to_array(),
                size: size.to_array(),
                color,
                rotation: Quat::IDENTITY.to_array(),
                surface: 0.0,
                skin_index: u32::MAX,
                thermal_emission: [0.0; 3],
            });
        }
    }
}
fn push_plant_lod(
    instances: &mut Vec<Instance>,
    occupied: &mut HashSet<(i32, i32, i32, u8, u8)>,
    voxel: &voxel_physics_lab::world::TerrainVoxel,
    lod: f32,
    lod_level: u8,
    opacity: f32,
) {
    if opacity <= 0.001 {
        return;
    }
    let material = match voxel.content {
        GeneratedCellContent::PlantWood => 0,
        GeneratedCellContent::PlantFoliage => 1,
        _ => 2,
    };
    if lod_level == 0 {
        let mut color = voxel.color;
        color[3] = opacity;
        instances.push(Instance {
            center: voxel.center.to_array(),
            size: voxel.size.to_array(),
            color,
            rotation: Quat::IDENTITY.to_array(),
            surface: surface_kind(voxel.content),
            skin_index: u32::MAX,
            thermal_emission: [0.0; 3],
        });
        return;
    }
    // A seed is a 1 cm biological state marker. Keep it at native display
    // scale nearby instead of expanding it into a tree-sized LOD billboard.
    if voxel.content == GeneratedCellContent::Seed {
        return;
    }
    // Coarse LOD cells summarize occupied plant voxels on a single world
    // aligned cubic lattice. Extending each branch along a quantized fiber
    // direction made long, thin bars that changed the tree silhouette and
    // opened gaps at diagonal joints. Sampling occupied cells into cubes keeps
    // the branch network connected and preserves the voxelized silhouette.
    let world_origin = Vec3::new(-WORLD_SIDE_M * 0.5, -WORLD_SIDE_M * 0.5, 0.0);
    let local_center = voxel.center - world_origin;
    let lod_cell = (
        (local_center.x / lod).floor() as i32,
        (local_center.y / lod).floor() as i32,
        (local_center.z / lod).floor() as i32,
    );
    let key = (lod_cell.0, lod_cell.1, lod_cell.2, lod_level, material);
    if !occupied.insert(key) {
        return;
    }
    let center = world_origin
        + Vec3::new(
            (key.0 as f32 + 0.5) * lod,
            (key.1 as f32 + 0.5) * lod,
            (key.2 as f32 + 0.5) * lod,
        );
    let mut color = voxel.color;
    color[3] = opacity;
    instances.push(Instance {
        center: center.to_array(),
        // Adjacent render cells should meet at their shared voxel boundary.
        // Shrinking coarse cells makes visible gaps between branch voxels,
        // especially at the distant LODs where one cell is already an
        // approximation of several source cells.
        size: [lod; 3],
        color,
        rotation: Quat::IDENTITY.to_array(),
        surface: surface_kind(voxel.content),
        skin_index: u32::MAX,
        thermal_emission: [0.0; 3],
    });
}
type WorldCacheKey = (u64, u64, i32, i32, i32);

#[derive(Default)]
struct WorldGeometryCache {
    world_instance_cache: Vec<Instance>,
    world_transition_cache: Vec<Instance>,
    world_terrain_instance_count: usize,
    world_transition_terrain_count: usize,
}

impl WorldGeometryCache {
    fn restrict_to_bounds(&mut self, world: &GeneratedWorld) {
        let Some((min, max)) = world.view_bounds else {
            return;
        };
        fn clip(instances: &mut Vec<Instance>, terrain: &mut usize, min: Vec3, max: Vec3) {
            let mut retained_terrain = 0;
            let old_terrain = *terrain;
            let mut index = 0;
            instances.retain_mut(|instance| {
                let was_terrain = index < old_terrain;
                index += 1;
                let center = Vec3::from_array(instance.center);
                let size = Vec3::from_array(instance.size);
                let lower = (center - size * 0.5).max(min);
                let upper = (center + size * 0.5).min(max);
                if upper.x <= lower.x || upper.y <= lower.y || upper.z <= lower.z {
                    return false;
                }
                instance.center = ((lower + upper) * 0.5).to_array();
                instance.size = (upper - lower).to_array();
                if was_terrain {
                    retained_terrain += 1;
                }
                true
            });
            *terrain = retained_terrain;
        }
        clip(
            &mut self.world_instance_cache,
            &mut self.world_terrain_instance_count,
            min,
            max,
        );
        clip(
            &mut self.world_transition_cache,
            &mut self.world_transition_terrain_count,
            min,
            max,
        );
    }
    fn overview(world: &GeneratedWorld, camera_position: Vec3) -> Self {
        let world_instance_cache = world
            .voxels
            .iter()
            .filter(|voxel| {
                let delta = voxel.center - camera_position;
                let radius = 1_000.0 + voxel.size.x.max(voxel.size.y) * 0.5;
                delta.x * delta.x + delta.y * delta.y <= radius * radius
            })
            .map(|voxel| Instance {
                center: voxel.center.to_array(),
                size: voxel.size.to_array(),
                color: voxel.color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
                skin_index: u32::MAX,
                thermal_emission: [0.0; 3],
            })
            .collect();
        let mut cache = Self {
            world_instance_cache,
            ..Self::default()
        };
        cache.restrict_to_bounds(world);
        cache
    }

    fn build(world: &GeneratedWorld, camera_position: Vec3, load_center: Vec3) -> Self {
        let mut cache = Self::default();
        cache.world_instance_cache.clear();
        cache.world_transition_cache.clear();
        cache.world_instance_cache.reserve(100_000);
        cache.world_transition_cache.reserve(16_000);
        for voxel in world.terrain_patch(load_center, 1_000.0) {
            // Render only the surface and a short skirt. The sampled
            // REV terrain still extends below this; hidden solid depth
            // does not need tens of metres of rasterized cube faces.
            let skirt = (voxel.size.x * 0.75).max(0.1);
            let instance = Instance {
                center: [voxel.center.x, voxel.center.y, voxel.size.z - skirt * 0.5],
                size: [voxel.size.x, voxel.size.y, skirt],
                color: voxel.color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
                skin_index: u32::MAX,
                thermal_emission: [0.0; 3],
            };
            if voxel.color[3] < 0.999 {
                cache.world_transition_cache.push(instance);
            } else {
                cache.world_instance_cache.push(instance);
            }
        }
        cache.world_terrain_instance_count = cache.world_instance_cache.len();
        cache.world_transition_terrain_count = cache.world_transition_cache.len();
        let mut occupied_lod_cells = HashSet::new();
        // Only refine cells actually rendered; the old 18 m query discarded
        // every result beyond 10.8 m after building it.
        let local_wood = world.plant_wood_voxels_near(camera_position, 10.8);
        for voxel in &world.voxels {
            let is_terrain = voxel.content == GeneratedCellContent::Terrain;
            let is_tree_cell = matches!(
                voxel.content,
                GeneratedCellContent::Seed
                    | GeneratedCellContent::PlantWood
                    | GeneratedCellContent::PlantFoliage
            );
            let delta = voxel.center - camera_position;
            let distance_squared = delta.length_squared();
            if is_terrain || distance_squared > 1_000_000.0 {
                continue;
            }
            // Close wood is replaced by its 1 cm local refinement. Keep
            // the coarse level in the blend band so the tree never fades
            // into empty space while crossing the detail boundary.
            if voxel.content == GeneratedCellContent::PlantWood && distance_squared <= 5.2 * 5.2 {
                continue;
            }
            if is_tree_cell {
                const LOD_RADII: [f32; 5] = [8.0, 32.0, 128.0, 512.0, 1_000.0];
                // Keep the coarsest plant cell below the scale of a
                // small tree crown. Multi-metre cubes erase branch
                // shape and turn distant trees into block silhouettes.
                const LOD_SIZES: [f32; 5] = [0.1, 0.2, 0.3, 0.4, 0.4];
                let mut transition = None;
                for boundary in 0..4 {
                    let radius = LOD_RADII[boundary];
                    let width = radius * 0.35;
                    let start = radius - width;
                    let end = radius + width;
                    if (start * start..=end * end).contains(&distance_squared) {
                        let distance = distance_squared.sqrt();
                        let t = ((distance - start) / (end - start)).clamp(0.0, 1.0);
                        let t = t * t * (3.0 - 2.0 * t);
                        transition = Some((boundary, t));
                        break;
                    }
                }
                if let Some((lower_level, blend)) = transition {
                    if lower_level != 0 || voxel.content != GeneratedCellContent::PlantWood {
                        push_plant_lod(
                            &mut cache.world_instance_cache,
                            &mut occupied_lod_cells,
                            voxel,
                            LOD_SIZES[lower_level],
                            lower_level as u8,
                            1.0 - blend,
                        );
                    }
                    push_plant_lod(
                        &mut cache.world_transition_cache,
                        &mut occupied_lod_cells,
                        voxel,
                        LOD_SIZES[lower_level + 1],
                        (lower_level + 1) as u8,
                        blend,
                    );
                } else {
                    let lod_level = LOD_RADII
                        .iter()
                        .filter(|&&radius| distance_squared >= radius * radius)
                        .count();
                    let lod_level = lod_level.min(4);
                    let fade = if distance_squared > 850.0 * 850.0 {
                        let distance = distance_squared.sqrt();
                        let t = ((distance - 850.0) / 150.0).clamp(0.0, 1.0);
                        let t = t * t * (3.0 - 2.0 * t);
                        1.0 - t
                    } else {
                        1.0
                    };
                    let destination = if fade < 0.999 {
                        &mut cache.world_transition_cache
                    } else {
                        &mut cache.world_instance_cache
                    };
                    push_plant_lod(
                        destination,
                        &mut occupied_lod_cells,
                        voxel,
                        LOD_SIZES[lod_level],
                        lod_level as u8,
                        fade,
                    );
                }
            } else {
                // Non-tree ground plants and fungi are sparse and keep
                // their original size at every LOD.
                cache.world_instance_cache.push(Instance {
                    center: voxel.center.to_array(),
                    size: voxel.size.to_array(),
                    color: voxel.color,
                    rotation: Quat::IDENTITY.to_array(),
                    surface: surface_kind(voxel.content),
                    skin_index: u32::MAX,
                    thermal_emission: [0.0; 3],
                });
            }
        }
        for voxel in local_wood {
            let distance_squared = (voxel.center - camera_position).length_squared();
            if distance_squared > 10.8 * 10.8 {
                continue;
            }
            let opacity = if distance_squared <= 5.2 * 5.2 {
                1.0
            } else {
                let distance = distance_squared.sqrt();
                let t = ((distance - 5.2) / (10.8 - 5.2)).clamp(0.0, 1.0);
                let t = t * t * (3.0 - 2.0 * t);
                1.0 - t
            };
            let mut color = voxel.color;
            color[3] = opacity;
            cache.world_instance_cache.push(Instance {
                center: voxel.center.to_array(),
                size: voxel.size.to_array(),
                color,
                rotation: Quat::IDENTITY.to_array(),
                surface: surface_kind(voxel.content),
                skin_index: u32::MAX,
                thermal_emission: [0.0; 3],
            });
        }
        cache.restrict_to_bounds(world);
        cache
    }
}

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub radius: f32,
    pub target: Vec3,
    free_position: Option<Vec3>,
    first_person: bool,
    pub free_speed_mps: f32,
    pub look_sensitivity: f32,
}

const CAMERA_VERTICAL_FOV_RADIANS: f32 = 50.0 * std::f32::consts::PI / 180.0;
const CAMERA_FIT_MARGIN: f32 = 1.2;
const CAMERA_MIN_ORBIT_DISTANCE_M: f32 = 0.04;
const CAMERA_FIT_FALLBACK_DISTANCE_M: f32 = 0.3;

impl Camera {
    pub fn new() -> Self {
        Self {
            yaw: -1.0,
            pitch: 0.35,
            radius: 5.0,
            target: Vec3::new(0., 0., 1.5),
            free_position: None,
            first_person: false,
            free_speed_mps: 20.0,
            look_sensitivity: 0.005,
        }
    }
    fn offset_direction(&self) -> Vec3 {
        Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
    }
    pub fn forward(&self) -> Vec3 {
        -self.offset_direction()
    }
    pub fn is_freecam(&self) -> bool {
        self.free_position.is_some()
    }
    pub fn is_first_person(&self) -> bool {
        self.first_person && self.free_position.is_none()
    }
    pub fn set_first_person(&mut self, eye_position: Vec3) {
        self.target = eye_position;
        self.first_person = true;
        self.free_position = None;
    }
    pub fn set_orbit(&mut self, target: Vec3, radius: f32) {
        self.target = target;
        self.radius = radius;
        self.first_person = false;
        self.free_position = None;
    }
    pub fn enable_freecam(&mut self) {
        if self.free_position.is_none() {
            let position = if self.first_person {
                self.target
            } else {
                self.target + self.offset_direction() * self.radius
            };
            self.free_position = Some(position);
            // Model-editor orbit radii can be only a few centimetres. Using
            // that radius directly made WASD freecam crawl at 3 cm/s.
            self.free_speed_mps = (self.radius * 0.8).clamp(0.5, 200.0);
        }
    }
    pub fn disable_freecam(&mut self) {
        if let Some(position) = self.free_position.take() {
            if self.first_person {
                self.target = position;
            } else {
                self.target = position + self.forward() * self.radius;
            }
        }
    }
    pub fn rotate_freecam(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx as f32 * self.look_sensitivity;
        // Mouse-down looks down: positive pitch lowers the view direction in
        // this camera convention because forward is the negative orbit offset.
        self.pitch = (self.pitch + dy as f32 * self.look_sensitivity).clamp(-1.55, 1.55);
    }
    pub fn position(&self) -> Vec3 {
        self.free_position.unwrap_or_else(|| {
            if self.first_person {
                self.target
            } else {
                self.target + self.offset_direction() * self.radius
            }
        })
    }
    pub fn move_freecam(&mut self, right: f32, forward: f32, up: f32, dt: f32, fast: bool) {
        if self.free_position.is_none() {
            return;
        }
        let view_forward = self.forward();
        let mut view_right = view_forward.cross(Vec3::Z);
        if view_right.length_squared() < 1.0e-6 {
            view_right = Vec3::X;
        } else {
            view_right = view_right.normalize();
        }
        let mut direction = view_right * right + view_forward * forward + Vec3::Z * up;
        if direction.length_squared() > 1.0 {
            direction = direction.normalize();
        }
        let speed_multiplier = if fast { 5.0 } else { 1.0 };
        if let Some(position) = self.free_position.as_mut() {
            *position += direction * self.free_speed_mps * speed_multiplier * dt;
        }
    }
    pub fn adjust_free_speed(&mut self, amount: f32) {
        self.free_speed_mps = (self.free_speed_mps * (amount * 0.2).exp()).clamp(0.005, 5000.0);
    }

    /// Return the orbit distance needed to fit a sphere in the current camera
    /// frustum, using the same vertical field of view as `matrix`.
    ///
    /// Invalid radii use a finite fallback distance. Invalid aspects use a
    /// square viewport, which avoids propagating NaNs into the camera matrix.
    pub fn fit_orbit_distance_for_sphere(radius_m: f32, viewport_aspect: f32) -> f32 {
        if !radius_m.is_finite() || radius_m <= 0.0 {
            return CAMERA_FIT_FALLBACK_DISTANCE_M;
        }

        let aspect = if viewport_aspect.is_finite() && viewport_aspect > 0.0 {
            f64::from(viewport_aspect)
        } else {
            1.0
        };
        let vertical_half_angle = f64::from(CAMERA_VERTICAL_FOV_RADIANS) * 0.5;
        let horizontal_half_angle = (aspect * vertical_half_angle.tan()).atan();
        let limiting_half_angle = vertical_half_angle.min(horizontal_half_angle);
        let distance = f64::from(radius_m) / limiting_half_angle.sin()
            * f64::from(CAMERA_FIT_MARGIN);
        distance
            .max(f64::from(CAMERA_MIN_ORBIT_DISTANCE_M))
            .min(f64::from(f32::MAX)) as f32
    }

    fn matrix(&self, aspect: f32) -> Mat4 {
        let offset = self.offset_direction() * self.radius;
        // Floating-point reverse Z retains precision on centimetre surfaces
        // while the same camera also sees the kilometre-scale world.
        let near = (self.radius * 0.00001).clamp(0.01, 0.5);
        let (eye, target) = match self.free_position {
            Some(position) => (position, position - self.offset_direction()),
            None if self.first_person => (self.target, self.target + self.forward()),
            None => (self.target + offset, self.target),
        };
        Mat4::perspective_infinite_reverse_rh(CAMERA_VERTICAL_FOV_RADIANS, aspect, near)
            * Mat4::look_at_rh(eye, target, Vec3::Z)
    }
}
pub struct Renderer {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    transition_pipeline: wgpu::RenderPipeline,
    ui_pipeline: wgpu::RenderPipeline,
    skinning_layout: wgpu::BindGroupLayout,
    skinning_influence_buffer: wgpu::Buffer,
    skinning_bone_buffer: wgpu::Buffer,
    skinning_influence_capacity: usize,
    skinning_bone_capacity: usize,
    skinning_bind_group: wgpu::BindGroup,
    skinning_default_bind_group: wgpu::BindGroup,
    skinning_weight_cache_key: Option<(u64, usize, usize, usize)>,
    skinning_weight_job: Option<(
        (u64, usize, usize, usize),
        std::sync::mpsc::Receiver<Vec<SkinInfluenceGpu>>,
    )>,
    bone_material_limits: Vec<BoneMaterialLimits>,
    bone_material_limits_cache_key: Option<(u64, usize, usize, usize, usize)>,
    vertices: wgpu::Buffer,
    terrain_vertices: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    selected_instances: wgpu::Buffer,
    selected_instance_capacity: usize,
    bone_instances: wgpu::Buffer,
    bone_instance_capacity: usize,
    bone_instance_cache: Vec<Instance>,
    bone_gizmo_cache_key: Option<u64>,
    physics_instances: wgpu::Buffer,
    physics_capacity: usize,
    physics_instance_cache: Vec<Instance>,
    physics_cache_key: Option<(u64, u64)>,
    authored_physics_instances: Vec<Instance>,
    authored_physics_instance_buffer: wgpu::Buffer,
    authored_physics_instance_capacity: usize,
    authored_physics_active: bool,
    authored_physics_cache_key: Option<(u64, u64, u64)>,
    scene_voxel_instance_count: usize,
    scene_surface_masks: Vec<Option<u8>>,
    scene_surface_mask_cache_key: Option<(usize, usize, usize, usize, u64, u64)>,
    scene_instance_cache: Vec<Instance>,
    scene_instance_cache_key: Option<u64>,
    world_instance_cache: Vec<Instance>,
    world_transition_cache: Vec<Instance>,
    world_terrain_instance_count: usize,
    world_transition_terrain_count: usize,
    cached_world_key: Option<WorldCacheKey>,
    world_cache_job: Option<std::sync::mpsc::Receiver<(WorldCacheKey, WorldGeometryCache)>>,
    world_snapshot: Option<((u64, u64), Arc<GeneratedWorld>)>,
    ui_instances: wgpu::Buffer,
    ui_capacity: usize,
    camera_buffer: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    screen_buffer: wgpu::Buffer,
    screen_group: wgpu::BindGroup,
    depth: wgpu::TextureView,
}
fn depth(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}
fn instance_buffer(device: &wgpu::Device, count: usize, stride: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("instances"),
        size: (count.max(1) * stride) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn terrain_surface_vertices() -> Vec<Vertex> {
    let mut vertices = Vec::with_capacity(30);
    let mut quad = |points: [Vec3; 4], normal: Vec3| {
        for index in [0, 1, 2, 0, 2, 3] {
            vertices.push(Vertex {
                position: points[index].to_array(),
                normal: normal.to_array(),
            });
        }
    };
    let half = 0.5;
    quad(
        [
            Vec3::new(-half, -half, half),
            Vec3::new(half, -half, half),
            Vec3::new(half, half, half),
            Vec3::new(-half, half, half),
        ],
        Vec3::Z,
    );
    for (normal, points) in [
        (
            Vec3::X,
            [
                Vec3::new(half, -half, half),
                Vec3::new(half, half, half),
                Vec3::new(half, half, -half),
                Vec3::new(half, -half, -half),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                Vec3::new(-half, half, half),
                Vec3::new(-half, -half, half),
                Vec3::new(-half, -half, -half),
                Vec3::new(-half, half, -half),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(half, half, half),
                Vec3::new(-half, half, half),
                Vec3::new(-half, half, -half),
                Vec3::new(half, half, -half),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                Vec3::new(-half, -half, half),
                Vec3::new(half, -half, half),
                Vec3::new(half, -half, -half),
                Vec3::new(-half, -half, -half),
            ],
        ),
    ] {
        quad(points, normal);
    }
    vertices
}
fn uniform(
    device: &wgpu::Device,
    size: u64,
) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniform"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, layout, group)
}
impl Renderer {
    /// Replace an authored scene's dynamic REV rendering with a worker frame.
    /// Keep this lightweight source snapshot while the full Scene is moved to
    /// its worker. Topology changes use a neutral dynamic color.
    pub fn set_authored_physics_frame(
        &mut self,
        source: &AuthoredPhysicsRenderSource,
        frame: Option<&voxel_physics_lab::world_physics::PhysicsFrame>,
    ) -> usize {
        let Some(frame) = frame else {
            self.authored_physics_instances.clear();
            self.authored_physics_active = false;
            self.authored_physics_cache_key = None;
            return 0;
        };
        let cache_key = (frame.generation, frame.revision, source.cache_key);
        if self.authored_physics_cache_key == Some(cache_key) {
            self.authored_physics_active = true;
            return self.authored_physics_instances.len();
        }
        self.authored_physics_instances.clear();

        let material_mapping_is_stable = authored_frame_indices_match(source, &frame.cells);
        self.authored_physics_instances
            .extend(frame.cells.iter().map(|cell| {
                let color = if material_mapping_is_stable {
                    source
                        .material_colors
                        .get(cell.scene_index)
                        .copied()
                        .unwrap_or([0.68, 0.70, 0.73, 1.0])
                } else {
                    [0.68, 0.70, 0.73, 1.0]
                };
                Instance {
                    center: cell.world_position_m.as_vec3().to_array(),
                    size: [CELL; 3],
                    color,
                    rotation: cell.rotation.to_array(),
                    surface: 0.0,
                    skin_index: u32::MAX,
                    thermal_emission: thermal_emission(
                        cell.temperature_kelvin,
                        cell.liquid_fraction,
                        cell.gas_fraction,
                    ),
                }
            }));
        // Build one combined pose list so exact face contacts are classified
        // consistently across dynamic and constrained authored REVs.
        let dynamic_count = self.authored_physics_instances.len();
        let fixed_thermal: HashMap<_, _> = frame
            .fixed_thermal
            .iter()
            .map(|state| (state.scene_index, state))
            .collect();
        self.authored_physics_instances
            .extend(source.kinematic_poses.iter().map(|pose| {
                let mut instance =
                    Instance::cube(pose.position.as_vec3(), pose.rotation, pose.color);
                instance.skin_index = u32::MAX;
                instance.thermal_emission = fixed_thermal
                    .get(&pose.scene_index)
                    .map(|state| {
                        thermal_emission(
                            state.temperature_kelvin,
                            state.liquid_fraction,
                            state.gas_fraction,
                        )
                    })
                    .unwrap_or(pose.thermal_emission);
                instance
            }));
        apply_surface_culling(&mut self.authored_physics_instances, |index| {
            if index < dynamic_count {
                let cell = &frame.cells[index];
                (cell.world_position_m, cell.rotation)
            } else {
                let pose = source.kinematic_poses[index - dynamic_count];
                (pose.position, pose.rotation)
            }
        });
        let authored_count = self.authored_physics_instances.len();
        compact_fully_hidden(&mut self.authored_physics_instances, authored_count);
        if self.authored_physics_instances.len() > self.authored_physics_instance_capacity {
            self.authored_physics_instance_capacity =
                self.authored_physics_instances.len().next_power_of_two();
            self.authored_physics_instance_buffer = instance_buffer(
                &self.device,
                self.authored_physics_instance_capacity,
                std::mem::size_of::<Instance>(),
            );
        }
        if !self.authored_physics_instances.is_empty() {
            self.queue.write_buffer(
                &self.authored_physics_instance_buffer,
                0,
                bytemuck::cast_slice(&self.authored_physics_instances),
            );
        }
        self.authored_physics_active = true;
        self.authored_physics_cache_key = Some(cache_key);
        self.authored_physics_instances.len()
    }

    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("No graphics adapter")?;
        let adapter_info = adapter.get_info();
        eprintln!(
            "wgpu renderer adapter: {} ({:?}, {:?}) — {}",
            adapter_info.name, adapter_info.device_type, adapter_info.backend, adapter_info.driver
        );
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("voxel lab"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await?;
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let (camera_buffer, camera_layout, camera_group) = uniform(&device, 64);
        let (screen_buffer, screen_layout, screen_group) = uniform(&device, 16);
        let skinning_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel skeleton skinning layout"),
            entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let skinning_influence_buffer =
            storage_buffer::<SkinInfluenceGpu>(&device, 1, "voxel skeleton influences");
        let skinning_bone_buffer =
            storage_buffer::<BoneSkinTransformGpu>(&device, 1, "voxel skeleton bone transforms");
        let identity_bone = BoneSkinTransformGpu {
            translation: [0.0; 4],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        queue.write_buffer(
            &skinning_influence_buffer,
            0,
            bytemuck::bytes_of(&SkinInfluenceGpu::zeroed()),
        );
        queue.write_buffer(&skinning_bone_buffer, 0, bytemuck::bytes_of(&identity_bone));
        let skinning_bind_group = create_skinning_bind_group(
            &device,
            &skinning_layout,
            &skinning_influence_buffer,
            &skinning_bone_buffer,
        );
        let fallback_influence = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback voxel skin influence"),
            contents: bytemuck::bytes_of(&SkinInfluenceGpu::zeroed()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let fallback_bone = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fallback voxel skin bone"),
            contents: bytemuck::bytes_of(&identity_bone),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let skinning_default_bind_group = create_skinning_bind_group(
            &device,
            &skinning_layout,
            &fallback_influence,
            &fallback_bone,
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/voxel.wgsl").into()),
        });
        let ui_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_layout, &skinning_layout],
            push_constant_ranges: &[],
        });
        let ui_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&screen_layout],
            push_constant_ranges: &[],
        });
        let vertex_attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let instance_attrs = wgpu::vertex_attr_array![2 => Float32x3, 3 => Float32x3, 4 => Float32x4, 5 => Float32x4, 6 => Float32, 7 => Uint32, 8 => Float32x3];
        let ui_attrs = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
        let target = Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("instanced cubes"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[target.clone()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Greater,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });
        let transition_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LOD transition blend"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &vertex_attrs,
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Instance>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &instance_attrs,
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs",
                targets: &[target.clone()],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
        });
        let ui_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI rectangles"),
            layout: Some(&ui_layout),
            vertex: wgpu::VertexState {
                module: &ui_shader,
                entry_point: "vs",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Rect>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &ui_attrs,
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ui_shader,
                entry_point: "fs",
                targets: &[target],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });
        let mut cube = Vec::new();
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut normal = Vec3::ZERO;
                normal[axis] = sign;
                let u = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 1) % 3];
                let v = [Vec3::X, Vec3::Y, Vec3::Z][(axis + 2) % 3];
                for [a, b] in [
                    [-1., -1.],
                    [1., -1.],
                    [-1., 1.],
                    [-1., 1.],
                    [1., -1.],
                    [1., 1.],
                ] {
                    cube.push(Vertex {
                        position: ((normal + a * u + b * v) * 0.5).to_array(),
                        normal: normal.to_array(),
                    });
                }
            }
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cube"),
            contents: bytemuck::cast_slice(&cube),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let terrain_mesh = terrain_surface_vertices();
        let terrain_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain surface and skirts"),
            contents: bytemuck::cast_slice(&terrain_mesh),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let instances = instance_buffer(&device, 100_000, std::mem::size_of::<Instance>());
        let selected_instances = instance_buffer(&device, 1024, std::mem::size_of::<Instance>());
        let bone_instances = instance_buffer(&device, 256, std::mem::size_of::<Instance>());
        let physics_instances = instance_buffer(&device, 1024, std::mem::size_of::<Instance>());
        let authored_physics_instance_buffer =
            instance_buffer(&device, 1, std::mem::size_of::<Instance>());
        let ui_instances = instance_buffer(&device, 10_000, std::mem::size_of::<Rect>());
        let depth = depth(&device, &config);
        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            transition_pipeline,
            ui_pipeline,
            skinning_layout,
            skinning_influence_buffer,
            skinning_bone_buffer,
            skinning_influence_capacity: 1,
            skinning_bone_capacity: 1,
            skinning_bind_group,
            skinning_default_bind_group,
            skinning_weight_cache_key: None,
            skinning_weight_job: None,
            bone_material_limits: Vec::new(),
            bone_material_limits_cache_key: None,
            vertices,
            terrain_vertices,
            instances,
            instance_capacity: 100_000,
            selected_instances,
            selected_instance_capacity: 1024,
            bone_instances,
            bone_instance_capacity: 256,
            bone_instance_cache: Vec::new(),
            bone_gizmo_cache_key: None,
            physics_instances,
            physics_capacity: 1024,
            physics_instance_cache: Vec::with_capacity(1024),
            physics_cache_key: None,
            authored_physics_instances: Vec::new(),
            authored_physics_instance_buffer,
            authored_physics_instance_capacity: 1,
            authored_physics_active: false,
            authored_physics_cache_key: None,
            scene_voxel_instance_count: 0,
            scene_surface_masks: Vec::new(),
            scene_surface_mask_cache_key: None,
            scene_instance_cache: Vec::new(),
            scene_instance_cache_key: None,
            world_instance_cache: Vec::new(),
            world_transition_cache: Vec::new(),
            world_terrain_instance_count: 0,
            world_transition_terrain_count: 0,
            cached_world_key: None,
            world_cache_job: None,
            world_snapshot: None,
            ui_instances,
            ui_capacity: 10_000,
            camera_buffer,
            camera_group,
            screen_buffer,
            screen_group,
            depth,
        })
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth = depth(&self.device, &self.config);
    }

    fn prepare_skeleton_skinning(
        &mut self,
        model: &VoxelModelDocument,
        voxel_count: usize,
        clip_index: Option<usize>,
        time_s: f64,
        weights_revision: u64,
        model_revision: u64,
    ) -> bool {
        let Some(skeleton) = model.skeleton.as_ref() else {
            return false;
        };
        if voxel_count == 0 || skeleton.bones.is_empty() || skeleton.voxel_weights.is_empty() {
            return false;
        }
        let material_limits_key = (
            model_revision,
            model as *const VoxelModelDocument as usize,
            model.materials.as_ptr() as usize,
            model.voxels.as_ptr() as usize,
            skeleton.voxel_weights.as_ptr() as usize,
        );
        if self.bone_material_limits_cache_key != Some(material_limits_key) {
            self.bone_material_limits = model.skeleton_material_limits().unwrap_or_default();
            self.bone_material_limits_cache_key = Some(material_limits_key);
        }
        let transforms = match clip_index {
            Some(index) => match skeleton.bone_skin_transforms_at_with_material_limits(
                index,
                time_s,
                Some(&self.bone_material_limits),
            ) {
                Ok(transforms) => transforms,
                Err(_) => return false,
            },
            None => vec![
                BoneSkinTransform {
                    translation_m: DVec3::ZERO,
                    rotation: DQuat::IDENTITY,
                };
                skeleton.bones.len()
            ],
        };
        if transforms.is_empty() {
            return false;
        }

        let weight_key = (
            weights_revision,
            voxel_count,
            skeleton.voxel_weights.len(),
            skeleton.voxel_weights.as_ptr() as usize,
        );
        let mut bind_group_dirty = false;
        if self.skinning_weight_cache_key != Some(weight_key) {
            let mut completed = None;
            if let Some((job_key, receiver)) = self.skinning_weight_job.take() {
                if job_key == weight_key {
                    match receiver.try_recv() {
                        Ok(packed) => completed = Some(packed),
                        Err(std::sync::mpsc::TryRecvError::Empty) => {
                            self.skinning_weight_job = Some((job_key, receiver));
                            return false;
                        }
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
                    }
                } else if matches!(
                    receiver.try_recv(),
                    Err(std::sync::mpsc::TryRecvError::Empty)
                ) {
                    self.skinning_weight_job = Some((job_key, receiver));
                    return false;
                }
            }
            let packed = if let Some(packed) = completed {
                packed
            } else {
                let weights = skeleton.voxel_weights.clone();
                let (sender, receiver) = std::sync::mpsc::channel();
                let spawn = std::thread::Builder::new()
                    .name("voxel-skin-weight-pack".into())
                    .spawn(move || {
                        let _ = sender.send(pack_skin_influences(voxel_count, &weights));
                    });
                match spawn {
                    Ok(_) => {
                        self.skinning_weight_job = Some((weight_key, receiver));
                        return false;
                    }
                    Err(_) => pack_skin_influences(voxel_count, &skeleton.voxel_weights),
                }
            };
            if packed.len() > self.skinning_influence_capacity {
                self.skinning_influence_capacity = packed.len().next_power_of_two();
                self.skinning_influence_buffer = storage_buffer::<SkinInfluenceGpu>(
                    &self.device,
                    self.skinning_influence_capacity,
                    "voxel skeleton influences",
                );
                bind_group_dirty = true;
            }
            self.queue.write_buffer(
                &self.skinning_influence_buffer,
                0,
                bytemuck::cast_slice(&packed),
            );
            self.skinning_weight_cache_key = Some(weight_key);
        }

        let packed_transforms = pack_bone_transforms(&transforms);
        if packed_transforms.len() > self.skinning_bone_capacity {
            self.skinning_bone_capacity = packed_transforms.len().next_power_of_two();
            self.skinning_bone_buffer = storage_buffer::<BoneSkinTransformGpu>(
                &self.device,
                self.skinning_bone_capacity,
                "voxel skeleton bone transforms",
            );
            bind_group_dirty = true;
        }
        self.queue.write_buffer(
            &self.skinning_bone_buffer,
            0,
            bytemuck::cast_slice(&packed_transforms),
        );
        if bind_group_dirty {
            self.skinning_bind_group = create_skinning_bind_group(
                &self.device,
                &self.skinning_layout,
                &self.skinning_influence_buffer,
                &self.skinning_bone_buffer,
            );
        }
        true
    }

    pub fn draw(
        &mut self,
        scene: &Scene,
        tool: &Option<Tool>,
        generated_world: Option<&GeneratedWorld>,
        physics: Option<&voxel_physics_lab::world_physics::PhysicsFrame>,
        camera: &Camera,
        ui: &[Rect],
        scene_viewport: Option<[u32; 4]>,
        hovered_face: Option<(Vec3, Vec3)>,
        selected_voxels: &[usize],
        selection_bounds: Option<(Vec3, Vec3)>,
        selection_preview: Option<(Vec3, Vec3)>,
        sculpt_preview_cells: &[Vec3],
        sculpt_preview_bounds: Option<(Vec3, Vec3)>,
        sculpt_preview_cell_size: f32,
        show_bonds: bool,
        rig: Option<(&VoxelModelDocument, Option<usize>, f32, Option<usize>, f64)>,
        show_rig_overlay: bool,
        skinning_weights_revision: u64,
        model_revision: u64,
    ) -> Result<(), wgpu::SurfaceError> {
        let authored_physics_override = self.authored_physics_active;
        let skinning_enabled = !authored_physics_override
            && rig.filter(|_| generated_world.is_none()).is_some_and(
                |(model, _, _, clip_index, time_s)| {
                    self.prepare_skeleton_skinning(
                        model,
                        scene.voxels.len(),
                        clip_index,
                        time_s,
                        skinning_weights_revision,
                        model_revision,
                    )
                },
            );
        let mut world_cache_changed = false;
        if let Some(frame) = physics {
            let cache_key = (frame.generation, frame.revision);
            if self.physics_cache_key != Some(cache_key) {
                self.physics_instance_cache.clear();
                self.physics_instance_cache
                    .extend(frame.cells.iter().map(|cell| Instance {
                        center: cell.world_position_m.as_vec3().to_array(),
                        size: [CELL; 3],
                        color: [0.62, 0.35, 0.13, 1.0],
                        rotation: cell.rotation.to_array(),
                        surface: 2.0,
                        skin_index: u32::MAX,
                        thermal_emission: thermal_emission(
                            cell.temperature_kelvin,
                            cell.liquid_fraction,
                            cell.gas_fraction,
                        ),
                    }));
                if self.physics_instance_cache.len() > self.physics_capacity {
                    self.physics_capacity = self.physics_instance_cache.len().next_power_of_two();
                    self.physics_instances = instance_buffer(
                        &self.device,
                        self.physics_capacity,
                        std::mem::size_of::<Instance>(),
                    );
                }
                if !self.physics_instance_cache.is_empty() {
                    self.queue.write_buffer(
                        &self.physics_instances,
                        0,
                        bytemuck::cast_slice(&self.physics_instance_cache),
                    );
                }
                self.physics_cache_key = Some(cache_key);
            }
        } else {
            self.physics_cache_key = None;
            self.physics_instance_cache.clear();
        }
        if let Some(world) = generated_world {
            let camera_position = camera.position();
            // Keep a seeded 1 km radius around the camera. The cache follows
            // the camera in 8 m increments to avoid rebuilding every frame.
            let camera_cell = (
                (camera_position.x / 8.0).floor() as i32,
                (camera_position.y / 8.0).floor() as i32,
                (camera_position.z / 8.0).floor() as i32,
            );
            let load_center = Vec3::new(
                (camera_cell.0 as f32 + 0.5) * 8.0,
                (camera_cell.1 as f32 + 0.5) * 8.0,
                (camera_cell.2 as f32 + 0.5) * 8.0,
            );
            let (seed_revision, geometry_revision) = world.render_cache_key();
            let cache_key = (
                seed_revision,
                geometry_revision,
                camera_cell.0,
                camera_cell.1,
                camera_cell.2,
            );
            let source_key = (seed_revision, geometry_revision);
            let reset_world = self.world_snapshot.as_ref().map_or(true, |(key, _)| {
                key.0 != seed_revision || key.1 > geometry_revision
            });
            if reset_world {
                // Show the existing overview immediately while local geometry
                // is prepared. No synchronous centimetre rasterization here.
                let overview = WorldGeometryCache::overview(world, camera_position);
                self.world_instance_cache = overview.world_instance_cache;
                self.world_transition_cache = overview.world_transition_cache;
                self.world_terrain_instance_count = overview.world_terrain_instance_count;
                self.world_transition_terrain_count = overview.world_transition_terrain_count;
                self.cached_world_key = None;
                self.world_snapshot = Some((source_key, Arc::new(world.clone())));
                world_cache_changed = true;
            }
            if let Some(receiver) = &self.world_cache_job {
                match receiver.try_recv() {
                    Ok((built_key, geometry)) => {
                        self.world_cache_job = None;
                        // Accept a completed earlier growth snapshot while
                        // requesting the latest one, so accelerated growth
                        // cannot starve all completed geometry updates.
                        if built_key.0 == seed_revision && built_key.1 <= geometry_revision {
                            self.world_instance_cache = geometry.world_instance_cache;
                            self.world_transition_cache = geometry.world_transition_cache;
                            self.world_terrain_instance_count =
                                geometry.world_terrain_instance_count;
                            self.world_transition_terrain_count =
                                geometry.world_transition_terrain_count;
                            self.cached_world_key = Some(built_key);
                            world_cache_changed = true;
                        }
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.world_cache_job = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
            }
            if self.cached_world_key != Some(cache_key) && self.world_cache_job.is_none() {
                if self.world_snapshot.as_ref().map(|(key, _)| *key) != Some(source_key) {
                    self.world_snapshot = Some((source_key, Arc::new(world.clone())));
                }
                let snapshot = Arc::clone(&self.world_snapshot.as_ref().unwrap().1);
                let (sender, receiver) = std::sync::mpsc::channel();
                match std::thread::Builder::new().name("world-render-geometry".into()).spawn(move || {
                    let start = std::time::Instant::now();
                    let geometry = WorldGeometryCache::build(&snapshot, camera_position, load_center);
                    if std::env::var_os("VOXEL_PROFILE_WORLD_GEOMETRY").is_some() {
                        eprintln!("world geometry: build_ms={:.2} instances={} transition_instances={}",
                            start.elapsed().as_secs_f64() * 1_000.0,
                            geometry.world_instance_cache.len(), geometry.world_transition_cache.len());
                    }
                    let _ = sender.send((cache_key, geometry));
                }) {
                    Ok(_) => self.world_cache_job = Some(receiver),
                    Err(error) => eprintln!("world geometry worker could not start: {error}"),
                }
            }
        } else {
            self.cached_world_key = None;
            self.world_snapshot = None;
            self.world_terrain_instance_count = 0;
        }
        let scene_cacheable =
            generated_world.is_none() && tool.is_none() && sculpt_preview_cells.is_empty();
        let mut scene_hasher = std::collections::hash_map::DefaultHasher::new();
        (scene.positions.as_ptr() as usize).hash(&mut scene_hasher);
        scene.positions.len().hash(&mut scene_hasher);
        (scene.voxels.as_ptr() as usize).hash(&mut scene_hasher);
        (scene.bonds[0].as_ptr() as usize).hash(&mut scene_hasher);
        (scene.bonds[1].as_ptr() as usize).hash(&mut scene_hasher);
        (scene.bonds[2].as_ptr() as usize).hash(&mut scene_hasher);
        scene.render_revision.hash(&mut scene_hasher);
        scene.substep_index.hash(&mut scene_hasher);
        scene.broken.hash(&mut scene_hasher);
        for material in &scene.materials {
            for channel in material_display_color(&material.name) {
                channel.to_bits().hash(&mut scene_hasher);
            }
        }
        show_bonds.hash(&mut scene_hasher);
        if let Some((center, normal)) = hovered_face {
            1_u8.hash(&mut scene_hasher);
            hash_vec3(center, &mut scene_hasher);
            hash_vec3(normal, &mut scene_hasher);
        } else {
            0_u8.hash(&mut scene_hasher);
        }
        for bounds in [selection_bounds, selection_preview, sculpt_preview_bounds] {
            if let Some((minimum, maximum)) = bounds {
                1_u8.hash(&mut scene_hasher);
                hash_vec3(minimum, &mut scene_hasher);
                hash_vec3(maximum, &mut scene_hasher);
            } else {
                0_u8.hash(&mut scene_hasher);
            }
        }
        sculpt_preview_cell_size.to_bits().hash(&mut scene_hasher);
        let scene_cache_key = scene_hasher.finish();
        let scene_cache_hit =
            scene_cacheable && self.scene_instance_cache_key == Some(scene_cache_key);
        if generated_world.is_none() && !show_bonds {
            let surface_cache_key = (
                scene.positions.as_ptr() as usize,
                scene.positions.len(),
                scene.rotations.as_ptr() as usize,
                scene.rotations.len(),
                scene.render_revision,
                scene.substep_index,
            );
            if self.scene_surface_mask_cache_key != Some(surface_cache_key) {
                let poses: Vec<_> = scene
                    .positions
                    .iter()
                    .copied()
                    .zip(scene.rotations.iter().copied())
                    .collect();
                self.scene_surface_masks = compute_surface_masks(&poses);
                self.scene_surface_mask_cache_key = Some(surface_cache_key);
            }
        }

        let mut instances: Vec<_> = if generated_world.is_some() || scene_cache_hit {
            Vec::new()
        } else {
            let damage = scene.incident_bond_damage();
            scene
                .positions
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    let color = if scene.voxels[i].kinematic {
                        [0.24, 0.28, 0.3, 1.]
                    } else if scene.axis_targets[i].iter().any(Option::is_some) {
                        [0.18, 0.55, 0.85, 1.]
                    } else if damage[i].1 >= 2 {
                        [0.83, 0.28, 0.24, 1.]
                    } else if damage[i].1 == 1 {
                        [0.92, 0.52, 0.24, 1.]
                    } else {
                        let d = damage[i].0;
                        let material = voxel_display_color(scene, i);
                        let amount = d.clamp(0.0, 1.0);
                        [
                            material[0] + (0.94 - material[0]) * amount,
                            material[1] + (0.75 - material[1]) * amount,
                            material[2] + (0.38 - material[2]) * amount,
                            1.0,
                        ]
                    };
                    let mut instance = Instance::cube(p.as_vec3(), scene.rotations[i], color);
                    if let Some(phase) = scene.phase_state(i) {
                        instance.thermal_emission = thermal_emission(
                            phase.temperature_kelvin,
                            phase.liquid_fraction,
                            phase.gas_fraction,
                        );
                    }
                    if !show_bonds {
                        instance.size = [CELL; 3];
                        if let Some(Some(world_mask)) = self.scene_surface_masks.get(i) {
                            let local_mask = local_face_mask(*world_mask, scene.rotations[i]);
                            instance.size = [CELL; 3];
                            instance.surface = -(f32::from(local_mask) + 1.0);
                        }
                    }
                    instance.skin_index = i as u32;
                    instance
                })
                .collect()
        };
        let mut scene_voxel_instance_count = if scene_cache_hit {
            self.scene_voxel_instance_count
        } else {
            0
        };
        if generated_world.is_none() && !scene_cache_hit {
            scene_voxel_instance_count = scene.voxels.len().min(instances.len());
            if !show_bonds {
                scene_voxel_instance_count =
                    compact_fully_hidden(&mut instances, scene_voxel_instance_count);
            }
        }
        if generated_world.is_none() && !scene_cache_hit {
            if let Some(t) = tool {
                if let Some(m) = t.mount {
                    // External guide visualization, not an additional collision body.
                    let direction = if m.direction == Vec3::ZERO {
                        Vec3::X
                    } else {
                        m.direction
                    };
                    let offset = Vec3::Y * 0.045;
                    instances.push(Instance {
                        center: (m.world_base + offset.as_dvec3()).as_vec3().to_array(),
                        size: [0.8, 0.008, 0.008],
                        color: [0.18, 0.55, 0.85, 1.0],
                        rotation: Quat::from_rotation_arc(Vec3::X, direction).to_array(),
                        surface: 0.0,
                        skin_index: u32::MAX,
                        thermal_emission: [0.0; 3],
                    });
                    let base = t.position + (t.rotation * m.local_base).as_dvec3();
                    instances.push(Instance::cube(
                        base.as_vec3(),
                        m.rotation,
                        [0.18, 0.55, 0.85, 1.0],
                    ));
                }
                instances.extend(
                    t.world_cells()
                        .map(|p| Instance::cube(p.as_vec3(), t.rotation, t.color)),
                );
            }
            if show_bonds {
                for axis in 0..3 {
                    for (i, b) in scene.bonds[axis].iter().enumerate() {
                        if b.neighbor == EMPTY || scene.voxels[i].kinematic {
                            continue;
                        }
                        let a = scene.positions[i];
                        let d = scene.positions[b.neighbor as usize] - a;
                        if d.length_squared() < 1e-12 {
                            continue;
                        }
                        instances.push(Instance {
                            center: (a + d * 0.5).as_vec3().to_array(),
                            size: [CELL * 0.08, CELL * 0.08, d.length() as f32],
                            rotation: Quat::from_rotation_arc(Vec3::Z, d.normalize().as_vec3())
                                .to_array(),
                            surface: 0.0,
                            color: [b.damage, 1. - b.damage, 0.2, 1.],
                            skin_index: u32::MAX,
                            thermal_emission: [0.0; 3],
                        });
                    }
                }
                // Shrink cubes to expose the lattice; absent bonds are never drawn.
                for i in 0..scene.voxels.len() {
                    if !scene.voxels[i].kinematic {
                        instances[i].size = [CELL * 0.35; 3];
                    }
                }
            }
            if let Some((center, normal)) = hovered_face {
                let normal = normal.normalize_or_zero();
                if normal.length_squared() > 0.5 {
                    instances.push(Instance {
                        center: (center + normal * (CELL * 0.52)).to_array(),
                        size: [CELL * 0.92, CELL * 0.92, CELL * 0.012],
                        color: [0.08, 0.95, 0.68, 1.0],
                        rotation: Quat::from_rotation_arc(Vec3::Z, normal).to_array(),
                        surface: 0.0,
                        skin_index: u32::MAX,
                        thermal_emission: [0.0; 3],
                    });
                }
            }
            if let Some(bounds) = selection_bounds {
                push_box_outline(&mut instances, bounds, [0.24, 1.0, 0.72, 1.0]);
            }
            if let Some(bounds) = selection_preview {
                push_box_outline(&mut instances, bounds, [1.0, 0.72, 0.18, 1.0]);
            }
            for &center in sculpt_preview_cells {
                instances.push(Instance {
                    center: center.to_array(),
                    size: [sculpt_preview_cell_size * 0.94; 3],
                    color: [0.12, 0.94, 1.0, 0.42],
                    rotation: Quat::IDENTITY.to_array(),
                    surface: 0.0,
                    skin_index: u32::MAX,
                    thermal_emission: [0.0; 3],
                });
            }
            if sculpt_preview_cells.is_empty() {
                if let Some(bounds) = sculpt_preview_bounds {
                    push_box_outline(&mut instances, bounds, [0.12, 0.94, 1.0, 0.95]);
                }
            }
        }
        let static_instance_count = if generated_world.is_some() {
            self.world_instance_cache.len() + self.world_transition_cache.len()
        } else if scene_cache_hit {
            self.scene_instance_cache.len()
        } else {
            instances.len()
        };
        let instance_count = static_instance_count;
        let upload_scene_instances = generated_world.is_none() && !scene_cache_hit;
        if upload_scene_instances && scene_cacheable {
            self.scene_instance_cache.clear();
            self.scene_instance_cache.extend_from_slice(&instances);
            self.scene_voxel_instance_count = scene_voxel_instance_count;
            self.scene_instance_cache_key = Some(scene_cache_key);
        } else if !scene_cacheable {
            self.scene_instance_cache_key = None;
        }
        if instance_count > self.instance_capacity {
            self.instance_capacity = instance_count.next_power_of_two();
            self.instances = instance_buffer(
                &self.device,
                self.instance_capacity,
                std::mem::size_of::<Instance>(),
            );
        }
        if ui.len() > self.ui_capacity {
            self.ui_capacity = ui.len().next_power_of_two();
            self.ui_instances =
                instance_buffer(&self.device, self.ui_capacity, std::mem::size_of::<Rect>());
        }
        if generated_world.is_some() {
            if world_cache_changed {
                self.queue.write_buffer(
                    &self.instances,
                    0,
                    bytemuck::cast_slice(&self.world_instance_cache),
                );
                if !self.world_transition_cache.is_empty() {
                    let transition_offset =
                        self.world_instance_cache.len() * std::mem::size_of::<Instance>();
                    self.queue.write_buffer(
                        &self.instances,
                        transition_offset as u64,
                        bytemuck::cast_slice(&self.world_transition_cache),
                    );
                }
            }
        } else if upload_scene_instances {
            self.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));
        }
        let selected_instances = if generated_world.is_none() {
            selected_voxels
                .iter()
                .copied()
                .filter(|&index| index < scene.positions.len())
                .map(|index| {
                    let mut instance = Instance::cube(
                        scene.positions[index].as_vec3(),
                        scene.rotations[index],
                        [0.08, 0.78, 0.59, 1.0],
                    );
                    instance.skin_index = index as u32;
                    if show_bonds && !scene.voxels[index].kinematic {
                        instance.size = [CELL * 0.35; 3];
                    }
                    instance
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        // Keep rig visualization in its own tiny cache and GPU buffer. Bone
        // selection changes therefore never invalidate/rebuild the voxel mesh.
        if let Some((model, selected_bone, cell_size, clip_index, time_s)) =
            rig.filter(|_| generated_world.is_none() && show_rig_overlay)
        {
            if let Some(skeleton) = model.skeleton.as_ref() {
                let posed_positions = clip_index.and_then(|index| {
                    skeleton
                        .bone_global_positions_at_with_material_limits(
                            index,
                            time_s,
                            Some(&self.bone_material_limits),
                        )
                        .ok()
                });
                let posed_positions = posed_positions.as_deref();
                let gizmo_key = bone_gizmo_key(skeleton, selected_bone, cell_size, posed_positions);
                if self.bone_gizmo_cache_key != Some(gizmo_key) {
                    self.bone_instance_cache =
                        build_bone_gizmos(skeleton, selected_bone, cell_size, posed_positions);
                    self.bone_gizmo_cache_key = Some(gizmo_key);
                    if self.bone_instance_cache.len() > self.bone_instance_capacity {
                        self.bone_instance_capacity =
                            self.bone_instance_cache.len().next_power_of_two();
                        self.bone_instances = instance_buffer(
                            &self.device,
                            self.bone_instance_capacity,
                            std::mem::size_of::<Instance>(),
                        );
                    }
                    if !self.bone_instance_cache.is_empty() {
                        self.queue.write_buffer(
                            &self.bone_instances,
                            0,
                            bytemuck::cast_slice(&self.bone_instance_cache),
                        );
                    }
                }
            }
        } else {
            self.bone_gizmo_cache_key = None;
            self.bone_instance_cache.clear();
        }
        if selected_instances.len() > self.selected_instance_capacity {
            self.selected_instance_capacity = selected_instances.len().next_power_of_two();
            self.selected_instances = instance_buffer(
                &self.device,
                self.selected_instance_capacity,
                std::mem::size_of::<Instance>(),
            );
        }
        if !selected_instances.is_empty() {
            self.queue.write_buffer(
                &self.selected_instances,
                0,
                bytemuck::cast_slice(&selected_instances),
            );
        }
        self.queue
            .write_buffer(&self.ui_instances, 0, bytemuck::cast_slice(ui));
        let camera_viewport =
            scene_viewport.unwrap_or([0, 0, self.config.width, self.config.height]);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(
                &camera
                    .matrix(camera_viewport[2] as f32 / camera_viewport[3].max(1) as f32)
                    .to_cols_array(),
            ),
        );
        self.queue.write_buffer(
            &self.screen_buffer,
            0,
            bytemuck::cast_slice(&[self.config.width as f32, self.config.height as f32, 0., 0.]),
        );
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.025,
                            g: 0.035,
                            b: 0.05,
                            a: 1.,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some([x, y, width, height]) = scene_viewport {
                pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, width, height);
            }
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_group, &[]);
            pass.set_bind_group(1, &self.skinning_default_bind_group, &[]);
            if authored_physics_override {
                if !self.authored_physics_instances.is_empty() {
                    pass.set_vertex_buffer(0, self.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.authored_physics_instance_buffer.slice(..));
                    pass.draw(0..36, 0..self.authored_physics_instances.len() as u32);
                }
            } else if generated_world.is_some() {
                if self.world_terrain_instance_count > 0 {
                    let terrain_bytes =
                        self.world_terrain_instance_count * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.terrain_vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(..terrain_bytes as u64));
                    pass.draw(0..30, 0..self.world_terrain_instance_count as u32);
                }
                // Transition instances have their own pass below. Including
                // them here draws them twice and treats terrain skirts as
                // full cubes in the opaque object pass.
                let object_count = self
                    .world_instance_cache
                    .len()
                    .saturating_sub(self.world_terrain_instance_count);
                if object_count > 0 {
                    let object_offset =
                        self.world_terrain_instance_count * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(object_offset as u64..));
                    pass.draw(0..36, 0..object_count as u32);
                }
            } else {
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                if skinning_enabled && scene_voxel_instance_count <= instance_count {
                    let voxel_count = scene_voxel_instance_count;
                    if voxel_count > 0 {
                        pass.set_bind_group(1, &self.skinning_bind_group, &[]);
                        let voxel_bytes = voxel_count * std::mem::size_of::<Instance>();
                        pass.set_vertex_buffer(1, self.instances.slice(..voxel_bytes as u64));
                        pass.draw(0..36, 0..voxel_count as u32);
                    }
                    let extra_count = instance_count - voxel_count;
                    if extra_count > 0 {
                        pass.set_bind_group(1, &self.skinning_default_bind_group, &[]);
                        let extra_offset = voxel_count * std::mem::size_of::<Instance>();
                        pass.set_vertex_buffer(1, self.instances.slice(extra_offset as u64..));
                        pass.draw(0..36, 0..extra_count as u32);
                    }
                } else {
                    pass.set_vertex_buffer(1, self.instances.slice(..));
                    pass.draw(0..36, 0..instance_count as u32);
                }
            }
            pass.set_bind_group(1, &self.skinning_default_bind_group, &[]);
            if !authored_physics_override && !selected_instances.is_empty() {
                if skinning_enabled {
                    pass.set_bind_group(1, &self.skinning_bind_group, &[]);
                }
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_vertex_buffer(1, self.selected_instances.slice(..));
                pass.draw(0..36, 0..selected_instances.len() as u32);
            }
            pass.set_bind_group(1, &self.skinning_default_bind_group, &[]);
            if !authored_physics_override && !self.physics_instance_cache.is_empty() {
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_vertex_buffer(1, self.physics_instances.slice(..));
                pass.draw(0..36, 0..self.physics_instance_cache.len() as u32);
            }
            if !authored_physics_override && !self.bone_instance_cache.is_empty() {
                pass.set_vertex_buffer(0, self.vertices.slice(..));
                pass.set_vertex_buffer(1, self.bone_instances.slice(..));
                pass.draw(0..36, 0..self.bone_instance_cache.len() as u32);
            }
            if !authored_physics_override
                && generated_world.is_some()
                && !self.world_transition_cache.is_empty()
            {
                pass.set_pipeline(&self.transition_pipeline);
                pass.set_bind_group(0, &self.camera_group, &[]);
                pass.set_bind_group(1, &self.skinning_default_bind_group, &[]);
                let transition_start = self.world_instance_cache.len();
                if self.world_transition_terrain_count > 0 {
                    let start_bytes = transition_start * std::mem::size_of::<Instance>();
                    let end_bytes = (transition_start + self.world_transition_terrain_count)
                        * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.terrain_vertices.slice(..));
                    pass.set_vertex_buffer(
                        1,
                        self.instances.slice(start_bytes as u64..end_bytes as u64),
                    );
                    pass.draw(0..30, 0..self.world_transition_terrain_count as u32);
                }
                let transition_object_start =
                    transition_start + self.world_transition_terrain_count;
                let transition_object_count = self
                    .world_transition_cache
                    .len()
                    .saturating_sub(self.world_transition_terrain_count);
                if transition_object_count > 0 {
                    let start_bytes = transition_object_start * std::mem::size_of::<Instance>();
                    pass.set_vertex_buffer(0, self.vertices.slice(..));
                    pass.set_vertex_buffer(1, self.instances.slice(start_bytes as u64..));
                    pass.draw(0..36, 0..transition_object_count as u32);
                }
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_bind_group(0, &self.screen_group, &[]);
            pass.set_vertex_buffer(0, self.ui_instances.slice(..));
            pass.draw(0..6, 0..ui.len() as u32);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

#[cfg(test)]
mod material_palette_tests {
    use super::{
        authored_frame_indices_match, has_axis_aligned_rotation, local_face_mask,
        material_display_color, AuthoredPhysicsRenderSource,
    };
    use glam::{DVec3, Quat};
    use voxel_physics_lab::world_physics::PhysicalCellPose;

    #[test]
    fn named_materials_have_distinct_display_colors() {
        let iron = material_display_color("iron");
        let wood = material_display_color("oak wood");
        let flesh = material_display_color("flesh");
        assert_ne!(iron, wood);
        assert_ne!(iron, flesh);
        assert_ne!(wood, flesh);
    }

    #[test]
    fn thermal_emission_preserves_cold_albedo_and_tracks_blackbody_hue() {
        use super::thermal_emission;

        assert_eq!(thermal_emission(293.15, 0.0, 0.0), [0.0; 3]);
        let onset = thermal_emission(900.0, 0.0, 0.0);
        let orange = thermal_emission(2_400.0, 0.0, 0.0);
        let blue_white = thermal_emission(9_000.0, 0.0, 0.0);
        assert!(onset[0] > onset[1] && onset[2] == 0.0);
        assert!(onset[0] < orange[0]);
        assert!(orange[1] > 0.0 && orange[0] > orange[1]);
        assert!(blue_white[2] > blue_white[0]);
        assert!(blue_white.iter().sum::<f32>() > orange.iter().sum::<f32>());
    }

    #[test]
    fn visible_incandescence_dominates_neutral_albedo_at_red_heat() {
        let emission = super::thermal_emission(1_500.0, 0.0, 0.0);
        let peak = emission.into_iter().fold(0.0_f32, f32::max);
        let displayed = emission.map(|channel| (0.4 + channel) / (1.0 + peak));
        assert!(displayed[0] > displayed[1] * 1.5);
        assert!(displayed[1] > displayed[2]);
        for temperature in [f32::NAN, f32::INFINITY, -1.0, 1.0e20] {
            assert!(super::thermal_emission(temperature, 0.0, 0.0)
                .iter().all(|channel| channel.is_finite() && *channel >= 0.0));
        }
    }

    #[test]
    fn thermal_emission_responds_to_phase_fraction() {
        use super::thermal_emission;

        let solid = thermal_emission(2_500.0, 0.0, 0.0);
        let liquid = thermal_emission(2_500.0, 1.0, 0.0);
        let gas = thermal_emission(2_500.0, 0.0, 1.0);
        assert!(liquid[0] > solid[0]);
        assert!(solid[0] > gas[0]);
    }

    #[test]
    fn authored_material_mapping_requires_same_dynamic_rev_indices() {
        let source = AuthoredPhysicsRenderSource {
            dynamic_scene_indices: vec![0, 2],
            material_colors: vec![[0.0; 4]; 3],
            kinematic_poses: Vec::new(),
            cache_key: 0,
        };
        let cell = |scene_index| PhysicalCellPose {
            scene_index,
            source_cell: None,
            world_position_m: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            temperature_kelvin: 293.15,
            liquid_fraction: 0.0,
            gas_fraction: 0.0,
        };

        assert!(authored_frame_indices_match(&source, &[cell(0), cell(2)]));
        assert!(!authored_frame_indices_match(&source, &[cell(0)]));
        assert!(!authored_frame_indices_match(&source, &[cell(0), cell(1)]));
    }

    #[test]
    fn axis_aligned_rotation_maps_world_occlusion_to_local_face_bits() {
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!(has_axis_aligned_rotation(rotation));
        assert_eq!(local_face_mask(1 << 0, rotation), 1 << 3);
        assert!(!has_axis_aligned_rotation(Quat::from_rotation_y(0.1)));
    }

    #[test]
    fn surface_culling_keeps_revs_and_preserves_gaps_and_rotated_cells() {
        use super::{apply_surface_culling, Instance};
        use crate::scene::CELL;
        use glam::{DVec3, Vec3};

        let mut touching = [
            Instance::cube(Vec3::ZERO, Quat::IDENTITY, [1.0; 4]),
            Instance::cube(Vec3::X * CELL, Quat::IDENTITY, [1.0; 4]),
        ];
        apply_surface_culling(&mut touching, |index| {
            (
                DVec3::new(index as f64 * f64::from(CELL), 0.0, 0.0),
                Quat::IDENTITY,
            )
        });
        assert_eq!(touching.len(), 2);
        assert_eq!(touching[0].size, [CELL; 3]);
        assert_eq!((-(touching[0].surface) - 1.0) as u8, 0b11_1101);
        assert_eq!((-(touching[1].surface) - 1.0) as u8, 0b11_1110);

        let mut gap = [
            Instance::cube(Vec3::ZERO, Quat::IDENTITY, [1.0; 4]),
            Instance::cube(Vec3::X * (CELL * 1.01), Quat::IDENTITY, [1.0; 4]),
        ];
        apply_surface_culling(&mut gap, |index| {
            (
                DVec3::new(index as f64 * f64::from(CELL * 1.01), 0.0, 0.0),
                Quat::IDENTITY,
            )
        });
        assert_eq!((-(gap[0].surface) - 1.0) as u8, 0b11_1111);

        let rotated = Quat::from_rotation_z(0.1);
        let mut rotated_pair = [
            Instance::cube(Vec3::ZERO, rotated, [1.0; 4]),
            Instance::cube(Vec3::X * CELL, Quat::IDENTITY, [1.0; 4]),
        ];
        apply_surface_culling(&mut rotated_pair, |index| {
            (
                DVec3::new(index as f64 * f64::from(CELL), 0.0, 0.0),
                if index == 0 { rotated } else { Quat::IDENTITY },
            )
        });
        assert_eq!(rotated_pair[0].surface, 0.0);
        assert_eq!((-(rotated_pair[1].surface) - 1.0) as u8, 0b11_1111);
        assert!(rotated_pair
            .iter()
            .all(|instance| instance.size == [CELL; 3]));
    }

    #[test]
    fn compaction_drops_only_hidden_render_cell_and_preserves_physics_count() {
        use super::{apply_surface_culling, compact_fully_hidden, Instance};
        use crate::scene::CELL;
        use glam::{DVec3, Quat};

        let mut physics_poses = Vec::new();
        let mut render_instances = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    let position = DVec3::new(
                        f64::from(x) * f64::from(CELL),
                        f64::from(y) * f64::from(CELL),
                        f64::from(z) * f64::from(CELL),
                    );
                    physics_poses.push((position, Quat::IDENTITY));
                    render_instances.push(Instance::cube(
                        position.as_vec3(),
                        Quat::IDENTITY,
                        [1.0; 4],
                    ));
                }
            }
        }

        let physics_count = physics_poses.len();
        apply_surface_culling(&mut render_instances, |index| physics_poses[index]);
        let render_count = compact_fully_hidden(&mut render_instances, physics_count);

        assert_eq!(physics_count, 27);
        assert_eq!(render_count, 26);
        assert_eq!(render_instances.len(), 26);
        assert!(render_instances
            .iter()
            .all(|instance| instance.surface != -1.0));
        assert_eq!(74_usize.pow(3) - 72_usize.pow(3), 31_976);
    }
}

#[cfg(test)]
mod depth_tests {
    use super::Camera;
    use glam::{Quat, Vec3};

    fn native_editor_scene_viewport(width: u32, height: u32) -> [u32; 4] {
        let mut ui = crate::ui::Ui::new(1.0, Vec3::ZERO, Vec3::Z);
        ui.world_active = true;
        ui.editor_mode = true;
        ui.set_active_scene_document(Some("framing test".into()), vec!["spider".into()]);
        ui.selected_scene_model = Some(0);
        ui.draw(0, 1, 60.0, 0.0, None, None, width as f32, height as f32, None);
        ui.scene_viewport().expect("native scene viewport")
    }

    #[test]
    fn reverse_depth_preserves_near_and_distant_surfaces_in_every_camera_mode() {
        for mode in 0..3 {
            let mut camera = Camera::new();
            if mode == 1 {
                camera.set_first_person(Vec3::new(0.0, 0.0, 1.7));
            } else if mode == 2 {
                camera.enable_freecam();
            }
            let matrix = camera.matrix(16.0 / 9.0);
            let depth = |distance: f32| {
                let point = camera.position() + camera.forward() * distance;
                let clip = matrix * point.extend(1.0);
                clip.z / clip.w
            };
            let close = depth(0.02);
            let middle = depth(10.0);
            let horizon = depth(10_000.0);
            assert!(close <= 1.0 && close > middle);
            assert!(middle > horizon && horizon > 0.0);
        }
    }

    #[test]
    fn orbit_fit_contains_rotated_tall_bounds_in_native_scene_viewports() {
        // A tall, narrow model turned diagonally produces a wider world-space
        // AABB than its unrotated local dimensions.
        let half_extents = Vec3::new(0.012, 0.008, 0.24);
        let rotation = Quat::from_rotation_y(0.23) * Quat::from_rotation_z(0.71);
        let (mut minimum, mut maximum) = (
            Vec3::splat(f32::INFINITY),
            Vec3::splat(f32::NEG_INFINITY),
        );
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let corner = rotation * (half_extents * Vec3::new(x, y, z));
                    minimum = minimum.min(corner);
                    maximum = maximum.max(corner);
                }
            }
        }
        let half_cell = 0.005;
        minimum -= Vec3::splat(half_cell);
        maximum += Vec3::splat(half_cell);
        let sphere_radius = (maximum - minimum).length() * 0.5;
        let expected_margin = f64::from(super::CAMERA_FIT_MARGIN);

        for (width, height) in [(1920, 1009), (1280, 720)] {
            let [_, _, viewport_width, viewport_height] =
                native_editor_scene_viewport(width, height);
            let aspect = viewport_width as f32 / viewport_height.max(1) as f32;
            let distance = Camera::fit_orbit_distance_for_sphere(sphere_radius, aspect);
            let vertical_half_angle = f64::from(super::CAMERA_VERTICAL_FOV_RADIANS) * 0.5;
            let horizontal_half_angle = (f64::from(aspect) * vertical_half_angle.tan()).atan();
            let limiting_half_angle = vertical_half_angle.min(horizontal_half_angle);

            assert!(
                distance.is_finite()
                    && f64::from(distance) * limiting_half_angle.sin()
                        >= f64::from(sphere_radius) * expected_margin - 1.0e-6,
                "sphere does not fit at {width}x{height}, viewport={viewport_width}x{viewport_height}, distance={distance}, radius={sphere_radius}"
            );
        }
    }

    #[test]
    fn orbit_fit_invalid_inputs_use_finite_fallbacks() {
        assert_eq!(
            Camera::fit_orbit_distance_for_sphere(0.0, 1.0),
            super::CAMERA_FIT_FALLBACK_DISTANCE_M
        );
        assert_eq!(
            Camera::fit_orbit_distance_for_sphere(f32::NAN, 1.0),
            super::CAMERA_FIT_FALLBACK_DISTANCE_M
        );
        let distance = Camera::fit_orbit_distance_for_sphere(0.1, f32::NAN);
        assert!(distance.is_finite() && distance >= super::CAMERA_MIN_ORBIT_DISTANCE_M);
    }
}
