use crate::{
    bond::{Bond, BondWorkspace, EMPTY},
    material::Material,
    voxel::{Composition, Microstructure, Voxel},
};
use glam::{Quat, Vec3};
use serde::Deserialize;
use std::{collections::HashMap, fs};

pub const CELL: f32 = 0.01;
pub const AREA: f32 = CELL * CELL;
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    pub dimensions: [usize; 3],
    pub branch_dimensions: [usize; 3],
    pub ground_dimensions: [usize; 3],
    pub tool_dimensions: [usize; 3],
    pub fibers: [f32; 3],
    pub anisotropy: f32,
    pub gravity: [f32; 3],
    pub fixed_dt: f32,
    pub substeps: usize,
    pub iterations: usize,
    pub solver_relative_tolerance: f32,
    pub corotation_tolerance: f32,
    pub corotation_iterations: usize,
    pub fracture: crate::fracture::FractureConfig,

    pub initial_speed: f32,
    pub tool_start: [f32; 3],
}

impl SceneConfig {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let result: Self = ron::from_str(&fs::read_to_string(path)?)?;
        if result.dimensions.contains(&0)
            || result.ground_dimensions.contains(&0)
            || result.tool_dimensions.iter().any(|&n| n < 2)
            || !(0.0..=1.0).contains(&result.anisotropy)
            || !Vec3::from_array(result.gravity).is_finite()
            || !Vec3::from_array(result.tool_start).is_finite()
            || !result.initial_speed.is_finite()
            || result.initial_speed < 0.0
            || !Vec3::from_array(result.fibers).is_finite()
            || !result.fixed_dt.is_finite()
            || result.fixed_dt <= 0.0
            || result.substeps == 0
            || result.iterations == 0
            || result.corotation_iterations == 0
            || !result.solver_relative_tolerance.is_finite()
            || result.solver_relative_tolerance <= 0.0
            || !result.corotation_tolerance.is_finite()
            || result.corotation_tolerance <= 0.0
            || !result.fracture.validate()
            || Vec3::from_array(result.fibers).length_squared() < 1e-8
        {
            return Err("Invalid scene parameters".into());
        }
        Ok(result)
    }
}
/// Positions are separate arrays; material identity exists only in composition.
pub struct Scene {
    pub config: SceneConfig,
    pub voxels: Vec<Voxel>,
    /// Persistent per-voxel forces in newtons, retained until explicitly reset.
    pub external_forces: Vec<Vec3>,
    /// Persistent per-voxel acceleration inputs in m/s².
    pub external_accelerations: Vec<Vec3>,
    /// Persistent world-coordinate constraints; None leaves an axis free.
    pub axis_targets: Vec<[Option<f32>; 3]>,
    pub positions: Vec<Vec3>,
    pub rest_positions: Vec<Vec3>,
    pub previous: Vec<Vec3>,
    pub bonds: [Vec<Bond>; 3],
    pub workspace: [Vec<BondWorkspace>; 3],
    pub colors: [Vec<usize>; 6],
    pub materials: Vec<Material>,
    pub properties: Vec<Material>,
    pub rotations: Vec<Quat>,
    /// Derived connectivity labels; rebuilt only when topology changes.
    pub clusters: Vec<u32>,
    pub broken: usize,
    pub cluster_count: usize,
    pub ground_top: f32,
    pub energy: crate::fracture::EnergyLedger,
    pub releases: Vec<crate::fracture::Release>,
    pub substep_index: u64,
    pub solver_residual: f32,
}
impl Scene {
    /// Derived display data for all incident bonds, including the negative-axis
    /// endpoint. Original outside faces are not damage. No visual state is stored.
    pub fn incident_bond_damage(&self) -> Vec<(f32, u8)> {
        let mut result = vec![(0.0f32, 0u8); self.voxels.len()];
        for axis in 0..3 {
            for (i, bond) in self.bonds[axis].iter().enumerate() {
                let original = self.workspace[axis][i].rest_neighbor;
                if original == EMPTY { continue; }
                let j = original as usize;
                let broken = bond.neighbor == EMPTY;
                let damage = if broken { 1.0 } else { bond.damage };
                for n in [i, j] {
                    result[n].0 = result[n].0.max(damage);
                    result[n].1 += u8::from(broken);
                }
            }
        }
        result
    }

    pub fn new(config: SceneConfig) -> Self {
        let materials = Material::table();
        let mut voxels = Vec::new();
        let mut positions = Vec::new();
        let mut keys = Vec::new();
        let mut occupancy = HashMap::new();
        let fibers = Vec3::from_array(config.fibers).normalize();
        let orientation = Quat::from_rotation_arc(Vec3::Z, fibers);
        let mut add =
            |key: [i32; 3], component: u16, anisotropy: f32, fixed: bool, orientation: Quat| {
                if occupancy.contains_key(&key) {
                    return;
                }
                let m = &materials[component as usize];
                let mass = m.density * CELL.powi(3);
                occupancy.insert(key, voxels.len());
                keys.push(key);
                positions.push(
                    Vec3::new(key[0] as f32, key[1] as f32, key[2] as f32) * CELL
                        + Vec3::splat(CELL * 0.5),
                );
                voxels.push(Voxel {
                    mass: if fixed { f32::INFINITY } else { mass },
                    composition: Composition::pure(component),
                    internal_energy: mass * m.heat_capacity * 293.15,
                    velocity: Vec3::ZERO,
                    microstructure: Microstructure {
                        orientation,
                        anisotropy,
                    },
                });
            };
        let [nx, ny, nz] = config.dimensions;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    add(
                        [x as i32 - nx as i32 / 2, y as i32 - ny as i32 / 2, z as i32],
                        1,
                        config.anisotropy,
                        false,
                        orientation,
                    );
                }
            }
        }
        let [bx, by, bz] = config.branch_dimensions;
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    add(
                        [
                            nx as i32 / 2 + x as i32,
                            y as i32 - by as i32 / 2,
                            (nz * 2 / 3 + z) as i32,
                        ],
                        1,
                        config.anisotropy,
                        false,
                        Quat::from_rotation_arc(Vec3::Z, Vec3::X),
                    );
                }
            }
        }
        let [gx, gy, gz] = config.ground_dimensions;
        for z in 0..gz {
            for y in 0..gy {
                for x in 0..gx {
                    add(
                        [
                            x as i32 - gx as i32 / 2,
                            y as i32 - gy as i32 / 2,
                            -(z as i32) - 1,
                        ],
                        3,
                        0.0,
                        true,
                        Quat::IDENTITY,
                    );
                }
            }
        }
        let n = voxels.len();
        let mut bonds: [Vec<Bond>; 3] = std::array::from_fn(|_| vec![Bond::default(); n]);
        let mut workspace: [Vec<BondWorkspace>; 3] = std::array::from_fn(|_| {
            vec![
                BondWorkspace {
                    rest_neighbor: EMPTY,
                    ..Default::default()
                };
                n
            ]
        });
        let mut colors: [Vec<usize>; 6] = std::array::from_fn(|_| Vec::new());
        for (i, key) in keys.iter().enumerate() {
            for axis in 0..3 {
                let mut neighbor = *key;
                neighbor[axis] += 1;
                if let Some(&j) = occupancy.get(&neighbor) {
                    // Ground supports by contact; do not glue different bodies together.
                    if voxels[i].mass.is_infinite() == voxels[j].mass.is_infinite() {
                        bonds[axis][i].neighbor = j as u32;
                        workspace[axis][i].rest_neighbor = j as u32;
                        colors[axis * 2 + key[axis].rem_euclid(2) as usize].push(i);
                    }
                }
            }
        }
        let properties = voxels
            .iter()
            .map(|v| Material::mixed(&v.composition, &materials))
            .collect();
        Self {
            config,
            external_forces: vec![Vec3::ZERO; n],
            external_accelerations: vec![Vec3::ZERO; n],
            axis_targets: vec![[None; 3]; n],
            previous: positions.clone(),
            rest_positions: positions.clone(),
            positions,
            voxels,
            bonds,
            workspace,
            colors,
            materials,
            properties,
            rotations: vec![Quat::IDENTITY; n],
            clusters: vec![0; n],
            broken: 0,
            cluster_count: 1,
            ground_top: 0.0,
            energy: Default::default(),
            releases: Vec::new(),
            substep_index: 0,
            solver_residual: 0.0,
        }
    }
}

#[cfg(test)]
mod damage_display_tests {
    use super::*;

    #[test]
    fn original_boundary_is_intact_and_both_crack_endpoints_show_damage() {
        let mut config = SceneConfig::load("scenes/wood_chop.ron").unwrap();
        config.dimensions = [2, 1, 1];
        config.branch_dimensions = [0; 3];
        config.ground_dimensions = [0; 3];
        let mut scene = Scene::new(config);
        assert_eq!(scene.incident_bond_damage(), vec![(0.0, 0); 2]);
        scene.bonds[0][0].damage = 0.4;
        assert_eq!(scene.incident_bond_damage(), vec![(0.4, 0); 2]);
        scene.bonds[0][0].neighbor = EMPTY;
        assert_eq!(scene.incident_bond_damage(), vec![(1.0, 1); 2]);
    }
}
