//! Normal contact constraints for the implicit mechanical solve.
//!
//! These are temporary unilateral constraints, not material bonds. Each row
//! equates one normal displacement component; tangential motion stays free.
use glam::{DVec3, Vec3};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PairContact {
    pub i: usize,
    pub j: usize,
    pub axis: usize,
    pub sign: f64,
    pub active: bool,
}

#[derive(Default)]
pub(crate) struct ContactProjection {
    nodes: Vec<(usize, usize, Option<usize>)>,
    groups: Vec<(Vec<usize>, bool)>,
    trees: Vec<Vec<(usize, usize)>>,
}

impl ContactProjection {
    pub fn build(pairs: &[PairContact], free: &[Vec3], active_index: &[u32]) -> Self {
        let mut result = Self::default();
        let mut ids = FxHashMap::default();
        for pair in pairs.iter().filter(|p| p.active) {
            for voxel in [pair.i, pair.j] {
                ids.entry((voxel, pair.axis)).or_insert_with(|| {
                    let id = result.nodes.len();
                    let slot =
                        (free[voxel][pair.axis] != 0.0).then_some(active_index[voxel] as usize);
                    result.nodes.push((voxel, pair.axis, slot));
                    id
                });
            }
        }
        let mut parent: Vec<_> = (0..result.nodes.len()).collect();
        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        result.trees.resize_with(parent.len(), Vec::new);
        for (edge, pair) in pairs.iter().enumerate().filter(|(_, p)| p.active) {
            let i = ids[&(pair.i, pair.axis)];
            let j = ids[&(pair.j, pair.axis)];
            let ri = root(&mut parent, i);
            let rj = root(&mut parent, j);
            if ri != rj {
                parent[rj] = ri;
                result.trees[i].push((j, edge));
                result.trees[j].push((i, edge));
            }
        }
        let mut groups = FxHashMap::<usize, Vec<usize>>::default();
        for i in 0..parent.len() {
            groups.entry(root(&mut parent, i)).or_default().push(i);
        }
        result.groups = groups
            .into_values()
            .map(|members| {
                let fixed = members.iter().any(|&i| result.nodes[i].2.is_none());
                (members, fixed)
            })
            .collect();
        result
    }

    /// Euclidean orthogonal projection onto the homogeneous contact nullspace.
    pub fn project(&self, vector: &mut [Vec3]) {
        for (members, fixed) in &self.groups {
            let mean = if *fixed {
                0.0
            } else {
                members
                    .iter()
                    .map(|&i| {
                        let (_, axis, slot) = self.nodes[i];
                        f64::from(vector[slot.unwrap()][axis])
                    })
                    .sum::<f64>()
                    / members.len() as f64
            } as f32;
            for &i in members {
                let (_, axis, slot) = self.nodes[i];
                if let Some(slot) = slot {
                    vector[slot][axis] = mean;
                }
            }
        }
    }

    /// The same orthogonal projection for cell vectors in double precision. Only
    /// the first three components, the translation, carry contact rows; the
    /// rotation components are untouched.
    pub fn project_cells(&self, vector: &mut [[f64; 6]]) {
        for (members, fixed) in &self.groups {
            let mean = if *fixed {
                0.0
            } else {
                members
                    .iter()
                    .map(|&i| {
                        let (_, axis, slot) = self.nodes[i];
                        vector[slot.unwrap()][axis]
                    })
                    .sum::<f64>()
                    / members.len() as f64
            };
            for &i in members {
                let (_, axis, slot) = self.nodes[i];
                if let Some(slot) = slot {
                    vector[slot][axis] = mean;
                }
            }
        }
    }

    /// A particular solution of C x = target. The tree keeps dependent
    /// manifold rows out of the solve, just as it does for reaction recovery.
    pub fn particular(
        &self,
        pairs: &[PairContact],
        targets: &[f64],
        active_count: usize,
    ) -> Vec<Vec3> {
        let mut result = vec![Vec3::ZERO; active_count];
        let mut values = vec![0.0; self.nodes.len()];
        let mut visited = vec![false; self.nodes.len()];
        for (members, fixed) in &self.groups {
            let start = members.iter().copied()
                .find(|&i| self.nodes[i].2.is_none()).unwrap_or(members[0]);
            let mut stack = vec![start];
            visited[start] = true;
            while let Some(i) = stack.pop() {
                for &(j, edge) in &self.trees[i] {
                    if visited[j] { continue; }
                    let pair = pairs[edge];
                    let direction = if self.nodes[i].0 == pair.i { 1.0 } else { -1.0 };
                    values[j] = values[i] + direction * pair.sign * targets[edge];
                    visited[j] = true;
                    stack.push(j);
                }
            }
            let mean = if *fixed { 0.0 } else {
                members.iter().map(|&i| values[i]).sum::<f64>() / members.len() as f64
            };
            for &i in members {
                let (_, axis, slot) = self.nodes[i];
                if let Some(slot) = slot { result[slot][axis] = (values[i] - mean) as f32; }
            }
        }
        result
    }

    /// Recover independent normal reactions by tree elimination. Positive is
    /// compressive. Redundant rows receive zero, avoiding a singular multiplier
    /// system without constraining any extra degree of freedom.
    pub fn reactions(&self, pairs: &[PairContact], gradient: &[DVec3]) -> Vec<f64> {
        let mut reactions = vec![0.0; pairs.len()];
        let mut sums = vec![0.0; self.nodes.len()];
        let mut parents = vec![usize::MAX; self.nodes.len()];
        let mut order = Vec::new();
        for (members, _) in &self.groups {
            let start = members
                .iter()
                .copied()
                .find(|&i| self.nodes[i].2.is_none())
                .unwrap_or(members[0]);
            order.clear();
            order.push(start);
            parents[start] = start;
            let mut cursor = 0;
            while cursor < order.len() {
                let i = order[cursor];
                let (voxel, axis, _) = self.nodes[i];
                sums[i] = gradient[voxel][axis];
                for &(j, _) in &self.trees[i] {
                    if j != parents[i] {
                        parents[j] = i;
                        order.push(j);
                    }
                }
                cursor += 1;
            }
            for &i in order.iter().rev() {
                if i == start {
                    continue;
                }
                let parent = parents[i];
                let edge = self.trees[i].iter().find(|&&(j, _)| j == parent).unwrap().1;
                let pair = pairs[edge];
                let coefficient = if self.nodes[i].0 == pair.j {
                    pair.sign
                } else {
                    -pair.sign
                };
                reactions[edge] = sums[i] / coefficient;
                sums[parent] += sums[i];
            }
        }
        reactions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normal_projection_is_orthogonal_idempotent_and_keeps_tangents() {
        let pairs = vec![
            PairContact {
                i: 0,
                j: 1,
                axis: 0,
                sign: 1.0,
                active: true,
            },
            PairContact {
                i: 1,
                j: 2,
                axis: 0,
                sign: 1.0,
                active: true,
            },
        ];
        let p = ContactProjection::build(&pairs, &[Vec3::ONE; 3], &[0, 1, 2]);
        let input = vec![
            Vec3::new(1., 7., 9.),
            Vec3::new(3., 2., 4.),
            Vec3::new(5., 8., 1.),
        ];
        let mut projected = input.clone();
        p.project(&mut projected);
        assert_eq!(
            projected.iter().map(|v| v.x).collect::<Vec<_>>(),
            vec![3.; 3]
        );
        assert!(projected
            .iter()
            .zip(&input)
            .all(|(a, b)| a.y == b.y && a.z == b.z));
        let twice = projected.clone();
        p.project(&mut projected);
        assert_eq!(projected, twice);
        let orthogonality: f32 = input
            .iter()
            .zip(&projected)
            .map(|(a, b)| (*a - *b).dot(*b))
            .sum();
        assert!(orthogonality.abs() < 1e-6);
        assert_eq!(
            p.reactions(&pairs, &[DVec3::X * -2., DVec3::X, DVec3::X]),
            vec![2., 1.]
        );
    }
    #[test]
    fn affine_contact_targets_are_satisfied_without_a_net_translation() {
        let pairs = vec![
            PairContact { i: 0, j: 1, axis: 0, sign: 1.0, active: true },
            PairContact { i: 1, j: 2, axis: 0, sign: 1.0, active: true },
        ];
        let projection = ContactProjection::build(&pairs, &[Vec3::ONE; 3], &[0, 1, 2]);
        let particular = projection.particular(&pairs, &[2.0, 3.0], 3);
        assert!((particular[1].x - particular[0].x - 2.0).abs() < 1.0e-6);
        assert!((particular[2].x - particular[1].x - 3.0).abs() < 1.0e-6);
        assert!(particular.iter().map(|v| v.x).sum::<f32>().abs() < 1.0e-6);
        assert!(particular.iter().all(|v| v.y == 0.0 && v.z == 0.0));
    }

    #[test]
    fn fixed_axis_pins_contact_group_and_redundant_cycles_are_safe() {
        let pairs = vec![
            PairContact {
                i: 0,
                j: 1,
                axis: 0,
                sign: 1.,
                active: true,
            },
            PairContact {
                i: 1,
                j: 2,
                axis: 0,
                sign: 1.,
                active: true,
            },
            PairContact {
                i: 0,
                j: 2,
                axis: 0,
                sign: 1.,
                active: true,
            },
        ];
        let p = ContactProjection::build(
            &pairs,
            &[Vec3::ZERO, Vec3::ONE, Vec3::ONE],
            &[u32::MAX, 0, 1],
        );
        let mut v = vec![Vec3::ONE; 2];
        p.project(&mut v);
        assert_eq!(v, vec![Vec3::new(0., 1., 1.); 2]);
        assert_eq!(
            p.reactions(&pairs, &[DVec3::ZERO, DVec3::X, DVec3::X]),
            vec![2., 1., 0.]
        );
    }
}
