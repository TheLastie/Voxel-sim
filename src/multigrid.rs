//! Algebraic aggregation V-cycle for the frozen vector elastic operator.
//!
//! This is a PCG preconditioner, not a change of physical resolution. Every
//! correction is still checked against the original fine REV system. Coarse
//! matrices are Galerkin projections, A_c = P^T A P, of connected aggregates.
//! Forward/backward block Gauss-Seidel and an exact bottom solve make the
//! cycle symmetric; no contact, fracture, or thermal state is aggregated here.

use glam::{DMat3, Mat3, Vec3};
use rayon::prelude::*;

const AGGREGATE_SIZE: usize = 8;
const DENSE_BOTTOM_ROWS: usize = 24;
const MAX_LEVELS: usize = 12;

pub(crate) struct Multigrid {
    levels: Vec<Level>,
    jacobi_smoother: bool,
}

struct Level {
    diagonal: Vec<Mat3>,
    inverse: Vec<Mat3>,
    free: Vec<Vec3>,
    offsets: Vec<usize>,
    columns: Vec<usize>,
    blocks: Vec<Mat3>,
    aggregate: Vec<usize>,
    prolong_offsets: Vec<usize>,
    prolong_columns: Vec<usize>,
    prolong_blocks: Vec<Mat3>,
    coarse_rhs: Vec<Vec3>,
    coarse_solution: Vec<Vec3>,
    bottom_factor: Vec<f64>,
    bottom_work: Vec<f64>,
    smooth_work: Vec<Vec3>,
}

impl Level {
    fn new(diagonal: Vec<Mat3>, free: Vec<Vec3>, offsets: Vec<usize>,
        columns: Vec<usize>, blocks: Vec<Mat3>) -> Result<Self, &'static str> {
        let inverse: Vec<_> = diagonal.iter().map(|d| d.as_dmat3().inverse().as_mat3()).collect();
        if inverse.iter().any(|d| !d.is_finite()) {
            return Err("nonfinite multigrid block inverse");
        }
        Ok(Self { diagonal, inverse, free, offsets, columns, blocks,
            aggregate: Vec::new(), coarse_rhs: Vec::new(), coarse_solution: Vec::new(),
            prolong_offsets: Vec::new(), prolong_columns: Vec::new(), prolong_blocks: Vec::new(),
            bottom_factor: Vec::new(), bottom_work: Vec::new(), smooth_work: Vec::new() })
    }

    fn smooth(&mut self, rhs: &[Vec3], x: &mut [Vec3], reverse: bool, jacobi: bool) {
        if jacobi {
            let omega = 0.65;
            self.smooth_work.resize(x.len(), Vec3::ZERO);
            let (diagonal, inverse, free, offsets, columns, blocks, smooth_work) =
                (&self.diagonal, &self.inverse, &self.free, &self.offsets,
                 &self.columns, &self.blocks, &mut self.smooth_work);
            let update = |(i, out): (usize, &mut Vec3)| {
                let mut residual = rhs[i] - diagonal[i] * x[i];
                for edge in offsets[i]..offsets[i + 1] {
                    residual -= blocks[edge] * x[columns[edge]];
                }
                *out = (x[i] + omega * (inverse[i] * residual)) * free[i];
            };
            if x.len() >= 8_192 {
                smooth_work.par_iter_mut().enumerate().for_each(update);
            } else {
                smooth_work.iter_mut().enumerate().for_each(update);
            }
            x.copy_from_slice(smooth_work);
            return;
        }
        let update = |i: usize, x: &mut [Vec3]| {
            let mut value = rhs[i];
            for edge in self.offsets[i]..self.offsets[i + 1] {
                value -= self.blocks[edge] * x[self.columns[edge]];
            }
            x[i] = (self.inverse[i] * value) * self.free[i];
        };
        if reverse {
            for i in (0..x.len()).rev() { update(i, x); }
        } else {
            for i in 0..x.len() { update(i, x); }
        }
    }

    fn factor_bottom(&mut self) -> Result<(), &'static str> {
        let n = self.diagonal.len() * 3;
        let mut matrix = vec![0.0_f64; n * n];
        for i in 0..self.diagonal.len() {
            for a in 0..3 {
                for b in 0..3 { matrix[(3*i+a)*n+3*i+b] = self.diagonal[i].col(b)[a] as f64; }
            }
            for edge in self.offsets[i]..self.offsets[i+1] {
                let j = self.columns[edge];
                for a in 0..3 {
                    for b in 0..3 { matrix[(3*i+a)*n+3*j+b] = self.blocks[edge].col(b)[a] as f64; }
                }
            }
        }
        for i in 0..n {
            for j in 0..=i {
                let mut value = matrix[i*n+j];
                for k in 0..j { value -= matrix[i*n+k] * matrix[j*n+k]; }
                if i == j {
                    if !value.is_finite() || value <= 0.0 {
                        return Err("Galerkin coarse operator is not positive definite");
                    }
                    matrix[i*n+j] = value.sqrt();
                } else {
                    matrix[i*n+j] = value / matrix[j*n+j];
                }
            }
        }
        self.bottom_factor = matrix;
        self.bottom_work.resize(n, 0.0);
        Ok(())
    }

    fn solve_bottom(&mut self, rhs: &[Vec3], out: &mut [Vec3]) {
        let n = self.bottom_work.len();
        for i in 0..n {
            let mut value = rhs[i/3][i%3] as f64;
            for j in 0..i { value -= self.bottom_factor[i*n+j] * self.bottom_work[j]; }
            self.bottom_work[i] = value / self.bottom_factor[i*n+i];
        }
        for i in (0..n).rev() {
            let mut value = self.bottom_work[i];
            for j in i+1..n { value -= self.bottom_factor[j*n+i] * self.bottom_work[j]; }
            self.bottom_work[i] = value / self.bottom_factor[i*n+i];
            out[i/3][i%3] = self.bottom_work[i] as f32;
        }
    }

    fn coarsen(&mut self, smooth_interpolation: bool) -> Result<Option<Level>, &'static str> {
        let n = self.diagonal.len();
        self.aggregate = vec![usize::MAX; n];
        let mut count = 0usize;
        let mut frontier = Vec::with_capacity(AGGREGATE_SIZE * 8);
        for seed in 0..n {
            if self.aggregate[seed] != usize::MAX { continue; }
            frontier.clear();
            frontier.push(seed);
            self.aggregate[seed] = count;
            let mut size = 1;
            let mut cursor = 0;
            while cursor < frontier.len() && size < AGGREGATE_SIZE {
                let i = frontier[cursor];
                cursor += 1;
                for edge in self.offsets[i]..self.offsets[i+1] {
                    let j = self.columns[edge];
                    if self.aggregate[j] == usize::MAX && self.blocks[edge] != Mat3::ZERO {
                        self.aggregate[j] = count;
                        frontier.push(j);
                        size += 1;
                        if size == AGGREGATE_SIZE { break; }
                    }
                }
            }
            count += 1;
        }
        // Disconnected singletons must not create an unbounded dense bottom.
        if count * 5 >= n * 4 {
            self.aggregate.clear();
            return Ok(None);
        }
        let mut free = vec![Vec3::ZERO; count];
        for i in 0..n {
            free[self.aggregate[i]] = free[self.aggregate[i]].max(self.free[i]);
        }
        // Smooth the finest interpolation once: P = (I - 0.5 D^-1 A) T.
        // Subsequent levels use connected unsmoothed aggregation to prevent
        // repeated stencil expansion and setup/memory blow-up on large grids.
        self.prolong_offsets.clear();
        self.prolong_columns.clear();
        self.prolong_blocks.clear();
        self.prolong_offsets.push(0);
        let mut terms = Vec::new();
        for i in 0..n {
            terms.clear();
            let fi = Mat3::from_diagonal(self.free[i]);
            terms.push((self.aggregate[i], fi.as_dmat3()
                * if smooth_interpolation { 0.5 } else { 1.0 }));
            if smooth_interpolation {
                for edge in self.offsets[i]..self.offsets[i+1] {
                    let j = self.columns[edge];
                    let pj = Mat3::from_diagonal(self.free[j]);
                    terms.push((self.aggregate[j],
                        (fi * self.inverse[i] * self.blocks[edge] * pj).as_dmat3() * -0.5));
                }
            }
            terms.sort_unstable_by_key(|&(j,_)| j);
            let mut cursor = 0;
            while cursor < terms.len() {
                let (j,mut block) = terms[cursor];
                cursor += 1;
                while cursor < terms.len() && terms[cursor].0 == j {
                    block += terms[cursor].1;
                    cursor += 1;
                }
                self.prolong_columns.push(j);
                self.prolong_blocks.push(block.as_mat3());
            }
            self.prolong_offsets.push(self.prolong_columns.len());
        }

        // Galerkin triple product. Accumulate one upper block triangle in f64
        // and reconstruct its transpose, keeping setup memory proportional to
        // the coarse stencil instead of storing O(N * stencil^2) triplets.
        let mut galerkin = rustc_hash::FxHashMap::<(usize,usize),DMat3>::default();
        for i in 0..n {
            terms.clear();
            for p in self.prolong_offsets[i]..self.prolong_offsets[i+1] {
                terms.push((self.prolong_columns[p],
                    self.diagonal[i].as_dmat3() * self.prolong_blocks[p].as_dmat3()));
            }
            for edge in self.offsets[i]..self.offsets[i+1] {
                let j = self.columns[edge];
                let a = self.blocks[edge].as_dmat3();
                for p in self.prolong_offsets[j]..self.prolong_offsets[j+1] {
                    terms.push((self.prolong_columns[p], a * self.prolong_blocks[p].as_dmat3()));
                }
            }
            terms.sort_unstable_by_key(|&(j,_)| j);
            let mut cursor = 0;
            while cursor < terms.len() {
                let (cj,mut ap) = terms[cursor];
                cursor += 1;
                while cursor < terms.len() && terms[cursor].0 == cj {
                    ap += terms[cursor].1;
                    cursor += 1;
                }
                for p in self.prolong_offsets[i]..self.prolong_offsets[i+1] {
                    let ci = self.prolong_columns[p];
                    if ci <= cj {
                        *galerkin.entry((ci,cj)).or_insert(DMat3::ZERO) +=
                            self.prolong_blocks[p].as_dmat3().transpose() * ap;
                    }
                }
            }
        }
        let mut diagonal = vec![Mat3::IDENTITY; count];
        let mut entries = Vec::with_capacity(galerkin.len()*2);
        for ((i,j),block) in galerkin {
            if i == j {
                diagonal[i] = ((block+block.transpose())*0.5).as_mat3()
                    + Mat3::from_diagonal(Vec3::ONE-free[i]);
            } else {
                let block = block.as_mat3();
                entries.push((i,j,block));
                entries.push((j,i,block.transpose()));
            }
        }
        entries.sort_unstable_by_key(|&(i,j,_)| (i,j));
        let mut offsets = vec![0usize; count+1];
        let mut columns = Vec::new();
        let mut blocks = Vec::new();
        for (i,j,block) in entries {
            offsets[i+1] += 1;
            columns.push(j);
            blocks.push(block);
        }
        for i in 0..count { offsets[i+1] += offsets[i]; }
        self.coarse_rhs.resize(count, Vec3::ZERO);
        self.coarse_solution.resize(count, Vec3::ZERO);
        Ok(Some(Level::new(diagonal, free, offsets, columns, blocks)?))
    }
}

impl Multigrid {
    /// Input blocks must form a symmetric positive definite fine operator,
    /// including identity diagonals on constrained coordinates.
    pub(crate) fn build(diagonal: Vec<Mat3>, free: Vec<Vec3>, offsets: Vec<usize>,
        columns: Vec<usize>, blocks: Vec<Mat3>) -> Result<Self, &'static str> {
        let mut levels = vec![Level::new(diagonal, free, offsets, columns, blocks)?];
        let jacobi_smoother = std::env::var("VOXEL_MULTIGRID_SMOOTHER")
            .map(|value| value.eq_ignore_ascii_case("jacobi"))
            .unwrap_or(false);
        let smooth_first_transfer = std::env::var("VOXEL_MG_SMOOTH_FIRST_TRANSFER")
            .map(|value| value != "0")
            .unwrap_or(true);
        loop {
            let smooth_interpolation = levels.len() == 1 && smooth_first_transfer;
            let level = levels.last_mut().unwrap();
            if std::env::var_os("VOXEL_PROFILE_MULTIGRID_SETUP").is_some() {
                eprintln!("multigrid level: rows={} off_diagonal_blocks={} smooth_interpolation={smooth_interpolation}",
                    level.diagonal.len(), level.blocks.len());
            }
            if level.diagonal.len() <= DENSE_BOTTOM_ROWS {
                level.factor_bottom()?;
                break;
            }
            if levels.len() >= MAX_LEVELS { break; }
            match levels.last_mut().unwrap().coarsen(smooth_interpolation)? {
                Some(coarse) => levels.push(coarse),
                None => break,
            }
        }
        Ok(Self { levels, jacobi_smoother })
    }

    pub(crate) fn apply(&mut self, rhs: &[Vec3], out: &mut [Vec3]) {
        Self::cycle(&mut self.levels, rhs, out, self.jacobi_smoother);
    }

    fn cycle(levels: &mut [Level], rhs: &[Vec3], out: &mut [Vec3], jacobi_smoother: bool) {
        let (level, coarser) = levels.split_first_mut().unwrap();
        out.fill(Vec3::ZERO);
        if !level.bottom_factor.is_empty() {
            level.solve_bottom(rhs, out);
            return;
        }
        level.smooth(rhs, out, false, jacobi_smoother);
        if !coarser.is_empty() {
            level.coarse_rhs.fill(Vec3::ZERO);
            for i in 0..rhs.len() {
                let mut residual = rhs[i] - level.diagonal[i] * out[i];
                for edge in level.offsets[i]..level.offsets[i+1] {
                    residual -= level.blocks[edge] * out[level.columns[edge]];
                }
                for p in level.prolong_offsets[i]..level.prolong_offsets[i+1] {
                    level.coarse_rhs[level.prolong_columns[p]] += level.prolong_blocks[p].transpose() * residual;
                }
            }
            Self::cycle(coarser, &level.coarse_rhs, &mut level.coarse_solution, jacobi_smoother);
            for (i,x) in out.iter_mut().enumerate() {
                for p in level.prolong_offsets[i]..level.prolong_offsets[i+1] {
                    *x += level.prolong_blocks[p] * level.coarse_solution[level.prolong_columns[p]];
                }
            }
        }
        level.smooth(rhs, out, true, jacobi_smoother);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice(side: usize) -> Level {
        let n = side.pow(3);
        let k = Mat3::from_cols(Vec3::new(4.0, 0.3, 0.1),
            Vec3::new(0.3, 2.0, 0.2), Vec3::new(0.1, 0.2, 3.0));
        let free: Vec<_> = (0..n).map(|i| {
            if i % side == 0 { Vec3::new(0.0, 1.0, 1.0) } else { Vec3::ONE }
        }).collect();
        let mut diagonal = vec![Mat3::IDENTITY * 0.01; n];
        let mut offsets = vec![0];
        let mut columns = Vec::new();
        let mut blocks = Vec::new();
        for i in 0..n {
            let xyz = [i%side, (i/side)%side, i/(side*side)];
            for (axis,stride) in [1,side,side*side].into_iter().enumerate() {
                for sign in [-1_isize,1] {
                    if (sign < 0 && xyz[axis] == 0) || (sign > 0 && xyz[axis]+1 == side) {
                        continue;
                    }
                    let j = (i as isize + sign * stride as isize) as usize;
                    diagonal[i] += k;
                    columns.push(j);
                    blocks.push(-Mat3::from_diagonal(free[i]) * k * Mat3::from_diagonal(free[j]));
                }
            }
            let p = Mat3::from_diagonal(free[i]);
            diagonal[i] = p * diagonal[i] * p + Mat3::IDENTITY - p;
            offsets.push(columns.len());
        }
        Level::new(diagonal, free, offsets, columns, blocks).unwrap()
    }

    fn apply(level: &Level, x: &[Vec3]) -> Vec<Vec3> {
        (0..x.len()).map(|i| {
            let mut y = level.diagonal[i] * x[i];
            for e in level.offsets[i]..level.offsets[i+1] { y += level.blocks[e] * x[level.columns[e]]; }
            y
        }).collect()
    }

    fn dot(a: &[Vec3], b: &[Vec3]) -> f64 {
        a.iter().zip(b).map(|(a,b)| a.as_dvec3().dot(b.as_dvec3())).sum()
    }

    fn vector(n: usize, phase: f32) -> Vec<Vec3> {
        (0..n).map(|i| Vec3::new((i as f32 * 0.13 + phase).sin(),
            (i as f32 * 0.29 + phase).cos(), (i as f32 * 0.071 + phase).sin())).collect()
    }

    fn hierarchy(side: usize) -> Multigrid {
        let fine = lattice(side);
        Multigrid::build(fine.diagonal, fine.free, fine.offsets, fine.columns, fine.blocks).unwrap()
    }

    #[test]
    fn galerkin_projection_preserves_quadratic_energy_with_partial_constraints() {
        let mut fine = lattice(8);
        let coarse = fine.coarsen(true).unwrap().unwrap();
        let mut c = vector(coarse.diagonal.len(), 0.5);
        for (i,x) in c.iter_mut().enumerate() { *x *= coarse.free[i]; }
        let p: Vec<_> = (0..fine.diagonal.len()).map(|i| {
            (fine.prolong_offsets[i]..fine.prolong_offsets[i+1]).fold(Vec3::ZERO, |sum,p| {
                sum+fine.prolong_blocks[p]*c[fine.prolong_columns[p]]
            })
        }).collect();
        let fine_energy = dot(&p, &apply(&fine, &p));
        let coarse_energy = dot(&c, &apply(&coarse, &c));
        assert!((fine_energy-coarse_energy).abs() / fine_energy.abs() < 2e-6,
            "fine={fine_energy} coarse={coarse_energy}");
    }

    #[test]
    fn v_cycle_is_symmetric_positive_and_keeps_constrained_coordinates_zero() {
        let mut mg = hierarchy(8);
        assert!(mg.levels.len() >= 3);
        let n = mg.levels[0].diagonal.len();
        let mut a = vector(n, 0.2);
        let mut b = vector(n, 1.4);
        for i in 0..n { a[i] *= mg.levels[0].free[i]; b[i] *= mg.levels[0].free[i]; }
        let mut ba = vec![Vec3::ZERO;n];
        let mut bb = ba.clone();
        mg.apply(&a, &mut ba);
        mg.apply(&b, &mut bb);
        let ab = dot(&a, &bb);
        let ba_dot = dot(&b, &ba);
        assert!((ab-ba_dot).abs() < 1e-5 * (ab.abs()+ba_dot.abs()).max(1.0));
        assert!(dot(&a,&ba) > 0.0);
        for i in 0..n { assert_eq!(ba[i] * (Vec3::ONE-mg.levels[0].free[i]), Vec3::ZERO); }
    }

    #[test]
    fn preconditioned_solve_reaches_original_fine_operator_residual() {
        let mut mg = hierarchy(8);
        let n = mg.levels[0].diagonal.len();
        let exact: Vec<_> = vector(n,0.7).into_iter().enumerate()
            .map(|(i,x)| x * mg.levels[0].free[i]).collect();
        let rhs = apply(&mg.levels[0], &exact);
        let mut x = vec![Vec3::ZERO;n];
        let mut r = rhs.clone();
        let mut z = x.clone();
        mg.apply(&r,&mut z);
        let mut p = z.clone();
        let mut rz = dot(&r,&z);
        let norm = dot(&rhs,&rhs).sqrt();
        for _ in 0..120 {
            if dot(&r,&r).sqrt()/norm < 1e-6 { break; }
            let ap = apply(&mg.levels[0],&p);
            let alpha = (rz/dot(&p,&ap)) as f32;
            for i in 0..n { x[i] += alpha*p[i]; r[i] -= alpha*ap[i]; }
            mg.apply(&r,&mut z);
            let next = dot(&r,&z);
            let beta = (next/rz) as f32;
            for i in 0..n { p[i] = z[i]+beta*p[i]; }
            rz = next;
        }
        let actual: Vec<_> = rhs.iter().zip(apply(&mg.levels[0],&x)).map(|(b,ax)| *b-ax).collect();
        let residual = dot(&actual,&actual).sqrt()/norm;
        assert!(residual < 1e-5, "true fine residual={residual}");
    }
}
