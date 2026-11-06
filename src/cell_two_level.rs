//! Parallel operator and two-level preconditioner for the 6x6 cell system.
//!
//! The cell matrix has a translation and a rotation per REV. Serial triangular
//! solves (block IC(0), a complete skyline factor) cannot use many cores and, for
//! bulk bodies, a skyline factor is too wide. This module provides
//!
//! * `CellSystem`: the matrix in row form, applied to a vector in parallel;
//! * `TwoLevel`: a symmetric two-level preconditioner. Smoothing is damped block
//!   Jacobi, which is parallel. The coarse space gives each aggregate of cells
//!   the six rigid-body modes, translation `U` and rotation `theta`, so a cell
//!   moves as `u = U + theta x r`, `phi = theta`. Stiff bulk and rotation of a
//!   whole body, the modes Jacobi damps worst, are solved on a small coarse
//!   system that reuses the skyline factor.
//!
//! The preconditioner is symmetric, so it can drive conjugate gradients, and it
//! never changes the matrix being solved: the residual is always checked against
//! the fine system.
use crate::solver::{
    block6_cholesky, block6_mul_vec, block6_solve_lower, block6_solve_lower_transpose,
    block6_transpose_mul_vec, Block6, CellFactor, Vec6,
};
use glam::DVec3;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Systems at least this large use rayon; smaller ones are faster serially.
const PARALLEL_MIN_CELLS: usize = 1024;
/// Cells per aggregate edge, in lattice cells: aggregates are 3x3x3 boxes.
const AGGREGATE_EDGE: f64 = 3.0;

const ZERO6: Block6 = [[0.0; 6]; 6];

/// Row-form 6x6 block matrix with a stored lower triangle.
pub(crate) struct CellSystem {
    pub(crate) count: usize,
    pub(crate) diagonal: Vec<Block6>,
    /// Lower blocks `H[row][col]`, `row > col`; the upper block is the transpose.
    pub(crate) blocks: Vec<Block6>,
    pub(crate) row_start: Vec<usize>,
    /// For each row entry: neighbour cell, block index, whether to transpose it.
    pub(crate) entries: Vec<(u32, u32, bool)>,
}

impl CellSystem {
    pub(crate) fn new(diagonal: &[Block6], edges: &[(u32, u32, Block6)]) -> Self {
        let count = diagonal.len();
        let mut degree = vec![0usize; count + 1];
        for &(row, col, _) in edges {
            degree[row as usize + 1] += 1;
            degree[col as usize + 1] += 1;
        }
        for i in 0..count {
            degree[i + 1] += degree[i];
        }
        let mut fill = degree.clone();
        let mut entries = vec![(0u32, 0u32, false); edges.len() * 2];
        let mut blocks = Vec::with_capacity(edges.len());
        for (index, &(row, col, block)) in edges.iter().enumerate() {
            blocks.push(block);
            entries[fill[row as usize]] = (col, index as u32, false);
            fill[row as usize] += 1;
            entries[fill[col as usize]] = (row, index as u32, true);
            fill[col as usize] += 1;
        }
        Self { count, diagonal: diagonal.to_vec(), blocks, row_start: degree, entries }
    }

    fn row_apply(&self, i: usize, v: &[Vec6]) -> Vec6 {
        let mut acc = block6_mul_vec(&self.diagonal[i], &v[i]);
        for &(other, block, transposed) in &self.entries[self.row_start[i]..self.row_start[i + 1]] {
            let b = &self.blocks[block as usize];
            let product = if transposed {
                block6_transpose_mul_vec(b, &v[other as usize])
            } else {
                block6_mul_vec(b, &v[other as usize])
            };
            for p in 0..6 {
                acc[p] += product[p];
            }
        }
        acc
    }

    /// `out = A v`.
    pub(crate) fn apply(&self, v: &[Vec6], out: &mut [Vec6]) {
        if self.count >= PARALLEL_MIN_CELLS {
            out.par_iter_mut().enumerate().for_each(|(i, o)| *o = self.row_apply(i, v));
        } else {
            for (i, o) in out.iter_mut().enumerate() {
                *o = self.row_apply(i, v);
            }
        }
    }
}

/// One bond of the cell matrix in factored form. With `S = skew(rest) / 2` the
/// bond strain is `e = u_j - u_i + S (phi_i + phi_j)` and its blocks are
/// `G^T k G` (`G_i = [-I, S]`, `G_j = [I, S]`) plus the face rotation stiffness
/// `stiff (phi_i - phi_j)`. Rigid motions give `e = 0` and equal rotations
/// exactly, so an operator applied in this form keeps them in its null space
/// even in single precision; assembled 6x6 blocks rounded to f32 do not.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellBond {
    pub(crate) i: u32,
    pub(crate) j: u32,
    pub(crate) k: glam::DMat3,
    pub(crate) rest: DVec3,
    pub(crate) stiff: glam::DMat3,
}

impl CellBond {
    /// The diagonal blocks `H_ii`, `H_jj` this bond adds, as in the solver's assembly.
    pub(crate) fn diagonal_blocks(&self) -> (Block6, Block6) {
        let r = self.rest;
        let skew = glam::DMat3::from_cols(
            DVec3::new(0.0, r.z, -r.y),
            DVec3::new(-r.z, 0.0, r.x),
            DVec3::new(r.y, -r.x, 0.0),
        );
        let half = skew * 0.5;
        let k_s = self.k * half;
        let s_k_s = half.transpose() * k_s;
        (
            crate::solver::block6_from(self.k, -k_s, -k_s.transpose(), s_k_s + self.stiff),
            crate::solver::block6_from(self.k, k_s, k_s.transpose(), s_k_s + self.stiff),
        )
    }
}

/// Inner product of two cell vectors.
pub(crate) fn dot(a: &[Vec6], b: &[Vec6]) -> f64 {
    let term = |(x, y): (&Vec6, &Vec6)| (0..6).map(|p| x[p] * y[p]).sum::<f64>();
    if a.len() >= PARALLEL_MIN_CELLS {
        a.par_iter().zip(b.par_iter()).map(term).sum()
    } else {
        a.iter().zip(b).map(term).sum()
    }
}

fn mul6(a: &Block6, b: &Block6) -> Block6 {
    let mut out = ZERO6;
    for i in 0..6 {
        for k in 0..6 {
            let value = a[i][k];
            if value != 0.0 {
                for j in 0..6 {
                    out[i][j] += value * b[k][j];
                }
            }
        }
    }
    out
}

fn transpose6(a: &Block6) -> Block6 {
    let mut out = ZERO6;
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// `P = [[I, -[r]x], [0, I]]` with rows of fixed degrees of freedom removed.
fn prolongation_block(offset: DVec3, mask: &[f64; 6]) -> Block6 {
    let mut p = ZERO6;
    for i in 0..3 {
        p[i][i] = 1.0;
        p[i + 3][i + 3] = 1.0;
    }
    // u = U + theta x r = U - [r]x theta.
    p[0][4] = offset.z;
    p[0][5] = -offset.y;
    p[1][3] = -offset.z;
    p[1][5] = offset.x;
    p[2][3] = offset.y;
    p[2][4] = -offset.x;
    for row in 0..6 {
        if mask[row] == 0.0 {
            p[row] = [0.0; 6];
        }
    }
    p
}

/// One level of the hierarchy. Every level except the last aggregates its cells
/// again; the last is solved exactly with a skyline factor.
pub(crate) struct LevelData {
    pub(crate) masks: Vec<[f64; 6]>,
    /// Damped inverse of each masked diagonal block, `omega * D^-1`.
    pub(crate) smoother: Vec<Block6>,
    pub(crate) transfer: Option<Transfer>,
}

/// Fine-to-coarse map of one level: cell -> aggregate, and the offset of the
/// cell from its aggregate centre, which fixes the rigid-body prolongation.
pub(crate) struct Transfer {
    pub(crate) aggregate: Vec<u32>,
    pub(crate) offset: Vec<DVec3>,
    pub(crate) coarse_count: usize,
    /// Cells of each aggregate, as `members[member_start[a]..member_start[a + 1]]`,
    /// so restriction can run one aggregate per task without shared writes.
    pub(crate) member_start: Vec<usize>,
    pub(crate) members: Vec<u32>,
}

/// Multilevel preconditioner (symmetric V-cycle, rigid-body aggregation).
/// Kept under the name `TwoLevel`: with a small body it has two levels.
pub(crate) struct TwoLevel {
    pub(crate) levels: Vec<LevelData>,
    /// Operators of the coarse levels; level 0 is the caller's system.
    pub(crate) systems: Vec<CellSystem>,
    bottom: CellFactor,
    graph_signature: (usize, usize),
    /// Unique per build, so a resident GPU copy can tell whether it is current.
    id: u64,
}

static NEXT_HIERARCHY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Buffers of one level, reused by every application.
#[derive(Default)]
struct LevelWork {
    residual: Vec<Vec6>,
    product: Vec<Vec6>,
    rhs: Vec<Vec6>,
    solution: Vec<Vec6>,
    coarse_work: Vec<Vec6>,
}

/// Buffers reused by every preconditioner application.
#[derive(Default)]
pub(crate) struct TwoLevelWork {
    levels: Vec<LevelWork>,
}

/// Block Jacobi damping. The smoother is `omega * D^-1`; `omega` stays below
/// 2 / lambda_max(D^-1 A) so the operator stays positive definite. Measured on captured
/// bulk systems (6 November): 0.5 gives 17 / 29 / 32 iterations at 360 / 5 488 / 16 000 cells,
/// 0.8 gives 14 / 23 / 26; 1.0 collapses (57 at 360), so 0.8 keeps a margin below the edge.
pub(crate) const DEFAULT_DAMPING: f64 = 0.8;

/// Stop coarsening when a level has at most this many cells.
const BOTTOM_CELLS: usize = 120;
/// Stop after this many levels in any case.
const MAX_LEVELS: usize = 8;

/// `out = a - b`, in parallel for large vectors.
fn subtract(a: &[Vec6], b: &[Vec6], out: &mut [Vec6]) {
    let row = |((o, x), y): ((&mut Vec6, &Vec6), &Vec6)| {
        for p in 0..6 {
            o[p] = x[p] - y[p];
        }
    };
    if out.len() >= PARALLEL_MIN_CELLS {
        out.par_iter_mut().zip(a.par_iter()).zip(b.par_iter()).for_each(row);
    } else {
        out.iter_mut().zip(a.iter()).zip(b.iter()).for_each(row);
    }
}

fn masked_inverse(diagonal: &[Block6], omega: f64) -> Option<Vec<Block6>> {
    let mut smoother = Vec::with_capacity(diagonal.len());
    for block in diagonal {
        let l = block6_cholesky(block)?;
        let mut inverse = ZERO6;
        for column in 0..6 {
            let mut unit = [0.0; 6];
            unit[column] = omega;
            let solved = block6_solve_lower_transpose(&l, &block6_solve_lower(&l, &unit));
            for row in 0..6 {
                inverse[row][column] = solved[row];
            }
        }
        smoother.push(inverse);
    }
    Some(smoother)
}

/// Galerkin product `P^T A P` for the aggregation `aggregate` with the given
/// offsets. Returns the coarse diagonal, lower edges and which coarse modes
/// have no free fine degree of freedom behind them (their rows become identity).
fn galerkin(
    diagonal: &[Block6],
    edges: &[(u32, u32, Block6)],
    aggregate: &[u32],
    offset: &[DVec3],
    masks: &[[f64; 6]],
    coarse_count: usize,
) -> (Vec<Block6>, Vec<(u32, u32, Block6)>, Vec<[f64; 6]>) {
    let prolong: Vec<Block6> = (0..diagonal.len())
        .map(|i| prolongation_block(offset[i], &masks[i]))
        .collect();
    let mut coarse_diagonal = vec![ZERO6; coarse_count];
    let mut coarse_lower = FxHashMap::<(u32, u32), Block6>::default();
    let add = |target: &mut Block6, value: &Block6| {
        for i in 0..6 {
            for j in 0..6 {
                target[i][j] += value[i][j];
            }
        }
    };
    // The products are independent and dominate the cost, so they run in parallel;
    // the sums into coarse blocks stay serial and in a fixed order.
    let diagonal_product = |i: usize| {
        let p = &prolong[i];
        mul6(&transpose6(p), &mul6(&diagonal[i], p))
    };
    let edge_product = |edge: &(u32, u32, Block6)| {
        mul6(
            &transpose6(&prolong[edge.0 as usize]),
            &mul6(&edge.2, &prolong[edge.1 as usize]),
        )
    };
    let (diagonal_values, edge_values): (Vec<Block6>, Vec<Block6>) =
        if diagonal.len() >= PARALLEL_MIN_CELLS {
            (
                (0..diagonal.len()).into_par_iter().map(diagonal_product).collect(),
                edges.par_iter().map(edge_product).collect(),
            )
        } else {
            (
                (0..diagonal.len()).map(diagonal_product).collect(),
                edges.iter().map(edge_product).collect(),
            )
        };
    for i in 0..diagonal.len() {
        add(&mut coarse_diagonal[aggregate[i] as usize], &diagonal_values[i]);
    }
    for (edge, value) in edges.iter().zip(&edge_values) {
        let (a, b) = (aggregate[edge.0 as usize], aggregate[edge.1 as usize]);
        let value = *value;
        if a == b {
            add(&mut coarse_diagonal[a as usize], &value);
            add(&mut coarse_diagonal[a as usize], &transpose6(&value));
        } else if a > b {
            add(coarse_lower.entry((a, b)).or_insert(ZERO6), &value);
        } else {
            add(coarse_lower.entry((b, a)).or_insert(ZERO6), &transpose6(&value));
        }
    }
    let scale = coarse_diagonal
        .iter()
        .flat_map(|block| (0..6).map(move |i| block[i][i]))
        .fold(0.0f64, f64::max);
    let mut coarse_masks = vec![[1.0f64; 6]; coarse_count];
    for a in 0..coarse_count {
        for p in 0..6 {
            if coarse_diagonal[a][p][p] <= 1.0e-12 * scale {
                coarse_masks[a][p] = 0.0;
            }
        }
    }
    let mut coarse_edges: Vec<(u32, u32, Block6)> =
        coarse_lower.into_iter().map(|((a, b), block)| (a, b, block)).collect();
    coarse_edges.sort_by_key(|edge| (edge.0, edge.1));
    for a in 0..coarse_count {
        for p in 0..6 {
            if coarse_masks[a][p] == 0.0 {
                for q in 0..6 {
                    coarse_diagonal[a][p][q] = 0.0;
                    coarse_diagonal[a][q][p] = 0.0;
                }
                coarse_diagonal[a][p][p] = 1.0;
            }
        }
    }
    for (row, col, block) in &mut coarse_edges {
        for p in 0..6 {
            for q in 0..6 {
                if coarse_masks[*row as usize][p] == 0.0 || coarse_masks[*col as usize][q] == 0.0 {
                    block[p][q] = 0.0;
                }
            }
        }
    }
    (coarse_diagonal, coarse_edges, coarse_masks)
}

impl TwoLevel {
    /// Build the hierarchy from cell geometry. `positions` are current cell
    /// centres, `lattice` the rest positions in metres, `bodies` a connectivity
    /// label per cell; aggregates never mix two bodies.
    pub(crate) fn build(
        system: &CellSystem,
        edges: &[(u32, u32, Block6)],
        positions: &[DVec3],
        lattice: &[DVec3],
        bodies: &[u32],
        masks: &[[f64; 6]],
        cell_m: f64,
        omega: f64,
    ) -> Option<Self> {
        let mut levels = Vec::<LevelData>::new();
        let mut systems = Vec::<CellSystem>::new();
        let mut diagonal = system.diagonal.clone();
        let mut level_edges = edges.to_vec();
        let mut position = positions.to_vec();
        let mut lat = lattice.to_vec();
        let mut body = bodies.to_vec();
        let mut level_masks = masks.to_vec();
        let mut edge_m = cell_m;
        loop {
            let count = diagonal.len();
            let smoother = masked_inverse(&diagonal, omega)?;
            let last = count <= BOTTOM_CELLS || levels.len() + 1 >= MAX_LEVELS;
            let mut next = None;
            if !last {
                let box_edge = AGGREGATE_EDGE * edge_m;
                let mut keys = FxHashMap::<(u32, [i64; 3]), u32>::default();
                let mut aggregate = vec![0u32; count];
                for i in 0..count {
                    let key = (
                        body[i],
                        [
                            (lat[i].x / box_edge).floor() as i64,
                            (lat[i].y / box_edge).floor() as i64,
                            (lat[i].z / box_edge).floor() as i64,
                        ],
                    );
                    let id = keys.len() as u32;
                    aggregate[i] = *keys.entry(key).or_insert(id);
                }
                let coarse_count = keys.len();
                // Stalled coarsening: solve this level exactly instead.
                if coarse_count * 4 > count * 3 {
                    // fall through with next = None
                } else {
                    let mut center = vec![DVec3::ZERO; coarse_count];
                    let mut lattice_center = vec![DVec3::ZERO; coarse_count];
                    let mut coarse_body = vec![0u32; coarse_count];
                    let mut members = vec![0.0f64; coarse_count];
                    for i in 0..count {
                        let a = aggregate[i] as usize;
                        center[a] += position[i];
                        lattice_center[a] += lat[i];
                        coarse_body[a] = body[i];
                        members[a] += 1.0;
                    }
                    for a in 0..coarse_count {
                        center[a] /= members[a];
                        lattice_center[a] /= members[a];
                    }
                    let offset: Vec<DVec3> =
                        (0..count).map(|i| position[i] - center[aggregate[i] as usize]).collect();
                    let (coarse_diagonal, coarse_edges, coarse_masks) =
                        galerkin(&diagonal, &level_edges, &aggregate, &offset, &level_masks, coarse_count);
                    let mut member_start = vec![0usize; coarse_count + 1];
                    for &a in &aggregate {
                        member_start[a as usize + 1] += 1;
                    }
                    for a in 0..coarse_count {
                        member_start[a + 1] += member_start[a];
                    }
                    let mut fill = member_start.clone();
                    let mut members = vec![0u32; count];
                    for i in 0..count {
                        let a = aggregate[i] as usize;
                        members[fill[a]] = i as u32;
                        fill[a] += 1;
                    }
                    next = Some((
                        Transfer { aggregate, offset, coarse_count, member_start, members },
                        coarse_diagonal,
                        coarse_edges,
                        coarse_masks,
                        center,
                        lattice_center,
                        coarse_body,
                    ));
                }
            }
            match next {
                None => {
                    let bottom = CellFactor::build(count, &diagonal, &level_edges, &level_masks)?;
                    levels.push(LevelData { masks: level_masks, smoother, transfer: None });
                    return Some(Self {
                        levels,
                        systems,
                        bottom,
                        graph_signature: (system.count, edges.len()),
                        id: NEXT_HIERARCHY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    });
                }
                Some((transfer, coarse_diagonal, coarse_edges, coarse_masks, center, lattice_center, coarse_body)) => {
                    levels.push(LevelData { masks: level_masks, smoother, transfer: Some(transfer) });
                    systems.push(CellSystem::new(&coarse_diagonal, &coarse_edges));
                    diagonal = coarse_diagonal;
                    level_edges = coarse_edges;
                    position = center;
                    lat = lattice_center;
                    body = coarse_body;
                    level_masks = coarse_masks;
                    edge_m *= AGGREGATE_EDGE;
                }
            }
        }
    }

    /// A built preconditioner stays valid while the graph and the set of free
    /// degrees of freedom are unchanged. A stale mask zeroes the correction of a
    /// degree of freedom that has become free, which makes the preconditioner
    /// singular there and stalls conjugate gradients.
    pub(crate) fn matches(&self, count: usize, edge_count: usize, masks: &[[f64; 6]]) -> bool {
        self.graph_signature == (count, edge_count) && self.levels[0].masks == masks
    }

    /// Identity of this build; a rebuilt hierarchy gets a new one.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The bottom solve as a dense matrix, row-major, `dofs x dofs` with
    /// `dofs = 6 * cells` of the last level: column `j` is the masked skyline solve
    /// of the unit vector `e_j`, so `z = B r` is exactly the CPU bottom step.
    /// Its size is quadratic in `bottom_dofs()`; callers check that first.
    pub(crate) fn bottom_dense(&self) -> (usize, Vec<f64>) {
        let masks = &self.levels.last().expect("bottom level").masks;
        let cells = masks.len();
        let dofs = cells * 6;
        let mut dense = vec![0.0f64; dofs * dofs];
        let columns: Vec<Vec<f64>> = (0..dofs)
            .into_par_iter()
            .map(|j| {
                let mut unit = vec![[0.0f64; 6]; cells];
                unit[j / 6][j % 6] = 1.0;
                let mut z = vec![[0.0f64; 6]; cells];
                let mut work = Vec::new();
                self.bottom.solve(&unit, &mut z, &mut work);
                (0..dofs).map(|row| z[row / 6][row % 6] * masks[row / 6][row % 6]).collect()
            })
            .collect();
        for (j, column) in columns.iter().enumerate() {
            for (row, value) in column.iter().enumerate() {
                dense[row * dofs + j] = *value;
            }
        }
        (dofs, dense)
    }

    /// Degrees of freedom of the bottom level.
    pub(crate) fn bottom_dofs(&self) -> usize {
        self.levels.last().map_or(0, |level| level.masks.len() * 6)
    }

    fn smooth(level: &LevelData, residual: &[Vec6], z: &mut [Vec6], add: bool) {
        let apply_row = |i: usize, out: &mut Vec6| {
            let value = block6_mul_vec(&level.smoother[i], &residual[i]);
            for p in 0..6 {
                let masked = value[p] * level.masks[i][p];
                if add {
                    out[p] += masked;
                } else {
                    out[p] = masked;
                }
            }
        };
        if z.len() >= PARALLEL_MIN_CELLS {
            z.par_iter_mut().enumerate().for_each(|(i, out)| apply_row(i, out));
        } else {
            for (i, out) in z.iter_mut().enumerate() {
                apply_row(i, out);
            }
        }
    }

    fn operator<'a>(&'a self, fine: &'a CellSystem, level: usize) -> &'a CellSystem {
        if level == 0 {
            fine
        } else {
            &self.systems[level - 1]
        }
    }

    fn cycle(
        &self,
        level: usize,
        fine: &CellSystem,
        r: &[Vec6],
        z: &mut [Vec6],
        works: &mut [LevelWork],
    ) {
        let data = &self.levels[level];
        let Some(transfer) = &data.transfer else {
            let (work, _) = works.split_first_mut().expect("bottom work");
            let started = std::time::Instant::now();
            self.bottom.solve(r, z, &mut work.coarse_work);
            crate::solver::CELL_PROFILE[7].fetch_add(started.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            for (value, mask) in z.iter_mut().zip(&data.masks) {
                for p in 0..6 {
                    value[p] *= mask[p];
                }
            }
            return;
        };
        let count = r.len();
        let parallel = count >= PARALLEL_MIN_CELLS;
        let (work, rest) = works.split_first_mut().expect("level work");
        let system = self.operator(fine, level);
        work.residual.resize(count, [0.0; 6]);
        work.product.resize(count, [0.0; 6]);
        Self::smooth(data, r, z, false);
        system.apply(z, &mut work.product);
        subtract(r, &work.product, &mut work.residual);
        // Restriction: F += f, M += r x f + m, one aggregate per task.
        work.rhs.clear();
        work.rhs.resize(transfer.coarse_count, [0.0; 6]);
        let coarse_masks = &self.levels[level + 1].masks;
        let residual = &work.residual;
        let restrict = |a: usize, target: &mut Vec6| {
            let mut force = DVec3::ZERO;
            let mut moment = DVec3::ZERO;
            for &i in &transfer.members[transfer.member_start[a]..transfer.member_start[a + 1]] {
                let i = i as usize;
                let res = &residual[i];
                let masked = DVec3::new(
                    res[0] * data.masks[i][0],
                    res[1] * data.masks[i][1],
                    res[2] * data.masks[i][2],
                );
                force += masked;
                moment += transfer.offset[i].cross(masked) + DVec3::new(res[3], res[4], res[5]);
            }
            let value = [force.x, force.y, force.z, moment.x, moment.y, moment.z];
            for p in 0..6 {
                target[p] = value[p] * coarse_masks[a][p];
            }
        };
        if parallel {
            work.rhs.par_iter_mut().enumerate().for_each(|(a, target)| restrict(a, target));
        } else {
            for (a, target) in work.rhs.iter_mut().enumerate() {
                restrict(a, target);
            }
        }
        work.solution.resize(transfer.coarse_count, [0.0; 6]);
        let coarse_rhs = std::mem::take(&mut work.rhs);
        let mut coarse_solution = std::mem::take(&mut work.solution);
        self.cycle(level + 1, fine, &coarse_rhs, &mut coarse_solution, rest);
        // Prolongation: u = U + theta x r, phi = theta.
        let prolong = |i: usize, out: &mut Vec6| {
            let s = &coarse_solution[transfer.aggregate[i] as usize];
            let theta = DVec3::new(s[3], s[4], s[5]);
            let shift = theta.cross(transfer.offset[i]);
            out[0] += (s[0] + shift.x) * data.masks[i][0];
            out[1] += (s[1] + shift.y) * data.masks[i][1];
            out[2] += (s[2] + shift.z) * data.masks[i][2];
            out[3] += s[3];
            out[4] += s[4];
            out[5] += s[5];
        };
        if parallel {
            z.par_iter_mut().enumerate().for_each(|(i, out)| prolong(i, out));
        } else {
            for (i, out) in z.iter_mut().enumerate() {
                prolong(i, out);
            }
        }
        work.rhs = coarse_rhs;
        work.solution = coarse_solution;
        system.apply(z, &mut work.product);
        subtract(r, &work.product, &mut work.residual);
        Self::smooth(data, &work.residual, z, true);
    }

    /// `z = M^-1 r`: smooth, correct on the coarser levels, smooth again.
    pub(crate) fn apply(
        &self,
        system: &CellSystem,
        r: &[Vec6],
        z: &mut [Vec6],
        work: &mut TwoLevelWork,
    ) {
        work.levels.resize_with(self.levels.len(), LevelWork::default);
        self.cycle(0, system, r, z, &mut work.levels);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A chain of `n` cells with SPD coupling, for checking the operator algebra.
    fn chain(n: usize) -> (Vec<Block6>, Vec<(u32, u32, Block6)>) {
        let mut diagonal = vec![ZERO6; n];
        let mut edges = Vec::new();
        for i in 0..n {
            for p in 0..6 {
                diagonal[i][p][p] = 4.0;
            }
        }
        for i in 1..n {
            let mut block = ZERO6;
            for p in 0..6 {
                block[p][p] = -1.0;
            }
            block[0][4] = 0.2;
            edges.push((i as u32, (i - 1) as u32, block));
        }
        (diagonal, edges)
    }

    #[test]
    fn apply_equals_dense_product() {
        let n = 7;
        let (diagonal, edges) = chain(n);
        let system = CellSystem::new(&diagonal, &edges);
        let v: Vec<Vec6> = (0..n).map(|i| std::array::from_fn(|p| (i * 6 + p) as f64 * 0.1 - 1.0)).collect();
        let mut out = vec![[0.0; 6]; n];
        system.apply(&v, &mut out);
        let mut expected = vec![[0.0; 6]; n];
        for i in 0..n {
            expected[i] = block6_mul_vec(&diagonal[i], &v[i]);
        }
        for (row, col, block) in &edges {
            let (row, col) = (*row as usize, *col as usize);
            let a = block6_mul_vec(block, &v[col]);
            let b = block6_transpose_mul_vec(block, &v[row]);
            for p in 0..6 {
                expected[row][p] += a[p];
                expected[col][p] += b[p];
            }
        }
        for i in 0..n {
            for p in 0..6 {
                assert!((out[i][p] - expected[i][p]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn two_level_preconditioner_is_symmetric_and_positive() {
        let n = 90;
        let (diagonal, edges) = chain(n);
        let system = CellSystem::new(&diagonal, &edges);
        let cell = 0.01;
        let positions: Vec<DVec3> = (0..n).map(|i| DVec3::new(i as f64 * cell, 0.0, 0.0)).collect();
        let masks = vec![[1.0; 6]; n];
        let bodies = vec![0u32; n];
        let two = TwoLevel::build(
            &system, &edges, &positions, &positions, &bodies, &masks, cell, DEFAULT_DAMPING,
        )
        .expect("coarse factor");
        let mut work = TwoLevelWork::default();
        let u: Vec<Vec6> = (0..n).map(|i| std::array::from_fn(|p| ((i + 3 * p) as f64).sin())).collect();
        let w: Vec<Vec6> = (0..n).map(|i| std::array::from_fn(|p| ((2 * i + p) as f64).cos())).collect();
        let (mut mu, mut mw) = (vec![[0.0; 6]; n], vec![[0.0; 6]; n]);
        two.apply(&system, &u, &mut mu, &mut work);
        two.apply(&system, &w, &mut mw, &mut work);
        let (a, b) = (dot(&w, &mu), dot(&u, &mw));
        assert!((a - b).abs() <= 1.0e-9 * a.abs().max(1.0), "not symmetric: {a} vs {b}");
        assert!(dot(&u, &mu) > 0.0 && dot(&w, &mw) > 0.0);
    }

    pub(crate) fn pcg(
        system: &CellSystem,
        rhs: &[Vec6],
        masks: &[[f64; 6]],
        mut precondition: impl FnMut(&[Vec6], &mut [Vec6]),
        limit: usize,
    ) -> (usize, f64) {
        let count = rhs.len();
        let norms = |v: &[Vec6]| {
            let f = v.iter().map(|x| x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sum::<f64>().sqrt();
            let m = v.iter().map(|x| x[3] * x[3] + x[4] * x[4] + x[5] * x[5]).sum::<f64>().sqrt();
            (f, m)
        };
        let (f0, m0) = norms(rhs);
        let mut x = vec![[0.0; 6]; count];
        let mut r = rhs.to_vec();
        let mut z = vec![[0.0; 6]; count];
        precondition(&r, &mut z);
        let mut p = z.clone();
        let mut ap = vec![[0.0; 6]; count];
        let mut rz = dot(&r, &z);
        let mut iterations = 0;
        loop {
            let (f, m) = norms(&r);
            if (f <= 1.0e-5 * f0 && m <= 1.0e-5 * m0) || iterations >= limit {
                return (iterations, f / f0.max(1e-300));
            }
            system.apply(&p, &mut ap);
            let denominator = dot(&p, &ap);
            iterations += 1;
            if !(denominator > 0.0) {
                return (usize::MAX, f64::NAN);
            }
            let alpha = rz / denominator;
            for i in 0..count {
                for t in 0..6 {
                    x[i][t] += alpha * p[i][t];
                    r[i][t] -= alpha * ap[i][t];
                }
            }
            precondition(&r, &mut z);
            for i in 0..count {
                for t in 0..6 {
                    z[i][t] *= masks[i][t];
                }
            }
            let next = dot(&r, &z);
            let beta = next / rz;
            rz = next;
            for i in 0..count {
                for t in 0..6 {
                    p[i][t] = z[i][t] + beta * p[i][t];
                }
            }
        }
    }

    /// The systems of one step of a bulk iron block tipping on a fixed floor, after
    /// `warmup` steps so friction patches are established.
    pub(crate) fn captured_bulk(width: i32, height: i32, warmup: usize) -> Vec<crate::solver::CapturedCellSystem> {
        use crate::material::Material;
        use crate::scene::{Scene, SceneConfig, VoxelSpawn};
        let iron = Material::component_id("iron").unwrap();
        let stone = Material::component_id("stone").unwrap();
        let mut config = SceneConfig::default();
        config.fixed_dt = 0.01;
        config.substeps = 4;
        config.gravity = [1.2 * 9.81, 0.0, -9.81];
        config.atmosphere.pressure_pa = 0.0;
        let mut spawns = Vec::new();
        let half = width / 2 + 8;
        for x in -half..half {
            for y in -half..half {
                spawns.push(VoxelSpawn::at_cell([x, y, 0], stone).fixed());
            }
        }
        let lo = -(width / 2);
        for z in 1..=height {
            for x in lo..lo + width {
                for y in lo..lo + width {
                    spawns.push(VoxelSpawn::at_cell([x, y, z], iron));
                }
            }
        }
        let mut scene = Scene::from_voxels(config, spawns).unwrap();
        crate::solver::FORCE_CELL_DOFS.with(|force| force.set(Some(true)));
        for _ in 0..warmup {
            scene.tick();
        }
        crate::solver::CAPTURED_CELL_SYSTEM.with(|slot| {
            *slot.borrow_mut() = Some(crate::solver::CellCapture { limit: 60, systems: Vec::new() });
        });
        scene.tick();
        crate::solver::CAPTURED_CELL_SYSTEM
            .with(|slot| slot.borrow_mut().take().unwrap().systems)
    }

    #[test]
    fn two_level_beats_jacobi_on_a_real_bulk_system() {
        check_bulk_sizes(&[(6, 10, 6)]);
    }

    #[test]
    #[ignore = "timing probe on 5 488 and 16 000 cells, about 25 s"]
    fn preconditioner_scales_to_large_bulk_systems() {
        check_bulk_sizes(&[(14, 28, 4), (20, 40, 4)]);
    }

    fn test_damping() -> f64 {
        std::env::var("VOXEL_TL_DAMPING").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_DAMPING)
    }

    fn check_bulk_sizes(sizes: &[(i32, i32, usize)]) {
        for &(width, height, warmup) in sizes {
            let systems = captured_bulk(width, height, warmup);
            assert!(!systems.is_empty(), "no system captured");
            let mut jacobi_counts = Vec::new();
            let mut two_counts = Vec::new();
            let mut masked = Vec::new();
            for captured in &systems {
                let system = CellSystem::new(&captured.diagonal, &captured.edges);
                let count = system.count;
                let masks = &captured.masks;
                masked.push(masks.iter().flatten().filter(|m| **m == 0.0).count());
                let diagonal_inverse: Vec<Block6> = captured
                    .diagonal
                    .iter()
                    .map(|d| {
                        let l = block6_cholesky(d).unwrap();
                        let mut inverse = ZERO6;
                        for c in 0..6 {
                            let mut unit = [0.0; 6];
                            unit[c] = 1.0;
                            let s = block6_solve_lower_transpose(&l, &block6_solve_lower(&l, &unit));
                            for r in 0..6 {
                                inverse[r][c] = s[r];
                            }
                        }
                        inverse
                    })
                    .collect();
                let (iterations, _) = pcg(
                    &system,
                    &captured.rhs,
                    masks,
                    |r, z| {
                        for i in 0..count {
                            z[i] = block6_mul_vec(&diagonal_inverse[i], &r[i]);
                        }
                    },
                    3000,
                );
                jacobi_counts.push(iterations);
                let two = TwoLevel::build(
                    &system,
                    &captured.edges,
                    &captured.positions,
                    &captured.lattice,
                    &captured.bodies,
                    masks,
                    0.01,
                    test_damping(),
                )
                .expect("coarse factor");
                let mut work = TwoLevelWork::default();
                let (iterations, _) = pcg(
                    &system,
                    &captured.rhs,
                    masks,
                    |r, z| two.apply(&system, r, z, &mut work),
                    3000,
                );
                two_counts.push(iterations);
            }
            eprintln!(
                "PRECOND {width}x{width}x{height} solves={} masked_dofs={:?}",
                systems.len(),
                masked
            );
            eprintln!("  jacobi   {jacobi_counts:?}");
            eprintln!("  twolevel {two_counts:?}");
            for (two, jacobi) in two_counts.iter().zip(&jacobi_counts) {
                assert!(*two <= 40, "{width}x{height}: {two} iterations");
                assert!(*two * 2 < *jacobi, "{width}x{height}: {two} against Jacobi {jacobi}");
            }
            // Cost of the pieces of one CG iteration on the last system.
            if let Some(captured) = systems.last() {
                let system = CellSystem::new(&captured.diagonal, &captured.edges);
                let count = system.count;
                let v: Vec<Vec6> = (0..count).map(|i| std::array::from_fn(|p| ((i + p) as f64).sin())).collect();
                let mut out = vec![[0.0; 6]; count];
                let time = |label: &str, reps: usize, mut f: Box<dyn FnMut() + '_>| {
                    let start = std::time::Instant::now();
                    for _ in 0..reps {
                        f();
                    }
                    eprintln!("  TIMING {label}: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0 / reps as f64);
                };
                time("spmv", 20, Box::new(|| system.apply(&v, &mut out)));
                let started = std::time::Instant::now();
                let two = TwoLevel::build(
                    &system, &captured.edges, &captured.positions, &captured.lattice,
                    &captured.bodies, &captured.masks, 0.01, test_damping(),
                ).unwrap();
                eprintln!("  TIMING build: {:.1} ms levels={}", started.elapsed().as_secs_f64() * 1000.0, two.levels.len());
                let mut work = TwoLevelWork::default();
                let mut z = vec![[0.0; 6]; count];
                time("vcycle", 10, Box::new(|| two.apply(&system, &v, &mut z, &mut work)));
                time("dot", 50, Box::new(|| { std::hint::black_box(dot(&v, &v)); }));
            }
        }
    }
}
