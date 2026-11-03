//! Exact block-banded LDLᵀ factorization for small mechanical systems.
use glam::{DMat3, DVec3, Mat3, Vec3};

const MAX_BLOCKS: usize = 2_000_000;

/// A unit-lower block-banded L and block-diagonal D, with `A = L D Lᵀ`.
pub(crate) struct BandedFactor {
    n: usize,
    bandwidth: usize,
    lower: Vec<DMat3>,
    inverse_diagonal: Vec<DMat3>,
}

impl BandedFactor {
    pub(crate) fn factor(
        diagonal: &[Mat3],
        lower_entries: &[(usize, usize, Mat3)],
        max_bandwidth: usize,
    ) -> Result<Self, &'static str> {
        let n = diagonal.len();
        let mut bandwidth = 0usize;
        for &(row, col, _) in lower_entries {
            if row >= n || col >= n || row <= col {
                return Err("invalid lower block index");
            }
            bandwidth = bandwidth.max(row - col);
        }
        if bandwidth > max_bandwidth {
            return Err("matrix bandwidth exceeds limit");
        }
        let stride = bandwidth.checked_add(1).ok_or("matrix too large")?;
        let blocks = n.checked_mul(stride).ok_or("matrix too large")?;
        if blocks > MAX_BLOCKS {
            return Err("matrix exceeds block storage limit");
        }
        let mut lower = vec![DMat3::ZERO; blocks];
        let idx = |row: usize, col: usize| row * stride + (row - col);
        for &(row, col, value) in lower_entries {
            if !finite_mat(value) {
                return Err("matrix contains non-finite values");
            }
            lower[idx(row, col)] += to_dmat(value);
        }
        let mut d = vec![DMat3::ZERO; n];
        let mut inv_d = vec![DMat3::ZERO; n];

        for i in 0..n {
            let first = i.saturating_sub(bandwidth);
            for j in first..i {
                let mut s = if i - j == 0 {
                    DMat3::ZERO
                } else {
                    lower[idx(i, j)]
                };
                // Input A is represented in lower; subtract all prior common factors.
                for k in i.saturating_sub(bandwidth)..j {
                    if i - k <= bandwidth && j - k <= bandwidth {
                        s -= lower[idx(i, k)] * d[k] * lower[idx(j, k)].transpose();
                    }
                }
                let lij = s * inv_d[j];
                if !finite_dmat(lij) {
                    return Err("factorization produced non-finite values");
                }
                lower[idx(i, j)] = lij;
            }

            let mut pivot = to_dmat(diagonal[i]);
            for k in first..i {
                let lik = lower[idx(i, k)];
                pivot -= lik * d[k] * lik.transpose();
            }
            pivot = (pivot + pivot.transpose()) * 0.5;
            if !finite_dmat(pivot) || !is_spd(pivot) {
                return Err("matrix is not positive definite");
            }
            let inverse = pivot.inverse();
            if !finite_dmat(inverse) {
                return Err("matrix is not positive definite");
            }
            d[i] = pivot;
            inv_d[i] = inverse;
        }
        Ok(Self {
            n,
            bandwidth,
            lower,
            inverse_diagonal: inv_d,
        })
    }

    pub(crate) fn solve(
        &self,
        rhs: &[Vec3],
        solution: &mut Vec<DVec3>,
    ) -> Result<(), &'static str> {
        if rhs.len() != self.n {
            return Err("right-hand side size mismatch");
        }
        if rhs.iter().any(|v| !v.is_finite()) {
            return Err("right-hand side contains non-finite values");
        }
        solution.resize(self.n, DVec3::ZERO);
        let stride = self.bandwidth + 1;
        for i in 0..self.n {
            let mut y = rhs[i].as_dvec3();
            for j in i.saturating_sub(self.bandwidth)..i {
                y -= self.lower[i * stride + (i - j)] * solution[j];
            }
            solution[i] = y;
        }
        for i in 0..self.n {
            solution[i] = self.inverse_diagonal[i] * solution[i];
        }
        for i in (0..self.n).rev() {
            let mut x = solution[i];
            let end = (i + self.bandwidth + 1).min(self.n);
            for row in i + 1..end {
                x -= self.lower[row * stride + (row - i)].transpose() * solution[row];
            }
            solution[i] = x;
        }
        if solution.iter().any(|v| !v.is_finite()) {
            return Err("solve produced non-finite values");
        }
        Ok(())
    }
}

fn finite_mat(m: Mat3) -> bool {
    m.to_cols_array().iter().all(|x| x.is_finite())
}
fn to_dmat(m: Mat3) -> DMat3 {
    DMat3::from_cols(
        m.x_axis.as_dvec3(),
        m.y_axis.as_dvec3(),
        m.z_axis.as_dvec3(),
    )
}
fn finite_dmat(m: DMat3) -> bool {
    m.to_cols_array().iter().all(|x| x.is_finite())
}

fn is_spd(m: DMat3) -> bool {
    let a = m.x_axis.x;
    let minor2 = m.x_axis.x * m.y_axis.y - m.y_axis.x * m.x_axis.y;
    a > 0.0 && minor2 > 0.0 && m.determinant() > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(a: [[f32; 3]; 3]) -> Mat3 {
        Mat3::from_cols_array(&[
            a[0][0], a[1][0], a[2][0], a[0][1], a[1][1], a[2][1], a[0][2], a[1][2], a[2][2],
        ])
    }

    fn residual(diag: &[Mat3], edges: &[(usize, usize, Mat3)], x: &[DVec3], b: &[Vec3]) -> f64 {
        let mut ax: Vec<DVec3> = diag.iter().zip(x).map(|(a, v)| to_dmat(*a) * *v).collect();
        for &(r, c, a) in edges {
            let a = to_dmat(a);
            ax[r] += a * x[c];
            ax[c] += a.transpose() * x[r];
        }
        ax.iter()
            .zip(b)
            .map(|(a, v)| (*a - v.as_dvec3()).length())
            .fold(0.0, f64::max)
    }

    #[test]
    fn anisotropic_banded_residual_and_resizes_output() {
        let diag = vec![
            mat([[8.0, 0.4, 0.2], [0.4, 7.0, 0.3], [0.2, 0.3, 6.0]]),
            mat([[9.0, 0.2, 0.1], [0.2, 8.0, 0.4], [0.1, 0.4, 7.0]]),
            mat([[7.0, 0.3, 0.1], [0.3, 9.0, 0.2], [0.1, 0.2, 8.0]]),
        ];
        let edges = vec![
            (
                1,
                0,
                mat([[-0.8, 0.1, 0.0], [0.2, -0.6, 0.1], [0.0, 0.1, -0.7]]),
            ),
            (
                2,
                1,
                mat([[-0.5, 0.0, 0.1], [0.1, -0.7, 0.0], [0.0, 0.2, -0.6]]),
            ),
        ];
        let b = vec![
            Vec3::new(1.0, -2.0, 0.5),
            Vec3::new(0.3, 1.2, -0.4),
            Vec3::new(-1.0, 0.2, 2.0),
        ];
        let f = BandedFactor::factor(&diag, &edges, 2).unwrap();
        let mut x = vec![DVec3::splat(99.0); 8];
        f.solve(&b, &mut x).unwrap();
        assert_eq!(x.len(), b.len());
        assert!(residual(&diag, &edges, &x, &b) < 1e-10);
    }

    #[test]
    fn one_row_and_disconnected_blocks() {
        let d = vec![Mat3::from_diagonal(Vec3::new(2.0, 3.0, 4.0))];
        let b = vec![Vec3::new(2.0, 6.0, 12.0)];
        let f = BandedFactor::factor(&d, &[], 0).unwrap();
        let mut x = Vec::new();
        f.solve(&b, &mut x).unwrap();
        assert!(residual(&d, &[], &x, &b) < 1e-12);

        let d = vec![Mat3::IDENTITY * 2.0, Mat3::IDENTITY * 3.0];
        let b = vec![Vec3::ONE, Vec3::splat(2.0)];
        let f = BandedFactor::factor(&d, &[], 4).unwrap();
        f.solve(&b, &mut x).unwrap();
        assert!(residual(&d, &[], &x, &b) < 1e-12);
    }

    #[test]
    fn factor_reuse_handles_new_rhs_and_refactor_handles_new_matrix() {
        let diagonal = vec![
            mat([[5.0, 0.2, 0.1], [0.2, 4.0, 0.3], [0.1, 0.3, 6.0]]),
            mat([[7.0, 0.1, 0.0], [0.1, 5.0, 0.2], [0.0, 0.2, 4.0]]),
        ];
        let lower = vec![(
            1,
            0,
            mat([[-0.6, 0.15, 0.05], [0.1, -0.7, 0.0], [0.0, 0.2, -0.5]]),
        )];
        let first_rhs = vec![Vec3::new(1.0, 0.5, -2.0), Vec3::new(-1.0, 2.0, 0.25)];
        let second_rhs = vec![Vec3::new(-0.3, 1.4, 0.8), Vec3::new(0.7, -0.5, 1.1)];
        let old_factor = BandedFactor::factor(&diagonal, &lower, 1).unwrap();
        let mut solution = Vec::new();
        old_factor.solve(&first_rhs, &mut solution).unwrap();
        assert!(residual(&diagonal, &lower, &solution, &first_rhs) < 1e-10);
        old_factor.solve(&second_rhs, &mut solution).unwrap();
        assert!(residual(&diagonal, &lower, &solution, &second_rhs) < 1e-10);

        let changed_diagonal = vec![diagonal[0] * 1.4, diagonal[1] * 0.8];
        let changed_lower = vec![(1, 0, lower[0].2 * 0.7)];
        let new_factor = BandedFactor::factor(&changed_diagonal, &changed_lower, 1).unwrap();
        new_factor.solve(&second_rhs, &mut solution).unwrap();
        assert!(residual(&changed_diagonal, &changed_lower, &solution, &second_rhs) < 1e-10);
    }

    #[test]
    fn rejects_invalid_matrices_and_sizes() {
        assert!(BandedFactor::factor(&[Mat3::ZERO], &[], 0).is_err());
        assert!(BandedFactor::factor(&[Mat3::IDENTITY; 3], &[(2, 0, Mat3::ZERO)], 1).is_err());
        assert!(BandedFactor::factor(&[Mat3::IDENTITY], &[], 0)
            .unwrap()
            .solve(&[], &mut Vec::new())
            .is_err());
        let huge = vec![Mat3::IDENTITY; 1_000_001];
        assert!(BandedFactor::factor(&huge, &[(1_000_000, 999_999, Mat3::ZERO)], 1).is_err());
    }
}
