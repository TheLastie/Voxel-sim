//! Reduced slender-body model (tier T2): a chain of linear Timoshenko beam elements.
//!
//! A trunk, branch, limb or bridge made of many REVs behaves, while its strains are small,
//! like a beam whose section properties follow from the REVs it replaces. This module
//! solves that beam implicitly (Newmark average acceleration, energy conserving for the
//! linear system) with a block tridiagonal factorisation, so a chain of `n` nodes costs
//! O(n) per step instead of the 0.3-0.6 ms per cell of the fine solver.
//!
//! It is only valid for small rotations and strains: `max_stress` tells the governor when
//! a region must open into cells (a crack or yield starts) and a falling body is handed to
//! the rigid tier. A beam never invents energy: the unit tests keep kinetic plus strain
//! energy constant in free vibration and compare static and dynamic results with the
//! analytic beam solution and with the fine cell solver.
use glam::DVec3;

pub type Dof6 = [f64; 6];
type Block = [[f64; 6]; 6];

const ZERO: Block = [[0.0; 6]; 6];

/// Section properties in the element's local axes: `x` along the beam, `y` and `z` across.
#[derive(Clone, Copy, Debug)]
pub struct Section {
    /// Area, m^2.
    pub area: f64,
    /// Second moment of area about local y (bending in the xz plane), m^4.
    pub iy: f64,
    /// Second moment of area about local z (bending in the xy plane), m^4.
    pub iz: f64,
    /// Torsion constant, m^4.
    pub torsion: f64,
    /// Shear correction factor (5/6 for a rectangle).
    pub shear_factor: f64,
    /// Largest distance of the section from its neutral axis in local y and z, m.
    pub extent_y: f64,
    pub extent_z: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct BeamMaterial {
    pub young: f64,
    pub shear: f64,
    pub density: f64,
    /// Stress above which the beam model is no longer valid and cells must take over, Pa.
    pub yield_stress: f64,
}

fn local_axes(direction: DVec3) -> [DVec3; 3] {
    let ex = direction.normalize();
    let reference = if ex.dot(DVec3::Y).abs() < 0.99 { DVec3::Y } else { DVec3::Z };
    let ey = (reference - ex * ex.dot(reference)).normalize();
    let ez = ex.cross(ey);
    [ex, ey, ez]
}

/// Local stiffness of one Timoshenko element, 12 x 12, DOF order per node `u v w tx ty tz`.
fn local_stiffness(length: f64, section: &Section, material: &BeamMaterial) -> [[f64; 12]; 12] {
    let (e, g) = (material.young, material.shear);
    let a = section.area;
    let mut k = [[0.0; 12]; 12];
    let axial = e * a / length;
    k[0][0] = axial;
    k[6][6] = axial;
    k[0][6] = -axial;
    k[6][0] = -axial;
    let torsion = g * section.torsion / length;
    k[3][3] = torsion;
    k[9][9] = torsion;
    k[3][9] = -torsion;
    k[9][3] = -torsion;
    let shear_area = section.shear_factor * g * a;
    // Bending in the xy plane: DOF v (1, 7) and rotation about z (5, 11).
    let phi_z = 12.0 * e * section.iz / (shear_area * length * length);
    let f = e * section.iz / ((1.0 + phi_z) * length.powi(3));
    let l = length;
    let xy = [
        [12.0, 6.0 * l, -12.0, 6.0 * l],
        [6.0 * l, (4.0 + phi_z) * l * l, -6.0 * l, (2.0 - phi_z) * l * l],
        [-12.0, -6.0 * l, 12.0, -6.0 * l],
        [6.0 * l, (2.0 - phi_z) * l * l, -6.0 * l, (4.0 + phi_z) * l * l],
    ];
    let xy_index = [1, 5, 7, 11];
    for r in 0..4 {
        for c in 0..4 {
            k[xy_index[r]][xy_index[c]] += f * xy[r][c];
        }
    }
    // Bending in the xz plane: DOF w (2, 8) and rotation about y (4, 10), opposite sign.
    let phi_y = 12.0 * e * section.iy / (shear_area * length * length);
    let f = e * section.iy / ((1.0 + phi_y) * length.powi(3));
    let xz = [
        [12.0, -6.0 * l, -12.0, -6.0 * l],
        [-6.0 * l, (4.0 + phi_y) * l * l, 6.0 * l, (2.0 - phi_y) * l * l],
        [-12.0, 6.0 * l, 12.0, 6.0 * l],
        [-6.0 * l, (2.0 - phi_y) * l * l, 6.0 * l, (4.0 + phi_y) * l * l],
    ];
    let xz_index = [2, 4, 8, 10];
    for r in 0..4 {
        for c in 0..4 {
            k[xz_index[r]][xz_index[c]] += f * xz[r][c];
        }
    }
    k
}

/// Rotation of the 12 DOFs from global to local axes (rows are the local axes).
fn transform(axes: &[DVec3; 3]) -> [[f64; 12]; 12] {
    let mut t = [[0.0; 12]; 12];
    for block in 0..4 {
        for row in 0..3 {
            let axis = axes[row];
            t[block * 3 + row][block * 3] = axis.x;
            t[block * 3 + row][block * 3 + 1] = axis.y;
            t[block * 3 + row][block * 3 + 2] = axis.z;
        }
    }
    t
}

fn mul12(a: &[[f64; 12]; 12], b: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0; 12]; 12];
    for i in 0..12 {
        for k in 0..12 {
            let value = a[i][k];
            if value != 0.0 {
                for j in 0..12 {
                    out[i][j] += value * b[k][j];
                }
            }
        }
    }
    out
}

fn transpose12(a: &[[f64; 12]; 12]) -> [[f64; 12]; 12] {
    let mut out = [[0.0; 12]; 12];
    for i in 0..12 {
        for j in 0..12 {
            out[i][j] = a[j][i];
        }
    }
    out
}

fn cholesky(a: &Block) -> Block {
    let mut l = ZERO;
    for i in 0..6 {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.max(1.0e-300).sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}

fn solve_with(l: &Block, b: &Dof6) -> Dof6 {
    let mut y = *b;
    for i in 0..6 {
        for k in 0..i {
            y[i] -= l[i][k] * y[k];
        }
        y[i] /= l[i][i];
    }
    for i in (0..6).rev() {
        for k in i + 1..6 {
            y[i] -= l[k][i] * y[k];
        }
        y[i] /= l[i][i];
    }
    y
}

fn mul_vec(a: &Block, v: &Dof6) -> Dof6 {
    std::array::from_fn(|i| (0..6).map(|j| a[i][j] * v[j]).sum())
}

fn transpose_mul_vec(a: &Block, v: &Dof6) -> Dof6 {
    std::array::from_fn(|j| (0..6).map(|i| a[i][j] * v[i]).sum())
}

/// A beam structure: nodes joined by elements into a tree, so a trunk can carry branches.
/// `parent[c] < c` for every node `c >= 1`; element `c - 1` joins `parent[c]` to `c`. Node 0
/// is the root and is clamped. A chain is the special case `parent[c] = c - 1`.
pub struct Beam {
    rest: Vec<DVec3>,
    parent: Vec<usize>,
    sections: Vec<Section>,
    materials: Vec<BeamMaterial>,
    stiffness_diagonal: Vec<Block>,
    /// `coupling[c]` is the block (row `parent[c]`, column `c`) of the stiffness matrix.
    coupling: Vec<Block>,
    mass: Vec<Dof6>,
    pub displacement: Vec<Dof6>,
    pub velocity: Vec<Dof6>,
    acceleration: Vec<Dof6>,
    dt: f64,
    effective: Option<Vec<EffectiveBlock>>,
}

/// Factor of one diagonal block of the effective matrix and `D'^-1 U^T` for its parent link.
struct EffectiveBlock {
    cholesky: Block,
    carry: Block,
}

impl Beam {
    /// A chain through `nodes` with one section and material per element.
    pub fn new(nodes: Vec<DVec3>, sections: Vec<Section>, materials: Vec<BeamMaterial>) -> Self {
        let parent: Vec<usize> = (0..nodes.len()).map(|c| c.saturating_sub(1)).collect();
        Self::new_tree(nodes, parent, sections, materials)
    }

    /// A tree of elements. `sections[c - 1]` and `materials[c - 1]` belong to the element
    /// joining `parent[c]` to node `c`.
    pub fn new_tree(
        nodes: Vec<DVec3>,
        parent: Vec<usize>,
        sections: Vec<Section>,
        materials: Vec<BeamMaterial>,
    ) -> Self {
        let n = nodes.len();
        assert!(n >= 2 && parent.len() == n && sections.len() == n - 1 && materials.len() == n - 1);
        assert!((1..n).all(|c| parent[c] < c), "parents must precede their children");
        let mut diagonal = vec![ZERO; n];
        let mut coupling = vec![ZERO; n];
        let mut mass = vec![[0.0; 6]; n];
        for c in 1..n {
            let p = parent[c];
            let direction = nodes[c] - nodes[p];
            let length = direction.length();
            let axes = local_axes(direction);
            let t = transform(&axes);
            let k_local = local_stiffness(length, &sections[c - 1], &materials[c - 1]);
            let k_global = mul12(&transpose12(&t), &mul12(&k_local, &t));
            for i in 0..6 {
                for j in 0..6 {
                    diagonal[p][i][j] += k_global[i][j];
                    diagonal[c][i][j] += k_global[6 + i][6 + j];
                    coupling[c][i][j] += k_global[i][6 + j];
                }
            }
            // Lumped mass: half of the element at each end, rotary inertia from the section.
            let m = materials[c - 1].density * sections[c - 1].area * length;
            let rotary = materials[c - 1].density * length * 0.5
                * (sections[c - 1].iy + sections[c - 1].iz + sections[c - 1].torsion)
                / 3.0;
            for node in [p, c] {
                for q in 0..3 {
                    mass[node][q] += 0.5 * m;
                    mass[node][q + 3] += rotary.max(1.0e-12 * m * length * length);
                }
            }
        }
        // Clamp the root: identity row, no coupling to its children.
        diagonal[0] = ZERO;
        for i in 0..6 {
            diagonal[0][i][i] = 1.0;
        }
        for c in 1..n {
            if parent[c] == 0 {
                coupling[c] = ZERO;
            }
        }
        Self {
            rest: nodes,
            parent,
            sections,
            materials,
            stiffness_diagonal: diagonal,
            coupling,
            mass,
            displacement: vec![[0.0; 6]; n],
            velocity: vec![[0.0; 6]; n],
            acceleration: vec![[0.0; 6]; n],
            dt: 0.0,
            effective: None,
        }
    }

    pub fn node_count(&self) -> usize {
        self.rest.len()
    }

    /// Eliminate nodes from the leaves to the root for `K + scale * M`.
    fn factor(&self, mass_scale: f64) -> Vec<EffectiveBlock> {
        let n = self.rest.len();
        let mut reduced: Vec<Block> = (0..n)
            .map(|i| {
                let mut d = self.stiffness_diagonal[i];
                for p in 0..6 {
                    d[p][p] += mass_scale * self.mass[i][p];
                }
                d
            })
            .collect();
        let mut out: Vec<EffectiveBlock> = (0..n)
            .map(|_| EffectiveBlock { cholesky: ZERO, carry: ZERO })
            .collect();
        for c in (1..n).rev() {
            let p = self.parent[c];
            let l = cholesky(&reduced[c]);
            let u = &self.coupling[c];
            let mut carry = ZERO;
            for column in 0..6 {
                // Column `column` of U^T is row `column` of U.
                let rhs: Dof6 = u[column];
                let solved = solve_with(&l, &rhs);
                for row in 0..6 {
                    carry[row][column] = solved[row];
                }
            }
            // D_p' -= U D_c'^-1 U^T = U * carry.
            for r in 0..6 {
                for col in 0..6 {
                    let value: f64 = (0..6).map(|k| u[r][k] * carry[k][col]).sum();
                    reduced[p][r][col] -= value;
                }
            }
            out[c] = EffectiveBlock { cholesky: l, carry };
        }
        out[0].cholesky = cholesky(&reduced[0]);
        out
    }

    fn solve_factored(&self, factor: &[EffectiveBlock], rhs: &[Dof6]) -> Vec<Dof6> {
        let n = self.rest.len();
        let mut b = rhs.to_vec();
        let mut y = vec![[0.0; 6]; n];
        for c in (1..n).rev() {
            y[c] = solve_with(&factor[c].cholesky, &b[c]);
            let p = self.parent[c];
            let contribution = mul_vec(&self.coupling[c], &y[c]);
            for q in 0..6 {
                b[p][q] -= contribution[q];
            }
        }
        let mut x = y;
        x[0] = solve_with(&factor[0].cholesky, &b[0]);
        for c in 1..n {
            let correction = mul_vec(&factor[c].carry, &x[self.parent[c]]);
            for q in 0..6 {
                x[c][q] -= correction[q];
            }
        }
        x
    }

    /// Static displacement under nodal loads `forces` (N and N m, global axes).
    pub fn solve_static(&mut self, forces: &[Dof6]) {
        let factor = self.factor(0.0);
        let mut rhs = forces.to_vec();
        rhs[0] = [0.0; 6];
        self.displacement = self.solve_factored(&factor, &rhs);
        self.velocity = vec![[0.0; 6]; self.rest.len()];
        self.acceleration = vec![[0.0; 6]; self.rest.len()];
    }

    fn apply_stiffness(&self, u: &[Dof6]) -> Vec<Dof6> {
        let n = self.rest.len();
        let mut out: Vec<Dof6> = (0..n).map(|i| mul_vec(&self.stiffness_diagonal[i], &u[i])).collect();
        for c in 1..n {
            let p = self.parent[c];
            let up = mul_vec(&self.coupling[c], &u[c]);
            let down = transpose_mul_vec(&self.coupling[c], &u[p]);
            for q in 0..6 {
                out[p][q] += up[q];
                out[c][q] += down[q];
            }
        }
        out[0] = [0.0; 6];
        out
    }

    /// Release a deformed state at rest: acceleration from `M a = F - K u`.
    pub fn release(&mut self, forces: &[Dof6]) {
        let ku = self.apply_stiffness(&self.displacement);
        for i in 0..self.rest.len() {
            for p in 0..6 {
                self.velocity[i][p] = 0.0;
                self.acceleration[i][p] =
                    if i == 0 { 0.0 } else { (forces[i][p] - ku[i][p]) / self.mass[i][p] };
            }
        }
    }

    /// Advance one step of `dt` seconds under nodal loads `forces`.
    pub fn step(&mut self, dt: f64, forces: &[Dof6]) {
        if self.effective.is_none() || (self.dt - dt).abs() > 1.0e-15 {
            self.dt = dt;
            self.effective = Some(self.factor(4.0 / (dt * dt)));
        }
        let n = self.rest.len();
        let scale_u = 4.0 / (dt * dt);
        let scale_v = 4.0 / dt;
        let mut rhs = vec![[0.0; 6]; n];
        for i in 0..n {
            for p in 0..6 {
                rhs[i][p] = forces[i][p]
                    + self.mass[i][p]
                        * (scale_u * self.displacement[i][p]
                            + scale_v * self.velocity[i][p]
                            + self.acceleration[i][p]);
            }
        }
        rhs[0] = [0.0; 6];
        let factor = self.effective.take().expect("factor");
        let next = self.solve_factored(&factor, &rhs);
        self.effective = Some(factor);
        for i in 0..n {
            for p in 0..6 {
                let a_new = scale_u * (next[i][p] - self.displacement[i][p])
                    - scale_v * self.velocity[i][p]
                    - self.acceleration[i][p];
                self.velocity[i][p] += 0.5 * dt * (self.acceleration[i][p] + a_new);
                self.acceleration[i][p] = a_new;
            }
        }
        self.displacement = next;
        self.displacement[0] = [0.0; 6];
        self.velocity[0] = [0.0; 6];
        self.acceleration[0] = [0.0; 6];
    }

    /// Kinetic plus strain energy, joules.
    pub fn energy(&self) -> f64 {
        let ku = self.apply_stiffness(&self.displacement);
        let mut total = 0.0;
        for i in 1..self.rest.len() {
            for p in 0..6 {
                total += 0.5 * self.mass[i][p] * self.velocity[i][p] * self.velocity[i][p];
                total += 0.5 * self.displacement[i][p] * ku[i][p];
            }
        }
        total
    }

    /// Rest position of node `i`.
    pub fn rest_position(&self, i: usize) -> DVec3 {
        self.rest[i]
    }

    /// Strain energy `1/2 u^T K u` of the current displacement, joules.
    pub fn strain_energy(&self) -> f64 {
        let ku = self.apply_stiffness(&self.displacement);
        (1..self.rest.len())
            .map(|i| (0..6).map(|p| 0.5 * self.displacement[i][p] * ku[i][p]).sum::<f64>())
            .sum()
    }

    /// Demote cells to this beam: fit each node to its slice. Translation is the mean
    /// displacement of the slice's cells; rotation is the small rigid rotation that best
    /// explains the rest of their displacement, `sum |u_i - u - theta x r_i|^2` minimised
    /// (a slice of one cell has no observable rotation and keeps zero). Velocities are fitted
    /// the same way. `groups` are the cell indices of each node, as returned by
    /// `reduce_lattice_along_x`; `rest` and the state arrays are indexed by cell.
    pub fn set_from_cells(
        &mut self,
        groups: &[Vec<usize>],
        rest: &[DVec3],
        positions: &[DVec3],
        velocities: &[DVec3],
    ) {
        for (node, cells) in groups.iter().enumerate() {
            let count = cells.len() as f64;
            let mean = |values: &dyn Fn(usize) -> DVec3| {
                cells.iter().fold(DVec3::ZERO, |sum, &i| sum + values(i)) / count
            };
            let u = mean(&|i| positions[i] - rest[i]);
            let v = mean(&|i| velocities[i]);
            let fit = |field: &dyn Fn(usize) -> DVec3, centre: DVec3| {
                let mut a = glam::DMat3::ZERO;
                let mut b = DVec3::ZERO;
                for &i in cells {
                    let r = rest[i] - self.rest[node];
                    a += glam::DMat3::from_diagonal(DVec3::splat(r.length_squared()))
                        - glam::DMat3::from_cols(r * r.x, r * r.y, r * r.z);
                    b += r.cross(field(i) - centre);
                }
                let trace = a.x_axis.x + a.y_axis.y + a.z_axis.z;
                if trace <= 0.0 {
                    return DVec3::ZERO;
                }
                (a + glam::DMat3::from_diagonal(DVec3::splat(trace * 1.0e-9))).inverse() * b
            };
            let theta = fit(&|i| positions[i] - rest[i], u);
            let omega = fit(&|i| velocities[i], v);
            self.displacement[node] = [u.x, u.y, u.z, theta.x, theta.y, theta.z];
            self.velocity[node] = [v.x, v.y, v.z, omega.x, omega.y, omega.z];
        }
        self.displacement[0] = [0.0; 6];
        self.velocity[0] = [0.0; 6];
        self.acceleration = vec![[0.0; 6]; self.rest.len()];
    }

    /// Promote this beam to cells: each cell is a rigid part of its section,
    /// `p = rest + u + theta x (rest - node_rest)`.
    pub fn expand_positions(&self, groups: &[Vec<usize>], rest: &[DVec3]) -> Vec<DVec3> {
        let mut out = rest.to_vec();
        for (node, cells) in groups.iter().enumerate() {
            let d = &self.displacement[node];
            let (u, theta) = (DVec3::new(d[0], d[1], d[2]), DVec3::new(d[3], d[4], d[5]));
            for &i in cells {
                out[i] = rest[i] + u + theta.cross(rest[i] - self.rest[node]);
            }
        }
        out
    }

    /// Cell velocities of the promoted beam: `v = v_node + omega x r`.
    pub fn expand_velocities(&self, groups: &[Vec<usize>], rest: &[DVec3]) -> Vec<DVec3> {
        let mut out = vec![DVec3::ZERO; rest.len()];
        for (node, cells) in groups.iter().enumerate() {
            let d = &self.velocity[node];
            let (v, omega) = (DVec3::new(d[0], d[1], d[2]), DVec3::new(d[3], d[4], d[5]));
            for &i in cells {
                out[i] = v + omega.cross(rest[i] - self.rest[node]);
            }
        }
        out
    }

    /// Largest normal stress at either end of element `e` (joining `parent[e + 1]` to
    /// node `e + 1`): axial plus bending, Pa.
    pub fn max_stress(&self, e: usize) -> f64 {
        let c = e + 1;
        let p = self.parent[c];
        let direction = self.rest[c] - self.rest[p];
        let length = direction.length();
        let axes = local_axes(direction);
        let t = transform(&axes);
        let k = local_stiffness(length, &self.sections[e], &self.materials[e]);
        let mut u = [0.0; 12];
        for q in 0..6 {
            u[q] = self.displacement[p][q];
            u[6 + q] = self.displacement[c][q];
        }
        let mut local_u = [0.0; 12];
        for i in 0..12 {
            local_u[i] = (0..12).map(|j| t[i][j] * u[j]).sum();
        }
        let forces: Vec<f64> = (0..12).map(|i| (0..12).map(|j| k[i][j] * local_u[j]).sum()).collect();
        let s = &self.sections[e];
        let mut worst: f64 = 0.0;
        for end in [0, 6] {
            let axial = forces[end].abs() / s.area;
            let bend_y = forces[end + 4].abs() * s.extent_z / s.iy;
            let bend_z = forces[end + 5].abs() * s.extent_y / s.iz;
            worst = worst.max(axial + bend_y + bend_z);
        }
        worst
    }

    /// Largest stress over all elements as a fraction of its material limit; 1.0 or more
    /// means the model is no longer valid and cells must take over.
    pub fn validity(&self) -> f64 {
        (0..self.rest.len() - 1)
            .map(|e| self.max_stress(e) / self.materials[e].yield_stress)
            .fold(0.0, f64::max)
    }
}

/// Section properties of a slice of square cells of side `cell_m` at the given offsets
/// `(y, z)` of their centres from the beam axis.
pub fn section_of_cells(centres: &[(f64, f64)], cell_m: f64) -> Section {
    let a = cell_m * cell_m;
    let own = cell_m.powi(4) / 12.0;
    let mut iy = 0.0;
    let mut iz = 0.0;
    let mut extent_y: f64 = 0.0;
    let mut extent_z: f64 = 0.0;
    for &(y, z) in centres {
        iz += own + a * y * y;
        iy += own + a * z * z;
        extent_y = extent_y.max(y.abs() + 0.5 * cell_m);
        extent_z = extent_z.max(z.abs() + 0.5 * cell_m);
    }
    // Torsion of a rectangle from its outline, St Venant approximation.
    let (min_y, max_y) = centres.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &(y, _)| (lo.min(y), hi.max(y)));
    let (min_z, max_z) = centres.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &(_, z)| (lo.min(z), hi.max(z)));
    let (w, h) = (max_y - min_y + cell_m, max_z - min_z + cell_m);
    let (long, short) = if w >= h { (w, h) } else { (h, w) };
    let torsion = long * short.powi(3) * (1.0 / 3.0 - 0.21 * short / long * (1.0 - short.powi(4) / (12.0 * long.powi(4))));
    Section {
        area: a * centres.len() as f64,
        iy,
        iz,
        torsion,
        shear_factor: 5.0 / 6.0,
        extent_y,
        extent_z,
    }
}

/// Reduce a lattice of cells to a beam along global X. Cells are grouped into slices
/// by their X coordinate; a slice becomes a node at its centroid and the section of
/// each element is the mean of its two slices. Returns the beam and, for each node,
/// the cell indices of its slice, so loads and results can be mapped back. The slice
/// with the smallest X is clamped. `None` if the cells do not form a connected chain.
pub fn reduce_lattice_along_x(
    positions: &[DVec3],
    cell_m: f64,
    material: BeamMaterial,
) -> Option<(Beam, Vec<Vec<usize>>)> {
    if positions.is_empty() {
        return None;
    }
    // Cluster cells into slices by X: a new slice starts where the gap exceeds half a cell.
    let mut order: Vec<usize> = (0..positions.len()).collect();
    order.sort_by(|&a, &b| positions[a].x.total_cmp(&positions[b].x));
    let mut slices: Vec<Vec<usize>> = Vec::new();
    let mut last_x = f64::NEG_INFINITY;
    for &index in &order {
        let x = positions[index].x;
        if slices.is_empty() || x - last_x > 0.5 * cell_m {
            slices.push(Vec::new());
        }
        slices.last_mut().expect("slice").push(index);
        last_x = x;
    }
    if slices.len() < 2 {
        return None;
    }
    let mean_x = |cells: &Vec<usize>| cells.iter().map(|&i| positions[i].x).sum::<f64>() / cells.len() as f64;
    if slices.windows(2).any(|pair| ((mean_x(&pair[1]) - mean_x(&pair[0])) / cell_m - 1.0).abs() > 0.25) {
        return None;
    }
    let mut nodes = Vec::new();
    let mut section_of_slice = Vec::new();
    let mut groups = Vec::new();
    for cells in &slices {
        let count = cells.len() as f64;
        let centroid = cells.iter().fold(DVec3::ZERO, |sum, &i| sum + positions[i]) / count;
        let offsets: Vec<(f64, f64)> =
            cells.iter().map(|&i| (positions[i].y - centroid.y, positions[i].z - centroid.z)).collect();
        nodes.push(centroid);
        section_of_slice.push(section_of_cells(&offsets, cell_m));
        groups.push(cells.clone());
    }
    let elements = nodes.len() - 1;
    let sections: Vec<Section> = (0..elements)
        .map(|e| {
            let (a, b) = (&section_of_slice[e], &section_of_slice[e + 1]);
            Section {
                area: 0.5 * (a.area + b.area),
                iy: 0.5 * (a.iy + b.iy),
                iz: 0.5 * (a.iz + b.iz),
                torsion: 0.5 * (a.torsion + b.torsion),
                shear_factor: a.shear_factor,
                extent_y: a.extent_y.max(b.extent_y),
                extent_z: a.extent_z.max(b.extent_z),
            }
        })
        .collect();
    Some((Beam::new(nodes, sections, vec![material; elements]), groups))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: f64 = 0.01;

    fn iron() -> BeamMaterial {
        BeamMaterial { young: 200e9, shear: 77e9, density: 7870.0, yield_stress: 350e6 }
    }

    /// 3 x 3 cell cross-section, `cells` slices along x, node spacing one cell.
    fn cantilever(cells: usize) -> Beam {
        let mut offsets = Vec::new();
        for a in -1..=1 {
            for b in -1..=1 {
                offsets.push((f64::from(a) * CELL, f64::from(b) * CELL));
            }
        }
        let section = section_of_cells(&offsets, CELL);
        let nodes: Vec<DVec3> = (0..cells).map(|i| DVec3::new(i as f64 * CELL, 0.0, 0.0)).collect();
        Beam::new(nodes, vec![section; cells - 1], vec![iron(); cells - 1])
    }

    fn tip_load(beam: &Beam, force: Dof6) -> Vec<Dof6> {
        let mut loads = vec![[0.0; 6]; beam.node_count()];
        *loads.last_mut().unwrap() = force;
        loads
    }

    #[test]
    fn static_tip_deflection_matches_the_analytic_beam_with_shear() {
        let mut beam = cantilever(31);
        let force = 90.0;
        beam.solve_static(&tip_load(&beam, [0.0, 0.0, force, 0.0, 0.0, 0.0]));
        let length = 30.0 * CELL;
        let side = 3.0 * CELL;
        let (a, i) = (side * side, side.powi(4) / 12.0);
        let expected = force * length.powi(3) / (3.0 * 200e9 * i) + force * length / (5.0 / 6.0 * 77e9 * a);
        let measured = beam.displacement.last().unwrap()[2];
        assert!((measured / expected - 1.0).abs() < 1.0e-3, "{measured:e} against {expected:e}");
    }

    #[test]
    fn axial_and_torsion_statics_match_closed_forms() {
        let mut beam = cantilever(21);
        let length = 20.0 * CELL;
        let a = 9.0 * CELL * CELL;
        beam.solve_static(&tip_load(&beam, [1000.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        let axial = beam.displacement.last().unwrap()[0];
        assert!((axial / (1000.0 * length / (200e9 * a)) - 1.0).abs() < 1.0e-9);
        beam.solve_static(&tip_load(&beam, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0]));
        let twist = beam.displacement.last().unwrap()[3];
        let j = beam.sections[0].torsion;
        assert!((twist / (5.0 * length / (77e9 * j)) - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn first_bending_frequency_matches_the_analytic_value() {
        let mut beam = cantilever(31);
        let length = 30.0 * CELL;
        let (a, i) = (9.0 * CELL * CELL, (3.0 * CELL).powi(4) / 12.0);
        let expected = 1.875_104_069_f64.powi(2) / (2.0 * std::f64::consts::PI)
            * (200e9 * i / (7870.0 * a * length.powi(4))).sqrt();
        beam.solve_static(&tip_load(&beam, [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]));
        beam.release(&vec![[0.0; 6]; beam.node_count()]);
        let dt = 1.0 / (expected * 400.0);
        let zero = vec![[0.0; 6]; beam.node_count()];
        let mut previous = beam.displacement.last().unwrap()[2];
        let mut crossings = Vec::new();
        for step in 1..4000 {
            beam.step(dt, &zero);
            let current = beam.displacement.last().unwrap()[2];
            if previous > 0.0 && current <= 0.0 {
                let fraction = previous / (previous - current);
                crossings.push((step as f64 - 1.0 + fraction) * dt);
            }
            previous = current;
            if crossings.len() == 3 {
                break;
            }
        }
        assert!(crossings.len() >= 2);
        let period = crossings[1] - crossings[0];
        assert!((1.0 / period / expected - 1.0).abs() < 0.03, "{} Hz against {expected} Hz", 1.0 / period);
    }

    #[test]
    fn free_vibration_conserves_energy() {
        let mut beam = cantilever(31);
        beam.solve_static(&tip_load(&beam, [0.0, 3.0, 20.0, 0.0, 0.0, 0.0]));
        beam.release(&vec![[0.0; 6]; beam.node_count()]);
        let zero = vec![[0.0; 6]; beam.node_count()];
        let start = beam.energy();
        assert!(start > 0.0);
        let mut worst: f64 = 0.0;
        for _ in 0..2000 {
            beam.step(2.0e-5, &zero);
            worst = worst.max((beam.energy() - start).abs() / start);
        }
        assert!(worst < 1.0e-8, "energy drifted by {worst}");
    }

    #[test]
    fn validity_reports_when_cells_must_take_over() {
        let mut beam = cantilever(31);
        beam.solve_static(&tip_load(&beam, [0.0, 0.0, 90.0, 0.0, 0.0, 0.0]));
        // M = F L at the root, sigma = M c / I.
        let (length, i, c) = (0.3, (0.03f64).powi(4) / 12.0, 0.015);
        let expected = 90.0 * length * c / i;
        let measured = beam.max_stress(0);
        assert!((measured / expected - 1.0).abs() < 0.02, "{measured:e} against {expected:e}");
        assert!(beam.validity() < 1.0);
        // 8000 N gives 5.3e8 Pa at the root, above the 3.5e8 Pa limit of iron.
        beam.solve_static(&tip_load(&beam, [0.0, 0.0, 8000.0, 0.0, 0.0, 0.0]));
        assert!(beam.validity() > 1.0, "a load beyond yield must hand over to cells");
    }

    #[test]
    fn tilted_beam_gives_the_same_deflection_as_an_axis_aligned_one() {
        let mut offsets = Vec::new();
        for a in -1..=1 {
            for b in -1..=1 {
                offsets.push((f64::from(a) * CELL, f64::from(b) * CELL));
            }
        }
        let section = section_of_cells(&offsets, CELL);
        let direction = DVec3::new(1.0, 2.0, 2.0).normalize();
        let nodes: Vec<DVec3> = (0..21).map(|i| direction * (i as f64 * CELL)).collect();
        let mut tilted = Beam::new(nodes, vec![section; 20], vec![iron(); 20]);
        let mut straight = cantilever(21);
        // A force perpendicular to each beam axis, of equal size.
        let perpendicular = direction.cross(DVec3::Z).normalize() * 50.0;
        let mut loads = vec![[0.0; 6]; 21];
        loads[20] = [perpendicular.x, perpendicular.y, perpendicular.z, 0.0, 0.0, 0.0];
        tilted.solve_static(&loads);
        straight.solve_static(&tip_load(&straight, [0.0, 50.0, 0.0, 0.0, 0.0, 0.0]));
        let tilted_tip = DVec3::new(tilted.displacement[20][0], tilted.displacement[20][1], tilted.displacement[20][2]);
        let straight_tip = straight.displacement[20][1];
        assert!((tilted_tip.length() / straight_tip - 1.0).abs() < 1.0e-6);
    }

    fn square_section() -> Section {
        let mut offsets = Vec::new();
        for a in -1..=1 {
            for b in -1..=1 {
                offsets.push((f64::from(a) * CELL, f64::from(b) * CELL));
            }
        }
        section_of_cells(&offsets, CELL)
    }

    /// A trunk along X of `trunk` nodes with a side branch of `branch` nodes along +Y
    /// starting at trunk node `at`.
    fn trunk_with_branch(trunk: usize, at: usize, branch: usize) -> Beam {
        let mut nodes: Vec<DVec3> = (0..trunk).map(|i| DVec3::new(i as f64 * CELL, 0.0, 0.0)).collect();
        let mut parent: Vec<usize> = (0..trunk).map(|c| c.saturating_sub(1)).collect();
        for j in 1..=branch {
            nodes.push(DVec3::new(at as f64 * CELL, j as f64 * CELL, 0.0));
            parent.push(if j == 1 { at } else { nodes.len() - 2 });
        }
        let elements = nodes.len() - 1;
        Beam::new_tree(nodes, parent, vec![square_section(); elements], vec![iron(); elements])
    }

    #[test]
    fn an_unloaded_branch_does_not_change_the_trunk_deflection() {
        let force = [0.0, 0.0, 90.0, 0.0, 0.0, 0.0];
        let mut plain = cantilever(31);
        plain.solve_static(&tip_load(&plain, force));
        let mut tree = trunk_with_branch(31, 15, 10);
        let mut loads = vec![[0.0; 6]; tree.node_count()];
        loads[30] = force;
        tree.solve_static(&loads);
        let (a, b) = (plain.displacement[30][2], tree.displacement[30][2]);
        assert!((a / b - 1.0).abs() < 1.0e-9, "{a:e} against {b:e}");
        // The unloaded branch carries no stress: it only moves with the node it grows from.
        assert!(tree.max_stress(31) < 1.0e-6 * tree.max_stress(0), "an unloaded branch carries stress");
    }

    #[test]
    fn an_l_shaped_frame_matches_the_analytic_deflection() {
        // Column of 31 nodes up Z (H = 0.3 m), then an arm of 20 elements along +X (A = 0.2 m).
        let mut nodes: Vec<DVec3> = (0..31).map(|i| DVec3::new(0.0, 0.0, i as f64 * CELL)).collect();
        let mut parent: Vec<usize> = (0..31usize).map(|c| c.saturating_sub(1)).collect();
        for j in 1..=20 {
            nodes.push(DVec3::new(j as f64 * CELL, 0.0, 0.3));
            parent.push(if j == 1 { 30 } else { nodes.len() - 2 });
        }
        let elements = nodes.len() - 1;
        let tip = nodes.len() - 1;
        let mut frame = Beam::new_tree(nodes, parent, vec![square_section(); elements], vec![iron(); elements]);
        let mut loads = vec![[0.0; 6]; frame.node_count()];
        let load = 50.0;
        loads[tip] = [0.0, 0.0, -load, 0.0, 0.0, 0.0];
        frame.solve_static(&loads);
        let (h, a) = (0.3f64, 0.2f64);
        let (e, g) = (200e9f64, 77e9f64);
        let (area, inertia) = (9.0e-4f64, 6.75e-8f64);
        let expected = load * a.powi(3) / (3.0 * e * inertia)
            + load * a * a * h / (e * inertia)
            + load * h / (e * area)
            + load * a / (5.0 / 6.0 * g * area);
        let measured = -frame.displacement[tip][2];
        assert!((measured / expected - 1.0).abs() < 0.01, "{measured:e} against {expected:e}");
    }

    #[test]
    fn a_branched_tree_conserves_energy_in_free_vibration() {
        let mut tree = trunk_with_branch(31, 15, 12);
        let mut loads = vec![[0.0; 6]; tree.node_count()];
        loads[30] = [0.0, 4.0, 15.0, 0.0, 0.0, 0.0];
        let last = tree.node_count() - 1;
        loads[last] = [6.0, 0.0, -5.0, 0.0, 0.0, 0.0];
        tree.solve_static(&loads);
        let zero = vec![[0.0; 6]; tree.node_count()];
        tree.release(&zero);
        let start = tree.energy();
        assert!(start > 0.0);
        let mut worst: f64 = 0.0;
        for _ in 0..2000 {
            tree.step(2.0e-5, &zero);
            worst = worst.max((tree.energy() - start).abs() / start);
        }
        assert!(worst < 1.0e-8, "energy drifted by {worst}");
    }

    #[test]
    fn a_step_costs_microseconds_per_node() {
        let mut beam = cantilever(201);
        let zero = vec![[0.0; 6]; beam.node_count()];
        beam.step(1.0e-3, &zero);
        let started = std::time::Instant::now();
        for _ in 0..200 {
            beam.step(1.0e-3, &zero);
        }
        let per_node_us = started.elapsed().as_secs_f64() * 1.0e6 / 200.0 / 201.0;
        eprintln!("BEAM cost {per_node_us:.2} us per node per step");
        assert!(per_node_us < 50.0, "{per_node_us} us per node");
    }
}
