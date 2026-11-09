//! Heat conduction with one temperature per REV (decision of 8 November 2026).
//!
//! The detailed model keeps 3×3×3 thermal control volumes inside each REV; this one treats
//! each REV as a single finite volume of heat capacity m·c joined to its bonded face
//! neighbours by the conductance G = k_eff·A/L = k_eff·CELL (A = CELL², L = CELL), with
//! k_eff the harmonic mean of the two conductivities, scaled by the bond's remaining
//! integrity. Fluxes are equal and opposite, so energy is conserved exactly; explicit
//! subcycles keep each step inside the stability bound. Lost: temperature gradients inside
//! one 1 cm REV (a heated edge, a melting corner), allowed by the 25% tolerance.
use super::*;

impl Scene {
    pub(crate) fn conduct_heat_per_rev(&mut self, elapsed_seconds: f32) -> Result<usize, &'static str> {
        let n = self.voxels.len();
        let cell = f64::from(CELL);
        let live = if self.thermal_scratch.sparse_disabled { None } else { self.step_live.clone() };
        let in_list = |j: usize| live.as_ref().map_or(true, |l| l.thermal_in_list[j]);
        let temperature = |scene: &Scene, i: usize| {
            let v = &scene.voxels[i];
            f64::from(scene.properties[i].temperature(v.mass, v.thermal_energy_f32()))
        };
        let mut t = vec![f64::NAN; n];
        for i in visit_thermal(&live, n) {
            t[i] = temperature(self, i);
        }
        // Every conductive bond whose two ends differ, plus the bonds next to those ends,
        // so heat can move on into an equal-temperature neighbour within this call.
        let mut all = Vec::new();
        let mut active = vec![false; n];
        for axis in 0..3 {
            for i in visit_thermal(&live, n) {
                let bond = &self.bonds[axis][i];
                if bond.neighbor == EMPTY || bond.damage >= 1.0 {
                    continue;
                }
                let j = bond.neighbor as usize;
                if !in_list(j) {
                    continue;
                }
                let (ki, kj) = (f64::from(self.properties[i].thermal_cond), f64::from(self.properties[j].thermal_cond));
                if ki <= 0.0 || kj <= 0.0 {
                    continue;
                }
                if t[j].is_nan() {
                    t[j] = temperature(self, j);
                }
                let g = 2.0 * ki * kj / (ki + kj) * cell * f64::from(1.0 - bond.damage);
                if t[i] != t[j] {
                    active[i] = true;
                    active[j] = true;
                }
                all.push((i, j, g));
            }
        }
        let links: Vec<(usize, usize, f64)> = all.into_iter().filter(|&(i, j, _)| active[i] || active[j]).collect();
        if links.is_empty() {
            return Ok(0);
        }
        let mut conductance = vec![0.0f64; n];
        for &(i, j, g) in &links {
            conductance[i] += g;
            conductance[j] += g;
        }
        let mut limit = f64::INFINITY;
        for (i, &g) in conductance.iter().enumerate() {
            if g > 0.0 {
                let capacity = f64::from(self.voxels[i].mass) * f64::from(self.properties[i].heat_capacity).max(1.0e-6);
                limit = limit.min(capacity / g);
            }
        }
        let elapsed = f64::from(elapsed_seconds);
        let subcycles = ((elapsed / (0.5 * limit)).ceil() as usize).clamp(1, 10_000);
        let h = elapsed / subcycles as f64;
        let mut touched: Vec<usize> = links.iter().flat_map(|&(i, j, _)| [i, j]).collect();
        touched.sort_unstable();
        touched.dedup();
        for _ in 0..subcycles {
            for &(i, j, g) in &links {
                let q = g * (t[i] - t[j]) * h;
                if q != 0.0 {
                    self.voxels[i].add_internal_energy_j(-q);
                    self.voxels[j].add_internal_energy_j(q);
                }
            }
            for &i in &touched {
                t[i] = temperature(self, i);
            }
        }
        Ok(subcycles)
    }
}
