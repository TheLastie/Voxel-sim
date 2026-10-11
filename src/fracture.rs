//! Local constitutive law and energy accounting. No tool or object identities.
use crate::{
    bond::{Bond, EMPTY},
    material::{FractureMode, Material},
    scene::{AREA, CELL},
};
use glam::{Mat3, Vec3};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FractureConfig {
    pub damage_rate: f32,
    pub energy_release: f32,
    /// Delay in physical substeps, not convergence iterations.
    pub propagation_delay: u64,
    pub fatigue_cycles: f32,
    pub fatigue_exponent: f32,
    pub fatigue_threshold: f32,
    pub plastic_heat_fraction: f32,
    pub fragment_min_size: usize,
    pub cluster_interval: u64,
    pub friction: f32,
    pub restitution: f32,
}
impl FractureConfig {
    pub fn validate(&self) -> bool {
        self.damage_rate.is_finite()
            && self.damage_rate > 0.0
            && (0.0..=1.0).contains(&self.energy_release)
            && self.fatigue_cycles.is_finite()
            && self.fatigue_cycles > 0.0
            && self.fatigue_exponent.is_finite()
            && self.fatigue_exponent >= 1.0
            && (0.0..1.0).contains(&self.fatigue_threshold)
            && (0.0..=1.0).contains(&self.plastic_heat_fraction)
            && self.fragment_min_size > 0
            && self.cluster_interval > 0
            && self.friction.is_finite()
            && self.friction >= 0.0
            && (0.0..=1.0).contains(&self.restitution)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub tension: f32,
    pub compression: f32,
    pub shear: f32,
    pub yield_stress: f32,
    pub hardening: f32,
    pub plastic_failure: f32,
    pub ductility: f32,
    pub fracture_energy: f32,
}
fn directional(parallel: f32, perpendicular: f32, alignment: f32, anisotropy: f32) -> f32 {
    parallel + anisotropy * (1.0 - alignment.abs().clamp(0.0, 1.0)) * (perpendicular - parallel)
}
pub fn pair_limits(
    a: &Material,
    b: &Material,
    alignment: [f32; 2],
    anisotropy: [f32; 2],
) -> Limits {
    let one = |m: &Material, k: usize| {
        let ductility = match m.fracture_mode {
            FractureMode::Brittle => 0.0,
            FractureMode::Ductile => 1.0,
            FractureMode::Mixed => {
                (1.0 - anisotropy[k]) * 0.5 + anisotropy[k] * alignment[k].abs().powi(2)
            }
        };
        Limits {
            tension: directional(
                m.tensile_strength,
                m.tensile_strength_perp,
                alignment[k],
                anisotropy[k],
            ),
            compression: directional(
                m.compressive_strength,
                m.compressive_strength_perp,
                alignment[k],
                anisotropy[k],
            ),
            shear: directional(
                m.shear_strength,
                m.shear_strength_perp,
                alignment[k],
                anisotropy[k],
            ),
            yield_stress: m.yield_strength,
            hardening: m.hardening_modulus,
            plastic_failure: m.failure_plastic_strain,
            ductility,
            fracture_energy: m.fracture_energy,
        }
    };
    let x = one(a, 0);
    let y = one(b, 1);
    Limits {
        tension: x.tension.min(y.tension),
        compression: x.compression.min(y.compression),
        shear: x.shear.min(y.shear),
        yield_stress: x.yield_stress.min(y.yield_stress),
        hardening: x.hardening.min(y.hardening),
        plastic_failure: x.plastic_failure.min(y.plastic_failure),
        ductility: x.ductility.min(y.ductility),
        fracture_energy: x.fracture_energy.min(y.fracture_energy),
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Dissipation {
    pub plastic_heat: f64,
    pub stored_plastic: f64,
    pub surface_work: f64,
    pub damage_heat: f64,
    pub released: f64,
}
impl Dissipation {
    pub fn total(self) -> f64 {
        self.plastic_heat
            + self.stored_plastic
            + self.surface_work
            + self.damage_heat
            + self.released
    }
}
#[derive(Clone, Debug)]
pub struct Release {
    pub i: usize,
    pub j: usize,
    pub energy: f64,
    pub due: u64,
}
#[derive(Clone, Debug, Default)]
pub struct EnergyLedger {
    pub plastic_heat: f64,
    pub stored_plastic: f64,
    pub surface_work: f64,
    pub damage_heat: f64,
    pub pending_release: f64,
    pub kinetic_release: f64,
}
impl EnergyLedger {
    pub fn record(&mut self, d: Dissipation) {
        self.plastic_heat += d.plastic_heat;
        self.stored_plastic += d.stored_plastic;
        self.surface_work += d.surface_work;
        self.damage_heat += d.damage_heat;
        self.pending_release += d.released;
    }
}
fn energy(u: Vec3, k: Mat3, damage: f32) -> f64 {
    (0.5 * (1.0 - damage) * u.dot(k * u)).max(0.0) as f64
}
/// Cyclic fatigue: only stress reversals consume life. Holding load does not age it.
fn fatigue(b: &mut Bond, signed_ratio: f32, c: &FractureConfig) {
    let delta = signed_ratio - b.previous_stress_ratio;
    if delta.abs() > 1e-6 {
        if b.loading_increment * delta < 0.0 {
            let amplitude = (b.previous_stress_ratio - b.reversal_stress_ratio).abs() * 0.5;
            if amplitude > c.fatigue_threshold {
                let level =
                    ((amplitude - c.fatigue_threshold) / (1.0 - c.fatigue_threshold)).min(1.0);
                b.fatigue_damage = (b.fatigue_damage
                    + 0.5 * level.powf(c.fatigue_exponent) / c.fatigue_cycles)
                    .min(1.0);
            }
            b.reversal_stress_ratio = b.previous_stress_ratio;
        }
        b.loading_increment = delta;
    }
    b.previous_stress_ratio = signed_ratio;
}
/// Update once per physical substep. `u_total` excludes the rest vector but includes
/// plastic displacement; K is the undamaged stiffness, N/m. Returns removed energy.
pub fn update(
    b: &mut Bond,
    u_total: Vec3,
    k: Mat3,
    l: Limits,
    dt: f32,
    c: &FractureConfig,
) -> Dissipation {
    if b.neighbor == EMPTY {
        return Dissipation::default();
    }
    let old_damage = b.damage;
    let mut u = u_total - Vec3::from_array(b.plastic_strain) * CELL;
    let before = energy(u, k, old_damage);
    let trial = k * u / AREA;
    let mut result = Dissipation::default();
    if l.ductility > 1e-5 {
        let equivalent = trial.length();
        let yield_now = l.yield_stress + l.hardening * b.equivalent_plastic_strain;
        if equivalent > yield_now {
            let direction = trial / equivalent;
            let compliance_direction = k.inverse() * (direction * AREA) / CELL;
            let elastic_modulus = 1.0 / compliance_direction.length().max(1e-20);
            let dep = (equivalent - yield_now) / (elastic_modulus + l.hardening) * l.ductility;
            let dp = compliance_direction.normalize_or_zero() * dep;
            b.plastic_strain = (Vec3::from_array(b.plastic_strain) + dp).to_array();
            b.equivalent_plastic_strain += dep;
            u = u_total - Vec3::from_array(b.plastic_strain) * CELL;
            let lost = (before - energy(u, k, old_damage)).max(0.0);
            b.plastic_work += lost as f32;
            result.plastic_heat = lost * c.plastic_heat_fraction as f64;
            result.stored_plastic = lost - result.plastic_heat;
        }
    }
    // Failure is evaluated after plastic return mapping, not on the elastic predictor.
    // Using the trial stress here tears ductile materials at every strain increment.
    let stress = k * u / AREA;
    let normal_limit = if stress.x >= 0.0 {
        l.tension
    } else {
        l.compression
    };
    let ratio = ((stress.x / normal_limit.max(1.0)).powi(2)
        + (stress.y * stress.y + stress.z * stress.z) / l.shear.max(1.0).powi(2))
    .sqrt();
    let principal = if stress.x.abs() >= stress.y.abs().max(stress.z.abs()) {
        stress.x
    } else if stress.y.abs() >= stress.z.abs() {
        stress.y
    } else {
        stress.z
    };
    fatigue(b, ratio.copysign(principal), c);
    if ratio >= 1.0 {
        if l.ductility <= 1e-5 {
            b.damage = 1.0; // Brittle: no plastic phase, immediate strength failure.
        } else {
            // Mixed materials can fail in shear or tension below their
            // compressive/plastic yield stress. Requiring prior plastic strain
            // here made wood bonds immune whenever (for example) the 6 MPa
            // shear limit was exceeded below the 25 MPa yield threshold.
            b.damage = (b.damage + c.damage_rate * dt * ratio.min(10.0)).min(1.0);
        }
    }
    if l.ductility > 1e-5 && b.equivalent_plastic_strain >= l.plastic_failure.max(1e-6) {
        b.damage = (b.damage + c.damage_rate * dt).min(1.0);
    }
    if b.fatigue_damage >= 1.0 {
        b.damage = 1.0;
    }
    let lost = (energy(u, k, old_damage) - energy(u, k, b.damage)).max(0.0);
    // Strength-driven discrete bond model: never spend more energy than released.
    // Gc is a surface-work budget, not a mesh-objective cohesive traction law yet.
    let surface_budget = l.fracture_energy as f64 * AREA as f64 * (b.damage - old_damage) as f64;
    result.surface_work = lost.min(surface_budget);
    let available = lost - result.surface_work;
    result.released = available * c.energy_release as f64;
    result.damage_heat = available - result.released;
    if b.damage >= 1.0 {
        b.neighbor = EMPTY;
    }
    result
}
/// Equal/opposite central impulse with exactly the requested kinetic-energy gain.
/// Returns zero for two immovable endpoints. No extra impulse is added elsewhere.
pub fn release_impulse(vi: Vec3, vj: Vec3, wi: f32, wj: f32, normal: Vec3, energy: f64) -> Vec3 {
    let w = (wi + wj) as f64;
    if w <= 0.0 || energy <= 0.0 {
        return Vec3::ZERO;
    }
    let relative = (vj - vi).dot(normal) as f64;
    // Choose the root continuous at zero released energy. The positive root
    // would reverse an approaching pair even for an arbitrarily tiny budget.
    // Rationalization also avoids cancellation for small energies.
    let sign = if relative < 0.0 { -1.0 } else { 1.0 };
    let magnitude = sign * 2.0 * energy
        / (relative.abs() + (relative * relative + 2.0 * w * energy).sqrt());
    normal * magnitude as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_release_does_not_reverse_approaching_pair() {
        let vi = Vec3::X;
        let vj = -Vec3::X;
        let impulse = release_impulse(vi, vj, 1.0, 1.0, Vec3::X, 1e-8);
        assert!(impulse.length() < 1e-7);
        assert!(((vj + impulse) - (vi - impulse)).x < 0.0);
        let energy = 0.25;
        let impulse = release_impulse(vi, vj, 1.0, 1.0, Vec3::X, energy);
        let delta = 0.5 * ((vi-impulse).length_squared() + (vj+impulse).length_squared()) - 1.0;
        assert!((delta as f64 - energy).abs() < 1e-6);
    }
    fn config() -> FractureConfig {
        crate::scene::SceneConfig::load("scenes/wood_chop.ron")
            .unwrap()
            .fracture
    }
    fn k(m: &Material) -> Mat3 {
        Mat3::from_diagonal(Vec3::new(m.young, m.shear, m.shear) * AREA / CELL)
    }
    #[test]
    fn brittle_failure_has_no_plasticity() {
        let m = &Material::table()[3];
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let lim = pair_limits(m, m, [1.; 2], [0.; 2]);
        update(
            &mut b,
            Vec3::X * (1.1 * m.tensile_strength / m.young * CELL),
            k(m),
            lim,
            0.001,
            &config(),
        );
        assert_eq!(b.neighbor, EMPTY);
        assert_eq!(b.plastic_strain, [0.; 3]);
    }
    #[test]
    fn metal_yields_heats_and_only_then_tears() {
        let m = &Material::table()[2];
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let lim = pair_limits(m, m, [1.; 2], [0.; 2]);
        let mut heat = 0.;
        for n in 1..=300 {
            let d = update(
                &mut b,
                Vec3::X * (n as f32 * 0.001 * CELL),
                k(m),
                lim,
                0.001,
                &config(),
            );
            heat += d.plastic_heat;
            if n == 2 {
                assert_ne!(b.neighbor, EMPTY);
                assert!(b.equivalent_plastic_strain > 0.);
            }
            if b.neighbor == EMPTY {
                break;
            }
        }
        assert_eq!(b.neighbor, EMPTY);
        assert!(heat > 0.);
        assert!(b.equivalent_plastic_strain > 0.1);
    }
    #[test]
    fn energy_removed_is_not_duplicated() {
        let m = &Material::table()[3];
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let u = Vec3::X * (2. * m.tensile_strength / m.young * CELL);
        let before = energy(u, k(m), 0.);
        let d = update(
            &mut b,
            u,
            k(m),
            pair_limits(m, m, [1.; 2], [0.; 2]),
            0.001,
            &config(),
        );
        assert!((d.total() - before).abs() < 1e-8);
    }
    #[test]
    fn release_conserves_momentum_and_adds_only_budgeted_energy() {
        let (mi, mj) = (0.02f32, 0.03f32);
        let vi = Vec3::new(1., 2., 3.);
        let vj = Vec3::new(-2., 0., 1.);
        let n = Vec3::new(1., 2., -1.).normalize();
        let budget = 0.017;
        let impulse = release_impulse(vi, vj, mi.recip(), mj.recip(), n, budget);
        let a = vi - impulse / mi;
        let b = vj + impulse / mj;
        assert!((a * mi + b * mj - vi * mi - vj * mj).length() < 1e-6);
        let gain = 0.5
            * ((mi * a.length_squared() + mj * b.length_squared())
                - (mi * vi.length_squared() + mj * vj.length_squared()));
        assert!((gain as f64 - budget).abs() < 1e-7);
        assert!(n.cross(impulse).length() < 1e-7);
    }
    #[test]
    fn sustained_subcritical_load_does_not_count_as_cycles() {
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let c = config();
        for _ in 0..10000 {
            fatigue(&mut b, 0.8, &c);
        }
        assert_eq!(b.fatigue_damage, 0.);
    }
    #[test]
    fn cyclic_subcritical_load_accumulates_damage() {
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let mut c = config();
        c.fatigue_cycles = 10.;
        c.fatigue_exponent = 1.;
        for _ in 0..50 {
            fatigue(&mut b, 0.8, &c);
            fatigue(&mut b, -0.8, &c);
        }
        assert_eq!(b.fatigue_damage, 1.);
    }
    #[test]
    fn wood_has_tension_and_compression_anisotropy() {
        let m = &Material::table()[1];
        let along = pair_limits(m, m, [1.; 2], [1.; 2]);
        let across = pair_limits(m, m, [0.; 2], [1.; 2]);
        assert!(along.tension / across.tension >= 10.);
        assert!(along.compression / across.compression >= 10.);
        assert_eq!(across.ductility, 0.);
        assert_eq!(along.ductility, 1.);
    }
    #[test]
    fn mixed_wood_shear_failure_does_not_wait_for_higher_yield_stress() {
        let m = &Material::table()[1];
        let mut b = Bond {
            neighbor: 1,
            ..Default::default()
        };
        let limits = pair_limits(m, m, [0.5; 2], [1.; 2]);
        assert!(limits.ductility > 0.0);
        let shear_strain = 1.1 * limits.shear / m.shear * CELL;
        update(
            &mut b,
            Vec3::Y * shear_strain,
            k(m),
            limits,
            0.001,
            &config(),
        );
        assert_eq!(b.equivalent_plastic_strain, 0.0);
        assert!(b.damage > 0.0, "above-limit shear must start bond damage");
        assert_eq!(b.neighbor, 1, "damage accumulates before separation");
    }
}
