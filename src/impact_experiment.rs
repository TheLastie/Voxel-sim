//! Reusable engine-side two-cube impact experiment.
//!
//! The callback receives the initial state (`step == 0`) and each actual
//! fixed-step state. Positions are the solver's positions and are never
//! interpolated.

use std::time::{Duration, Instant};

use glam::{DVec3, Vec3};

use crate::{
    material::Material,
    scene::{Scene, SceneBuilder, VoxelSpawn, CELL},
};

#[derive(Clone, Copy, Debug)]
pub struct ImpactConfig {
    pub side: usize,
    pub steps: usize,
    pub dt: f32,
    pub speed: f32,
    pub gap: f64,
    pub substeps: usize,
}

impl Default for ImpactConfig {
    fn default() -> Self {
        Self {
            side: 74,
            steps: 4,
            dt: 0.01,
            speed: 2.0,
            gap: 0.08,
            substeps: 4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImpactReport {
    pub passed: bool,
    pub collision_observed: bool,
    pub momentum_conserved: bool,
    pub no_energy_creation: bool,
    pub detail: String,
    pub simulated_seconds: f64,
    pub voxel_count: usize,
    pub max_step_ms: f64,
}

struct ImpactSetup {
    scene: Scene,
    left: Vec<usize>,
    right: Vec<usize>,
}

/// Build the production scene used by the impact experiment.
pub fn build_impact_scene(config: ImpactConfig) -> Result<Scene, String> {
    build_setup(config).map(|setup| setup.scene)
}

fn build_setup(config: ImpactConfig) -> Result<ImpactSetup, String> {
    validate_config(config)?;
    let side = i32::try_from(config.side).map_err(|_| "side exceeds the voxel coordinate range")?;
    let half = config
        .side
        .checked_pow(3)
        .ok_or("voxel count overflow for one cube")?;
    half.checked_mul(2).ok_or("total voxel count overflow")?;
    let gap_cells = (((config.gap / f64::from(CELL)) - 1.0e-6).ceil() as i32).max(1);
    let right_start = side
        .checked_add(gap_cells)
        .ok_or("right cube coordinate exceeds i32")?;
    right_start
        .checked_add(side - 1)
        .ok_or("right cube coordinate exceeds i32")?;
    let iron = Material::component_id("iron").ok_or("iron material is unavailable")?;
    let mut builder = SceneBuilder::standard()
        .with_gravity(Vec3::ZERO)
        .with_timestep(config.dt, config.substeps);
    let left_velocity = Vec3::new(config.speed, 0.0, 0.0);
    let right_velocity = -left_velocity;
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                builder = builder
                    .add_voxel(VoxelSpawn::at_cell([x, y, z], iron).with_velocity(left_velocity));
            }
        }
    }
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                builder = builder.add_voxel(
                    VoxelSpawn::at_cell([right_start + x, y, z], iron)
                        .with_velocity(right_velocity),
                );
            }
        }
    }
    let mut scene = builder.build()?;
    let mut x_centers: Vec<_> = scene.positions.iter().map(|position| position.x).collect();
    x_centers.sort_by(f64::total_cmp);
    let split_x = (x_centers[half - 1] + x_centers[half]) * 0.5;
    let (left, right): (Vec<_>, Vec<_>) = scene
        .positions
        .iter()
        .enumerate()
        .partition(|(_, position)| position.x < split_x);
    let left: Vec<_> = left.into_iter().map(|(index, _)| index).collect();
    let right: Vec<_> = right.into_iter().map(|(index, _)| index).collect();
    if left.len() != half || right.len() != half {
        return Err("scene builder did not preserve the two spatial cube groups".into());
    }
    // Coherent nonzero initial velocity prevents quiescent auto-discovery.
    // Register each intact component through the normal conservative rigid tier.
    scene
        .register_rigid_aggregate_candidate(&left)
        .map_err(|error| error.to_string())?;
    scene
        .register_rigid_aggregate_candidate(&right)
        .map_err(|error| error.to_string())?;
    scene.set_adaptive_substeps(true);
    Ok(ImpactSetup { scene, left, right })
}

fn validate_config(config: ImpactConfig) -> Result<(), String> {
    if config.side == 0
        || config.steps == 0
        || config.substeps == 0
        || !config.dt.is_finite()
        || config.dt <= 0.0
        || !config.speed.is_finite()
        || config.speed <= 0.0
        || !config.gap.is_finite()
        || config.gap <= 0.0
    {
        return Err("side, steps, substeps, dt, speed, and gap must be positive and finite".into());
    }
    Ok(())
}

fn center_velocity(scene: &Scene, members: &[usize]) -> DVec3 {
    let (momentum, mass) = members
        .iter()
        .copied()
        .fold((DVec3::ZERO, 0.0), |(sum, total), i| {
            let m = f64::from(scene.voxels[i].mass);
            (sum + scene.voxels[i].velocity.as_dvec3() * m, total + m)
        });
    momentum / mass
}

fn system_momentum(scene: &Scene) -> DVec3 {
    scene.voxels.iter().fold(DVec3::ZERO, |sum, voxel| {
        sum + voxel.velocity.as_dvec3() * f64::from(voxel.mass)
    })
}

fn kinetic_energy_j(scene: &Scene) -> f64 {
    scene.voxels.iter().fold(0.0, |sum, voxel| {
        sum + 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared())
    })
}

fn accounted_energy_j(scene: &Scene) -> f64 {
    kinetic_energy_j(scene)
        + scene.elastic_bond_energy_j()
        + scene.stored_plastic_bond_energy_j()
        + scene.energy.surface_work
        + scene.energy.pending_release
}

fn internal_energy_delta_j(scene: &Scene, initial: &[f64]) -> f64 {
    scene
        .voxels
        .iter()
        .zip(initial)
        .map(|(voxel, baseline)| voxel.total_internal_energy_j() - baseline)
        .sum()
}

fn surface_gap_x(scene: &Scene, left: &[usize], right: &[usize]) -> f64 {
    let left_max = left
        .iter()
        .map(|&i| scene.positions[i].x + f64::from(CELL) * 0.5)
        .fold(f64::NEG_INFINITY, f64::max);
    let right_min = right
        .iter()
        .map(|&i| scene.positions[i].x - f64::from(CELL) * 0.5)
        .fold(f64::INFINITY, f64::min);
    right_min - left_max
}

fn state_is_finite(scene: &Scene) -> bool {
    scene.positions.iter().all(|position| position.is_finite())
        && scene.voxels.iter().all(|voxel| voxel.velocity.is_finite())
}

/// Run two free iron cubes through the normal scene tick and report measured checks.
pub fn run_impact(
    config: ImpactConfig,
    mut on_step: impl FnMut(usize, &Scene, f64),
) -> Result<ImpactReport, String> {
    validate_config(config)?;
    let mut setup = build_setup(config)?;
    let scene = &mut setup.scene;
    let start = Instant::now();
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let voxel_count = scene.voxels.len();
    let actual_gap = surface_gap_x(scene, &setup.left, &setup.right);
    let relative_speed = 2.0 * f64::from(config.speed);
    let first_contact_s = actual_gap / relative_speed;
    let initial_momentum = system_momentum(scene);
    let initial_energy = accounted_energy_j(scene);
    let initial_kinetic = kinetic_energy_j(scene);
    // Compare per-REV changes so the large thermal baseline cannot mask impact energy.
    let initial_internal: Vec<_> = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .collect();
    let initial_relative_speed =
        center_velocity(scene, &setup.left).x - center_velocity(scene, &setup.right).x;
    let mut max_energy_excess = 0.0_f64;
    let mut max_momentum_error = 0.0_f64;
    let mut minimum_gap = actual_gap;
    let mut max_step_ms = 0.0_f64;
    let live = crate::live_tests::enabled();

    on_step(0, scene, setup_ms);
    if live {
        crate::live_tests::observe_scene_force(scene, "cubes-impact", 0.0, setup_ms);
    }
    let mut tool = None;
    let mut finite = true;
    let mut last_wall_ms = 0.0;
    for step in 1..=config.steps {
        let clock = Instant::now();
        scene.tick_with_tool(&mut tool);
        last_wall_ms = clock.elapsed().as_secs_f64() * 1000.0;
        max_step_ms = max_step_ms.max(last_wall_ms);
        finite = state_is_finite(scene);
        on_step(step, scene, last_wall_ms);
        if live {
            crate::live_tests::observe_scene_force(
                scene,
                "cubes-impact",
                scene.simulated_seconds,
                last_wall_ms,
            );
        }
        if !finite {
            break;
        }
        minimum_gap = minimum_gap.min(surface_gap_x(scene, &setup.left, &setup.right));
        max_momentum_error =
            max_momentum_error.max((system_momentum(scene) - initial_momentum).length());
        let energy_excess = accounted_energy_j(scene) - initial_energy
            + internal_energy_delta_j(scene, &initial_internal);
        max_energy_excess = max_energy_excess.max(energy_excess);
    }
    if live {
        crate::live_tests::observe_scene_force(
            scene,
            "cubes-impact",
            scene.simulated_seconds,
            last_wall_ms,
        );
        if !crate::live_tests::flush(Duration::from_secs(2)) {
            eprintln!("live viewer: final impact frame was not confirmed written");
        }
    }

    let momentum_error = (system_momentum(scene) - initial_momentum).length();
    let energy_excess = accounted_energy_j(scene) - initial_energy
        + internal_energy_delta_j(scene, &initial_internal);
    let final_relative_speed =
        center_velocity(scene, &setup.left).x - center_velocity(scene, &setup.right).x;
    let ran_past_contact = scene.simulated_seconds >= first_contact_s;
    let momentum_tolerance = (scene.total_mass_kg() * relative_speed * 1.0e-4).max(1e-8);
    let energy_tolerance = (initial_kinetic.abs() * 1.0e-5).max(1e-8);
    let collision_observed = finite
        && ran_past_contact
        && minimum_gap <= 0.002
        && (final_relative_speed - initial_relative_speed).abs() >= 1.0e-3;
    let momentum_conserved = finite && max_momentum_error <= momentum_tolerance;
    let no_energy_creation = finite && max_energy_excess <= energy_tolerance;
    let mut failures = Vec::new();
    if !finite {
        failures.push("nonfinite physical state".to_string());
    }
    if !collision_observed {
        failures.push("collision was not observed".to_string());
    }
    if !momentum_conserved {
        failures.push(format!(
            "momentum error {max_momentum_error:.6e} kg m/s exceeds tolerance {momentum_tolerance:.6e}"
        ));
    }
    if !no_energy_creation {
        failures.push(format!(
            "energy excess {max_energy_excess:.6e} J exceeds tolerance {energy_tolerance:.6e}"
        ));
    }
    let passed = failures.is_empty();
    let detail = format!(
        "{}; expected_contact_s={first_contact_s:.6}; minimum_x_gap_m={minimum_gap:.9}; relative_speed_x_m_s={initial_relative_speed:.6}->{final_relative_speed:.6}; final_momentum_error_kgm_s={momentum_error:.6e}; max_momentum_error_kgm_s={max_momentum_error:.6e}; momentum_tolerance_kgm_s={momentum_tolerance:.6e}; momentum_conserved={momentum_conserved}; energy_excess_j={energy_excess:.6e}; max_energy_excess_j={max_energy_excess:.6e}; energy_tolerance_j={energy_tolerance:.6e}; initial_kinetic_energy_j={initial_kinetic:.6e}; no_energy_creation={no_energy_creation}; finite={finite}; broken={}; clusters={}",
        if passed { "passed".to_string() } else { format!("failed: {}", failures.join(", ")) },
        scene.broken,
        scene.cluster_count,
    );
    Ok(ImpactReport {
        passed,
        collision_observed,
        momentum_conserved,
        no_energy_creation,
        detail,
        simulated_seconds: scene.simulated_seconds,
        voxel_count,
        max_step_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_has_equal_opposite_impulse_and_requested_gap() {
        let config = ImpactConfig {
            side: 2,
            steps: 1,
            dt: 0.001,
            speed: 1.25,
            gap: 0.02,
            substeps: 4,
        };
        let setup = build_setup(config).unwrap();
        let scene = &setup.scene;
        let p_left = center_velocity(scene, &setup.left)
            * setup
                .left
                .iter()
                .map(|&i| f64::from(scene.voxels[i].mass))
                .sum::<f64>();
        let p_right = center_velocity(scene, &setup.right)
            * setup
                .right
                .iter()
                .map(|&i| f64::from(scene.voxels[i].mass))
                .sum::<f64>();
        assert!((p_left + p_right).length() < 1e-12);
        assert!(
            (surface_gap_x(scene, &setup.left, &setup.right) - config.gap).abs()
                < f64::from(CELL) * 1e-6
        );
        assert!(scene.adaptive_substeps());
        assert_eq!(setup.left.len(), config.side.pow(3));
        assert_eq!(setup.right.len(), config.side.pow(3));
        assert!(scene.positions.iter().all(|position| position.is_finite()));
    }

    #[test]
    fn callback_observes_initial_and_actual_steps() {
        let config = ImpactConfig {
            side: 2,
            steps: 2,
            dt: 0.001,
            ..ImpactConfig::default()
        };
        let mut observed = Vec::new();
        let report = run_impact(config, |step, scene, _| {
            observed.push((step, scene.simulated_seconds));
        })
        .unwrap();
        assert_eq!(observed.len(), 3);
        assert_eq!(observed[0], (0, 0.0));
        assert_eq!(observed[1].0, 1);
        assert_eq!(observed[2].0, 2);
        assert_eq!(report.voxel_count, 16);
    }
}
