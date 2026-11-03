//! Reproduce the native contact scene and expose its actual numerical state.
use glam::{DVec3, Vec3};
use serde::Serialize;
use voxel_physics_lab::{
    bond::{Bond, BondWorkspace, EMPTY},
    scene::{CpuPreconditioner, Scene, SceneConfig, VoxelSpawn, CELL},
    solver,
    world::GeneratedWorld,
    world_physics::contact_experiment,
};

#[derive(Serialize)]
struct Snapshot {
    schema_version: u32,
    seed: &'static str,
    global_coordinates: bool,
    steps: usize,
    simulated_seconds: f64,
    fixed_dt_s: f32,
    substeps: usize,
    adaptive_substeps: bool,
    timing_only: bool,
    mechanical_substeps_per_step: Vec<usize>,
    solver_iterations: usize,
    solver_relative_tolerance: f32,
    cpu_preconditioner: String,
    voxel_count: usize,
    dynamic_voxel_count: usize,
    total_mass_kg: f64,
    initial: StateSnapshot,
    final_state: StateSnapshot,
    step_ms: Vec<f64>,
    timing: TimingSummary,
}

#[derive(Serialize)]
struct StateSnapshot {
    positions_m: Vec<[f64; 3]>,
    velocities_m_s: Vec<[f32; 3]>,
    masses_kg: Vec<f32>,
    internal_energy_j: Vec<f64>,
    total_internal_energy_j: f64,
    kinetic_energy_j: f64,
    elastic_energy_j: f64,
    broken_bonds: usize,
    bond_damage: Vec<f32>,
    bond_stress_proxy_pa: Vec<f32>,
    finite: bool,
}

#[derive(Serialize)]
struct TimingSummary {
    wall_seconds: f64,
    solver_compute_seconds: f64,
    diagnostic_overhead_seconds: f64,
    mean_step_ms: f64,
    p95_step_ms: f64,
    max_step_ms: f64,
    wall_to_simulated_time_ratio: f64,
    physics_realtime_1x: bool,
    solver_compute_realtime_1x: bool,
}

fn state_snapshot(scene: &Scene) -> StateSnapshot {
    let kinetic_energy_j = scene.voxels.iter().fold(0.0, |sum, voxel| {
        sum + 0.5 * f64::from(voxel.mass) * f64::from(voxel.velocity.length_squared())
    });
    let internal_energy_j: Vec<_> = scene
        .voxels
        .iter()
        .map(|voxel| voxel.total_internal_energy_j())
        .collect();
    let mut bond_damage = Vec::new();
    let mut bond_stress_proxy_pa = Vec::new();
    let mut broken_bonds = 0;
    for axis in 0..3 {
        for (i, bond) in scene.bonds[axis].iter().enumerate() {
            if scene.workspace[axis][i].rest_neighbor == EMPTY {
                continue;
            }
            let broken = bond.neighbor == EMPTY;
            broken_bonds += usize::from(broken);
            bond_damage.push(if broken { 1.0 } else { bond.damage });
            bond_stress_proxy_pa.push(scene.workspace[axis][i].last_force.length() / (CELL * CELL));
        }
    }
    let finite = scene.positions.iter().all(|p| p.is_finite())
        && scene.voxels.iter().all(|v| v.velocity.is_finite())
        && internal_energy_j.iter().all(|e| e.is_finite())
        && bond_damage.iter().all(|d| d.is_finite())
        && bond_stress_proxy_pa.iter().all(|s| s.is_finite());
    StateSnapshot {
        positions_m: scene.positions.iter().map(|p| p.to_array()).collect(),
        velocities_m_s: scene.voxels.iter().map(|v| v.velocity.to_array()).collect(),
        masses_kg: scene.voxels.iter().map(|v| v.mass).collect(),
        total_internal_energy_j: internal_energy_j.iter().sum(),
        internal_energy_j,
        kinetic_energy_j,
        elastic_energy_j: scene.elastic_bond_energy_j(),
        broken_bonds,
        bond_damage,
        bond_stress_proxy_pa,
        finite,
    }
}

fn percentile95(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[(sorted.len() * 95).div_ceil(100) - 1]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--memory-layout") {
        use std::mem::size_of;
        use voxel_physics_lab::{material::Material, voxel::Voxel};
        // Inline slots in Scene's persistent arrays only. Excludes allocator
        // overhead, strings/composition heaps, capacities, solver/thermal
        // scratch, aggregates, spatial indices, meshes and GPU buffers.
        let persistent = size_of::<Voxel>()
            + 3 * size_of::<Bond>() + 3 * size_of::<BondWorkspace>()
            + size_of::<Material>() + 3 * size_of::<DVec3>()
            + size_of::<glam::Quat>() + 2 * size_of::<Vec3>()
            + size_of::<[Option<f64>; 3]>() + size_of::<f32>()
            + size_of::<u32>() + size_of::<Option<[i32; 3]>>()
            + size_of::<Option<usize>>() + 3 * size_of::<usize>();
        println!("{}", serde_json::json!({
            "voxel_inline_bytes": size_of::<Voxel>(),
            "bond_inline_bytes": size_of::<Bond>(),
            "bond_workspace_inline_bytes": size_of::<BondWorkspace>(),
            "derived_material_inline_bytes": size_of::<Material>(),
            "persistent_array_slots_bytes_per_rev": persistent,
            "nonuniform_thermal_extra_bytes_per_rev": size_of::<[f64; 27]>(),
            "scope": "inline persistent Scene array slots; excludes heap payloads, spare capacity, all scratch, geometry and GPU allocations; uniform thermal fields do not allocate 27 cells"
        }));
        return Ok(());
    }
    let steps = args
        .iter()
        .skip(1)
        .find_map(|arg| {
            arg.strip_prefix("--steps=")
                .and_then(|s| s.parse::<usize>().ok())
        })
        .or_else(|| {
            args.windows(2)
                .find(|pair| pair[0] == "--steps")
                .and_then(|pair| pair[1].parse::<usize>().ok())
        })
        .or_else(|| {
            args.iter()
                .skip(1)
                .find_map(|arg| arg.parse::<usize>().ok())
        })
        .unwrap_or(40);
    let state_out = args
        .windows(2)
        .find(|p| p[0] == "--state-out")
        .map(|p| p[1].as_str());
    let global = args.iter().any(|s| s == "--global");
    let world = GeneratedWorld::from_seed("1001");
    let region = contact_experiment(
        &world,
        Vec3::ZERO,
        SceneConfig::load("scenes/mounted_axe.ron")?,
    )?;
    eprintln!(
        "origin={:?} global={} REVs={}",
        region.origin,
        global,
        region.scene.voxels.len()
    );
    let mut scene = if global {
        let offset = (region.origin / CELL).round().as_ivec3();
        let spawns = region
            .scene
            .voxels
            .iter()
            .zip(&region.scene.positions)
            .map(|(v, &p)| {
                let cell = (p / (CELL as f64) - DVec3::splat(0.5)).round().as_ivec3() + offset;
                VoxelSpawn {
                    center: VoxelSpawn::at_cell(cell.to_array(), 0).center,
                    composition: v.composition.clone(),
                    velocity: v.velocity,
                    microstructure: v.microstructure.clone(),
                    kinematic: v.kinematic,
                    temperature_kelvin: 293.15,
                }
            })
            .collect();
        Scene::from_voxels(region.scene.config.clone(), spawns)?
    } else {
        region.scene
    };
    if args.iter().any(|s| s == "--block-jacobi") {
        scene.config.cpu_preconditioner = CpuPreconditioner::BlockJacobi;
    }
    eprintln!("cpu_preconditioner={:?}", scene.config.cpu_preconditioner);
    if let Some(substeps) = args.windows(2).find(|p| p[0] == "--substeps") {
        scene.config.substeps = substeps[1].parse()?;
        scene.set_adaptive_substeps(false);
    }
    if args.iter().any(|s| s == "--adaptive") {
        scene.set_adaptive_substeps(true);
    }
    if args.iter().any(|s| s == "--fixed-substeps") {
        scene.set_adaptive_substeps(false);
    }
    let mut max_error = 0.0_f64;
    for (axis, bonds) in scene.bonds.iter().enumerate() {
        for (i, bond) in bonds.iter().enumerate() {
            if bond.neighbor == EMPTY || scene.voxels[i].kinematic {
                continue;
            }
            let mut ideal = DVec3::ZERO;
            ideal[axis] = CELL as f64;
            max_error = max_error.max(
                (scene.positions[bond.neighbor as usize] - scene.positions[i] - ideal).length(),
            );
        }
    }
    eprintln!(
        "max_initial_bond_offset_error_m={max_error:e} substeps={}",
        scene.config.substeps
    );
    let initial_positions = scene.positions.clone();
    let initial_snapshot = state_snapshot(&scene);
    let initial_internal: f64 = scene
        .voxels
        .iter()
        .map(|v| v.total_internal_energy_j())
        .sum();
    let initial_elastic = scene.elastic_bond_energy_j();
    let timing_only = args.iter().any(|arg| arg == "--timing-only");
    if timing_only {
        println!("step,time_s,step_ms,mechanical_substeps");
    } else {
        println!("step,time_s,broken,max_speed_mps,kinetic_j,damage_heat_j,surface_work_j,plastic_heat_j,contact_heat_j,pending_j,release_j,residual,step_ms,solid_heat_j,internal_delta_j,elastic_j,potential_delta_j,balance_error_j");
    }
    let mut tool = None;
    let wall_start = std::time::Instant::now();
    let mut step_times_ms = Vec::with_capacity(steps);
    let mut mechanical_substeps = Vec::with_capacity(steps);
    for step in 1..=steps {
        let start = std::time::Instant::now();
        solver::step(&mut scene, &mut tool);
        let step_ms = start.elapsed().as_secs_f64() * 1000.0;
        step_times_ms.push(step_ms);
        mechanical_substeps.push(scene.last_mechanical_substeps());
        if scene.positions.iter().any(|p| !p.is_finite()) {
            return Err("nonfinite state".into());
        }
        if timing_only {
            println!("{step},{:.6},{step_ms:.3},{}",
                step as f64 * f64::from(scene.config.fixed_dt),
                scene.last_mechanical_substeps());
            continue;
        }
        let kinetic: f64 = scene
            .voxels
            .iter()
            .map(|v| 0.5 * f64::from(v.mass) * f64::from(v.velocity.length_squared()))
            .sum();
        let speed = scene
            .voxels
            .iter()
            .map(|v| v.velocity.length())
            .fold(0.0_f32, f32::max);
        let internal_delta = scene
            .voxels
            .iter()
            .map(|v| v.total_internal_energy_j())
            .sum::<f64>()
            - initial_internal;
        let elastic = scene.elastic_bond_energy_j();
        let gravity = Vec3::from_array(scene.config.gravity);
        let potential_delta: f64 = scene
            .positions
            .iter()
            .zip(&initial_positions)
            .zip(&scene.voxels)
            .filter(|(_, v)| !v.kinematic)
            .map(|((&p, &old), v)| -f64::from(v.mass) * gravity.as_dvec3().dot(p - old))
            .sum();
        let balance = kinetic + elastic - initial_elastic
            + internal_delta
            + potential_delta
            + scene.stored_plastic_bond_energy_j()
            + scene.energy.surface_work
            + scene.energy.pending_release;
        println!("{step},{:.6},{},{speed:.6},{kinetic:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.4e},{:.3},{:.9e},{internal_delta:.9e},{elastic:.9e},{potential_delta:.9e},{balance:.9e}",
            step as f64*f64::from(scene.config.fixed_dt),scene.broken,scene.energy.damage_heat,
            scene.energy.surface_work,scene.energy.plastic_heat,
            scene.energy.contact_friction_heat+scene.energy.contact_restitution_heat,
            scene.energy.pending_release,scene.energy.kinetic_release,scene.solver_residual,
            step_ms,scene.energy.solid_viscoelastic_heat);
    }
    let wall_seconds = wall_start.elapsed().as_secs_f64();
    let simulated_seconds = steps as f64 * f64::from(scene.config.fixed_dt);
    let final_snapshot = state_snapshot(&scene);
    let solver_compute_seconds = step_times_ms.iter().sum::<f64>() / 1000.0;
    let mean_step_ms = if step_times_ms.is_empty() {
        0.0
    } else {
        step_times_ms.iter().sum::<f64>() / step_times_ms.len() as f64
    };
    let summary = TimingSummary {
        wall_seconds,
        solver_compute_seconds,
        diagnostic_overhead_seconds: (wall_seconds - solver_compute_seconds).max(0.0),
        mean_step_ms,
        p95_step_ms: percentile95(&step_times_ms),
        max_step_ms: step_times_ms.iter().copied().fold(0.0_f64, f64::max),
        wall_to_simulated_time_ratio: wall_seconds / simulated_seconds.max(f64::MIN_POSITIVE),
        physics_realtime_1x: wall_seconds <= simulated_seconds,
        solver_compute_realtime_1x: solver_compute_seconds <= simulated_seconds,
    };
    eprintln!("timing wall_s={wall_seconds:.6} solver_s={solver_compute_seconds:.6} simulated_s={simulated_seconds:.6} wall_to_sim_ratio={:.3} mean_step_ms={mean_step_ms:.3} p95_step_ms={:.3} max_step_ms={:.3} physics_realtime_1x={} solver_compute_realtime_1x={} timing_only={timing_only} (headless; no FPS claim)", summary.wall_to_simulated_time_ratio, summary.p95_step_ms, summary.max_step_ms, summary.physics_realtime_1x, summary.solver_compute_realtime_1x);
    let last_step_ms = step_times_ms.last().copied().unwrap_or(0.0);
    if let Some(path) = state_out {
        let snapshot = Snapshot {
            schema_version: 1,
            seed: "1001",
            global_coordinates: global,
            steps,
            simulated_seconds,
            fixed_dt_s: scene.config.fixed_dt,
            substeps: scene.config.substeps,
            adaptive_substeps: scene.adaptive_substeps(),
            timing_only,
            mechanical_substeps_per_step: mechanical_substeps,
            solver_iterations: scene.config.iterations,
            solver_relative_tolerance: scene.config.solver_relative_tolerance,
            cpu_preconditioner: format!("{:?}", scene.config.cpu_preconditioner),
            voxel_count: scene.voxels.len(),
            dynamic_voxel_count: scene.voxels.iter().filter(|v| !v.kinematic).count(),
            total_mass_kg: scene.total_mass_kg(),
            initial: initial_snapshot,
            final_state: final_snapshot,
            step_ms: step_times_ms,
            timing: summary,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&snapshot)?)?;
        eprintln!("snapshot={path}");
    }
    if voxel_physics_lab::live_tests::enabled() {
        voxel_physics_lab::live_tests::observe_scene_force(
            &scene, "", scene.simulated_seconds, last_step_ms,
        );
        if !voxel_physics_lab::live_tests::flush(std::time::Duration::from_secs(2)) {
            eprintln!("live viewer: final frame was not confirmed written");
        }
    }
    Ok(())
}
