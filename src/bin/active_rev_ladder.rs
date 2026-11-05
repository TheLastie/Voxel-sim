//! Repeatable headless benchmark for the 1k/16k/50k active REV gates.
//!
//! Each case is an ordinary generated impact scene: two equal iron cubes that
//! touch and approach each other, with automatic aggregation disabled. It times
//! `solver::step` only, without worker delivery, metrics or rendering, so it is
//! not the full 5 ms acceptance measurement. Solver stage timings come from the
//! engine's own step diagnostics; stderr carries the engine's profile lines.
//!
//! Usage: active_rev_ladder [--sides 8,20,30] [--steps 20] [--pre ic0,amg]
//!                          [--limit 80] [--gap-m 0] [--speed 0.2] [--out DIR]
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
use voxel_physics_lab::{
    scene::CpuPreconditioner,
    scene_presets::{create_impact_scene_with_options, ImpactSceneOptions},
    solver,
};

struct Args {
    sides: Vec<usize>,
    steps: usize,
    preconditioners: Vec<String>,
    limit: usize,
    gap_m: f64,
    speed: f32,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        sides: vec![8, 20, 30],
        steps: 20,
        preconditioners: vec!["ic0".into(), "amg".into()],
        limit: 80,
        gap_m: 0.0,
        speed: 0.2,
        out: PathBuf::from("target/active-rev-ladder"),
    };
    let mut iter = std::env::args().skip(1);
    while let Some(key) = iter.next() {
        let value = iter.next().ok_or_else(|| format!("{key} requires a value"))?;
        match key.as_str() {
            "--sides" => {
                args.sides = value
                    .split(',')
                    .map(|side| side.parse().map_err(|e| format!("invalid side: {e}")))
                    .collect::<Result<_, _>>()?
            }
            "--steps" => args.steps = value.parse().map_err(|e| format!("invalid steps: {e}"))?,
            "--pre" => args.preconditioners = value.split(',').map(str::to_owned).collect(),
            "--limit" => args.limit = value.parse().map_err(|e| format!("invalid limit: {e}"))?,
            "--gap-m" => args.gap_m = value.parse().map_err(|e| format!("invalid gap: {e}"))?,
            "--speed" => args.speed = value.parse().map_err(|e| format!("invalid speed: {e}"))?,
            "--out" => args.out = PathBuf::from(value),
            _ => return Err(format!("unknown argument: {key}")),
        }
    }
    if args.steps == 0 {
        return Err("--steps must be positive".into());
    }
    Ok(args)
}

fn preconditioner(name: &str) -> Result<CpuPreconditioner, String> {
    match name {
        "ic0" => Ok(CpuPreconditioner::IncompleteCholesky0),
        "jacobi" => Ok(CpuPreconditioner::BlockJacobi),
        "amg" => Ok(CpuPreconditioner::AggregationMultigrid),
        other => Err(format!("unknown preconditioner: {other}")),
    }
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    std::env::set_var("VOXEL_DISABLE_AUTO_AGGREGATION", "1");
    std::env::set_var("VOXEL_PROFILE_STEP", "1");
    std::fs::create_dir_all(&args.out).map_err(|e| e.to_string())?;
    for &side in &args.sides {
        let path = args
            .out
            .join(format!("cubes-{side}-{}.vxscene", std::process::id()));
        let document = create_impact_scene_with_options(
            &path,
            &ImpactSceneOptions {
                side,
                gap_m: args.gap_m,
                approach_speed_m_s: args.speed,
                ..Default::default()
            },
        )?;
        for name in &args.preconditioners {
            let mut scene = document.build_scene(&path)?;
            scene.config.cpu_preconditioner = preconditioner(name)?;
            scene.config.iterations = args.limit;
            let active = scene.voxels.iter().filter(|v| !v.kinematic).count();
            let mut wall = Vec::with_capacity(args.steps);
            let mut iterations = Vec::with_capacity(args.steps);
            let mut contact_iterations = Vec::with_capacity(args.steps);
            let mut worst_residual = 0.0f32;
            let mut peak_contacts = 0;
            let mut stages = BTreeMap::<&'static str, f64>::new();
            for _ in 0..args.steps {
                let start = Instant::now();
                solver::step(&mut scene, &mut None);
                wall.push(start.elapsed().as_secs_f64() * 1000.0);
                worst_residual = worst_residual.max(scene.solver_residual);
                if let Some(status) = &scene.last_step_diagnostics {
                    iterations.push(status.pcg_iterations_total);
                    contact_iterations.push(status.contact_pcg_iterations_including_restitution);
                    peak_contacts = peak_contacts.max(status.contacts_peak);
                    for (stage, ms) in &status.timings_ms {
                        *stages.entry(*stage).or_default() += ms;
                    }
                }
                if !scene.solver_residual.is_finite() {
                    return Err(format!("non-finite solve: side={side} pre={name}"));
                }
            }
            let first = wall[0];
            let mut sorted = wall.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "CASE side={side} active={active} pre={name} limit={} steps={} first_ms={first:.1} p50_ms={:.1} p95_ms={:.1} max_ms={:.1} pcg_iters_total={} contact_iters_total={} peak_contacts={peak_contacts} worst_residual={worst_residual:.3e} scope=headless_solver",
                args.limit,
                args.steps,
                percentile(&sorted, 0.5),
                percentile(&sorted, 0.95),
                sorted[sorted.len() - 1],
                iterations.iter().sum::<usize>(),
                contact_iterations.iter().sum::<usize>(),
            );
            let steps = args.steps as f64;
            let line: Vec<_> = [
                "prediction_tool",
                "mechanics_total",
                "matrix_assembly",
                "factorization",
                "pcg",
                "rotation_fit",
                "collision",
                "fracture",
                "finish_total",
                "thermal",
            ]
            .iter()
            .map(|stage| format!("{stage}={:.1}", stages.get(stage).copied().unwrap_or(0.0) / steps))
            .collect();
            println!("MEAN_STAGES_MS side={side} pre={name} {}", line.join(" "));
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
