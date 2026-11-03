//! Thin command-line runner for the engine impact experiment.
use std::env;
use voxel_physics_lab::impact_experiment::{run_impact, ImpactConfig};

fn parse_args() -> Result<ImpactConfig, String> {
    let mut config = ImpactConfig::default();
    let mut iter = env::args().skip(1);
    while let Some(key) = iter.next() {
        if key == "--help" || key == "-h" {
            println!("cubes_impact_probe [--side N] [--steps N] [--dt SECONDS] [--speed M_PER_S] [--gap METERS] [--substeps N]");
            std::process::exit(0);
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("{key} requires a value"))?;
        match key.as_str() {
            "--side" => config.side = value.parse().map_err(|e| format!("invalid --side: {e}"))?,
            "--steps" => {
                config.steps = value.parse().map_err(|e| format!("invalid --steps: {e}"))?
            }
            "--dt" => config.dt = value.parse().map_err(|e| format!("invalid --dt: {e}"))?,
            "--speed" => {
                config.speed = value.parse().map_err(|e| format!("invalid --speed: {e}"))?
            }
            "--gap" => config.gap = value.parse().map_err(|e| format!("invalid --gap: {e}"))?,
            "--substeps" => {
                config.substeps = value
                    .parse()
                    .map_err(|e| format!("invalid --substeps: {e}"))?
            }
            _ => return Err(format!("unknown argument: {key}")),
        }
    }
    Ok(config)
}

fn main() {
    let config = match parse_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };
    let report = match run_impact(config, |step, scene, wall_ms| {
        println!(
            "step={step} time_s={:.9} wall_ms={wall_ms:.3}",
            scene.simulated_seconds
        );
    }) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("impact setup/run error: {error}");
            std::process::exit(2);
        }
    };
    println!(
        "passed={} collision_observed={} momentum_conserved={} no_energy_creation={} simulated_seconds={:.9} voxel_count={} max_step_ms={:.3}",
        report.passed,
        report.collision_observed,
        report.momentum_conserved,
        report.no_energy_creation,
        report.simulated_seconds,
        report.voxel_count,
        report.max_step_ms,
    );
    println!("detail={}", report.detail);
    if !report.passed {
        std::process::exit(1);
    }
}
