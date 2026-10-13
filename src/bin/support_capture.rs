//! Real solver capture of floor, gravity and the mounted axe (no scripted trajectories).
use glam::Vec3;
use std::{
    fs::File,
    io::{BufWriter, Write},
};
use voxel_physics_lab::{
    bond::EMPTY,
    scene::{Scene, SceneConfig, CELL},
    solver,
    tool::{Shape, Tool},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or("validation/supports.csv".into());
    let steps: usize = std::env::args().nth(2).unwrap_or("1500".into()).parse()?;
    let saved_frames: usize = std::env::args()
        .nth(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or_else(|| steps.div_ceil(10) + 1);
    if steps == 0 || saved_frames < 2 || saved_frames > steps + 1 {
        return Err("steps must be positive and saved_frames must be in 2..=steps+1".into());
    }
    let mut cfg = SceneConfig::load("scenes/mounted_axe.ron")?;
    if let Ok(raw) = std::env::var("VOXEL_CAPTURE_SUBSTEPS") {
        let substeps: usize = raw.parse()?;
        if substeps == 0 {
            return Err("VOXEL_CAPTURE_SUBSTEPS must be greater than zero".into());
        }
        cfg.substeps = substeps;
    }
    // A separate 3 cm coupon starts over a clear part of the finite floor so
    // its fall exercises vertical support, independently of the axe and trunk.
    cfg.branch_dimensions = [3; 3];
    let mut scene = Scene::new(cfg);
    let probe_start = 4320;
    let probe_end = 4347;
    for axis in 0..3 {
        for i in 0..scene.voxels.len() {
            let j = scene.bonds[axis][i].neighbor;
            if j != EMPTY
                && (i >= probe_start && i < probe_end)
                    != (j as usize >= probe_start && (j as usize) < probe_end)
            {
                scene.bonds[axis][i].neighbor = EMPTY;
                scene.workspace[axis][i].rest_neighbor = EMPTY;
            }
        }
    }
    for i in probe_start..probe_end {
        scene.positions[i] += Vec3::new(-0.25, -0.25, 0.22);
        scene.previous[i] = scene.positions[i];
        scene.rest_positions[i] = scene.positions[i];
    }
    let trunk_base: Vec<_> = (0..probe_start)
        .filter(|&i| scene.positions[i].z <= CELL * 0.5 + 1e-6)
        .collect();
    scene.fix_voxels(&trunk_base)?;
    solver::rebuild_clusters(&mut scene);
    let mut axe = Tool::new(
        Shape::Wedge,
        &scene,
        Vec3::from_array(scene.config.tool_start),
        Vec3::X,
        scene.config.initial_speed,
    );
    let rear = axe.cells.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    axe.mount_base(Vec3::new(rear, 0.0, 0.0), Vec3::X)?;
    let mount = axe.mount.unwrap();
    let mut tool = Some(axe);
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "H,{},{},{}",
        scene.config.fixed_dt, steps, saved_frames
    )?;
    writeln!(
        out,
        "M,{},{},{},{},{},{}",
        mount.world_base.x,
        mount.world_base.y,
        mount.world_base.z,
        mount.local_base.x,
        mount.local_base.y,
        mount.local_base.z
    )?;
    for p in &tool.as_ref().unwrap().cells {
        writeln!(out, "T,{},{},{}", p.x, p.y, p.z)?;
    }
    for i in probe_end..scene.voxels.len() {
        let p = scene.positions[i];
        writeln!(out, "P,{},{},{}", p.x, p.y, p.z)?;
    }
    // Make the initial geometry visible to the browser before solving starts.
    out.flush()?;
    let clock = std::time::Instant::now();
    let mut drift = 0.0f32;
    for frame in 0..=steps {
        if frame > 0 {
            // Mild ramp: 12 N/s, cap 18 N; same generic persistent-force API.
            let force = (frame as f32 * scene.config.fixed_dt * 12.0).min(18.0);
            tool.as_mut().unwrap().set_external_force(Vec3::X * force)?;
            solver::step(&mut scene, &mut tool);
        }
        let t = tool.as_ref().unwrap();
        let base = t.position + t.rotation * mount.local_base;
        drift = drift.max(
            Vec3::new(
                0.0,
                base.y - mount.world_base.y,
                base.z - mount.world_base.z,
            )
            .length(),
        );
        if frame % 100 == 0 {
            eprintln!(
                "frame={frame}/{steps} wall={:.1}s substeps={} broken={} drift={drift:.3e} tool_x={:.5}",
                clock.elapsed().as_secs_f64(),
                scene.config.substeps,
                scene.broken,
                t.position.x
            );
        }
        let intervals = saved_frames - 1;
        let is_sample = frame == 0
            || frame == steps
            || (frame as u128 * intervals as u128 / steps as u128)
                > ((frame - 1) as u128 * intervals as u128 / steps as u128);
        if !is_sample {
            continue;
        }
        assert!(scene.positions.iter().all(|p| p.is_finite()));
        writeln!(
            out,
            "F,{frame},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            t.position.x,
            t.position.y,
            t.position.z,
            t.rotation.x,
            t.rotation.y,
            t.rotation.z,
            t.rotation.w,
            t.velocity.x,
            t.external_force.x,
            scene.broken,
            scene.cluster_count,
            drift,
            scene.energy.contact_friction_heat
        )?;
        let damage = scene.incident_bond_damage();
        for i in 0..probe_end {
            let p = scene.positions[i];
            writeln!(
                out,
                "V,{frame},{i},{},{},{},{},{}",
                p.x, p.y, p.z, damage[i].0, damage[i].1
            )?;
        }
        // Flush each sparse snapshot so the live page can render it immediately.
        out.flush()?;
    }
    out.flush()?;
    eprintln!(
        "complete drift={drift:.3e} gravity={:?}",
        scene.config.gravity
    );
    Ok(())
}
