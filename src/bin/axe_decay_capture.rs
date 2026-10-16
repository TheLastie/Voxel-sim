use glam::Vec3;
use std::{env, error::Error};
use voxel_physics_lab::{material::Material, scene::{Scene, SceneConfig}, solver, tool::{Shape, Tool}};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let scene_path = args.get(1).map(String::as_str).unwrap_or("scenes/mounted_axe_impact.ron");
    let load_steps = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(100usize);
    let coast_steps = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(300usize);
    let force = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(150.0f32);

    let config = SceneConfig::load(scene_path)?;
    let mut scene = Scene::new(config.clone());
    let wood_ids: Vec<_> = ["wood", "oak"]
        .into_iter()
        .filter_map(Material::component_id)
        .collect();
    let wood: Vec<_> = scene.voxels.iter().enumerate()
        .filter(|(_, voxel)| voxel.composition.components.iter()
            .any(|&(component, fraction)| wood_ids.contains(&component) && fraction > 0.0))
        .map(|(index, _)| index).collect();
    if let Some(lowest_z) = wood.iter().map(|&i| scene.positions[i].z).min_by(f32::total_cmp) {
        let base: Vec<_> = wood.into_iter()
            .filter(|&i| (scene.positions[i].z - lowest_z).abs() <= voxel_physics_lab::scene::CELL * 0.1)
            .collect();
        scene.fix_voxels(&base)?;
    }

    let direction = Vec3::from_array(config.tool_direction).normalize_or_zero();
    let mut tool = Tool::new(Shape::Wedge, &scene, Vec3::from_array(config.tool_start), direction, config.initial_speed);
    let rear = tool.cells.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
    tool.mount_base(Vec3::new(rear, 0.0, 0.0), direction)?;
    let mut tool = Some(tool);
    println!("phase,step,broken,clusters,tool_axis,tool_speed,wood_max_speed,tool_kinetic_j,wood_kinetic_j,pending_release_j,kinetic_release_j,damage_heat_j,surface_work_j,plastic_heat_j,contact_friction_heat_j,contact_restitution_heat_j,external_work_j");

    let total_steps = load_steps + coast_steps;
    let mut previous_broken = scene.broken;
    for step in 1..=total_steps {
        let phase = if step <= load_steps { "load" } else { "coast" };
        if let Some(active) = tool.as_mut() {
            active.set_external_force(if phase == "load" { direction * force } else { Vec3::ZERO })?;
        }
        solver::step(&mut scene, &mut tool);
        if step == 1 || step == load_steps || step == total_steps || step % 10 == 0 || scene.broken != previous_broken {
            let wood_max_speed = scene.voxels.iter()
                .filter(|v| v.composition.components.iter().any(|&(id, f)| wood_ids.contains(&id) && f > 0.0))
                .map(|v| v.velocity.length()).fold(0.0f32, f32::max);
            let wood_kinetic = scene.voxels.iter()
                .filter(|v| v.composition.components.iter().any(|&(id, f)| wood_ids.contains(&id) && f > 0.0))
                .map(|v| 0.5 * v.mass as f64 * v.velocity.length_squared() as f64)
                .sum::<f64>();
            let active = tool.as_ref().unwrap();
            let tool_kinetic = 0.5 * active.mass as f64 * active.velocity.length_squared() as f64;
            println!("{phase},{step},{},{},{:.6},{:.6},{:.6},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}",
                scene.broken, scene.cluster_count, active.position.dot(direction), active.velocity.length(), wood_max_speed,
                tool_kinetic, wood_kinetic,
                scene.energy.pending_release, scene.energy.kinetic_release, scene.energy.damage_heat,
                scene.energy.surface_work, scene.energy.plastic_heat, scene.energy.contact_friction_heat,
                scene.energy.contact_restitution_heat, scene.energy.external_work);
        }
        previous_broken = scene.broken;
    }
    Ok(())
}
