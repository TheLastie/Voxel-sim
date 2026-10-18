//! Reproduce the native contact scene and expose its actual numerical state.
use glam::{DVec3, Vec3};
use voxel_physics_lab::{bond::EMPTY, scene::{CpuPreconditioner,Scene,SceneConfig,VoxelSpawn,CELL},
    solver, world::GeneratedWorld, world_physics::contact_experiment};

fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let steps = args.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(40);
    let global = args.iter().any(|s| s=="--global");
    let world = GeneratedWorld::from_seed("1001");
    let region = contact_experiment(&world,Vec3::ZERO,SceneConfig::load("scenes/mounted_axe.ron")?)?;
    eprintln!("origin={:?} global={} REVs={}",region.origin,global,region.scene.voxels.len());
    let mut scene = if global {
        let offset = (region.origin/CELL).round().as_ivec3();
        let spawns = region.scene.voxels.iter().zip(&region.scene.positions).map(|(v,&p)| {
            let cell = (p/(CELL as f64)-DVec3::splat(0.5)).round().as_ivec3()+offset;
            VoxelSpawn { center: VoxelSpawn::at_cell(cell.to_array(),0).center,
                composition:v.composition.clone(), velocity:v.velocity,
                microstructure:v.microstructure.clone(),kinematic:v.kinematic,temperature_kelvin:293.15 }
        }).collect();
        Scene::from_voxels(region.scene.config.clone(),spawns)?
    } else { region.scene };
    if args.iter().any(|s| s == "--block-jacobi") {
        scene.config.cpu_preconditioner = CpuPreconditioner::BlockJacobi;
    }
    eprintln!("cpu_preconditioner={:?}",scene.config.cpu_preconditioner);
    if let Some(substeps) = args.windows(2).find(|p|p[0]=="--substeps") {
        scene.config.substeps = substeps[1].parse()?;
    }
    let mut max_error = 0.0_f64;
    for (axis,bonds) in scene.bonds.iter().enumerate() {
        for (i,bond) in bonds.iter().enumerate() {
            if bond.neighbor==EMPTY || scene.voxels[i].kinematic { continue; }
            let mut ideal = DVec3::ZERO; ideal[axis]=CELL as f64;
            max_error=max_error.max((scene.positions[bond.neighbor as usize]-scene.positions[i]-ideal).length());
        }
    }
    eprintln!("max_initial_bond_offset_error_m={max_error:e} substeps={}",scene.config.substeps);
    let initial_positions=scene.positions.clone();
    let initial_internal: f64=scene.voxels.iter().map(|v|v.total_internal_energy_j()).sum();
    let initial_elastic=scene.elastic_bond_energy_j();
    println!("step,time_s,broken,max_speed_mps,kinetic_j,damage_heat_j,surface_work_j,plastic_heat_j,contact_heat_j,pending_j,release_j,residual,step_ms,solid_heat_j,internal_delta_j,elastic_j,potential_delta_j,balance_error_j");
    let mut tool=None;
    for step in 1..=steps {
        let start=std::time::Instant::now();
        solver::step(&mut scene,&mut tool);
        let kinetic: f64=scene.voxels.iter().map(|v|0.5*f64::from(v.mass)*f64::from(v.velocity.length_squared())).sum();
        let speed=scene.voxels.iter().map(|v|v.velocity.length()).fold(0.0_f32,f32::max);
        let internal_delta=scene.voxels.iter().map(|v|v.total_internal_energy_j()).sum::<f64>()-initial_internal;
        let elastic=scene.elastic_bond_energy_j();
        let gravity=Vec3::from_array(scene.config.gravity);
        let potential_delta: f64=scene.positions.iter().zip(&initial_positions).zip(&scene.voxels)
            .filter(|(_,v)|!v.kinematic).map(|((&p,&old),v)|-f64::from(v.mass)*gravity.as_dvec3().dot(p-old)).sum();
        let balance=kinetic+elastic-initial_elastic+internal_delta+potential_delta
            +scene.stored_plastic_bond_energy_j()+scene.energy.surface_work+scene.energy.pending_release;
        println!("{step},{:.6},{},{speed:.6},{kinetic:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.4e},{:.3},{:.9e},{internal_delta:.9e},{elastic:.9e},{potential_delta:.9e},{balance:.9e}",
            step as f64*f64::from(scene.config.fixed_dt),scene.broken,scene.energy.damage_heat,
            scene.energy.surface_work,scene.energy.plastic_heat,
            scene.energy.contact_friction_heat+scene.energy.contact_restitution_heat,
            scene.energy.pending_release,scene.energy.kinetic_release,scene.solver_residual,
            start.elapsed().as_secs_f64()*1000.0,scene.energy.solid_viscoelastic_heat);
        if scene.positions.iter().any(|p| !p.is_finite()) { return Err("nonfinite state".into()); }
    }
    Ok(())
}
