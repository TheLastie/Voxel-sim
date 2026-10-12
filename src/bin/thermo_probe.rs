use voxel_physics_lab::material::Material;

fn print_state(label: &str, material: &Material, mass: f32, energy: f32) {
    let phase = material.phase_state(mass, energy);
    println!(
        "{label:>20}: {:7.2} K | solid {:.3} liquid {:.3} gas {:.3}",
        phase.temperature_kelvin, phase.solid_fraction, phase.liquid_fraction, phase.gas_fraction
    );
}

fn main() {
    let water = Material::table().remove(4);
    let mass = 1.0;
    let melting_energy = water.energy_at_temperature(mass, water.melt_temp);
    let boiling_energy = water.energy_at_temperature(mass, water.boil_temp);

    println!("1 kg water: enthalpy-driven equilibrium phase state");
    print_state(
        "room temperature",
        &water,
        mass,
        water.energy_at_temperature(mass, 293.15),
    );
    print_state(
        "half melted",
        &water,
        mass,
        melting_energy + water.latent_fusion * 0.5,
    );
    print_state(
        "half vaporized",
        &water,
        mass,
        boiling_energy + water.latent_vaporization * 0.5,
    );
}
