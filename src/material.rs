use crate::voxel::Composition;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FractureMode {
    Brittle,
    Ductile,
    Mixed,
}

/// Thermodynamic state derived from a voxel's conserved internal energy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhaseState {
    pub temperature_kelvin: f32,
    pub solid_fraction: f32,
    pub liquid_fraction: f32,
    pub gas_fraction: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Material {
    pub name: String,
    pub density: f32,
    pub young: f32,
    pub young_perp: f32,
    pub shear: f32,
    pub poisson: f32,
    pub tensile_strength: f32,
    pub tensile_strength_perp: f32,
    pub compressive_strength: f32,
    pub compressive_strength_perp: f32,
    pub fracture_mode: FractureMode,
    pub hardening_modulus: f32,
    pub failure_plastic_strain: f32,
    pub shear_strength: f32,
    pub shear_strength_perp: f32,
    pub yield_strength: f32,
    pub hardness: f32,
    pub heat_capacity: f32,
    pub thermal_cond: f32,
    /// Dynamic viscosity for the liquid phase, pascal-seconds (Pa·s).
    pub dynamic_viscosity_pa_s: f32,
    /// Material-side contribution to dry Coulomb friction at a surface.
    /// Contact friction adds the two mixed material contributions.
    pub surface_friction_coefficient: f32,
    /// Reference temperature for the liquid viscosity model, kelvin (K).
    pub viscosity_reference_temperature_kelvin: f32,
    /// Liquid viscosity activation energy, joules per mole (J/mol).
    /// Zero preserves a temperature-independent viscosity.
    pub viscosity_activation_energy_j_mol: f32,
    /// Electrical conductivity, siemens per metre (S/m).
    pub electrical_conductivity: f32,
    /// Relative dielectric permittivity (dimensionless, vacuum = 1).
    pub relative_permittivity: f32,
    /// Linear thermal expansion parallel to the material fiber axis (1/K).
    pub thermal_expansion_parallel: f32,
    /// Linear thermal expansion perpendicular to the material fiber axis (1/K).
    pub thermal_expansion_perpendicular: f32,
    pub melt_temp: f32,
    /// Fraction of the melting temperature where progressive strength loss starts.
    /// A value of 1.0 disables pre-melt softening. Illustrative built-in values
    /// are not calibrated material data.
    pub strength_softening_start_ratio: f32,
    pub boil_temp: f32,
    pub latent_fusion: f32,
    pub latent_vaporization: f32,
    /// Fracture energy per newly separated area, J/m².
    pub fracture_energy: f32,
    /// Fracture energy for a bond perpendicular to the material's fiber axis, J/m².
    /// `fracture_energy` is the parallel value; isotropic materials use equal values.
    pub fracture_energy_perp: f32,
    /// Moisture content on the oven-dry mass basis, kg water / kg dry wood.
    /// Zero means unspecified for materials that are not hygroscopic solids.
    pub moisture_content_dry_basis: f32,
    /// Kelvin–Voigt time coefficient η/E (seconds) for solid viscoelasticity.
    /// This is calibrated from tan(delta) at a stated reference frequency.
    pub solid_kelvin_voigt_time_s: f32,
    /// Internal friction slope tan(phi) in the pressure-dependent shear
    /// envelope. This is distinct from surface friction between bodies.
    #[serde(default)]
    pub internal_friction_coefficient: f32,
    /// Maximum relative angular excursion of a skeleton joint whose bound
    /// REVs are made from this material. This is an authoring/animation limit,
    /// separate from the bulk continuum constitutive law.
    #[serde(default = "default_skeletal_joint_limit_rad")]
    pub skeletal_joint_limit_rad: f32,
    /// Chemical (formation) energy per kilogram, J/kg, relative to a common
    /// reference chosen so that every built-in value is nonnegative. A
    /// reaction releases `-sum(nu_k * h_k)` per unit extent; because the
    /// recipes conserve total mass, a constant shift of this reference cancels.
    /// It is part of the equation of state: temperature and phase fractions
    /// use `internal energy - mass * formation energy`, so heat comes from a
    /// change of composition at constant internal energy.
    /// Not serialized, to keep every document layout unchanged: the built-in
    /// values are restored by component name when a scene is built
    /// (`Material::restore_builtin_chemistry`). Zero for all pre-existing
    /// components, which keeps their behaviour bit-identical.
    #[serde(skip)]
    pub formation_energy_j_per_kg: f32,
}

fn default_skeletal_joint_limit_rad() -> f32 {
    std::f32::consts::PI
}
impl Material {
    /// Resolve a built-in component name to its composition-table index.
    pub fn component_id(name: &str) -> Option<u16> {
        Self::table()
            .iter()
            .position(|material| material.name.eq_ignore_ascii_case(name))
            .and_then(|index| u16::try_from(index).ok())
    }

    /// Built-in formation energy (J/kg) by component name; `None` means zero.
    /// Reference: liquid water = 0. Vapour sits above liquid by the latent
    /// heat of vaporization (2.256 MJ/kg); CO2 and O2 follow from standard
    /// formation enthalpies (-393.5 and 0 kJ/mol against -241.8 kJ/mol for
    /// water vapour); dry_wood is solved so that burning one kg releases
    /// 17.0 MJ with the recipe in `crate::combustion` (cellulose C6H10O5
    /// plus 1% ash, ash at zero). Illustrative, not calibrated species data.
    pub fn builtin_formation_energy(name: &str) -> Option<f32> {
        match name {
            "dry_wood" => Some(10_723_700.0),
            "oxygen" => Some(15_678_100.0),
            "carbon_dioxide" => Some(6_736_800.0),
            "water_vapor" => Some(2_256_000.0),
            _ => None,
        }
    }

    /// Fill in the non-serialized formation energies of built-in components in
    /// a table that was just deserialized or assembled. Entries that already
    /// carry a nonzero value are left alone.
    pub fn restore_builtin_chemistry(table: &mut [Self]) {
        for material in table {
            if material.formation_energy_j_per_kg == 0.0 {
                if let Some(value) = Self::builtin_formation_energy(&material.name) {
                    material.formation_energy_j_per_kg = value;
                }
            }
        }
    }

    pub fn table() -> Vec<Self> {
        // Illustrative engineering inputs, not a calibrated species database.
        let mut table = vec![
            Self::new(
                "air",
                1.225,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1005.0,
                0.026,
                f32::INFINITY,
                0.0,
                0.0,
                f32::INFINITY,
                0.0,
                0.0,
            ),
            Self::new(
                "wood",
                650.0,
                10e9,
                0.6e9,
                0.65e9,
                // Longitudinal/transverse limits create strong grain directionality.
                100e6,
                2.5e6,
                50e6,
                12e6,
                25e6,
                1700.0,
                0.15,
                f32::INFINITY,
                350.0,
                80.0,
                f32::INFINITY,
                0.0,
                0.0,
            ),
            Self::new(
                "iron",
                7870.0,
                200e9,
                200e9,
                77e9,
                350e6,
                350e6,
                350e6,
                200e6,
                200e6,
                450.0,
                80.0,
                1811.0,
                30000.0,
                30000.0,
                3134.0,
                247_000.0,
                6_300_000.0,
            ),
            Self::new(
                "stone",
                2700.0,
                50e9,
                50e9,
                20e9,
                7e6,
                7e6,
                120e6,
                12e6,
                100e6,
                790.0,
                2.5,
                1500.0,
                80.0,
                80.0,
                f32::INFINITY,
                300_000.0,
                0.0,
            ),
            Self::new(
                "water",
                1000.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4184.0,
                0.6,
                273.15,
                0.0,
                0.0,
                373.15,
                333_500.0,
                2_256_000.0,
            ),
            Self::new(
                "aluminum",
                2700.0,
                69e9,
                69e9,
                26e9,
                310e6,
                310e6,
                310e6,
                180e6,
                276e6,
                897.0,
                237.0,
                933.15,
                15_000.0,
                15_000.0,
                2743.0,
                397_000.0,
                10_500_000.0,
            ),
            Self::new(
                "copper",
                8960.0,
                117e9,
                117e9,
                44e9,
                220e6,
                220e6,
                210e6,
                120e6,
                70e6,
                385.0,
                401.0,
                1357.77,
                30_000.0,
                30_000.0,
                2835.0,
                205_000.0,
                4_730_000.0,
            ),
            Self::new(
                "glass",
                2500.0,
                70e9,
                70e9,
                29e9,
                45e6,
                45e6,
                1000e6,
                35e6,
                45e6,
                840.0,
                1.05,
                1700.0,
                10.0,
                10.0,
                f32::INFINITY,
                0.0,
                0.0,
            ),
            Self::new(
                "concrete",
                2400.0,
                30e9,
                30e9,
                12.5e9,
                3.5e6,
                3.5e6,
                40e6,
                5e6,
                3.5e6,
                880.0,
                1.7,
                1773.0,
                100.0,
                100.0,
                f32::INFINITY,
                0.0,
                0.0,
            ),
            Self::new(
                "rubber",
                1100.0,
                10e6,
                10e6,
                3.3e6,
                20e6,
                20e6,
                20e6,
                10e6,
                5e6,
                1900.0,
                0.2,
                f32::INFINITY,
                5_000.0,
                5_000.0,
                f32::INFINITY,
                0.0,
                0.0,
            ),
        ];
        // Illustrative dry-surface contributions. A pair's coefficient is
        // their sum; moisture, lubrication, and wear can refine these later.
        for (name, coefficient) in [
            ("air", 0.0),
            ("wood", 0.50),
            ("iron", 0.35),
            ("stone", 0.45),
            ("water", 0.0),
            ("aluminum", 0.30),
            ("copper", 0.30),
            ("glass", 0.10),
            ("concrete", 0.50),
            ("rubber", 0.85),
        ] {
            let material = table
                .iter_mut()
                .find(|material| material.name.eq_ignore_ascii_case(name))
                .expect("surface-friction entry names a built-in material");
            material.surface_friction_coefficient = coefficient;
        }
        table[1].fracture_mode = FractureMode::Mixed;
        table[1].electrical_conductivity = 1.0e-4;
        table[1].relative_permittivity = 3.0;
        table[1].thermal_expansion_parallel = 3.0e-6;
        table[1].thermal_expansion_perpendicular = 30.0e-6;
        table[1].compressive_strength_perp = 2e6;
        table[1].shear_strength_perp = 3e6;
        table[1].hardening_modulus = 2e9;
        table[1].failure_plastic_strain = 0.04;
        table[2].fracture_mode = FractureMode::Ductile;
        table[2].electrical_conductivity = 1.0e7;
        table[2].dynamic_viscosity_pa_s = 6.0e-3;
        table[2].relative_permittivity = 1.0;
        table[2].thermal_expansion_parallel = 12.0e-6;
        table[2].thermal_expansion_perpendicular = 12.0e-6;
        table[2].strength_softening_start_ratio = 0.7;
        table[2].hardening_modulus = 1e9;
        table[2].failure_plastic_strain = 0.15;
        table[5].fracture_mode = FractureMode::Ductile;
        table[5].electrical_conductivity = 3.5e7;
        table[5].dynamic_viscosity_pa_s = 1.3e-3;
        table[5].relative_permittivity = 1.0;
        table[5].thermal_expansion_parallel = 23.0e-6;
        table[5].thermal_expansion_perpendicular = 23.0e-6;
        table[5].strength_softening_start_ratio = 0.7;
        table[5].hardening_modulus = 0.7e9;
        table[5].failure_plastic_strain = 0.18;
        table[6].fracture_mode = FractureMode::Ductile;
        table[6].electrical_conductivity = 5.96e7;
        table[6].dynamic_viscosity_pa_s = 4.0e-3;
        table[6].relative_permittivity = 1.0;
        table[6].thermal_expansion_parallel = 16.5e-6;
        table[6].thermal_expansion_perpendicular = 16.5e-6;
        table[6].strength_softening_start_ratio = 0.7;
        table[6].hardening_modulus = 0.45e9;
        table[6].failure_plastic_strain = 0.30;
        table[7].strength_softening_start_ratio = 0.7;
        table[7].dynamic_viscosity_pa_s = 1.0e3;
        table[8].strength_softening_start_ratio = 0.7;
        table[3].electrical_conductivity = 1.0e-10;
        table[3].relative_permittivity = 5.0;
        table[3].thermal_expansion_parallel = 8.0e-6;
        table[3].thermal_expansion_perpendicular = 8.0e-6;
        table[4].electrical_conductivity = 5.0e-2;
        table[4].relative_permittivity = 80.0;
        table[4].thermal_expansion_parallel = 69.0e-6;
        table[4].thermal_expansion_perpendicular = 69.0e-6;
        table[4].dynamic_viscosity_pa_s = 1.002e-3;
        table[7].electrical_conductivity = 1.0e-12;
        table[7].relative_permittivity = 7.0;
        table[7].thermal_expansion_parallel = 9.0e-6;
        table[7].thermal_expansion_perpendicular = 9.0e-6;
        table[8].electrical_conductivity = 1.0e-4;
        table[8].relative_permittivity = 5.0;
        table[8].thermal_expansion_parallel = 10.0e-6;
        table[8].thermal_expansion_perpendicular = 10.0e-6;
        table[9].fracture_mode = FractureMode::Ductile;
        table[9].electrical_conductivity = 1.0e-13;
        table[9].relative_permittivity = 2.5;
        table[9].thermal_expansion_parallel = 100.0e-6;
        table[9].thermal_expansion_perpendicular = 100.0e-6;
        table[9].hardening_modulus = 20e6;
        table[9].failure_plastic_strain = 1.0;

        // European oak starter profile (Quercus robur). The sourced values are
        // density at 12% MC and longitudinal dynamic MOE / loss factor measured
        // near 11% MC. Other strength and transverse properties still inherit
        // the illustrative generic-wood estimates above.
        let mut oak = table[1].clone();
        oak.name = "oak".into();
        oak.density = 625.0;
        oak.young = 12.088e9;
        oak.moisture_content_dry_basis = 0.12;
        // tan(delta)=0.0087 at an estimated 2.05 kHz first bending mode.
        // Kelvin–Voigt η/E = tan(delta)/(2πf_ref), seconds.
        oak.solid_kelvin_voigt_time_s = 0.0087 / (std::f32::consts::TAU * 2_050.0);
        table.push(oak);

        // Skeleton limits are authoring presets, not natural anatomical joint
        // limits. The linked REV material controls the maximum pose excursion.
        for (name, degrees) in [
            ("air", 180.0_f32),
            ("wood", 115.0),
            ("iron", 75.0),
            ("stone", 25.0),
            ("water", 180.0),
            ("aluminum", 80.0),
            ("copper", 75.0),
            ("glass", 18.0),
            ("concrete", 20.0),
            ("rubber", 165.0),
            ("oak", 115.0),
        ] {
            let material = table
                .iter_mut()
                .find(|material| material.name.eq_ignore_ascii_case(name))
                .expect("skeleton-limit entry names a built-in material");
            material.skeletal_joint_limit_rad = degrees.to_radians();
        }

        // Biological presets use the same complete physical record as the
        // original simulation's materials. Values are coarse, low-rate starter
        // estimates; the current linear constitutive law cannot reproduce the
        // full nonlinear, anisotropic, viscoelastic behavior of living tissue.
        let mut flesh = Self::new(
            "flesh",
            1_050.0,
            1.0e6,
            1.0e6,
            0.34e6,
            0.5e6,
            0.5e6,
            0.3e6,
            0.17e6,
            0.2e6,
            3_500.0,
            0.5,
            f32::INFINITY,
            100.0,
            100.0,
            f32::INFINITY,
            0.0,
            0.0,
        );
        flesh.poisson = 0.45;
        flesh.fracture_mode = FractureMode::Ductile;
        flesh.hardening_modulus = 0.1e6;
        flesh.failure_plastic_strain = 0.5;
        flesh.hardness = 0.2e6;
        flesh.surface_friction_coefficient = 0.55;
        flesh.electrical_conductivity = 0.5;
        flesh.relative_permittivity = 100.0;
        flesh.thermal_expansion_parallel = 60.0e-6;
        flesh.thermal_expansion_perpendicular = 60.0e-6;
        flesh.moisture_content_dry_basis = 0.70;
        flesh.solid_kelvin_voigt_time_s = 0.025;
        flesh.internal_friction_coefficient = 0.35;
        flesh.skeletal_joint_limit_rad = 130.0_f32.to_radians();
        table.push(flesh);

        let mut muscle = Self::new(
            "muscle",
            1_060.0,
            1.5e6,
            1.0e6,
            0.5e6,
            0.6e6,
            0.4e6,
            0.5e6,
            0.2e6,
            0.25e6,
            3_500.0,
            0.5,
            f32::INFINITY,
            120.0,
            80.0,
            f32::INFINITY,
            0.0,
            0.0,
        );
        muscle.poisson = 0.45;
        muscle.fracture_mode = FractureMode::Ductile;
        muscle.hardening_modulus = 0.15e6;
        muscle.failure_plastic_strain = 0.45;
        muscle.hardness = 0.25e6;
        muscle.surface_friction_coefficient = 0.45;
        muscle.electrical_conductivity = 0.5;
        muscle.relative_permittivity = 100.0;
        muscle.thermal_expansion_parallel = 60.0e-6;
        muscle.thermal_expansion_perpendicular = 60.0e-6;
        muscle.moisture_content_dry_basis = 0.70;
        muscle.solid_kelvin_voigt_time_s = 0.04;
        muscle.internal_friction_coefficient = 0.35;
        muscle.skeletal_joint_limit_rad = 135.0_f32.to_radians();
        table.push(muscle);

        let mut tendon = Self::new(
            "tendon",
            1_120.0,
            0.8e9,
            80.0e6,
            50.0e6,
            80.0e6,
            15.0e6,
            100.0e6,
            30.0e6,
            60.0e6,
            3_500.0,
            0.45,
            f32::INFINITY,
            5_000.0,
            1_000.0,
            f32::INFINITY,
            0.0,
            0.0,
        );
        tendon.poisson = 0.35;
        tendon.fracture_mode = FractureMode::Mixed;
        tendon.compressive_strength_perp = 20.0e6;
        tendon.shear_strength_perp = 12.0e6;
        tendon.hardening_modulus = 80.0e6;
        tendon.failure_plastic_strain = 0.12;
        tendon.hardness = 60.0e6;
        tendon.surface_friction_coefficient = 0.35;
        tendon.electrical_conductivity = 0.4;
        tendon.relative_permittivity = 100.0;
        tendon.thermal_expansion_parallel = 30.0e-6;
        tendon.thermal_expansion_perpendicular = 60.0e-6;
        tendon.moisture_content_dry_basis = 0.65;
        tendon.solid_kelvin_voigt_time_s = 0.02;
        tendon.internal_friction_coefficient = 0.2;
        tendon.skeletal_joint_limit_rad = 60.0_f32.to_radians();
        table.push(tendon);

        let mut cartilage = Self::new(
            "cartilage",
            1_100.0,
            10.0e6,
            6.0e6,
            3.5e6,
            5.0e6,
            3.0e6,
            20.0e6,
            4.0e6,
            3.0e6,
            3_500.0,
            0.5,
            f32::INFINITY,
            300.0,
            200.0,
            f32::INFINITY,
            0.0,
            0.0,
        );
        cartilage.poisson = 0.45;
        cartilage.fracture_mode = FractureMode::Mixed;
        cartilage.compressive_strength_perp = 15.0e6;
        cartilage.shear_strength_perp = 2.5e6;
        cartilage.hardening_modulus = 1.0e6;
        cartilage.failure_plastic_strain = 0.3;
        cartilage.hardness = 3.0e6;
        cartilage.surface_friction_coefficient = 0.05;
        cartilage.electrical_conductivity = 0.5;
        cartilage.relative_permittivity = 100.0;
        cartilage.thermal_expansion_parallel = 50.0e-6;
        cartilage.thermal_expansion_perpendicular = 50.0e-6;
        cartilage.moisture_content_dry_basis = 0.75;
        cartilage.solid_kelvin_voigt_time_s = 0.08;
        cartilage.internal_friction_coefficient = 0.25;
        cartilage.skeletal_joint_limit_rad = 100.0_f32.to_radians();
        table.push(cartilage);

        let mut bone = Self::new(
            "bone",
            1_900.0,
            17.0e9,
            12.0e9,
            5.0e9,
            100.0e6,
            70.0e6,
            170.0e6,
            50.0e6,
            80.0e6,
            1_300.0,
            0.3,
            f32::INFINITY,
            100.0,
            70.0,
            f32::INFINITY,
            0.0,
            0.0,
        );
        bone.fracture_mode = FractureMode::Brittle;
        bone.compressive_strength_perp = 120.0e6;
        bone.shear_strength_perp = 35.0e6;
        bone.failure_plastic_strain = 0.02;
        bone.hardness = 80.0e6;
        bone.surface_friction_coefficient = 0.45;
        bone.electrical_conductivity = 0.02;
        bone.relative_permittivity = 15.0;
        bone.thermal_expansion_parallel = 10.0e-6;
        bone.thermal_expansion_perpendicular = 20.0e-6;
        bone.moisture_content_dry_basis = 0.15;
        bone.solid_kelvin_voigt_time_s = 0.004;
        bone.internal_friction_coefficient = 0.15;
        bone.skeletal_joint_limit_rad = 26.0_f32.to_radians();
        table.push(bone);

        // Combustion chemistry (see `crate::combustion`), appended after every
        // earlier entry so existing composition ids keep their meaning. Dry wood fuel is a
        // new component rather than a change to `wood`/`oak`, so every existing
        // scene keeps zero formation energy. Cellulose-like, ~1% mineral ash.
        let mut dry_wood = table[1].clone();
        dry_wood.name = "dry_wood".into();
        dry_wood.moisture_content_dry_basis = 0.0;
        dry_wood.skeletal_joint_limit_rad = 115.0_f32.to_radians();
        table.push(dry_wood);
        // Gases and ash: illustrative room-condition values.
        for (name, density, cp, conductivity) in [
            ("oxygen", 1.331_f32, 918.0_f32, 0.0263_f32),
            ("carbon_dioxide", 1.842, 844.0, 0.0166),
            ("water_vapor", 0.598, 2010.0, 0.025),
        ] {
            table.push(Self::new(
                name,
                density,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                cp,
                conductivity,
                f32::INFINITY,
                0.0,
                0.0,
                f32::INFINITY,
                0.0,
                0.0,
            ));
        }
        // Loose mineral ash: dense solid grains with almost no cohesion.
        let mut ash = Self::new(
            "ash", 2400.0, 20e6, 20e6, 8e6, 0.05e6, 0.05e6, 1e6, 0.1e6, 0.1e6, 840.0, 0.3, 1500.0,
            1.0, 1.0, f32::INFINITY, 300_000.0, 0.0,
        );
        ash.surface_friction_coefficient = 0.5;
        table.push(ash);

        // Organic processing (see `crate::recipes`: tanning and decay),
        // appended after `ash` so every earlier composition id keeps its
        // meaning. ILLUSTRATIVE engineering values in the right order of
        // magnitude for wet raw skin, vegetable tannin extract, vegetable-
        // tanned leather and humus; NOT calibrated material data. `water`
        // already exists (id 4). Animal tissue (`flesh`, `muscle`) is used as
        // is. Like those, all four have formation energy zero (the same
        // reference), so tanning is thermally neutral and the heat of rot
        // comes from oxygen and carbon dioxide (`builtin_formation_energy`
        // therefore needs no new entries). The water that drives the recipes
        // is the `water` component of the REV composition; the
        // `moisture_content_dry_basis` field is descriptive, as for flesh.
        // Raw hide: ~64% water by mass (1.8 kg/kg dry), weak, high rate of rot.
        let mut hide = Self::new(
            "hide", 1_080.0, 8.0e6, 4.0e6, 2.5e6, 12.0e6, 8.0e6, 3.0e6, 4.0e6, 6.0e6, 3_200.0,
            0.4, f32::INFINITY, 15_000.0, 10_000.0, f32::INFINITY, 0.0, 0.0,
        );
        hide.poisson = 0.45;
        hide.fracture_mode = FractureMode::Ductile;
        hide.hardening_modulus = 1.0e6;
        hide.failure_plastic_strain = 0.4;
        hide.hardness = 3.0e6;
        hide.surface_friction_coefficient = 0.5;
        hide.electrical_conductivity = 0.3;
        hide.relative_permittivity = 80.0;
        hide.thermal_expansion_parallel = 60.0e-6;
        hide.thermal_expansion_perpendicular = 60.0e-6;
        hide.moisture_content_dry_basis = 1.8;
        hide.solid_kelvin_voigt_time_s = 0.03;
        hide.internal_friction_coefficient = 0.3;
        hide.skeletal_joint_limit_rad = 140.0_f32.to_radians();
        table.push(hide);
        // Vegetable tannin (polyphenol extract powder): loose, weak grains.
        let mut tannin = Self::new(
            "tannin", 1_300.0, 5.0e6, 5.0e6, 2.0e6, 0.02e6, 0.02e6, 1.0e6, 0.1e6, 0.1e6, 1_500.0,
            0.15, f32::INFINITY, 1.0, 1.0, f32::INFINITY, 0.0, 0.0,
        );
        tannin.surface_friction_coefficient = 0.5;
        tannin.relative_permittivity = 5.0;
        tannin.moisture_content_dry_basis = 0.1;
        table.push(tannin);
        // Vegetable-tanned leather: stronger and stiffer than wet hide, little
        // water (0.15 kg/kg dry), resists rot (see the decay recipes). Its heat
        // capacity is the mass-weighted mean of the tanning reactants (1 kg hide
        // at 3200 + 0.35 kg tannin at 1500 per 1.35 kg), so tanning, whose
        // formation-energy heat is zero, leaves the derived temperature alone at
        // constant internal energy: enthalpy here is `cp*T + h_f` from 0 K, so a
        // cp mismatch across a reaction would show up as spurious heating.
        let mut leather = Self::new(
            "leather", 1_000.0, 150.0e6, 100.0e6, 50.0e6, 25.0e6, 18.0e6, 8.0e6, 12.0e6, 15.0e6,
            2_759.26, 0.15, f32::INFINITY, 30_000.0, 25_000.0, f32::INFINITY, 0.0, 0.0,
        );
        leather.poisson = 0.4;
        leather.fracture_mode = FractureMode::Ductile;
        leather.hardening_modulus = 20.0e6;
        leather.failure_plastic_strain = 0.3;
        leather.hardness = 10.0e6;
        leather.surface_friction_coefficient = 0.6;
        leather.electrical_conductivity = 1.0e-8;
        leather.relative_permittivity = 4.0;
        leather.thermal_expansion_parallel = 80.0e-6;
        leather.thermal_expansion_perpendicular = 80.0e-6;
        leather.moisture_content_dry_basis = 0.15;
        leather.solid_kelvin_voigt_time_s = 0.01;
        leather.internal_friction_coefficient = 0.3;
        leather.skeletal_joint_limit_rad = 150.0_f32.to_radians();
        table.push(leather);
        // Humus: the rot product (organic waste), crumbly with almost no cohesion.
        let mut humus = Self::new(
            "humus", 1_200.0, 5.0e6, 5.0e6, 2.0e6, 0.02e6, 0.02e6, 0.5e6, 0.05e6, 0.05e6, 1_900.0,
            0.25, f32::INFINITY, 1.0, 1.0, f32::INFINITY, 0.0, 0.0,
        );
        humus.surface_friction_coefficient = 0.6;
        humus.electrical_conductivity = 1.0e-3;
        humus.relative_permittivity = 15.0;
        humus.moisture_content_dry_basis = 0.3;
        humus.internal_friction_coefficient = 0.5;
        table.push(humus);
        Self::restore_builtin_chemistry(&mut table);
        table
    }
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: &str,
        density: f32,
        young: f32,
        young_perp: f32,
        shear: f32,
        tensile_strength: f32,
        tensile_strength_perp: f32,
        compressive_strength: f32,
        shear_strength: f32,
        yield_strength: f32,
        heat_capacity: f32,
        thermal_cond: f32,
        melt_temp: f32,
        fracture_energy: f32,
        fracture_energy_perp: f32,
        boil_temp: f32,
        latent_fusion: f32,
        latent_vaporization: f32,
    ) -> Self {
        Self {
            name: name.into(),
            density,
            young,
            young_perp,
            shear,
            poisson: 0.3,
            tensile_strength,
            tensile_strength_perp,
            compressive_strength,
            compressive_strength_perp: compressive_strength,
            fracture_mode: FractureMode::Brittle,
            hardening_modulus: 0.0,
            failure_plastic_strain: 0.0,
            shear_strength,
            shear_strength_perp: shear_strength,
            yield_strength,
            hardness: yield_strength,
            heat_capacity,
            thermal_cond,
            dynamic_viscosity_pa_s: 0.0,
            surface_friction_coefficient: 0.0,
            viscosity_reference_temperature_kelvin: 293.15,
            viscosity_activation_energy_j_mol: 0.0,
            electrical_conductivity: 1.0e-14,
            relative_permittivity: 1.0,
            thermal_expansion_parallel: 0.0,
            thermal_expansion_perpendicular: 0.0,
            melt_temp,
            strength_softening_start_ratio: 1.0,
            boil_temp,
            latent_fusion,
            latent_vaporization,
            fracture_energy,
            fracture_energy_perp,
            moisture_content_dry_basis: 0.0,
            solid_kelvin_voigt_time_s: 0.0,
            internal_friction_coefficient: 0.0,
            skeletal_joint_limit_rad: default_skeletal_joint_limit_rad(),
            formation_energy_j_per_kg: 0.0,
        }
    }
    pub fn mixed(c: &Composition, table: &[Self]) -> Self {
        let mut result = table[c.components[0].0 as usize].clone();
        // Composition fractions are mass fractions. Specific volumes therefore
        // add linearly: v_mix = Σ(w_i / ρ_i), ρ_mix = 1 / v_mix.
        // An arithmetic density average would violate the mass/volume meaning
        // of the composition and give every mixed REV the wrong physical mass.
        let specific_volume = c
            .components
            .iter()
            .map(|&(id, fraction)| fraction / table[id as usize].density)
            .sum::<f32>();
        result.density = 1.0 / specific_volume;
        macro_rules! mix { ($($field:ident),*) => { $(result.$field = c.components.iter()
            .map(|&(id, f)| table[id as usize].$field * f).sum();)* }; }
        mix!(
            young,
            young_perp,
            shear,
            poisson,
            tensile_strength,
            tensile_strength_perp,
            compressive_strength,
            compressive_strength_perp,
            shear_strength,
            shear_strength_perp,
            yield_strength,
            hardening_modulus,
            failure_plastic_strain,
            hardness,
            heat_capacity,
            thermal_cond,
            surface_friction_coefficient,
            viscosity_reference_temperature_kelvin,
            viscosity_activation_energy_j_mol,
            thermal_expansion_parallel,
            thermal_expansion_perpendicular,
            melt_temp,
            strength_softening_start_ratio,
            boil_temp,
            latent_fusion,
            latent_vaporization,
            fracture_energy,
            fracture_energy_perp,
            moisture_content_dry_basis,
            solid_kelvin_voigt_time_s,
            internal_friction_coefficient,
            skeletal_joint_limit_rad,
            formation_energy_j_per_kg
        );
        // Starter mixture estimate: dynamic viscosity is mass-fraction weighted.
        result.dynamic_viscosity_pa_s = c
            .components
            .iter()
            .map(|&(id, fraction)| table[id as usize].dynamic_viscosity_pa_s * fraction)
            .sum();
        // Series-mixture (lower-bound) rule: component resistivities add by
        // volume fraction, keeping composition and geometry explicit.
        let contains_insulator = c
            .components
            .iter()
            .any(|&(id, _)| table[id as usize].electrical_conductivity == 0.0);
        result.electrical_conductivity = if contains_insulator {
            0.0
        } else {
            specific_volume
                / c.components
                    .iter()
                    .map(|&(id, mass_fraction)| {
                        let material = &table[id as usize];
                        mass_fraction / material.density / material.electrical_conductivity
                    })
                    .sum::<f32>()
        };
        // The same series orientation gives a volume-weighted harmonic
        // permittivity bound for layered component mixtures.
        result.relative_permittivity = specific_volume
            / c.components
                .iter()
                .map(|&(id, mass_fraction)| {
                    let material = &table[id as usize];
                    mass_fraction / material.density / material.relative_permittivity
                })
                .sum::<f32>();
        if c.components
            .iter()
            .any(|&(id, _)| table[id as usize].fracture_mode != result.fracture_mode)
        {
            result.fracture_mode = FractureMode::Mixed;
        }
        result
    }

    /// Return the pre-melt factor applied to bond strength and fracture energy.
    /// Temperature and melting point are in kelvin. Values outside the useful
    /// ratio range disable this optional softening instead of producing NaNs.
    pub fn thermal_strength_scale(&self, temperature_kelvin: f32) -> f32 {
        let onset_ratio = self.strength_softening_start_ratio;
        if !temperature_kelvin.is_finite()
            || !self.melt_temp.is_finite()
            || self.melt_temp <= 0.0
            || !onset_ratio.is_finite()
            || !(0.0..1.0).contains(&onset_ratio)
        {
            return 1.0;
        }
        let onset = self.melt_temp * onset_ratio;
        if temperature_kelvin <= onset {
            1.0
        } else {
            ((self.melt_temp - temperature_kelvin) / (self.melt_temp - onset)).clamp(0.0, 1.0)
        }
    }

    /// Enthalpy relative to 0 K for the compact constant-heat-capacity EOS.
    pub fn energy_at_temperature(&self, mass: f32, temperature_kelvin: f32) -> f32 {
        let mass = mass.max(0.0);
        let temperature = temperature_kelvin.max(0.0);
        let mut specific = self.heat_capacity * temperature + self.formation_energy_j_per_kg;
        if temperature > self.melt_temp {
            specific += self.latent_fusion;
        }
        if temperature > self.boil_temp {
            specific += self.latent_vaporization;
        }
        mass * specific
    }

    /// Invert conserved energy into temperature and equilibrium phase fractions.
    pub fn phase_state(&self, mass: f32, energy: f32) -> PhaseState {
        let mass = mass.max(1e-20);
        // The chemical energy of the current composition is not thermal.
        let specific = (energy / mass - self.formation_energy_j_per_kg).max(0.0);
        let cp = self.heat_capacity.max(1e-12);
        let mut state = PhaseState {
            temperature_kelvin: specific / cp,
            solid_fraction: 1.0,
            liquid_fraction: 0.0,
            gas_fraction: 0.0,
        };

        if self.melt_temp.is_finite() {
            let melt_enthalpy = cp * self.melt_temp;
            if specific >= melt_enthalpy {
                let fusion = self.latent_fusion.max(0.0);
                if fusion > 0.0 && specific < melt_enthalpy + fusion {
                    state.temperature_kelvin = self.melt_temp;
                    state.solid_fraction =
                        (1.0 - (specific - melt_enthalpy) / fusion).clamp(0.0, 1.0);
                    state.liquid_fraction = 1.0 - state.solid_fraction;
                    return state;
                }
                state.solid_fraction = 0.0;
                state.liquid_fraction = 1.0;
                state.temperature_kelvin = (specific - fusion) / cp;
            }
        }

        if self.boil_temp.is_finite() {
            let vapor_start = cp * self.boil_temp + self.latent_fusion.max(0.0);
            if specific >= vapor_start {
                let vaporization = self.latent_vaporization.max(0.0);
                if vaporization > 0.0 && specific < vapor_start + vaporization {
                    state.temperature_kelvin = self.boil_temp;
                    state.liquid_fraction =
                        (1.0 - (specific - vapor_start) / vaporization).clamp(0.0, 1.0);
                    state.gas_fraction = 1.0 - state.liquid_fraction;
                    return state;
                }
                state.solid_fraction = 0.0;
                state.liquid_fraction = 0.0;
                state.gas_fraction = 1.0;
                state.temperature_kelvin =
                    (specific - self.latent_fusion.max(0.0) - vaporization) / cp;
            }
        }
        state
    }

    pub fn temperature(&self, mass: f32, energy: f32) -> f32 {
        self.phase_state(mass, energy).temperature_kelvin
    }

    /// Dynamic viscosity at a liquid temperature using an Arrhenius law.
    /// The reference viscosity is `dynamic_viscosity_pa_s`; the gas and solid
    /// phases do not use this property. Built-in materials default to zero
    /// activation energy until a calibrated temperature law is supplied.
    pub fn liquid_viscosity_at_temperature(&self, temperature_kelvin: f32) -> f32 {
        let reference = self.dynamic_viscosity_pa_s.max(0.0);
        if reference == 0.0 || self.viscosity_activation_energy_j_mol <= 0.0 {
            return reference;
        }
        if !temperature_kelvin.is_finite() || temperature_kelvin <= 0.0 {
            return f32::MAX;
        }
        const GAS_CONSTANT_J_MOL_K: f64 = 8.314_462_618_153_24;
        let exponent = (self.viscosity_activation_energy_j_mol as f64 / GAS_CONSTANT_J_MOL_K)
            * (1.0 / temperature_kelvin as f64
                - 1.0 / self.viscosity_reference_temperature_kelvin.max(1.0) as f64);
        let maximum_exponent = (f32::MAX as f64 / reference as f64).ln();
        (reference as f64 * exponent.clamp(-80.0, maximum_exponent).exp()) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::Material;
    use crate::voxel::Composition;

    #[test]
    fn starter_material_names_resolve_to_physical_entries() {
        let table = Material::table();
        for name in [
            "aluminum",
            "copper",
            "glass",
            "concrete",
            "rubber",
            "flesh",
            "muscle",
            "tendon",
            "cartilage",
            "bone",
        ] {
            let id = Material::component_id(name).expect("starter component must resolve");
            let material = &table[id as usize];
            assert_eq!(material.name, name);
            assert!(material.density > 0.0);
            assert!(material.heat_capacity > 0.0);
            assert!((0.0..=std::f32::consts::PI).contains(&material.skeletal_joint_limit_rad));
        }
        let water_id = Material::component_id("water").unwrap();
        assert!((table[water_id as usize].dynamic_viscosity_pa_s - 1.002e-3).abs() < 1e-8);
    }

    #[test]
    fn biological_materials_use_distinct_joint_limits_and_keep_original_fields() {
        let table = Material::table();
        let flesh = &table[Material::component_id("flesh").unwrap() as usize];
        let bone = &table[Material::component_id("bone").unwrap() as usize];
        assert!(flesh.young < bone.young);
        assert!(flesh.skeletal_joint_limit_rad > bone.skeletal_joint_limit_rad);
        for material in [flesh, bone] {
            assert!(material.young_perp > 0.0);
            assert!(material.shear > 0.0);
            assert!(material.tensile_strength > 0.0);
            assert!(material.compressive_strength > 0.0);
            assert!(material.shear_strength > 0.0);
            assert!(material.yield_strength > 0.0);
            assert!(material.fracture_energy > 0.0);
            assert!(material.thermal_cond > 0.0);
            assert!(material.relative_permittivity > 0.0);
        }
    }

    #[test]
    fn liquid_viscosity_arrhenius_law_uses_kelvin_and_reference_value() {
        let mut water =
            Material::table()[Material::component_id("water").unwrap() as usize].clone();
        water.viscosity_reference_temperature_kelvin = 293.15;
        water.viscosity_activation_energy_j_mol = 17_000.0;
        let reference = water.dynamic_viscosity_pa_s;
        assert!((water.liquid_viscosity_at_temperature(293.15) - reference).abs() < 1e-9);
        let hot = water.liquid_viscosity_at_temperature(333.15);
        let cold = water.liquid_viscosity_at_temperature(273.15);
        assert!(hot < reference && cold > reference);
        assert!(hot > 0.0 && cold.is_finite());

        water.viscosity_activation_energy_j_mol = 0.0;
        assert_eq!(water.liquid_viscosity_at_temperature(333.15), reference);
    }

    #[test]
    fn material_strength_softens_linearly_before_melting() {
        let iron_id = Material::component_id("iron").unwrap();
        let iron = &Material::table()[iron_id as usize];
        let onset = iron.melt_temp * iron.strength_softening_start_ratio;
        assert_eq!(iron.thermal_strength_scale(onset - 1.0), 1.0);
        assert!((iron.thermal_strength_scale((onset + iron.melt_temp) * 0.5) - 0.5).abs() < 1e-6);
        assert_eq!(iron.thermal_strength_scale(iron.melt_temp), 0.0);

        let wood_id = Material::component_id("wood").unwrap();
        let wood = &Material::table()[wood_id as usize];
        assert_eq!(wood.thermal_strength_scale(2000.0), 1.0);
    }

    #[test]
    fn mass_fraction_mixture_preserves_specific_volume_and_heat_capacity() {
        let table = Material::table();
        let iron = Material::component_id("iron").unwrap();
        let copper = Material::component_id("copper").unwrap();
        let composition = Composition {
            components: smallvec::smallvec![(iron, 0.25), (copper, 0.75)],
        };
        let mixture = Material::mixed(&composition, &table);
        let iron_material = &table[iron as usize];
        let copper_material = &table[copper as usize];

        let expected_density =
            1.0 / (0.25 / iron_material.density + 0.75 / copper_material.density);
        let expected_heat_capacity =
            0.25 * iron_material.heat_capacity + 0.75 * copper_material.heat_capacity;
        let expected_viscosity = 0.25 * iron_material.dynamic_viscosity_pa_s
            + 0.75 * copper_material.dynamic_viscosity_pa_s;
        let expected_expansion_parallel = 0.25 * iron_material.thermal_expansion_parallel
            + 0.75 * copper_material.thermal_expansion_parallel;
        let expected_expansion_perpendicular = 0.25 * iron_material.thermal_expansion_perpendicular
            + 0.75 * copper_material.thermal_expansion_perpendicular;
        let expected_joint_limit = 0.25 * iron_material.skeletal_joint_limit_rad
            + 0.75 * copper_material.skeletal_joint_limit_rad;
        let iron_volume = 0.25 / iron_material.density;
        let copper_volume = 0.75 / copper_material.density;
        let expected_conductivity = (iron_volume + copper_volume)
            / (iron_volume / iron_material.electrical_conductivity
                + copper_volume / copper_material.electrical_conductivity);
        let expected_permittivity = (iron_volume + copper_volume)
            / (iron_volume / iron_material.relative_permittivity
                + copper_volume / copper_material.relative_permittivity);
        assert!((mixture.density - expected_density).abs() <= expected_density * 1e-6);
        assert!((mixture.heat_capacity - expected_heat_capacity).abs() <= 1e-5);
        assert!((mixture.dynamic_viscosity_pa_s - expected_viscosity).abs() <= 1e-8);
        assert!((mixture.thermal_expansion_parallel - expected_expansion_parallel).abs() <= 1e-10);
        assert!(
            (mixture.thermal_expansion_perpendicular - expected_expansion_perpendicular).abs()
                <= 1e-10
        );
        assert!((mixture.skeletal_joint_limit_rad - expected_joint_limit).abs() <= 1e-6);
        assert!(
            (mixture.electrical_conductivity - expected_conductivity).abs()
                <= expected_conductivity * 1e-6
        );
        assert!(
            (mixture.relative_permittivity - expected_permittivity).abs()
                <= expected_permittivity * 1e-6
        );

        let mass_kg = 0.02;
        let temperature_k = 300.0;
        let energy_j = mixture.energy_at_temperature(mass_kg, temperature_k);
        let phase = mixture.phase_state(mass_kg, energy_j);
        assert!((phase.temperature_kelvin - temperature_k).abs() < 1e-4);
        assert_eq!(phase.solid_fraction, 1.0);
        assert_eq!(phase.liquid_fraction, 0.0);
        assert_eq!(phase.gas_fraction, 0.0);
    }

    #[test]
    fn zero_formation_energy_leaves_the_equation_of_state_unchanged() {
        let table = Material::table();
        // Every component that existed before the chemistry additions keeps
        // its composition id (saved scenes refer to table positions).
        assert_eq!(table[10].name, "oak");
        assert_eq!(table[15].name, "bone");
        assert_eq!(table[16].name, "dry_wood");
        assert_eq!(table[20].name, "ash");
        // Organic-processing components are appended after `ash`.
        assert_eq!(
            table[21..].iter().map(|m| m.name.as_str()).collect::<Vec<_>>(),
            ["hide", "tannin", "leather", "humus"]
        );
        assert_eq!(table.len(), 25);
        for material in table.iter().take(16) {
            assert_eq!(material.formation_energy_j_per_kg, 0.0, "{}", material.name);
        }
        let iron = &table[Material::component_id("iron").unwrap() as usize];
        for kelvin in [10.0_f32, 293.15, 1811.0, 2000.0, 3134.0, 4000.0] {
            let energy = iron.energy_at_temperature(0.01, kelvin);
            let reference = {
                let mut specific = iron.heat_capacity * kelvin;
                if kelvin > iron.melt_temp {
                    specific += iron.latent_fusion;
                }
                if kelvin > iron.boil_temp {
                    specific += iron.latent_vaporization;
                }
                0.01_f32 * specific
            };
            assert_eq!(energy.to_bits(), reference.to_bits());
        }
        let mixture = Material::mixed(
            &Composition {
                components: smallvec::smallvec![
                    (Material::component_id("iron").unwrap(), 0.5),
                    (Material::component_id("water").unwrap(), 0.5)
                ],
            },
            &table,
        );
        assert_eq!(mixture.formation_energy_j_per_kg, 0.0);
    }

    #[test]
    fn formation_energy_mixes_by_mass_and_is_removed_from_the_thermal_state() {
        let table = Material::table();
        let fuel = Material::component_id("dry_wood").unwrap();
        let oxygen = Material::component_id("oxygen").unwrap();
        let composition = Composition {
            components: smallvec::smallvec![(fuel, 0.25), (oxygen, 0.75)],
        };
        let mixture = Material::mixed(&composition, &table);
        let expected = 0.25 * table[fuel as usize].formation_energy_j_per_kg
            + 0.75 * table[oxygen as usize].formation_energy_j_per_kg;
        assert!((mixture.formation_energy_j_per_kg - expected).abs() <= 2.0);
        let mass = 0.003;
        let energy = mixture.energy_at_temperature(mass, 650.0);
        // The stored energy is dominated by the chemical part...
        assert!(energy > mass * mixture.formation_energy_j_per_kg);
        // ...but temperature is derived from the thermal remainder only.
        let phase = mixture.phase_state(mass, energy);
        assert!((phase.temperature_kelvin - 650.0).abs() < 0.05);
        assert_eq!(phase.solid_fraction, 1.0);
        // Less stored energy than the chemical offset is no thermal energy.
        assert_eq!(
            mixture.temperature(mass, 0.5 * mass * mixture.formation_energy_j_per_kg),
            0.0
        );
    }

    #[test]
    fn formation_energy_is_not_serialized_and_is_restored_by_name() {
        let table = Material::table();
        let bytes = bincode::serialize(&table).unwrap();
        let mut loaded: Vec<Material> = bincode::deserialize(&bytes).unwrap();
        let fuel = Material::component_id("dry_wood").unwrap() as usize;
        assert_eq!(loaded[fuel].formation_energy_j_per_kg, 0.0);
        Material::restore_builtin_chemistry(&mut loaded);
        for (a, b) in table.iter().zip(&loaded) {
            assert_eq!(a.formation_energy_j_per_kg, b.formation_energy_j_per_kg);
        }
        assert!(loaded[fuel].formation_energy_j_per_kg > 0.0);
    }

    #[test]
    fn organic_processing_components_are_complete_and_leather_beats_hide() {
        let table = Material::table();
        let get = |name: &str| &table[Material::component_id(name).unwrap() as usize];
        for name in ["hide", "tannin", "leather", "humus"] {
            let m = get(name);
            assert!(m.density > 0.0 && m.heat_capacity > 0.0 && m.thermal_cond > 0.0, "{name}");
            assert!(m.young > 0.0 && m.shear > 0.0 && m.tensile_strength > 0.0, "{name}");
            assert!(m.fracture_energy > 0.0 && m.relative_permittivity > 0.0, "{name}");
            assert!((0.0..=std::f32::consts::PI).contains(&m.skeletal_joint_limit_rad));
            // No tabulated formation energy: same reference as flesh and muscle.
            assert_eq!(m.formation_energy_j_per_kg, 0.0, "{name}");
        }
        let (hide, leather) = (get("hide"), get("leather"));
        assert!(leather.tensile_strength > 1.5 * hide.tensile_strength);
        assert!(leather.young > 5.0 * hide.young);
        assert!(leather.moisture_content_dry_basis < 0.2 * hide.moisture_content_dry_basis);
        assert!(get("humus").tensile_strength < 0.01 * hide.tensile_strength);
        // Restored by name after a round trip, like the other chemistry.
        let bytes = bincode::serialize(&table).unwrap();
        let mut loaded: Vec<Material> = bincode::deserialize(&bytes).unwrap();
        Material::restore_builtin_chemistry(&mut loaded);
        for (a, b) in table.iter().zip(&loaded) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.formation_energy_j_per_kg, b.formation_energy_j_per_kg);
        }
    }

    #[test]
    fn internal_friction_slope_mixes_with_the_mass_fractions() {
        let mut table = Material::table();
        let iron = Material::component_id("iron").unwrap();
        let copper = Material::component_id("copper").unwrap();
        table[iron as usize].internal_friction_coefficient = 0.4;
        table[copper as usize].internal_friction_coefficient = 0.8;
        let composition = Composition {
            components: smallvec::smallvec![(iron, 0.25), (copper, 0.75)],
        };
        let mixture = Material::mixed(&composition, &table);
        assert!((mixture.internal_friction_coefficient - 0.7).abs() < 1e-6);
        assert_eq!(
            Material::table()[iron as usize].internal_friction_coefficient,
            0.0
        );
    }
}
