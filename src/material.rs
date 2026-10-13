use crate::voxel::Composition;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FractureMode {
    Brittle,
    Ductile,
    Mixed,
}

/// Thermodynamic state derived from a voxel's conserved internal energy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhaseState {
    pub temperature_kelvin: f32,
    pub solid_fraction: f32,
    pub liquid_fraction: f32,
    pub gas_fraction: f32,
}

#[derive(Clone, Debug)]
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
}
impl Material {
    /// Resolve a built-in component name to its composition-table index.
    pub fn component_id(name: &str) -> Option<u16> {
        Self::table()
            .iter()
            .position(|material| material.name.eq_ignore_ascii_case(name))
            .and_then(|index| u16::try_from(index).ok())
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
        table[2].relative_permittivity = 1.0;
        table[2].thermal_expansion_parallel = 12.0e-6;
        table[2].thermal_expansion_perpendicular = 12.0e-6;
        table[2].strength_softening_start_ratio = 0.7;
        table[2].hardening_modulus = 1e9;
        table[2].failure_plastic_strain = 0.15;
        table[5].fracture_mode = FractureMode::Ductile;
        table[5].electrical_conductivity = 3.5e7;
        table[5].relative_permittivity = 1.0;
        table[5].thermal_expansion_parallel = 23.0e-6;
        table[5].thermal_expansion_perpendicular = 23.0e-6;
        table[5].strength_softening_start_ratio = 0.7;
        table[5].hardening_modulus = 0.7e9;
        table[5].failure_plastic_strain = 0.18;
        table[6].fracture_mode = FractureMode::Ductile;
        table[6].electrical_conductivity = 5.96e7;
        table[6].relative_permittivity = 1.0;
        table[6].thermal_expansion_parallel = 16.5e-6;
        table[6].thermal_expansion_perpendicular = 16.5e-6;
        table[6].strength_softening_start_ratio = 0.7;
        table[6].hardening_modulus = 0.45e9;
        table[6].failure_plastic_strain = 0.30;
        table[7].strength_softening_start_ratio = 0.7;
        table[8].strength_softening_start_ratio = 0.7;
        table[3].electrical_conductivity = 1.0e-10;
        table[3].relative_permittivity = 5.0;
        table[3].thermal_expansion_parallel = 8.0e-6;
        table[3].thermal_expansion_perpendicular = 8.0e-6;
        table[4].electrical_conductivity = 5.0e-2;
        table[4].relative_permittivity = 80.0;
        table[4].thermal_expansion_parallel = 69.0e-6;
        table[4].thermal_expansion_perpendicular = 69.0e-6;
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
            thermal_expansion_parallel,
            thermal_expansion_perpendicular,
            melt_temp,
            strength_softening_start_ratio,
            boil_temp,
            latent_fusion,
            latent_vaporization,
            fracture_energy,
            fracture_energy_perp
        );
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
        let mut specific = self.heat_capacity * temperature;
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
        let specific = (energy / mass).max(0.0);
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
}

#[cfg(test)]
mod tests {
    use super::Material;
    use crate::voxel::Composition;

    #[test]
    fn starter_material_names_resolve_to_physical_entries() {
        let table = Material::table();
        for name in ["aluminum", "copper", "glass", "concrete", "rubber"] {
            let id = Material::component_id(name).expect("starter component must resolve");
            let material = &table[id as usize];
            assert_eq!(material.name, name);
            assert!(material.density > 0.0);
            assert!(material.heat_capacity > 0.0);
        }
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
        let expected_expansion_parallel = 0.25 * iron_material.thermal_expansion_parallel
            + 0.75 * copper_material.thermal_expansion_parallel;
        let expected_expansion_perpendicular = 0.25 * iron_material.thermal_expansion_perpendicular
            + 0.75 * copper_material.thermal_expansion_perpendicular;
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
        assert!((mixture.thermal_expansion_parallel - expected_expansion_parallel).abs() <= 1e-10);
        assert!(
            (mixture.thermal_expansion_perpendicular - expected_expansion_perpendicular).abs()
                <= 1e-10
        );
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
}
