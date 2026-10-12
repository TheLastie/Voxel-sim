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
    pub melt_temp: f32,
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
        ];
        table[1].fracture_mode = FractureMode::Mixed;
        table[1].compressive_strength_perp = 2e6;
        table[1].shear_strength_perp = 3e6;
        table[1].hardening_modulus = 2e9;
        table[1].failure_plastic_strain = 0.04;
        table[2].fracture_mode = FractureMode::Ductile;
        table[2].hardening_modulus = 1e9;
        table[2].failure_plastic_strain = 0.15;
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
            melt_temp,
            boil_temp,
            latent_fusion,
            latent_vaporization,
            fracture_energy,
            fracture_energy_perp,
        }
    }
    pub fn mixed(c: &Composition, table: &[Self]) -> Self {
        let mut result = table[c.components[0].0 as usize].clone();
        macro_rules! mix { ($($field:ident),*) => { $(result.$field = c.components.iter()
            .map(|&(id, f)| table[id as usize].$field * f).sum();)* }; }
        mix!(
            density,
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
            melt_temp,
            boil_temp,
            latent_fusion,
            latent_vaporization,
            fracture_energy,
            fracture_energy_perp
        );
        if c.components
            .iter()
            .any(|&(id, _)| table[id as usize].fracture_mode != result.fracture_mode)
        {
            result.fracture_mode = FractureMode::Mixed;
        }
        result
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
