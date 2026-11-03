//! Authored thermal bonding settings for native `.vxscene` documents.

use crate::scene::Scene;
use serde::{Deserialize, Serialize};

/// A per-instance initial temperature override, applied to each REV in the model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneObjectTemperature {
    pub model_index: usize,
    pub temperature_kelvin: f32,
}

/// Thresholds for creating real thermal bonds between contacting REVs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThermalBondingSettings {
    /// Material component that must be present in both contacting REVs.
    pub material_name: String,
    pub minimum_mass_fraction: f32,
    pub minimum_temperature_kelvin: f32,
    pub minimum_impact_energy_j: f64,
    pub minimum_contact_pressure_pa: f32,
    pub minimum_dwell_seconds: f32,
}

impl ThermalBondingSettings {
    /// Enable the existing contact-driven bond rule on a running scene.
    pub fn apply(&self, scene: &mut Scene) -> Result<(), String> {
        scene
            .set_thermal_bonding_rule(
                &self.material_name,
                self.minimum_mass_fraction,
                self.minimum_temperature_kelvin,
                self.minimum_impact_energy_j,
                self.minimum_contact_pressure_pa,
                self.minimum_dwell_seconds,
            )
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.material_name.trim().is_empty()
            || !self.minimum_mass_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_mass_fraction)
            || self.minimum_mass_fraction == 0.0
            || !self.minimum_temperature_kelvin.is_finite()
            || self.minimum_temperature_kelvin < 0.0
            || !self.minimum_impact_energy_j.is_finite()
            || self.minimum_impact_energy_j < 0.0
            || !self.minimum_contact_pressure_pa.is_finite()
            || self.minimum_contact_pressure_pa < 0.0
            || !self.minimum_dwell_seconds.is_finite()
            || self.minimum_dwell_seconds < 0.0
        {
            return Err("thermal bonding settings contain an invalid threshold or material name".into());
        }
        Ok(())
    }
}

/// Optional simulation authoring values saved in a native scene.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SceneSimulationSettings {
    /// Sorted by model index; omitted objects use their model's authored REV temperatures.
    pub object_initial_temperatures: Vec<SceneObjectTemperature>,
    /// `None` leaves automatic thermal bond formation disabled.
    pub thermal_bonding: Option<ThermalBondingSettings>,
}

impl SceneSimulationSettings {
    pub fn object_temperature_kelvin(&self, model_index: usize) -> Option<f32> {
        self.object_initial_temperatures
            .binary_search_by_key(&model_index, |entry| entry.model_index)
            .ok()
            .map(|index| self.object_initial_temperatures[index].temperature_kelvin)
    }

    pub fn set_object_temperature_kelvin(
        &mut self,
        model_index: usize,
        temperature_kelvin: Option<f32>,
    ) -> Result<(), String> {
        if temperature_kelvin.is_some_and(|temperature| {
            !temperature.is_finite() || temperature <= 0.0
        }) {
            return Err("initial object temperature must be finite and above absolute zero".into());
        }
        match self
            .object_initial_temperatures
            .binary_search_by_key(&model_index, |entry| entry.model_index)
        {
            Ok(index) => {
                if let Some(temperature_kelvin) = temperature_kelvin {
                    self.object_initial_temperatures[index].temperature_kelvin = temperature_kelvin;
                } else {
                    self.object_initial_temperatures.remove(index);
                }
            }
            Err(index) => {
                if let Some(temperature_kelvin) = temperature_kelvin {
                    self.object_initial_temperatures.insert(
                        index,
                        SceneObjectTemperature {
                            model_index,
                            temperature_kelvin,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    pub fn validate(&self, model_count: usize) -> Result<(), String> {
        if self
            .object_initial_temperatures
            .windows(2)
            .any(|pair| pair[0].model_index >= pair[1].model_index)
        {
            return Err("initial object temperatures must have unique sorted model indices".into());
        }
        for entry in &self.object_initial_temperatures {
            if entry.model_index >= model_count
                || !entry.temperature_kelvin.is_finite()
                || entry.temperature_kelvin <= 0.0
            {
                return Err("initial object temperature has an invalid index or value".into());
            }
        }
        if let Some(rule) = &self.thermal_bonding {
            rule.validate()?;
        }
        Ok(())
    }

    pub fn remove_object(&mut self, model_index: usize) {
        self.object_initial_temperatures.retain_mut(|entry| {
            if entry.model_index == model_index {
                false
            } else {
                if entry.model_index > model_index {
                    entry.model_index -= 1;
                }
                true
            }
        });
    }
}
