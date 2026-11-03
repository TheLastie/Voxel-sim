//! Versioned `.vxmat` material assets using SI units.
//!
//! The file is a RON wrapper containing a magic identifier, schema version,
//! and every serialized field of [`Material`].

use crate::material::Material;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const MAGIC: &str = "VXMAT";
const SCHEMA_VERSION: u32 = 1;
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize)]
struct MaterialAsset {
    magic: String,
    schema_version: u32,
    units: String,
    material: Material,
}

/// Loads and validates a version 1 `.vxmat` material asset.
pub fn load_material(path: &Path) -> Result<Material, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("cannot inspect material asset: {error}"))?;
    if metadata.len() > MAX_FILE_BYTES {
        return Err("material asset exceeds 8 MiB limit".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read material asset: {error}"))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err("material asset exceeds 8 MiB limit".into());
    }
    decode_material(&bytes)
}

fn decode_material(bytes: &[u8]) -> Result<Material, String> {
    let asset: MaterialAsset =
        ron::de::from_bytes(bytes).map_err(|error| format!("invalid material asset: {error}"))?;
    if asset.magic != MAGIC {
        return Err("invalid material asset magic".into());
    }
    if asset.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "unsupported material asset schema version {}",
            asset.schema_version
        ));
    }
    if asset.units != "SI" {
        return Err("material asset units must be SI".into());
    }
    validate_material(&asset.material)?;
    Ok(asset.material)
}

/// Validates and saves a version 1 `.vxmat` material asset.
pub fn save_material(path: &Path, material: &Material) -> Result<(), String> {
    let contents = encode_material(material)?;
    write_atomically(path, contents.as_bytes())
}

fn write_atomically(path: &Path, contents: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path
        .file_name()
        .ok_or_else(|| "material asset path has no file name".to_string())?;
    let mut temporary = PathBuf::new();
    let mut temporary_file = None;
    for _ in 0..16 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut candidate_name = file_name.to_os_string();
        candidate_name.push(format!(".{}.{}.tmp", std::process::id(), sequence));
        temporary = parent.join(candidate_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => {
                temporary_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create temporary material asset: {error}")),
        }
    }
    let mut file = temporary_file
        .ok_or_else(|| "could not reserve a temporary material asset file".to_string())?;
    let write_result = file
        .write_all(contents)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("write temporary material asset: {error}"));
    drop(file);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("replace material asset atomically: {error}"));
    }
    Ok(())
}

fn encode_material(material: &Material) -> Result<String, String> {
    validate_material(material)?;
    let asset = MaterialAsset {
        magic: MAGIC.into(),
        schema_version: SCHEMA_VERSION,
        units: "SI".into(),
        material: material.clone(),
    };
    let contents = ron::ser::to_string_pretty(&asset, PrettyConfig::default())
        .map_err(|error| format!("cannot encode material asset: {error}"))?;
    if contents.len() as u64 > MAX_FILE_BYTES {
        return Err("material asset exceeds 8 MiB limit".into());
    }
    Ok(contents)
}

fn validate_material(material: &Material) -> Result<(), String> {
    if material.name.trim().is_empty() {
        return Err("material name must not be empty".into());
    }
    let fields = [
        ("density", material.density),
        ("young", material.young),
        ("young_perp", material.young_perp),
        ("shear", material.shear),
        ("poisson", material.poisson),
        ("tensile_strength", material.tensile_strength),
        ("tensile_strength_perp", material.tensile_strength_perp),
        ("compressive_strength", material.compressive_strength),
        (
            "compressive_strength_perp",
            material.compressive_strength_perp,
        ),
        ("hardening_modulus", material.hardening_modulus),
        ("failure_plastic_strain", material.failure_plastic_strain),
        ("shear_strength", material.shear_strength),
        ("shear_strength_perp", material.shear_strength_perp),
        ("yield_strength", material.yield_strength),
        ("hardness", material.hardness),
        ("heat_capacity", material.heat_capacity),
        ("thermal_cond", material.thermal_cond),
        ("dynamic_viscosity_pa_s", material.dynamic_viscosity_pa_s),
        (
            "surface_friction_coefficient",
            material.surface_friction_coefficient,
        ),
        (
            "viscosity_reference_temperature_kelvin",
            material.viscosity_reference_temperature_kelvin,
        ),
        (
            "viscosity_activation_energy_j_mol",
            material.viscosity_activation_energy_j_mol,
        ),
        ("electrical_conductivity", material.electrical_conductivity),
        ("relative_permittivity", material.relative_permittivity),
        (
            "thermal_expansion_parallel",
            material.thermal_expansion_parallel,
        ),
        (
            "thermal_expansion_perpendicular",
            material.thermal_expansion_perpendicular,
        ),
        ("melt_temp", material.melt_temp),
        (
            "strength_softening_start_ratio",
            material.strength_softening_start_ratio,
        ),
        ("boil_temp", material.boil_temp),
        ("latent_fusion", material.latent_fusion),
        ("latent_vaporization", material.latent_vaporization),
        ("fracture_energy", material.fracture_energy),
        ("fracture_energy_perp", material.fracture_energy_perp),
        (
            "moisture_content_dry_basis",
            material.moisture_content_dry_basis,
        ),
        (
            "solid_kelvin_voigt_time_s",
            material.solid_kelvin_voigt_time_s,
        ),
        (
            "internal_friction_coefficient",
            material.internal_friction_coefficient,
        ),
        (
            "skeletal_joint_limit_rad",
            material.skeletal_joint_limit_rad,
        ),
    ];
    for (name, value) in fields {
        let positive_infinity_is_no_transition = matches!(name, "melt_temp" | "boil_temp");
        if value.is_nan()
            || value == f32::NEG_INFINITY
            || (!value.is_finite()
                && !(positive_infinity_is_no_transition && value == f32::INFINITY))
        {
            return Err(format!("material field {name} must be finite"));
        }
    }
    let nonnegative = [
        ("density", material.density),
        ("young", material.young),
        ("young_perp", material.young_perp),
        ("shear", material.shear),
        ("tensile_strength", material.tensile_strength),
        ("tensile_strength_perp", material.tensile_strength_perp),
        ("compressive_strength", material.compressive_strength),
        (
            "compressive_strength_perp",
            material.compressive_strength_perp,
        ),
        ("hardening_modulus", material.hardening_modulus),
        ("failure_plastic_strain", material.failure_plastic_strain),
        ("shear_strength", material.shear_strength),
        ("shear_strength_perp", material.shear_strength_perp),
        ("yield_strength", material.yield_strength),
        ("hardness", material.hardness),
        ("heat_capacity", material.heat_capacity),
        ("thermal_cond", material.thermal_cond),
        ("dynamic_viscosity_pa_s", material.dynamic_viscosity_pa_s),
        (
            "viscosity_activation_energy_j_mol",
            material.viscosity_activation_energy_j_mol,
        ),
        ("electrical_conductivity", material.electrical_conductivity),
        ("latent_fusion", material.latent_fusion),
        ("latent_vaporization", material.latent_vaporization),
        ("fracture_energy", material.fracture_energy),
        ("fracture_energy_perp", material.fracture_energy_perp),
        (
            "moisture_content_dry_basis",
            material.moisture_content_dry_basis,
        ),
        (
            "solid_kelvin_voigt_time_s",
            material.solid_kelvin_voigt_time_s,
        ),
        (
            "internal_friction_coefficient",
            material.internal_friction_coefficient,
        ),
    ];
    for (name, value) in nonnegative {
        if value < 0.0 {
            return Err(format!("material field {name} must be nonnegative"));
        }
    }
    if !(-1.0..=0.5).contains(&material.poisson) {
        return Err("material poisson must be in [-1, 0.5]".into());
    }
    if material.density <= 0.0
        || material.heat_capacity <= 0.0
        || material.surface_friction_coefficient < 0.0
        || material.viscosity_reference_temperature_kelvin <= 0.0
        || material.relative_permittivity <= 0.0
        || material.melt_temp <= 0.0
        || material.boil_temp < material.melt_temp
        || material.strength_softening_start_ratio < 0.0
        || material.strength_softening_start_ratio > 1.0
        || material.skeletal_joint_limit_rad < 0.0
        || material.skeletal_joint_limit_rad > std::f32::consts::PI
    {
        return Err("material contains a value outside its physical range".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Material;

    fn valid_material() -> Material {
        let mut material = Material::table()
            .into_iter()
            .find(|m| m.name == "iron")
            .unwrap();
        material.name = "asset test iron".into();
        material
    }

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("vxmat-{label}-{}.vxmat", std::process::id()))
    }

    #[test]
    fn roundtrip_preserves_all_material_parameters() {
        let path = temp_path("roundtrip");
        let original = valid_material();
        save_material(&path, &original).unwrap();
        let loaded = load_material(&path).unwrap();
        let _ = fs::remove_file(path);
        assert_eq!(
            ron::to_string(&original).unwrap(),
            ron::to_string(&loaded).unwrap()
        );
    }

    #[test]
    fn overwriting_material_asset_replaces_a_complete_file() {
        let path = temp_path("atomic-overwrite");
        let original = valid_material();
        save_material(&path, &original).unwrap();
        let mut replacement = original.clone();
        replacement.density *= 1.25;
        save_material(&path, &replacement).unwrap();
        let loaded = load_material(&path).unwrap();
        assert_eq!(loaded.density, replacement.density);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn every_builtin_material_roundtrips_through_the_asset_codec() {
        for material in Material::table() {
            let encoded = encode_material(&material)
                .unwrap_or_else(|error| panic!("{} failed validation: {error}", material.name));
            let decoded = decode_material(encoded.as_bytes())
                .unwrap_or_else(|error| panic!("{} failed to decode: {error}", material.name));
            assert_eq!(
                ron::to_string(&material).unwrap(),
                ron::to_string(&decoded).unwrap(),
                "roundtrip changed {}",
                material.name
            );
        }
    }

    #[test]
    fn rejects_invalid_schema_version() {
        let path = temp_path("version");
        let asset = MaterialAsset {
            magic: MAGIC.into(),
            schema_version: SCHEMA_VERSION + 1,
            units: "SI".into(),
            material: valid_material(),
        };
        fs::write(&path, ron::to_string(&asset).unwrap()).unwrap();
        let result = load_material(&path);
        let _ = fs::remove_file(path);
        assert!(result.unwrap_err().contains("schema version"));
    }

    #[test]
    fn rejects_non_finite_material_values() {
        let mut material = valid_material();
        material.density = f32::NAN;
        assert!(save_material(&temp_path("nonfinite"), &material)
            .unwrap_err()
            .contains("density"));
    }

    #[test]
    fn allows_positive_infinity_for_materials_without_phase_transitions() {
        let material = Material::table()
            .into_iter()
            .find(|material| material.name == "air")
            .unwrap();
        let decoded = decode_material(encode_material(&material).unwrap().as_bytes()).unwrap();
        assert_eq!(decoded.melt_temp, f32::INFINITY);
        assert_eq!(decoded.boil_temp, f32::INFINITY);
    }

    #[test]
    fn rejects_negative_infinity_even_in_phase_thresholds() {
        let mut material = valid_material();
        material.melt_temp = f32::NEG_INFINITY;
        assert!(encode_material(&material)
            .unwrap_err()
            .contains("melt_temp"));
    }
}
