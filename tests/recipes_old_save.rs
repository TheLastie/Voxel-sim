//! Saves written before the reaction recipes existed still load, keep their
//! material tables, and are not disturbed by chemistry.

use std::path::Path;

use voxel_physics_lab::{
    documents::{load_document, SceneDocument},
    material::Material,
    recipes::RecipeRegistry,
    scene::Scene,
};

fn load(relative: &str) -> Scene {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    let document: SceneDocument = load_document(&path).expect("load saved scene document");
    document.build_scene(&path).expect("restore saved scene")
}

fn ledger(scene: &Scene) -> Vec<(u32, u64, Vec<(u16, u32)>)> {
    scene
        .voxels
        .iter()
        .map(|v| {
            (
                v.mass.to_bits(),
                v.total_internal_energy_j().to_bits(),
                v.composition.components.iter().map(|&(c, f)| (c, f.to_bits())).collect(),
            )
        })
        .collect()
}

#[test]
fn saved_scenes_from_before_the_recipes_load_and_react_as_no_ops() {
    let builtin = Material::table();
    for relative in [
        "scenes/hot_cold_iron_contact.vxscene",
        "scenes/spider_flat_physics_2026-11-04.vxscene",
    ] {
        let mut scene = load(relative);
        assert!(!scene.voxels.is_empty(), "{relative}");
        // The save carries its own, older, material table: no organic-processing
        // components were added to it and the names of its entries are unchanged.
        let organic = ["hide", "tannin", "leather", "humus"];
        assert!(
            scene.materials.iter().all(|m| !organic.contains(&m.name.as_str())),
            "{relative}: unexpected organic components in an old save"
        );
        for (old, new) in scene.materials.iter().zip(&builtin) {
            if old.name == new.name {
                assert_eq!(old.density, new.density, "{relative}: {}", old.name);
            }
        }
        // The built-in registry resolves against the old table without error
        // (recipes whose components are absent are simply not there).
        let registry = RecipeRegistry::builtin(&scene.materials);
        assert!(registry.recipes().iter().all(|r| !r.name.contains("tanning")), "{relative}");
        let before = ledger(&scene);
        // A day of chemistry on a scene with no reactive components: nothing
        // changes, not even a bit, and nothing is reported.
        let report = scene.react(86_400.0).unwrap();
        assert_eq!(report.reacting_revs, 0, "{relative}");
        assert!(report.recipes.is_empty(), "{relative}");
        assert_eq!(before, ledger(&scene), "{relative}");
        // And with the registry built for the old table.
        let report = scene.react_with(&registry, 3_600.0).unwrap();
        assert_eq!(report.reacting_revs, 0, "{relative}");
        assert_eq!(before, ledger(&scene), "{relative}");
    }
}

#[test]
fn an_old_material_table_is_a_prefix_of_the_new_one() {
    // The first 21 ids keep their meaning: a save that stores only ids and its
    // own table cannot be shifted by the appended components.
    let table = Material::table();
    let names: Vec<&str> = table.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(&names[20..], ["ash", "hide", "tannin", "leather", "humus"]);
    assert_eq!(names[16], "dry_wood");
}
