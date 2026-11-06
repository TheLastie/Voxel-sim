//! Read-only creature inspector, shared by the editor window and the MCP
//! `inspect_creatures` tool.
//!
//! A scene has a creature only when `<scene>.creatures.json` says so: it lists
//! a name, the index of a skeletal actuator of the scene and the creature's
//! [`Genome`] (and optionally its [`LearnedState`]). Without that file there is
//! no creature and the inspector does not exist.
//!
//! Inspecting never changes anything. Sensors are read with
//! [`SensorReading::read_with_motion`] directly, so the creature's own
//! cumulative [`SensorCost`] counters are not advanced by looking at it. The
//! ledger is read from the creature as it is. The editor does not drive
//! creatures (it never calls `Creature::drive_step`), so a creature built from
//! the file shows its birth ledger and a network that has not run; the network
//! cost counters are `None` unless the caller owns a running network and
//! passes its counters.

use crate::{
    creature::{
        Creature, Genome, LearnedState, NetworkCost, SensorCost, SensorReading,
    },
    scene::Scene,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub const CREATURE_FILE_VERSION: u32 = 1;
const MAX_FILE_BYTES: u64 = 8 * 1024 * 1024;
/// Rows returned by [`CreatureInspection::summary_lines`].
pub const CREATURE_SUMMARY_ROWS: usize = 7;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreatureSpec {
    pub name: String,
    /// Index into `Scene::skeletal_actuators`.
    pub actuator_index: usize,
    pub genome: Genome,
    #[serde(default)]
    pub learned: Option<LearnedState>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreatureFile {
    pub version: u32,
    pub creatures: Vec<CreatureSpec>,
}

pub struct NamedCreature {
    pub name: String,
    pub creature: Creature,
}

impl CreatureFile {
    /// `scenes/a.vxscene` -> `scenes/a.creatures.json`.
    pub fn sidecar_path(scene_path: &Path) -> PathBuf {
        scene_path.with_extension("creatures.json")
    }

    /// The creatures saved beside the scene; `None` when there is no file.
    pub fn load(scene_path: &Path) -> Result<Option<Self>, String> {
        let path = Self::sidecar_path(scene_path);
        match fs::metadata(&path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(format!("cannot read {}: {error}", path.display())),
            Ok(meta) if meta.len() > MAX_FILE_BYTES => {
                return Err(format!("{} is larger than {MAX_FILE_BYTES} bytes", path.display()));
            }
            Ok(_) => {}
        }
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let file: Self = serde_json::from_str(&text)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        file.validate().map_err(|error| format!("{}: {error}", path.display()))?;
        Ok(Some(file))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.version != CREATURE_FILE_VERSION {
            return Err(format!("unsupported creature file version {}", self.version));
        }
        for spec in &self.creatures {
            if spec.name.trim().is_empty() {
                return Err("a creature has an empty name".into());
            }
            spec.genome.validate().map_err(|error| format!("creature \"{}\": {error}", spec.name))?;
        }
        Ok(())
    }

    /// Build every creature against `scene`. The first creature that does not
    /// fit the scene is an error naming it.
    pub fn build(&self, scene: &Scene) -> Result<Vec<NamedCreature>, String> {
        self.creatures
            .iter()
            .map(|spec| {
                Creature::new(
                    spec.genome.clone(),
                    spec.learned.clone(),
                    scene,
                    spec.actuator_index,
                )
                .map(|creature| NamedCreature { name: spec.name.clone(), creature })
                .map_err(|error| format!("creature \"{}\": {error}", spec.name))
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LedgerView {
    pub stored_j: f64,
    pub capacity_j: f64,
    pub stomach_j: f64,
    pub hunger: f64,
    /// Net mechanical work done by the muscles so far (J).
    pub work_j: f64,
    /// Heat the muscles returned to the body so far (J).
    pub heat_j: f64,
    /// Chemical energy debited so far, basal included (J).
    pub debited_j: f64,
    pub unfunded_work_j: f64,
    pub conservation_residual_j: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensorRow {
    pub label: String,
    pub unit: &'static str,
    /// `None` where the quantity is unavailable at this instant.
    pub value: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkView {
    pub neurons: usize,
    pub synapses: usize,
    pub inputs: usize,
    /// Counters of a running network; `None` when none is running.
    pub cost: Option<NetworkCost>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CreatureInspection {
    pub name: String,
    pub ledger: LedgerView,
    pub sensors: Vec<SensorRow>,
    /// Cost of the read that produced these values.
    pub read_cost: SensorCost,
    /// Cumulative cost of every read the creature itself has made.
    pub creature_sensor_cost: SensorCost,
    pub network: NetworkView,
}

/// Read one creature now. Read-only: the scene and the creature are borrowed
/// shared and none of their counters advance.
pub fn inspect(
    name: &str,
    creature: &Creature,
    scene: &Scene,
    network_cost: Option<NetworkCost>,
) -> Result<CreatureInspection, String> {
    let layout = creature.sensor_layout();
    let ledger = creature.ledger();
    let reading = SensorReading::read_with_motion(scene, layout, ledger, None)?;
    let values = reading.values();
    let catalog = layout.catalog();
    if values.len() != catalog.len() {
        return Err(format!(
            "sensor catalogue lists {} channels but the reading has {}",
            catalog.len(),
            values.len()
        ));
    }
    let sensors = catalog
        .iter()
        .zip(values)
        .map(|(descriptor, value)| SensorRow {
            label: descriptor.id.label(),
            unit: descriptor.unit,
            value: value.filter(|v| v.is_finite()),
        })
        .collect();
    let network = &creature.genome().network;
    Ok(CreatureInspection {
        name: name.to_owned(),
        ledger: LedgerView {
            stored_j: ledger.stored_j(),
            capacity_j: ledger.params().store_capacity_j,
            stomach_j: ledger.stomach_j(),
            hunger: ledger.hunger(),
            work_j: ledger.mechanical_net_total_j(),
            heat_j: ledger.heat_total_j(),
            debited_j: ledger.debited_total_j(),
            unfunded_work_j: ledger.unfunded_work_total_j(),
            conservation_residual_j: ledger.conservation_residual_j(),
        },
        sensors,
        read_cost: reading.cost,
        creature_sensor_cost: creature.sensor_cost_total(),
        network: NetworkView {
            neurons: network.neuron_count(),
            synapses: network.synapses.len(),
            inputs: network.input_count as usize,
            cost: network_cost,
        },
    })
}

fn cost_json(cost: &SensorCost) -> Value {
    json!({
        "reads": cost.reads,
        "values": cost.values,
        "rev_reads": cost.rev_reads,
        "bond_reads": cost.bond_reads,
        "pair_tests": cost.pair_tests,
        "operations": cost.operations(),
    })
}

fn number(value: f64) -> String {
    if value != 0.0 && (value.abs() >= 1.0e5 || value.abs() < 1.0e-3) {
        format!("{value:.3e}")
    } else {
        format!("{value:.4}")
    }
}

impl CreatureInspection {
    pub fn to_json(&self) -> Value {
        let ledger = &self.ledger;
        json!({
            "name": self.name,
            "ledger": {
                "stored_energy_j": ledger.stored_j,
                "store_capacity_j": ledger.capacity_j,
                "stomach_j": ledger.stomach_j,
                "hunger": ledger.hunger,
                "work_j": ledger.work_j,
                "heat_j": ledger.heat_j,
                "debited_j": ledger.debited_j,
                "unfunded_work_j": ledger.unfunded_work_j,
                "conservation_residual_j": ledger.conservation_residual_j,
            },
            "sensors": self.sensors.iter().map(|row| json!({
                "id": row.label,
                "unit": row.unit,
                "value": row.value,
            })).collect::<Vec<_>>(),
            "sensor_cost": {
                "this_read": cost_json(&self.read_cost),
                "creature_total": cost_json(&self.creature_sensor_cost),
            },
            "network": {
                "neurons": self.network.neurons,
                "synapses": self.network.synapses,
                "inputs": self.network.inputs,
                "cost": self.network.cost.map(|cost| json!({
                    "steps": cost.steps,
                    "neuron_updates": cost.neuron_updates,
                    "spikes": cost.spikes,
                    "synaptic_events": cost.synaptic_events,
                    "input_ops": cost.input_ops,
                    "operations": cost.operations(),
                })),
                "cost_scope": if self.network.cost.is_some() {
                    "counters of the running network"
                } else {
                    "no network is running in this process"
                },
            },
        })
    }

    /// Ledger and cost rows for the window (sensor channels are paged
    /// separately through [`CreatureInspection::sensor_line`]).
    pub fn summary_lines(&self) -> Vec<String> {
        let l = &self.ledger;
        let network_cost = self.network.cost.map_or_else(
            || "Network cost: not running".to_owned(),
            |c| format!("Network cost: {} steps, {} ops", c.steps, c.operations()),
        );
        vec![
            format!("Energy {} / {} J", number(l.stored_j), number(l.capacity_j)),
            format!("Stomach {} J   hunger {:.2}", number(l.stomach_j), l.hunger),
            format!("Work {} J   Heat {} J", number(l.work_j), number(l.heat_j)),
            format!("Debited {} J   Unfunded {} J", number(l.debited_j), number(l.unfunded_work_j)),
            format!(
                "Sensor read {} ops (total {})",
                self.read_cost.operations(),
                self.creature_sensor_cost.operations()
            ),
            format!(
                "Network: {} neurons, {} synapses",
                self.network.neurons, self.network.synapses
            ),
            network_cost,
        ]
    }

    pub fn sensor_line(&self, index: usize) -> Option<String> {
        let row = self.sensors.get(index)?;
        let value = row.value.map_or_else(|| "—".to_owned(), number);
        Some(format!("{} = {} {}", row.label, value, row.unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        creature::{genome_for_body, OrganParams, SensorLayout},
        documents::{
            BoneKeyframeDocument, EditorDocument, SkeletonDocument, VoxelBoneWeightDocument,
            VoxelModelDocument,
        },
        scene::{SceneBuilder, VoxelSpawn},
        skeletal_actuator::SkeletalActuator,
    };
    use glam::{DVec3, Quat};

    const GROUND: usize = 8;

    fn cells() -> Vec<[i32; 3]> {
        vec![[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0], [3, 1, 0]]
    }

    fn model() -> VoxelModelDocument {
        let mut builder = SceneBuilder::standard();
        for cell in cells() {
            builder = builder.add_voxel(VoxelSpawn::at_cell(cell, 1));
        }
        let scene = builder.build().unwrap();
        let mut model = VoxelModelDocument::from_scene("creature-fixture", &scene);
        let mut skeleton = SkeletonDocument::with_root_bone("root", DVec3::new(0.01, 0.0, 0.0));
        skeleton.add_bone("limb", Some(0), DVec3::new(0.02, 0.0, 0.0)).unwrap();
        skeleton.voxel_weights = (0..8)
            .map(|i| VoxelBoneWeightDocument {
                voxel_index: i,
                bone_index: u32::from(i >= 4),
                weight: 1.0,
            })
            .collect();
        let clip = skeleton.add_animation_clip("bend", 1.0).unwrap();
        for (time_s, angle) in [(0.0, 0.0), (1.0, 0.3)] {
            skeleton
                .set_keyframe(
                    clip,
                    1,
                    BoneKeyframeDocument {
                        time_s,
                        translation_m: DVec3::new(0.02, 0.0, 0.0),
                        rotation: Quat::from_rotation_z(angle),
                    },
                )
                .unwrap();
        }
        model.skeleton = Some(skeleton);
        model.validate().unwrap();
        model
    }

    fn world() -> Scene {
        let mut builder = SceneBuilder::standard();
        for x in 0..4 {
            for y in 0..2 {
                builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, -1], 1).fixed());
            }
        }
        for [x, y, z] in cells() {
            builder = builder.add_voxel(VoxelSpawn::at_cell([x, y, z], 1));
        }
        let mut scene = builder.build().unwrap();
        let actuator = SkeletalActuator::new(model(), GROUND..GROUND + 8, Quat::IDENTITY, 16).unwrap();
        scene.skeletal_actuators.push(actuator);
        scene
    }

    fn organs() -> OrganParams {
        OrganParams {
            store_capacity_j: 100.0,
            birth_store_fraction: 0.5,
            stomach_capacity_j: 20.0,
            digestion_power_w: 2.0,
            absorption_efficiency: 0.8,
            muscle_efficiency: 0.25,
            basal_power_w: 0.0,
            peak_mechanical_power_w: 1.0,
            weak_fraction: 0.1,
            exhaustion_fraction: 0.0,
        }
    }

    fn file(scene: &Scene) -> CreatureFile {
        let layout = SensorLayout::new(&scene.skeletal_actuators[0], vec![0, 4]).unwrap();
        let genome = genome_for_body("creature-fixture", &layout, organs(), 3, 8, 2).unwrap();
        CreatureFile {
            version: CREATURE_FILE_VERSION,
            creatures: vec![CreatureSpec {
                name: "Walker".to_owned(),
                actuator_index: 0,
                genome,
                learned: None,
            }],
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!(
            "E:/voxel-creature-test-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn no_file_means_no_creature() {
        let dir = temp_dir("none");
        assert!(CreatureFile::load(&dir.join("a.vxscene")).unwrap().is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_creature_file_round_trips_builds_and_inspects_without_changing_anything() {
        let dir = temp_dir("inspect");
        let scene_path = dir.join("a.vxscene");
        let scene = world();
        let written = file(&scene);
        fs::write(
            CreatureFile::sidecar_path(&scene_path),
            serde_json::to_string(&written).unwrap(),
        )
        .unwrap();
        let loaded = CreatureFile::load(&scene_path).unwrap().expect("creature file");
        assert_eq!(loaded, written);

        let creatures = loaded.build(&scene).unwrap();
        assert_eq!(creatures.len(), 1);
        let before = creatures[0].creature.sensor_cost_total();
        let first = inspect("Walker", &creatures[0].creature, &scene, None).unwrap();
        let second = inspect("Walker", &creatures[0].creature, &scene, None).unwrap();
        assert_eq!(first, second, "inspecting twice gives the same values");
        assert_eq!(creatures[0].creature.sensor_cost_total(), before, "looking does not count as a creature read");

        // Ledger: a creature the editor does not drive shows its birth ledger.
        assert_eq!(first.ledger.stored_j, 50.0);
        assert_eq!(first.ledger.capacity_j, 100.0);
        assert_eq!(first.ledger.work_j, 0.0);
        assert_eq!(first.ledger.heat_j, 0.0);
        // Sensors: every channel has a label and a unit, values match the catalogue.
        assert_eq!(first.sensors.len(), creatures[0].creature.sensor_layout().catalog().len());
        assert!(first.sensors.iter().all(|row| !row.label.is_empty() && !row.unit.is_empty()));
        let stored = first.sensors.iter().find(|row| row.label == "stored_energy").unwrap();
        assert_eq!((stored.value, stored.unit), (Some(0.5), "1"));
        let height = first.sensors.iter().find(|row| row.label == "com_height").unwrap();
        assert_eq!(height.unit, "m");
        assert!(height.value.is_some());
        assert!(first.sensors.iter().any(|row| row.unit == "N"));
        assert!(first.read_cost.reads == 1 && first.read_cost.operations() > 0);
        assert!(first.network.neurons > 0 && first.network.cost.is_none());

        let json = first.to_json();
        assert_eq!(json["name"], "Walker");
        assert_eq!(json["ledger"]["stored_energy_j"], 50.0);
        assert_eq!(json["sensors"].as_array().unwrap().len(), first.sensors.len());
        assert!(json["network"]["cost"].is_null());
        assert_eq!(json["sensor_cost"]["this_read"]["reads"], 1);

        let with_network = inspect(
            "Walker",
            &creatures[0].creature,
            &scene,
            Some(NetworkCost { steps: 4, neuron_updates: 40, spikes: 3, synaptic_events: 9, input_ops: 2 }),
        )
        .unwrap();
        assert_eq!(with_network.to_json()["network"]["cost"]["operations"], 51);
        let lines = with_network.summary_lines().join("\n");
        assert!(lines.contains("Energy 50.0000 / 100.0000 J"), "{lines}");
        assert!(lines.contains("Network cost: 4 steps, 51 ops"), "{lines}");
        assert!(first.summary_lines().join("\n").contains("not running"));
        assert_eq!(first.summary_lines().len(), CREATURE_SUMMARY_ROWS);
        assert!(first.sensor_line(0).unwrap().contains('='));
        assert!(first.sensor_line(first.sensors.len()).is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn creatures_that_do_not_fit_the_scene_or_the_file_are_errors() {
        let scene = world();
        let mut bad_actuator = file(&scene);
        bad_actuator.creatures[0].actuator_index = 3;
        let error = bad_actuator.build(&scene).err().unwrap();
        assert!(error.contains("Walker"), "{error}");

        let mut bad_version = file(&scene);
        bad_version.version = 5;
        assert!(bad_version.validate().is_err());
        let mut unnamed = file(&scene);
        unnamed.creatures[0].name = " ".into();
        assert!(unnamed.validate().is_err());

        let dir = temp_dir("broken");
        let scene_path = dir.join("a.vxscene");
        fs::write(CreatureFile::sidecar_path(&scene_path), "{ nope").unwrap();
        assert!(CreatureFile::load(&scene_path).is_err());
        let _ = fs::remove_dir_all(dir);
    }
}
