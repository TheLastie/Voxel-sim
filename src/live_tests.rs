//! Low-rate snapshots of real solver state for the native live-test viewer.
//! Enable by setting `VOXEL_LIVE_TESTS_DIR` to a local directory.

use crate::scene::Scene;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Condvar, Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

const FRAME_VERSION: u32 = 1;
const SAMPLE_INTERVAL: Duration = Duration::from_millis(100);

/// Complete physical REV snapshot. `positions` are relative to `origin` to
/// retain f32 precision in the viewer; no REVs are dropped or merged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LiveTestFrame {
    pub version: u32,
    pub sequence: u64,
    pub test_name: String,
    pub simulated_seconds: f64,
    pub wall_ms: f64,
    pub voxel_count: usize,
    pub cell_m: f32,
    pub origin: [f64; 3],
    pub positions: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
}

struct Writer {
    latest: std::sync::Arc<(Mutex<Option<LiveTestFrame>>, Condvar)>,
    last_sample: Mutex<Instant>,
    sequence: std::sync::atomic::AtomicU64,
    directory: PathBuf,
    written: std::sync::Arc<(Mutex<u64>, Condvar)>,
}

static WRITER: OnceLock<Option<Writer>> = OnceLock::new();
static ENABLED: OnceLock<bool> = OnceLock::new();
// 0 follows the launch environment, 1/2 are explicit editor enable/disable.
static RECORDING_MODE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether live native snapshots are enabled by the local output-directory
/// environment setting. Checking this does not create the directory or thread.
pub fn enabled() -> bool {
    match RECORDING_MODE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => *ENABLED.get_or_init(|| std::env::var_os("VOXEL_LIVE_TESTS_DIR").is_some()),
    }
}

/// Enable snapshots when entering the engine's test workspace, including
/// after a normal scene has already checked the optional observer setting.
pub fn enable(directory: &Path) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    if let Some(Some(current)) = WRITER.get() {
        if current.directory != directory {
            return Err("the active snapshot writer uses another directory".into());
        }
    }
    std::env::set_var("VOXEL_LIVE_TESTS_DIR", directory);
    RECORDING_MODE.store(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Stop capturing when returning to ordinary scene editing.
pub fn disable() {
    RECORDING_MODE.store(2, std::sync::atomic::Ordering::Relaxed);
}

fn writer() -> Option<&'static Writer> {
    if !enabled() {
        return None;
    }
    let directory = std::env::var_os("VOXEL_LIVE_TESTS_DIR").map(PathBuf::from)?;
    WRITER
        .get_or_init(|| {
            fs::create_dir_all(&directory).ok()?;
            let latest = std::sync::Arc::new((Mutex::new(None), Condvar::new()));
            let written = std::sync::Arc::new((Mutex::new(0), Condvar::new()));
            let worker_directory = directory.clone();
            let worker_latest = latest.clone();
            let worker_written = written.clone();
            thread::Builder::new()
                .name("voxel-live-test-writer".into())
                .spawn(move || writer_loop(worker_directory, worker_latest, worker_written))
                .ok()?;
            Some(Writer {
                latest,
                last_sample: Mutex::new(Instant::now() - SAMPLE_INTERVAL),
                sequence: std::sync::atomic::AtomicU64::new(0),
                directory,
                written,
            })
        })
        .as_ref()
}

/// Capture a throttled, complete current scene snapshot for the external
/// native viewer. This does not advance physics or modify Scene state.
pub fn observe_scene(scene: &Scene, test_name: &str, simulated_seconds: f64, wall_ms: f64) {
    observe_scene_impl(scene, test_name, simulated_seconds, wall_ms, false);
}

/// Write a final snapshot immediately, even if the regular throttle interval
/// has not elapsed. Intended for end-of-test capture.
pub fn observe_scene_force(scene: &Scene, test_name: &str, simulated_seconds: f64, wall_ms: f64) {
    observe_scene_impl(scene, test_name, simulated_seconds, wall_ms, true);
}

fn observe_scene_impl(
    scene: &Scene,
    test_name: &str,
    simulated_seconds: f64,
    wall_ms: f64,
    force: bool,
) {
    let Some(writer) = writer() else { return };
    let now = Instant::now();
    if !force {
        let Ok(mut last) = writer.last_sample.lock() else {
            return;
        };
        if now.duration_since(*last) < SAMPLE_INTERVAL {
            return;
        }
        *last = now;
    }

    let origin = scene
        .positions
        .first()
        .copied()
        .unwrap_or(glam::DVec3::ZERO);
    let material_colors: Vec<[f32; 3]> = scene
        .materials
        .iter()
        .map(|material| palette_color(&material.name))
        .collect();
    let mut positions = Vec::with_capacity(scene.positions.len());
    let mut colors = Vec::with_capacity(scene.voxels.len());
    for (index, position) in scene.positions.iter().enumerate() {
        let relative = *position - origin;
        positions.push([relative.x as f32, relative.y as f32, relative.z as f32]);
        let voxel = &scene.voxels[index];
        let mut color = [0.0; 3];
        let mut total = 0.0_f32;
        for &(component, fraction) in &voxel.composition.components {
            let component_color = material_colors
                .get(component as usize)
                .copied()
                .unwrap_or([0.7, 0.7, 0.7]);
            for axis in 0..3 {
                color[axis] += component_color[axis] * fraction;
            }
            total += fraction;
        }
        if total > 0.0 {
            for channel in &mut color {
                *channel /= total;
            }
        } else {
            color = [0.7, 0.7, 0.7];
        }
        colors.push(color);
    }

    let test_name = if test_name.is_empty() {
        std::env::var("VOXEL_LIVE_TEST_NAME")
            .ok()
            .filter(|name| !name.is_empty())
            .or_else(|| {
                std::env::current_exe()
                    .ok()?
                    .file_stem()?
                    .to_str()
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| "physics test process".to_owned())
    } else {
        test_name.to_owned()
    };
    let mut frame = LiveTestFrame {
        version: FRAME_VERSION,
        sequence: 0,
        test_name,
        simulated_seconds,
        wall_ms,
        voxel_count: scene.voxels.len(),
        cell_m: crate::scene::CELL,
        origin: origin.to_array(),
        positions,
        colors,
    };

    // Keep only the newest pending frame. If encoding falls behind, it will
    // replace this slot rather than queueing stale snapshots.
    if let Ok(mut pending) = writer.latest.0.lock() {
        // Order frames when publishing, after snapshot construction. Parallel
        // test threads must not replace a newer sequence with an older one.
        frame.sequence = writer
            .sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            .wrapping_add(1);
        *pending = Some(frame);
        writer.latest.1.notify_one();
    }
}

/// Exposed for the viewer process to discover the configured local directory.
pub fn snapshot_directory() -> Option<PathBuf> {
    writer().map(|writer| writer.directory.clone())
}

/// Wait until all snapshots submitted before this call are visible to the
/// viewer. Intended for final capture outside the measured physics interval.
pub fn flush(timeout: Duration) -> bool {
    let Some(writer) = writer() else { return false };
    let expected = writer.sequence.load(std::sync::atomic::Ordering::Relaxed);
    let Ok(written) = writer.written.0.lock() else {
        return false;
    };
    let result = writer
        .written
        .1
        .wait_timeout_while(written, timeout, |written| *written < expected);
    result
        .map(|(written, _)| *written >= expected)
        .unwrap_or(false)
}

fn writer_loop(
    directory: PathBuf,
    latest: std::sync::Arc<(Mutex<Option<LiveTestFrame>>, Condvar)>,
    written_state: std::sync::Arc<(Mutex<u64>, Condvar)>,
) {
    loop {
        let frame = {
            let Ok(mut pending) = latest.0.lock() else {
                return;
            };
            while pending.is_none() {
                let Ok(guard) = latest.1.wait(pending) else {
                    return;
                };
                pending = guard;
            }
            pending.take().expect("frame slot was checked above")
        };
        let slot = frame.sequence % 2;
        let final_path = directory.join(format!("frame-{slot}.bin"));
        let temp_path = directory.join(format!(
            "frame-{slot}-{}-{}.tmp",
            std::process::id(),
            frame.sequence
        ));
        if write_frame_atomic(&temp_path, &final_path, &frame).is_err() {
            let _ = fs::remove_file(&temp_path);
        } else if let Ok(mut written) = written_state.0.lock() {
            *written = (*written).max(frame.sequence);
            written_state.1.notify_all();
        }
    }
}

fn write_frame_atomic(
    temp: &Path,
    final_path: &Path,
    frame: &LiveTestFrame,
) -> std::io::Result<()> {
    // bincode emits small scalar writes. Buffer them instead of issuing a
    // filesystem write for each coordinate of a large physical scene.
    let mut file = BufWriter::with_capacity(
        1024 * 1024,
        OpenOptions::new().write(true).create_new(true).open(temp)?,
    );
    bincode::serialize_into(&mut file, frame)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    file.flush()?;
    drop(file);

    // Windows does not replace an existing destination with std::fs::rename.
    // The viewer retries transient read errors while the two-slot snapshot is
    // switched; unique temp names ensure no partial frame is ever read.
    if final_path.exists() {
        fs::remove_file(final_path)?;
    }
    fs::rename(temp, final_path)
}

fn palette_color(name: &str) -> [f32; 3] {
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "wood" | "oak" | "pine" | "birch" | "spruce" | "flesh" => [0.69, 0.39, 0.17],
        "iron" | "steel" | "copper" | "aluminum" => [0.55, 0.62, 0.70],
        "water" | "ice" => [0.17, 0.56, 0.87],
        "air" | "steam" => [0.78, 0.83, 0.89],
        "stone" | "concrete" => [0.48, 0.49, 0.51],
        _ => {
            let hash = name.bytes().fold(0x811c_9dc5_u32, |hash, byte| {
                hash.wrapping_mul(0x0100_0193) ^ u32::from(byte)
            });
            let hue = (hash % 360) as f32 / 60.0;
            let chroma = 0.58;
            let x = chroma * (1.0 - ((hue % 2.0) - 1.0).abs());
            let (r, g, b) = match hue as u32 {
                0 => (chroma, x, 0.0),
                1 => (x, chroma, 0.0),
                2 => (0.0, chroma, x),
                3 => (0.0, x, chroma),
                4 => (x, 0.0, chroma),
                _ => (chroma, 0.0, x),
            };
            [r + 0.2, g + 0.2, b + 0.2]
        }
    }
}
