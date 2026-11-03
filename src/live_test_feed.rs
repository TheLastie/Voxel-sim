//! Background reader for live-test snapshots and runner status.
//!
//! This module only reads local files. Snapshot decoding and surface-mask
//! generation happen on its worker thread so engine/UI frame handlers only
//! take ownership of the latest prepared values.

use crate::{live_tests::LiveTestFrame, voxel_surface::exterior_face_masks};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}},
    thread,
    time::{Duration, SystemTime},
};

const FRAME_VERSION: u32 = 1;
const POLL_INTERVAL: Duration = Duration::from_millis(40);

/// Status published by the local test runner in `run-status.json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct RunStatus {
    pub label: String,
    pub status: String,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub detail: String,
}

impl Default for RunStatus {
    fn default() -> Self {
        Self {
            label: "test run".to_owned(),
            status: "starting".to_owned(),
            passed: 0,
            failed: 0,
            ignored: 0,
            detail: String::new(),
        }
    }
}

/// A real physics snapshot plus data prepared for surface-only rendering.
pub struct FrameSnapshot {
    pub frame: LiveTestFrame,
    pub process_epoch: u64,
    pub surface_masks: Vec<u8>,
}

/// Polls one local directory for the latest test frame and runner status.
///
/// The receiver is intentionally not cloneable: `take_frame` moves the large
/// frame and mask vectors to the caller instead of copying them every UI tick.
pub struct LiveTestFeed {
    frame: Arc<Mutex<Option<FrameSnapshot>>>,
    status: Arc<Mutex<Option<RunStatus>>>,
    stop: Arc<AtomicBool>,
}

impl LiveTestFeed {
    /// Start a background reader for snapshots in `directory`.
    pub fn start(directory: PathBuf) -> Self {
        let frame = Arc::new(Mutex::new(None));
        let status = Arc::new(Mutex::new(None));
        let reader_frame = Arc::clone(&frame);
        let reader_status = Arc::clone(&status);
        let stop = Arc::new(AtomicBool::new(false));
        let reader_stop = Arc::clone(&stop);

        thread::Builder::new()
            .name("voxel-live-test-feed".to_owned())
            .spawn(move || {
                let mut latest_sequence = 0_u64;
                let mut process_epoch = 0_u64;
                let mut latest_modified = None;
                let mut observed_frames: [Option<(SystemTime, u64)>; 2] = [None, None];
                let mut observed_status: Option<RunStatus> = None;
                let status_path = directory.join("run-status.json");

                while !reader_stop.load(Ordering::Relaxed) {
                    for (slot, observed) in observed_frames.iter_mut().enumerate() {
                        let path = directory.join(format!("frame-{slot}.bin"));
                        let Ok(metadata) = fs::metadata(&path) else {
                            continue;
                        };
                        let Ok(modified) = metadata.modified() else {
                            continue;
                        };
                        let signature = (modified, metadata.len());
                        if *observed == Some(signature) {
                            continue;
                        }
                        let Ok(bytes) = fs::read(&path) else {
                            continue;
                        };
                        let Ok(snapshot) = bincode::deserialize::<LiveTestFrame>(&bytes) else {
                            continue;
                        };
                        *observed = Some(signature);

                        if snapshot.version != FRAME_VERSION
                            || !is_newer_frame(
                                modified,
                                snapshot.sequence,
                                latest_modified,
                                latest_sequence,
                            )
                        {
                            continue;
                        }

                        if is_new_process_snapshot(
                            snapshot.sequence,
                            latest_sequence,
                            latest_modified.is_some(),
                        ) {
                            process_epoch = process_epoch.wrapping_add(1);
                        }
                        latest_sequence = snapshot.sequence;
                        latest_modified = Some(modified);
                        let surface_masks =
                            exterior_face_masks(&snapshot.positions, snapshot.cell_m);
                        if let Ok(mut latest) = reader_frame.lock() {
                            *latest = Some(FrameSnapshot {
                                frame: snapshot,
                                process_epoch,
                                surface_masks,
                            });
                        }
                    }

                    if let Ok(bytes) = fs::read(&status_path) {
                        if let Ok(next_status) = serde_json::from_slice::<RunStatus>(&bytes) {
                            if observed_status.as_ref() != Some(&next_status) {
                                observed_status = Some(next_status.clone());
                                if let Ok(mut latest) = reader_status.lock() {
                                    *latest = Some(next_status);
                                }
                            }
                        }
                    }

                    thread::sleep(POLL_INTERVAL);
                }
            })
            .expect("start live-test feed reader");

        Self { frame, status, stop }
    }

    /// Move out the most recent prepared snapshot, if one is available.
    pub fn take_frame(&self) -> Option<FrameSnapshot> {
        self.frame.lock().ok()?.take()
    }

    /// Move out the latest changed runner status, if one is available.
    pub fn take_status(&self) -> Option<RunStatus> {
        self.status.lock().ok()?.take()
    }
}

impl Drop for LiveTestFeed {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn is_newer_frame(
    candidate_modified: SystemTime,
    candidate_sequence: u64,
    latest_modified: Option<SystemTime>,
    latest_sequence: u64,
) -> bool {
    match latest_modified {
        Some(latest) if candidate_modified < latest => false,
        Some(latest) if candidate_modified == latest => candidate_sequence > latest_sequence,
        _ => true,
    }
}

fn is_new_process_snapshot(
    candidate_sequence: u64,
    latest_sequence: u64,
    has_latest: bool,
) -> bool {
    has_latest && candidate_sequence <= latest_sequence
}
