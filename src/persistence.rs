use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::types::{BatchMode, FileJob, Settings};

/// Everything needed to pick a session back up: settings, every file's
/// current stage/segments, and which batch mode (if any) was running.
/// Written to disk next to the executable so the app can resume after it
/// was closed, crashed, or the machine was rebooted — not just while it
/// stays open (that case is handled by Pause/Resume instead).
#[derive(Serialize, Deserialize)]
pub struct SavedState {
    pub settings: Settings,
    pub jobs: Vec<FileJob>,
    pub batch_mode: Option<BatchMode>,
}

fn state_path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("voxdub_state.json")))
        .unwrap_or_else(|| PathBuf::from("voxdub_state.json"))
}

pub fn save(state: &SavedState) -> anyhow::Result<()> {
    let path = state_path();
    let json = serde_json::to_vec_pretty(state)?;
    std::fs::write(&path, json)?;
    Ok(())
}

/// Loads the last saved session, if any. Returns `None` on first run, or if
/// the file is missing/unreadable/corrupt — never panics, a bad or absent
/// state file just means "start fresh."
pub fn load() -> Option<SavedState> {
    let bytes = std::fs::read(state_path()).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub fn clear() {
    let _ = std::fs::remove_file(state_path());
}
