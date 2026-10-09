mod app;
mod audio;
mod external;
mod persistence;
mod pipeline;
mod proc_util;
mod types;
mod whisper_engine;

use app::VoxDubApp;

fn main() -> eframe::Result<()> {
    // scripts/ is expected next to the executable (or the crate root when
    // running via `cargo run`).
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()));
    let candidates = [
        exe_dir.map(|d| d.join("scripts")),
        Some(std::path::PathBuf::from("scripts")),
    ];
    let scripts_dir = candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("scripts"));

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1100.0, 720.0]),
        ..Default::default()
    };

    eframe::run_native(
        "VoxDub — transcription, traduction & clonage vocal",
        options,
        Box::new(|_cc| Ok(Box::new(VoxDubApp::new(scripts_dir)))),
    )
}
