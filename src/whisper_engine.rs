use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::ffi::OsString;
use std::path::Path;

use crate::proc_util::command_with_priority;
use crate::types::Segment;

/// Transcribes `wav16k_path` (16kHz mono wav) by shelling out to a prebuilt
/// whisper.cpp CLI binary (`whisper-cli.exe` / `whisper-cli` / `main.exe`,
/// depending on the release). No compilation involved: just download the
/// official binary release and point Settings at it + at a ggml model file.
///
/// `language`: an ISO 639-1 code (e.g. "fr"), or "auto" for auto-detection.
///
/// GPU note: whisper-cli's use of a GPU depends entirely on which
/// whisper.cpp binary you downloaded (a plain CPU build vs a cuBLAS/Vulkan
/// build) — there is nothing to toggle here, see the README.
pub fn transcribe(
    whisper_exe: &Path,
    model_path: &Path,
    wav16k_path: &Path,
    language: &str,
    low_priority: bool,
) -> Result<Vec<Segment>> {
    // Output prefix: same dir/stem as the wav, whisper.cpp will write
    // "<prefix>.json" when passed -oj -of <prefix>.
    let out_prefix = wav16k_path.with_extension("");
    let lang = if language.trim().is_empty() {
        "auto"
    } else {
        language
    };

    let whisper_exe_str = whisper_exe
        .to_str()
        .context("le chemin de whisper-cli n'est pas de l'UTF-8 valide")?;

    let args: Vec<OsString> = vec![
        "-m".into(),
        model_path.as_os_str().to_os_string(),
        "-f".into(),
        wav16k_path.as_os_str().to_os_string(),
        "-l".into(),
        lang.into(),
        "-oj".into(),
        "-of".into(),
        out_prefix.as_os_str().to_os_string(),
    ];

    let output = command_with_priority(whisper_exe_str, &args, low_priority)
        .output()
        .with_context(|| {
            format!(
                "impossible de lancer {whisper_exe:?} (chemin correct vers whisper-cli.exe ?)"
            )
        })?;

    if !output.status.success() {
        bail!(
            "whisper-cli a échoué (code {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let json_path = out_prefix.with_extension("json");
    let content = std::fs::read_to_string(&json_path).with_context(|| {
        format!(
            "fichier JSON de sortie introuvable ({json_path:?}) ; \
             ta version de whisper-cli utilise peut-être des options différentes (-oj/-of)"
        )
    })?;
    let root: Value = serde_json::from_str(&content)
        .with_context(|| format!("JSON invalide dans {json_path:?}"))?;

    let entries = root
        .get("transcription")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut out = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let text = entry
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if text.is_empty() {
            continue;
        }

        let (start, end) = if let Some(offsets) = entry.get("offsets") {
            let from_ms = offsets.get("from").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let to_ms = offsets.get("to").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (from_ms as f32 / 1000.0, to_ms as f32 / 1000.0)
        } else if let Some(timestamps) = entry.get("timestamps") {
            let from = timestamps
                .get("from")
                .and_then(|v| v.as_str())
                .map(parse_srt_timestamp)
                .unwrap_or(0.0);
            let to = timestamps
                .get("to")
                .and_then(|v| v.as_str())
                .map(parse_srt_timestamp)
                .unwrap_or(0.0);
            (from, to)
        } else {
            (0.0, 0.0)
        };

        out.push(Segment {
            id: i,
            start,
            end: end.max(start + 0.1),
            original_text: text,
            translated_text: String::new(),
            synth_path: None,
            synth_duration: None,
        });
    }

    if out.is_empty() {
        bail!("whisper-cli n'a produit aucun segment de texte (fichier vide ou silencieux ?)");
    }

    Ok(out)
}

/// Parses a "HH:MM:SS,mmm" timestamp (whisper.cpp's srt-style format) into seconds.
fn parse_srt_timestamp(s: &str) -> f32 {
    let s = s.trim();
    let (hms, ms) = s.split_once(',').unwrap_or((s, "0"));
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return 0.0;
    }
    let h: f32 = parts[0].parse().unwrap_or(0.0);
    let m: f32 = parts[1].parse().unwrap_or(0.0);
    let sec: f32 = parts[2].parse().unwrap_or(0.0);
    let msec: f32 = ms.parse().unwrap_or(0.0);
    h * 3600.0 + m * 60.0 + sec + msec / 1000.0
}
