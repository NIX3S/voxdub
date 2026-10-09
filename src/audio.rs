use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::proc_util::command_with_priority;

/// Converts any input audio (ogg/mp3/wav/m4a/...) to 16kHz mono PCM16 wav,
/// the format whisper.cpp expects.
pub fn convert_to_wav16k_mono(input: &Path, output: &Path, low_priority: bool) -> Result<()> {
    run_ffmpeg(
        &[
            "-y",
            "-i",
            path_str(input)?,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            path_str(output)?,
        ],
        low_priority,
    )
}

/// Extracts a short reference clip (for XTTS speaker cloning) starting at
/// `start_s`, `len_s` seconds long, at the sample rate the TTS engine wants.
pub fn extract_reference_clip(
    input: &Path,
    output: &Path,
    start_s: f32,
    len_s: f32,
    low_priority: bool,
) -> Result<()> {
    run_ffmpeg(
        &[
            "-y",
            "-ss",
            &start_s.to_string(),
            "-t",
            &len_s.to_string(),
            "-i",
            path_str(input)?,
            "-ar",
            "24000",
            "-ac",
            "1",
            path_str(output)?,
        ],
        low_priority,
    )
}

/// Extracts `[start_s, start_s+len_s]` from `input` as a playable wav clip
/// (used for the before/after preview player, not the TTS speaker ref).
pub fn extract_clip_to_wav(
    input: &Path,
    output: &Path,
    start_s: f32,
    len_s: f32,
    low_priority: bool,
) -> Result<()> {
    run_ffmpeg(
        &[
            "-y",
            "-ss",
            &start_s.to_string(),
            "-t",
            &len_s.max(0.05).to_string(),
            "-i",
            path_str(input)?,
            "-c:a",
            "pcm_s16le",
            path_str(output)?,
        ],
        low_priority,
    )
}

/// Re-encodes `input` (any format ffmpeg understands) into `output`, whose
/// container/codec is inferred by ffmpeg from `output`'s extension. Used to
/// deliver the dubbed file in the same format as the original input
/// (ogg -> ogg, mp3 -> mp3, wav -> wav, ...).
pub fn transcode(input: &Path, output: &Path, low_priority: bool) -> Result<()> {
    run_ffmpeg(&["-y", "-i", path_str(input)?, path_str(output)?], low_priority)
}

/// Returns the duration of an audio file in seconds, via ffprobe.
pub fn probe_duration(input: &Path) -> Result<f64> {
    let out = command_with_priority(
        "ffprobe",
        &[
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path_str(input)?,
        ],
        false, // probing is quick, no need to deprioritize
    )
    .output()
    .context("failed to run ffprobe (is ffmpeg/ffprobe installed and on PATH?)")?;
    if !out.status.success() {
        bail!(
            "ffprobe failed on {:?}: {}",
            input,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim()
        .parse::<f64>()
        .with_context(|| format!("could not parse ffprobe duration output: {s:?}"))
}

/// Time-stretches/compresses `input` so its duration matches `target_s`,
/// writing the result to `output`. Uses ffmpeg's `atempo` filter, chained as
/// needed since a single atempo stage only covers factors in [0.5, 2.0].
pub fn stretch_to_duration(
    input: &Path,
    output: &Path,
    target_s: f32,
    low_priority: bool,
) -> Result<()> {
    let current = probe_duration(input)? as f32;
    if current <= 0.01 {
        bail!("source clip {:?} has ~zero duration", input);
    }
    // factor > 1 => speed up (audio is longer than target)
    // factor < 1 => slow down (audio is shorter than target)
    let mut factor = current / target_s;
    factor = factor.clamp(0.25, 4.0);

    let mut stages = Vec::new();
    let mut remaining = factor;
    while remaining > 2.0 {
        stages.push(2.0_f32);
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        stages.push(0.5_f32);
        remaining /= 0.5;
    }
    stages.push(remaining);

    let filter = stages
        .iter()
        .map(|f| format!("atempo={f:.4}"))
        .collect::<Vec<_>>()
        .join(",");

    run_ffmpeg(
        &[
            "-y",
            "-i",
            path_str(input)?,
            "-filter:a",
            &filter,
            "-t",
            &target_s.to_string(),
            path_str(output)?,
        ],
        low_priority,
    )
}

/// Builds a silent track of `total_s` seconds at `sample_rate`, then mixes
/// each (path, start_s) clip on top at its own timeline offset, producing a
/// dubbed track whose total length and per-line timing follow the original.
pub fn stitch_on_timeline(
    clips: &[(PathBuf, f32)],
    total_s: f32,
    sample_rate: u32,
    output: &Path,
    low_priority: bool,
) -> Result<()> {
    let mut args: Vec<String> = vec!["-y".into()];
    args.push("-f".into());
    args.push("lavfi".into());
    args.push("-i".into());
    args.push(format!("anullsrc=r={sample_rate}:cl=mono"));

    for (p, _) in clips {
        args.push("-i".into());
        args.push(path_str(p)?.to_string());
    }

    let mut filter_parts = Vec::new();
    let mut mix_labels = vec!["0:a".to_string()];
    for (i, (_, start)) in clips.iter().enumerate() {
        let delay_ms = (start * 1000.0).max(0.0) as i64;
        let label = format!("d{i}");
        filter_parts.push(format!("[{}:a]adelay={delay_ms}|{delay_ms}[{label}]", i + 1));
        mix_labels.push(format!("{label}"));
    }
    let mix_inputs = mix_labels
        .iter()
        .map(|l| format!("[{l}]"))
        .collect::<String>();
    filter_parts.push(format!(
        "{mix_inputs}amix=inputs={}:duration=first:dropout_transition=0[out]",
        mix_labels.len()
    ));
    let filter_complex = filter_parts.join(";");

    args.push("-filter_complex".into());
    args.push(filter_complex);
    args.push("-map".into());
    args.push("[out]".into());
    args.push("-t".into());
    args.push(total_s.to_string());
    args.push(path_str(output)?.to_string());

    let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    run_ffmpeg(&arg_refs, low_priority)
}

/// Simple back-to-back concatenation of clips, no timeline sync (used for
/// OutputMode::JustTranslated).
pub fn concat_sequential(
    clips: &[PathBuf],
    output: &Path,
    workdir: &Path,
    low_priority: bool,
) -> Result<()> {
    let list_path = workdir.join("concat_list.txt");
    let mut content = String::new();
    for p in clips {
        content.push_str(&format!("file '{}'\n", p.display()));
    }
    std::fs::write(&list_path, content)?;
    run_ffmpeg(
        &[
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            path_str(&list_path)?,
            "-c",
            "copy",
            path_str(output)?,
        ],
        low_priority,
    )
}

fn path_str(p: &Path) -> Result<&str> {
    p.to_str()
        .with_context(|| format!("non-UTF8 path: {p:?}"))
}

fn run_ffmpeg(args: &[&str], low_priority: bool) -> Result<()> {
    let out = command_with_priority("ffmpeg", args, low_priority)
        .output()
        .context("failed to run ffmpeg (is it installed and on PATH?)")?;
    if !out.status.success() {
        bail!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}
