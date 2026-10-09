use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;

use crate::proc_util::command_with_priority;

#[derive(Serialize)]
struct TranslateRequest<'a> {
    texts: &'a [String],
    src: &'a str,
    tgt: &'a str,
}

#[derive(Deserialize)]
struct TranslateResponse {
    translations: Vec<String>,
}

/// Forces the Python child process to use UTF-8 for stdin/stdout/stderr
/// regardless of the OS locale/console code page. On Windows, Python
/// defaults to the console's code page (often cp1252), which both corrupts
/// accented text we exchange over stdin/stdout and crashes on characters it
/// can't encode (e.g. combining diacritics) when a library does a stray
/// print(). PYTHONUTF8 switches the interpreter's default encoding to
/// UTF-8; PYTHONIOENCODING pins the std streams specifically, with
/// `:replace` so an unencodable character degrades instead of crashing.
fn configure_utf8(cmd: &mut std::process::Command) {
    cmd.env("PYTHONUTF8", "1");
    cmd.env("PYTHONIOENCODING", "utf-8:replace");
}

/// Calls scripts/translate.py (Argos Translate, fully offline) with a batch
/// of segment texts and returns the translated texts, same order.
pub fn translate_segments(
    python_bin: &str,
    script_path: &Path,
    texts: &[String],
    src: &str,
    tgt: &str,
    low_priority: bool,
) -> Result<Vec<String>> {
    let req = TranslateRequest { texts, src, tgt };
    let payload = serde_json::to_vec(&req)?;

    let script_str = script_path
        .to_str()
        .context("le chemin de translate.py n'est pas de l'UTF-8 valide")?;
    let mut cmd = command_with_priority(python_bin, &[script_str], low_priority);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_utf8(&mut cmd);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch {python_bin} {script_path:?}"))?;

    child
        .stdin
        .take()
        .context("no stdin on translate.py child")?
        .write_all(&payload)?;

    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!(
            "translate.py failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let resp: TranslateResponse = serde_json::from_slice(&out.stdout).with_context(|| {
        format!(
            "bad JSON from translate.py: {}",
            String::from_utf8_lossy(&out.stdout)
        )
    })?;
    Ok(resp.translations)
}

#[derive(Serialize)]
struct CloneItemReq<'a> {
    text: &'a str,
    out: String,
}

#[derive(Serialize)]
struct CloneBatchReq<'a> {
    lang: &'a str,
    speaker: String,
    items: Vec<CloneItemReq<'a>>,
}

#[derive(Deserialize)]
struct CloneResultLine {
    index: usize,
    ok: bool,
    out: Option<String>,
    error: Option<String>,
}

/// Calls scripts/clone_tts.py (Coqui XTTS-v2, fully offline) to synthesize
/// an ENTIRE file's worth of segments in ONE Python process / ONE model
/// load, instead of spawning a fresh process (and reloading the ~2GB XTTS
/// model, which can take 10-30s+) for every single segment. This was the
/// single biggest performance bottleneck in the per-segment version: with
/// dozens of segments per file across thousands of files, the model reload
/// overhead alone could dwarf actual synthesis time.
///
/// `items`: (text, output_wav_path) pairs, in segment order.
/// `on_result` is called once per segment AS IT COMPLETES (the script
/// streams one JSON result line per segment to stdout), so the caller can
/// still report live per-segment progress.
pub fn clone_tts_batch(
    python_bin: &str,
    script_path: &Path,
    items: &[(String, PathBuf)],
    lang: &str,
    speaker_wav: &Path,
    use_gpu: bool,
    low_priority: bool,
    on_result: &mut dyn FnMut(usize, Result<PathBuf>),
) -> Result<()> {
    let req = CloneBatchReq {
        lang,
        speaker: speaker_wav.to_string_lossy().to_string(),
        items: items
            .iter()
            .map(|(t, p)| CloneItemReq {
                text: t,
                out: p.to_string_lossy().to_string(),
            })
            .collect(),
    };
    let payload = serde_json::to_vec(&req)?;

    let script_str = script_path
        .to_str()
        .context("le chemin de clone_tts.py n'est pas de l'UTF-8 valide")?;
    let mut script_args: Vec<&str> = vec![script_str];
    if use_gpu {
        script_args.push("--gpu");
    }

    let mut cmd = command_with_priority(python_bin, &script_args, low_priority);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    configure_utf8(&mut cmd);

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to launch {python_bin} {script_path:?}"))?;

    let mut stdin = child
        .stdin
        .take()
        .context("no stdin on clone_tts.py child")?;
    let writer = std::thread::spawn(move || {
        // Errors here surface anyway via the exit status / stderr below.
        let _ = stdin.write_all(&payload);
        // `stdin` drops at the end of this closure, closing the pipe and
        // signaling EOF to Python so it can start processing.
    });

    let mut stderr_pipe = child
        .stderr
        .take()
        .context("no stderr on clone_tts.py child")?;
    let stderr_reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr_pipe.read_to_string(&mut buf);
        buf
    });

    let stdout = child
        .stdout
        .take()
        .context("no stdout on clone_tts.py child")?;
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let line = line.context("failed to read clone_tts.py output")?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parsed: CloneResultLine = match serde_json::from_str(line) {
            Ok(p) => p,
            Err(_) => continue, // stray non-JSON line (shouldn't happen, ignore defensively)
        };
        if parsed.ok {
            let path = PathBuf::from(parsed.out.unwrap_or_default());
            on_result(parsed.index, Ok(path));
        } else {
            on_result(
                parsed.index,
                Err(anyhow!(parsed.error.unwrap_or_else(|| "erreur inconnue".into()))),
            );
        }
    }

    let _ = writer.join();
    let stderr_buf = stderr_reader.join().unwrap_or_default();
    let status = child
        .wait()
        .context("failed to wait on clone_tts.py")?;
    if !status.success() {
        let lines: Vec<&str> = stderr_buf.lines().collect();
        let tail = lines[lines.len().saturating_sub(12)..].join("\n");
        bail!("clone_tts.py a échoué ({status}):\n{tail}");
    }
    Ok(())
}
