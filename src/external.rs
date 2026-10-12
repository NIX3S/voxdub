use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::{Arc, Mutex};

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
struct CloneFileReq<'a> {
    req_id: u64,
    lang: &'a str,
    speaker: String,
    items: Vec<CloneItemReq<'a>>,
}

/// One line of clone_tts.py's persistent-worker protocol: either the
/// one-off device announcement right after model load, a per-segment
/// result, the "this file is done" marker, or a fatal error. Fields are all
/// optional because which ones are present depends on which kind of line
/// it is — there's no tag, callers just check what's `Some`.
#[derive(Deserialize, Default)]
struct WorkerLine {
    device: Option<String>,
    index: Option<usize>,
    ok: Option<bool>,
    out: Option<String>,
    error: Option<String>,
    done: Option<bool>,
    fatal: Option<String>,
}

const STDERR_TAIL_LINES: usize = 200;

/// A long-lived `clone_tts.py` subprocess that loads the XTTS model ONCE
/// and then stays alive to synthesize many files one after another, over a
/// small newline-delimited JSON protocol on stdin/stdout — instead of the
/// previous design, which spawned a fresh Python process (reloading the
/// ~2GB model, 10-30s+) for every single FILE. With thousands of files in a
/// batch, that per-file reload cost alone could dwarf actual synthesis
/// time; keeping one worker alive per `max_parallel` "slot" for the whole
/// batch amortizes the load to (ideally) once per slot instead of once per
/// file. One segment-level model reload per file was already eliminated
/// earlier (see the git history / README) — this removes the next layer.
pub struct CloneWorker {
    child: Child,
    /// `None` once `run_file` detects the process is dead/misbehaving, so
    /// the caller knows to stop reusing this worker and spawn a fresh one.
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    /// "cuda" or "cpu" — captured once right after the model finished
    /// loading. Fixed for the worker's whole lifetime (a process can't
    /// change device mid-run), so every file it processes reuses this
    /// instead of re-parsing a device line each time.
    pub device: String,
}

impl CloneWorker {
    /// Spawns clone_tts.py and blocks until it reports the device it loaded
    /// the model on (or dies trying) — i.e. this returns only once the
    /// (potentially slow, 10-30s+, or much longer on first run / slow CPU)
    /// model load has finished.
    ///
    /// `on_log` is called with every stderr line clone_tts.py prints WHILE
    /// loading (torch/CUDA diagnostics, download progress, the CPML-license
    /// notice, ...), as it happens — not just dumped after the fact on
    /// failure — so the caller can forward it to the UI and the user isn't
    /// staring at a frozen "chargement…" message with zero information for
    /// however long this takes.
    pub fn spawn(
        python_bin: &str,
        script_path: &Path,
        use_gpu: bool,
        low_priority: bool,
        on_log: &mut dyn FnMut(&str),
    ) -> Result<Self> {
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
        // Coqui TTS gates the XTTS-v2 model (non-commercial CPML license)
        // behind an interactive confirmation prompt (`input()`) the very
        // first time it's loaded. Our stdin is a pipe, not a real terminal,
        // so that prompt would block forever waiting for a keystroke that
        // can never arrive — this is the documented opt-out (clone_tts.py
        // also sets this itself, and patches `input()` as a further
        // fallback, in case some other prompt shows up).
        cmd.env("COQUI_TOS_AGREED", "1");

        let mut child = cmd
            .spawn()
            .with_context(|| format!("failed to launch {python_bin} {script_path:?}"))?;

        let stdin = child.stdin.take().context("no stdin on clone_tts.py child")?;
        let stdout = child.stdout.take().context("no stdout on clone_tts.py child")?;
        let mut stdout = BufReader::new(stdout);

        let stderr_pipe = child.stderr.take().context("no stderr on clone_tts.py child")?;
        let stderr_tail: Arc<Mutex<VecDeque<String>>> = Arc::new(Mutex::new(VecDeque::new()));
        // Every stderr line also goes out on this channel, live, so the
        // startup loop below can forward it to `on_log` as it arrives
        // instead of only after the fact.
        let (log_tx, log_rx) = std::sync::mpsc::channel::<String>();
        {
            let stderr_tail = stderr_tail.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr_pipe);
                for line in reader.lines().map_while(std::io::Result::ok) {
                    let mut buf = stderr_tail.lock().unwrap();
                    if buf.len() >= STDERR_TAIL_LINES {
                        buf.pop_front();
                    }
                    buf.push_back(line.clone());
                    drop(buf);
                    let _ = log_tx.send(line);
                }
            });
        }

        // The device-line read is a blocking call, so it runs on its own
        // thread; this lets the loop below keep forwarding stderr lines
        // (via `log_rx`) to `on_log` WHILE that blocking read is still in
        // progress, instead of only being able to show anything once it's
        // already finished (or failed). On success the thread hands back
        // `stdout` itself (it owns it temporarily), so the caller can keep
        // using it afterwards for `run_file`.
        type DeviceOutcome = Result<(String, BufReader<ChildStdout>), String>;
        let (device_tx, device_rx) = std::sync::mpsc::channel::<DeviceOutcome>();
        {
            let stderr_tail = stderr_tail.clone();
            std::thread::spawn(move || {
                let result: DeviceOutcome = (|| {
                    loop {
                        let mut raw = String::new();
                        let n = stdout
                            .read_line(&mut raw)
                            .map_err(|e| format!("lecture interrompue : {e}"))?;
                        if n == 0 {
                            return Err(format!(
                                "clone_tts.py s'est arrêté avant d'avoir fini de charger le modèle :\n{}",
                                Self::tail_text(&stderr_tail)
                            ));
                        }
                        let raw = raw.trim();
                        if raw.is_empty() {
                            continue;
                        }
                        let parsed: WorkerLine = serde_json::from_str(raw).unwrap_or_default();
                        if let Some(d) = parsed.device {
                            return Ok((d, stdout));
                        }
                        // anything else before the device line is unexpected; ignore defensively
                    }
                })();
                let _ = device_tx.send(result);
            });
        }

        // Interleave: forward stderr lines to `on_log` as they arrive,
        // while waiting (with no overall timeout — a first-run model
        // download can legitimately take several minutes) for the
        // device-ready signal from the thread above.
        let (device, stdout) = loop {
            match device_rx.try_recv() {
                Ok(Ok(pair)) => break pair,
                Ok(Err(e)) => bail!("{e}"),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    bail!(
                        "clone_tts.py s'est arrêté de façon inattendue pendant le chargement :\n{}",
                        Self::tail_text(&stderr_tail)
                    );
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            match log_rx.recv_timeout(std::time::Duration::from_millis(300)) {
                Ok(line) => on_log(&line),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                // stderr pipe closed (process likely dying/dead): keep
                // looping on `device_rx`, which will report the real
                // outcome (success or the process having died) shortly.
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => match device_rx.recv() {
                    Ok(Ok(pair)) => break pair,
                    Ok(Err(e)) => bail!("{e}"),
                    Err(_) => bail!(
                        "clone_tts.py s'est arrêté de façon inattendue pendant le chargement :\n{}",
                        Self::tail_text(&stderr_tail)
                    ),
                },
            }
        };

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr_tail,
            device,
        })
    }

    fn tail_text(buf: &Mutex<VecDeque<String>>) -> String {
        buf.lock().unwrap().iter().cloned().collect::<Vec<_>>().join("\n")
    }

    /// Runs ONE file's worth of segments through the already-loaded model.
    /// Blocks until this file's results have all streamed back — the
    /// Python side fully finishes one file before reading its next request
    /// line, so this is a simple synchronous round-trip from here.
    ///
    /// `items`: (text, output_wav_path) pairs, in segment order.
    /// `on_result` is called once per segment AS IT COMPLETES, so the
    /// caller can still report live per-segment progress.
    pub fn run_file(
        &mut self,
        req_id: u64,
        lang: &str,
        speaker_wav: &Path,
        items: &[(String, PathBuf)],
        on_result: &mut dyn FnMut(usize, Result<PathBuf>),
    ) -> Result<()> {
        let req = CloneFileReq {
            req_id,
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
        let mut line = serde_json::to_vec(&req)?;
        line.push(b'\n');

        if self.stdin.is_none() {
            bail!("worker clone_tts.py déjà arrêté");
        }
        // Scoped so the mutable borrow of `self.stdin` is fully released
        // before we potentially need to write to `self.stdin` again (set it
        // to `None`) in the error branch below.
        let write_result = {
            let stdin = self.stdin.as_mut().unwrap();
            stdin.write_all(&line).and_then(|_| stdin.flush())
        };
        if let Err(e) = write_result {
            self.stdin = None; // process is gone, don't try to reuse this worker
            bail!("échec d'écriture vers clone_tts.py (process mort ?) : {e}");
        }

        loop {
            let mut raw = String::new();
            let n = self.stdout.read_line(&mut raw);
            let n = match n {
                Ok(n) => n,
                Err(e) => {
                    self.stdin = None;
                    bail!("lecture de clone_tts.py interrompue (process mort ?) : {e}");
                }
            };
            if n == 0 {
                self.stdin = None;
                bail!(
                    "clone_tts.py s'est arrêté de façon inattendue :\n{}",
                    Self::tail_text(&self.stderr_tail)
                );
            }
            let raw = raw.trim();
            if raw.is_empty() {
                continue;
            }
            let parsed: WorkerLine = match serde_json::from_str(raw) {
                Ok(p) => p,
                Err(_) => continue, // stray non-JSON line, ignore defensively
            };
            if let Some(msg) = parsed.fatal {
                self.stdin = None;
                bail!("clone_tts.py: {msg}");
            }
            if parsed.done == Some(true) {
                return Ok(());
            }
            if let (Some(index), Some(ok)) = (parsed.index, parsed.ok) {
                if ok {
                    on_result(index, Ok(PathBuf::from(parsed.out.unwrap_or_default())));
                } else {
                    on_result(
                        index,
                        Err(anyhow!(parsed.error.unwrap_or_else(|| "erreur inconnue".into()))),
                    );
                }
            }
        }
    }
}

impl Drop for CloneWorker {
    fn drop(&mut self) {
        // Close stdin FIRST (before waiting): that's what signals EOF to
        // the Python side's `for line in sys.stdin:` loop, letting it exit
        // on its own. Waiting without doing this first would deadlock
        // forever on a worker that's just idle, waiting for a next file
        // that's never coming.
        self.stdin.take();
        // Reap the process so we don't leave it running in the background
        // (still holding the model, and GPU memory if applicable) after
        // this worker is no longer going to be used. Best-effort: if the
        // process is somehow stuck exiting, this can block briefly.
        let _ = self.child.wait();
    }
}
