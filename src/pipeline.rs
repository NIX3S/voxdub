use anyhow::{bail, Context, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio;
use crate::external;
use crate::types::{BatchMode, OutputMode, PipelineMsg, Segment, Settings, Stage, ValidateMsg, VoiceSource};
use crate::whisper_engine;

pub fn workdir_for(input: &Path) -> Result<PathBuf> {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "job".to_string());
    let dir = std::env::temp_dir().join("voxdub").join(stem);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Stage 1: convert + transcribe. Reports progress via `tx` and stops
/// (logically) at Stage::ReviewTranscript for the user to validate. Returns
/// the segments on success, so a caller (e.g. the batch coordinator) can
/// chain straight into translation without waiting on the UI.
pub fn run_transcription(
    job_idx: usize,
    input_path: PathBuf,
    voice_override: Option<PathBuf>,
    settings: Settings,
    tx: Sender<PipelineMsg>,
) -> Option<Vec<Segment>> {
    let result: Result<Vec<Segment>> = (|| {
        let dir = workdir_for(&input_path)?;
        let wav16k = dir.join("16k.wav");

        tx.send(PipelineMsg::Stage(job_idx, Stage::Converting)).ok();
        audio::convert_to_wav16k_mono(&input_path, &wav16k, settings.low_priority)
            .context("conversion to 16kHz mono wav failed")?;

        // Reference clip for voice cloning later: first 12s (or less) of
        // either this file's own audio (default) or a dedicated voice
        // sample, per-file override taking priority over the global
        // Settings::voice_source.
        let voice_src: Option<PathBuf> = voice_override.or_else(|| match &settings.voice_source {
            VoiceSource::CustomFile(p) => Some(p.clone()),
            VoiceSource::FromInputFile => None,
        });
        let ref_source: &Path = voice_src.as_deref().unwrap_or(&input_path);
        let total = audio::probe_duration(ref_source).unwrap_or(12.0);
        let ref_len = total.min(12.0).max(2.0) as f32;
        let speaker_ref = dir.join("speaker_ref.wav");
        audio::extract_reference_clip(ref_source, &speaker_ref, 0.0, ref_len, settings.low_priority)
            .with_context(|| format!("extracting speaker reference clip from {ref_source:?} failed"))?;

        tx.send(PipelineMsg::Stage(job_idx, Stage::Transcribing)).ok();
        let model_path = settings
            .whisper_model_path
            .clone()
            .context("no whisper model configured (set it in Settings)")?;
        let whisper_exe = settings
            .whisper_exe_path
            .clone()
            .context("no whisper-cli executable configured (set it in Settings)")?;
        let segments = whisper_engine::transcribe(
            &whisper_exe,
            &model_path,
            &wav16k,
            &settings.source_lang,
            settings.low_priority,
        )
        .context("transcription failed")?;

        tx.send(PipelineMsg::TranscriptReady(job_idx, segments.clone())).ok();
        tx.send(PipelineMsg::Stage(job_idx, Stage::ReviewTranscript)).ok();
        Ok(segments)
    })();

    match result {
        Ok(segments) => Some(segments),
        Err(e) => {
            tx.send(PipelineMsg::Error(job_idx, format!("{e:#}"))).ok();
            None
        }
    }
}

/// Stage 2: translate validated segment texts. Stops (logically) at
/// ReviewTranslation. Returns the segments (with `translated_text` filled
/// in) on success, so a caller can chain straight into synthesis.
pub fn run_translation(
    job_idx: usize,
    mut segments: Vec<Segment>,
    settings: Settings,
    scripts_dir: PathBuf,
    tx: Sender<PipelineMsg>,
) -> Option<Vec<Segment>> {
    let result: Result<Vec<Segment>> = (|| {
        tx.send(PipelineMsg::Stage(job_idx, Stage::Translating)).ok();
        let texts: Vec<String> = segments.iter().map(|s| s.original_text.clone()).collect();
        let translate_script = scripts_dir.join("translate.py");
        let translations = external::translate_segments(
            &settings.python_bin,
            &translate_script,
            &texts,
            &settings.source_lang,
            &settings.target_lang,
            settings.low_priority,
        )
        .context("translation failed (is argostranslate installed, and the language pair downloaded?)")?;

        for (seg, t) in segments.iter_mut().zip(translations.iter()) {
            seg.translated_text = t.clone();
        }

        tx.send(PipelineMsg::TranslationReady(job_idx, translations)).ok();
        tx.send(PipelineMsg::Stage(job_idx, Stage::ReviewTranslation)).ok();
        Ok(segments)
    })();

    match result {
        Ok(segments) => Some(segments),
        Err(e) => {
            tx.send(PipelineMsg::Error(job_idx, format!("{e:#}"))).ok();
            None
        }
    }
}

/// Stage 3: synthesize each validated translated segment (voice cloning),
/// then stitch them into the final output file. Shared between the manual
/// single-file path (`run_synthesis`, which spawns a one-shot worker) and
/// the batch coordinator (which passes in a worker it keeps alive across
/// many files — see `external::CloneWorker`).
fn run_synthesis_with_worker(
    job_idx: usize,
    input_path: PathBuf,
    segments: Vec<Segment>,
    settings: Settings,
    tx: Sender<PipelineMsg>,
    worker: &mut external::CloneWorker,
) -> Result<PathBuf> {
    tx.send(PipelineMsg::Stage(job_idx, Stage::Synthesizing)).ok();
    let dir = workdir_for(&input_path)?;
    let speaker_ref = dir.join("speaker_ref.wav");

    let total = segments.len();
    let items: Vec<(String, PathBuf)> = segments
        .iter()
        .enumerate()
        .map(|(i, seg)| {
            (
                seg.translated_text.clone(),
                dir.join(format!("seg_{i:04}_raw.wav")),
            )
        })
        .collect();

    let mut raw_by_index: HashMap<usize, PathBuf> = HashMap::new();
    let mut synth_errors: Vec<String> = Vec::new();
    let mut done_count = 0usize;

    worker
        .run_file(
            job_idx as u64,
            &settings.target_lang,
            &speaker_ref,
            &items,
            &mut |i, res| {
                done_count += 1;
                match res {
                    Ok(path) => {
                        raw_by_index.insert(i, path);
                    }
                    Err(e) => synth_errors.push(format!("segment {i}: {e}")),
                }
                tx.send(PipelineMsg::SynthesisProgress(job_idx, done_count, total))
                    .ok();
            },
        )
        .context("voice synthesis failed")?;

    // Fais savoir à l'utilisateur, dans l'app, sur quel device la synthèse
    // a réellement tourné — sans ça, un GPU demandé mais pas obtenu (torch
    // CPU-only installé, par ex.) retombe sur CPU en silence et rien ne le
    // signale. `worker.device` est fixé une fois pour toutes au chargement
    // du modèle, donc identique pour tous les fichiers que ce worker traite.
    match worker.device.as_str() {
        "cuda" if settings.use_gpu => {
            tx.send(PipelineMsg::Log(job_idx, "🟢 Synthèse vocale : GPU (CUDA) utilisé.".into()))
                .ok();
        }
        other if settings.use_gpu => {
            tx.send(PipelineMsg::Log(
                job_idx,
                format!(
                    "⚠ GPU demandé mais la synthèse a tourné sur « {other} ». \
                     Vérifie qu'un torch avec support CUDA est bien installé \
                     (voir README § Performance) : \
                     python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\""
                ),
            ))
            .ok();
        }
        _ => {}
    }

    if !synth_errors.is_empty() {
        bail!(
            "échec de synthèse sur {} segment(s) : {}",
            synth_errors.len(),
            synth_errors.join("; ")
        );
    }

    let mut clip_paths: Vec<(PathBuf, f32)> = Vec::with_capacity(total);
    for (i, seg) in segments.iter().enumerate() {
        let raw_clip = raw_by_index
            .remove(&i)
            .with_context(|| format!("segment {i} manquant en sortie de synthèse"))?;

        let final_clip = match settings.output_mode {
            OutputMode::FitOriginalDuration => {
                let fitted = dir.join(format!("seg_{i:04}_fit.wav"));
                audio::stretch_to_duration(&raw_clip, &fitted, seg.duration(), settings.low_priority)
                    .with_context(|| format!("time-stretch failed for segment {i}"))?;
                fitted
            }
            OutputMode::JustTranslated => raw_clip,
        };

        let clip_duration = audio::probe_duration(&final_clip).unwrap_or(seg.duration() as f64) as f32;
        tx.send(PipelineMsg::SegmentSynthesized(job_idx, i, final_clip.clone(), clip_duration))
            .ok();
        clip_paths.push((final_clip, seg.start));
    }

    tx.send(PipelineMsg::Stage(job_idx, Stage::Stitching)).ok();

    let out_dir = settings
        .output_dir
        .clone()
        .context("no output folder configured (set it in Settings)")?;
    std::fs::create_dir_all(&out_dir)?;
    let stem = input_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());

    // Intermediate mix, always wav (stitching filters need a PCM target).
    let stitched_wav = dir.join("stitched.wav");

    match settings.output_mode {
        OutputMode::FitOriginalDuration => {
            let total_s = segments
                .last()
                .map(|s| s.end)
                .unwrap_or(0.0)
                .max(audio::probe_duration(&input_path).unwrap_or(0.0) as f32);
            audio::stitch_on_timeline(&clip_paths, total_s, 24000, &stitched_wav, settings.low_priority)
                .context("stitching (timeline mix) failed")?;
        }
        OutputMode::JustTranslated => {
            let just_paths: Vec<PathBuf> = clip_paths.into_iter().map(|(p, _)| p).collect();
            audio::concat_sequential(&just_paths, &stitched_wav, &dir, settings.low_priority)
                .context("stitching (concat) failed")?;
        }
    }

    // Deliver in the SAME format as the input file (ogg -> ogg, mp3 -> mp3, ...).
    let target_ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("wav")
        .to_lowercase();
    let out_path = out_dir.join(format!("{stem}_{}.{target_ext}", settings.target_lang));
    if target_ext == "wav" {
        std::fs::copy(&stitched_wav, &out_path).context("copying final wav failed")?;
    } else {
        audio::transcode(&stitched_wav, &out_path, settings.low_priority).with_context(|| {
            format!(
                "conversion to .{target_ext} failed (ton build ffmpeg a-t-il l'encodeur pour ce format ?)"
            )
        })?;
    }

    let orig_dur = audio::probe_duration(&input_path).unwrap_or(0.0) as f32;
    let out_dur = audio::probe_duration(&out_path).unwrap_or(0.0) as f32;
    tx.send(PipelineMsg::FileDurations(job_idx, orig_dur, out_dur)).ok();

    Ok(out_path)
}

/// Manual (non-batch) single-file entry point: spawns a one-shot XTTS
/// worker just for this file, synthesizes + stitches, then lets the worker
/// shut itself down (its `Drop` closes stdin and reaps the process) once
/// this function returns. For the batch coordinator, which wants to keep
/// one worker alive across many files instead, see `run_synthesis_with_worker`.
pub fn run_synthesis(
    job_idx: usize,
    input_path: PathBuf,
    segments: Vec<Segment>,
    settings: Settings,
    scripts_dir: PathBuf,
    tx: Sender<PipelineMsg>,
) {
    // Envoyé ICI, avant le (potentiellement long, 10-30s+ voire plus au
    // tout premier lancement le temps de télécharger le modèle de ~2 Go)
    // chargement du modèle : sinon l'UI reste sur l'étape précédente sans
    // rien afficher pendant tout ce temps, ce qui donne l'impression que
    // l'app est bloquée alors qu'elle charge simplement XTTS.
    tx.send(PipelineMsg::Stage(job_idx, Stage::Synthesizing)).ok();

    let result: Result<PathBuf> = (|| {
        let clone_script = scripts_dir.join("clone_tts.py");
        let log_tx = tx.clone();
        let mut worker = external::CloneWorker::spawn(
            &settings.python_bin,
            &clone_script,
            settings.use_gpu,
            settings.low_priority,
            &mut |line| {
                log_tx.send(PipelineMsg::Log(job_idx, format!("⏳ {line}"))).ok();
            },
        )
        .context("échec du lancement de clone_tts.py")?;
        run_synthesis_with_worker(job_idx, input_path, segments, settings, tx.clone(), &mut worker)
    })();

    match result {
        Ok(path) => {
            tx.send(PipelineMsg::Stage(job_idx, Stage::Done(path))).ok();
        }
        Err(e) => {
            tx.send(PipelineMsg::Error(job_idx, format!("{e:#}"))).ok();
        }
    }
}

/// A file's starting point for the batch coordinator: either fresh
/// (`Stage::Queued`) or resuming mid-pipeline after a restart (persisted
/// session), in which case `stage` is one of `ReviewTranscript` /
/// `ReviewTranslation` and `segments` already holds that stage's output.
#[derive(Clone)]
pub struct BatchSeed {
    pub idx: usize,
    pub path: PathBuf,
    pub stage: Stage,
    pub segments: Vec<Segment>,
    pub validated: bool,
    /// Per-file voice-cloning override, see `FileJob::voice_override`.
    pub voice_override: Option<PathBuf>,
}

/// Spawns `max_parallel` worker threads (at least 1) that pull job indices
/// off a shared queue and hand each to `make_job()`'s returned closure.
/// `make_job` is called once PER WORKER THREAD (not per item), so it's the
/// right place to clone per-thread-owned resources (a `Sender`, `Settings`,
/// ...) exactly once instead of once per file.
fn run_phase_with<Make, Job>(
    ids: &[usize],
    max_parallel: usize,
    stop_flag: &AtomicBool,
    pause_flag: &AtomicBool,
    make_job: Make,
) where
    Make: Fn() -> Job,
    Job: FnMut(usize) + Send,
{
    if ids.is_empty() {
        return;
    }
    let queue: Mutex<VecDeque<usize>> = Mutex::new(ids.iter().copied().collect());
    let n_workers = max_parallel.max(1).min(ids.len());
    std::thread::scope(|scope| {
        // Take the shared reference ONCE, outside the loop: each spawned
        // `move` closure then just copies this `&Mutex<..>` (references are
        // Copy), instead of trying to move the one-and-only `queue` itself
        // into the first thread and leaving nothing for the rest.
        let queue_ref = &queue;
        for _ in 0..n_workers {
            let mut job = make_job();
            scope.spawn(move || loop {
                if stop_flag.load(Ordering::Relaxed) {
                    return;
                }
                while pause_flag.load(Ordering::Relaxed) {
                    if stop_flag.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(200));
                }
                let next = queue_ref.lock().unwrap().pop_front();
                match next {
                    Some(idx) => job(idx),
                    None => return,
                }
            });
        }
    });
}

/// Blocks until a `ValidateMsg` for `idx` is available (either already
/// parked in `pending` from an earlier out-of-order validation, or freshly
/// received), or returns `None` if the batch is stopped / the channel
/// closes. Safe to call from multiple worker threads concurrently: the
/// shared receiver is behind a `Mutex`, so only one thread polls it at a
/// time; a message meant for another file is parked in `pending` for
/// whichever thread is waiting on it.
fn wait_for_validation(
    idx: usize,
    pending: &Mutex<HashMap<usize, Vec<Segment>>>,
    validate_rx: &Mutex<Receiver<ValidateMsg>>,
    stop_flag: &AtomicBool,
    pause_flag: &AtomicBool,
) -> Option<Vec<Segment>> {
    loop {
        if let Some(s) = pending.lock().unwrap().remove(&idx) {
            return Some(s);
        }
        if stop_flag.load(Ordering::Relaxed) {
            return None;
        }
        if pause_flag.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
            continue;
        }
        let msg = validate_rx.lock().unwrap().recv_timeout(Duration::from_millis(200));
        match msg {
            Ok(m) if m.job_idx == idx => return Some(m.segments),
            Ok(m) => {
                pending.lock().unwrap().insert(m.job_idx, m.segments);
            }
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}

/// Runs an entire batch (all files, all stages) on dedicated background
/// thread(s), independent of the GUI's frame loop — this is what makes the
/// batch keep progressing even while the window is minimized, occluded, or
/// simply not being redrawn. Up to `settings.max_parallel` files are
/// processed concurrently WITHIN each stage (all transcriptions, then all
/// translations, then all syntheses, each stage fully finishing before the
/// next starts) — simpler and safer to reason about than fully
/// cross-stage parallelism, while still giving the speedup that matters
/// most: several whisper-cli / XTTS processes running at once instead of
/// one at a time.
///
/// `seeds` carries each file's starting point, so a resumed session (one
/// reloaded from disk after the app was closed or the PC rebooted) can
/// skip stages it already completed: a seed at `Stage::ReviewTranscript`
/// already has its transcript and jumps straight to translation; one at
/// `Stage::ReviewTranslation` jumps straight to synthesis. A seed whose
/// `validated` flag was already true for that stage (the user had clicked
/// "Valider" before the app closed) is treated as pre-validated, so it
/// doesn't wait for a fresh click.
///
/// - `BatchMode::Auto`: no waiting, each stage's raw output feeds the next.
/// - `BatchMode::WithReview`: for each file, blocks until a `ValidateMsg`
///   for that file arrives from the UI before moving it to the next stage,
///   using the edited segments the message carries.
pub fn run_batch(
    mode: BatchMode,
    seeds: Vec<BatchSeed>,
    settings: Settings,
    scripts_dir: PathBuf,
    tx: Sender<PipelineMsg>,
    validate_rx: Receiver<ValidateMsg>,
    stop_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
) {
    let max_parallel = settings.max_parallel.max(1);
    let path_by_idx: HashMap<usize, PathBuf> =
        seeds.iter().map(|s| (s.idx, s.path.clone())).collect();
    let voice_by_idx: HashMap<usize, Option<PathBuf>> = seeds
        .iter()
        .map(|s| (s.idx, s.voice_override.clone()))
        .collect();

    let pending: Mutex<HashMap<usize, Vec<Segment>>> = Mutex::new(HashMap::new());
    if mode == BatchMode::WithReview {
        let mut p = pending.lock().unwrap();
        for seed in &seeds {
            if seed.validated
                && matches!(seed.stage, Stage::ReviewTranscript | Stage::ReviewTranslation)
            {
                p.insert(seed.idx, seed.segments.clone());
            }
        }
    }
    let validate_rx = Mutex::new(validate_rx);

    // --- Phase 1: transcribe files still at Queued ---
    let transcripts: Mutex<HashMap<usize, Vec<Segment>>> = Mutex::new(HashMap::new());
    {
        let mut t = transcripts.lock().unwrap();
        for s in &seeds {
            if s.stage != Stage::Queued {
                t.insert(s.idx, s.segments.clone());
            }
        }
    }
    let to_transcribe: Vec<usize> = seeds
        .iter()
        .filter(|s| s.stage == Stage::Queued)
        .map(|s| s.idx)
        .collect();

    run_phase_with(&to_transcribe, max_parallel, &stop_flag, &pause_flag, || {
        let tx = tx.clone();
        let settings = settings.clone();
        let path_by_idx = &path_by_idx;
        let voice_by_idx = &voice_by_idx;
        let transcripts = &transcripts;
        move |idx: usize| {
            let Some(path) = path_by_idx.get(&idx).cloned() else {
                return;
            };
            let voice_override = voice_by_idx.get(&idx).cloned().flatten();
            if let Some(segs) =
                run_transcription(idx, path, voice_override, settings.clone(), tx.clone())
            {
                transcripts.lock().unwrap().insert(idx, segs);
            }
        }
    });
    if stop_flag.load(Ordering::Relaxed) {
        return;
    }

    // --- Phase 2: translate files that have a transcript ---
    let translations: Mutex<HashMap<usize, Vec<Segment>>> = Mutex::new(HashMap::new());
    {
        let mut tr = translations.lock().unwrap();
        for s in &seeds {
            if s.stage == Stage::ReviewTranslation {
                tr.insert(s.idx, s.segments.clone());
            }
        }
    }
    let to_translate: Vec<usize> = seeds
        .iter()
        .filter(|s| matches!(s.stage, Stage::Queued | Stage::ReviewTranscript))
        .map(|s| s.idx)
        .collect();

    run_phase_with(&to_translate, max_parallel, &stop_flag, &pause_flag, || {
        let tx = tx.clone();
        let settings = settings.clone();
        let scripts_dir = scripts_dir.clone();
        let transcripts = &transcripts;
        let translations = &translations;
        let pending = &pending;
        let validate_rx = &validate_rx;
        let stop_flag = &stop_flag;
        let pause_flag = &pause_flag;
        move |idx: usize| {
            let orig = transcripts.lock().unwrap().remove(&idx);
            let Some(orig) = orig else { return };
            let input_segments = if mode == BatchMode::WithReview {
                match wait_for_validation(idx, pending, validate_rx, stop_flag, pause_flag) {
                    Some(s) => s,
                    None => return,
                }
            } else {
                orig
            };
            if let Some(segs) = run_translation(idx, input_segments, settings.clone(), scripts_dir.clone(), tx.clone())
            {
                translations.lock().unwrap().insert(idx, segs);
            }
        }
    });
    if stop_flag.load(Ordering::Relaxed) {
        return;
    }

    // --- Phase 3: synthesize files that have a translation ---
    let to_synthesize: Vec<usize> = seeds
        .iter()
        .filter(|s| matches!(s.stage, Stage::Queued | Stage::ReviewTranscript | Stage::ReviewTranslation))
        .map(|s| s.idx)
        .collect();

    run_phase_with(&to_synthesize, max_parallel, &stop_flag, &pause_flag, || {
        let tx = tx.clone();
        let settings = settings.clone();
        let clone_script = scripts_dir.join("clone_tts.py");
        let translations = &translations;
        let path_by_idx = &path_by_idx;
        let pending = &pending;
        let validate_rx = &validate_rx;
        let stop_flag = &stop_flag;
        let pause_flag = &pause_flag;
        // Persistent XTTS worker for THIS thread (one of up to
        // `max_parallel`), spawned lazily on first use and then reused for
        // every file this thread processes afterwards — the whole point
        // being to pay the ~2GB model load ONCE per parallel "slot" instead
        // of once per file. If a call reports the process died mid-batch,
        // it's dropped here so the next file gets a freshly spawned one
        // instead of repeating the same failure down the line.
        let mut worker: Option<external::CloneWorker> = None;
        move |idx: usize| {
            let translated = translations.lock().unwrap().remove(&idx);
            let Some(translated) = translated else { return };
            let Some(path) = path_by_idx.get(&idx).cloned() else {
                return;
            };
            let input_segments = if mode == BatchMode::WithReview {
                match wait_for_validation(idx, pending, validate_rx, stop_flag, pause_flag) {
                    Some(s) => s,
                    None => return,
                }
            } else {
                translated
            };

            tx.send(PipelineMsg::Stage(idx, Stage::Synthesizing)).ok();

            if worker.is_none() {
                let log_tx = tx.clone();
                match external::CloneWorker::spawn(
                    &settings.python_bin,
                    &clone_script,
                    settings.use_gpu,
                    settings.low_priority,
                    &mut |line| {
                        log_tx.send(PipelineMsg::Log(idx, format!("⏳ {line}"))).ok();
                    },
                ) {
                    Ok(w) => worker = Some(w),
                    Err(e) => {
                        tx.send(PipelineMsg::Error(idx, format!("échec du lancement de clone_tts.py : {e:#}")))
                            .ok();
                        return;
                    }
                }
            }
            let w = worker.as_mut().expect("worker just ensured Some above");

            match run_synthesis_with_worker(idx, path, input_segments, settings.clone(), tx.clone(), w) {
                Ok(out_path) => {
                    tx.send(PipelineMsg::Stage(idx, Stage::Done(out_path))).ok();
                }
                Err(e) => {
                    tx.send(PipelineMsg::Error(idx, format!("{e:#}"))).ok();
                    // Don't assume the worker is still usable after a
                    // failure — a fresh one will be spawned for the next
                    // file on this thread instead of repeating the error.
                    worker = None;
                }
            }
        }
    });
}
