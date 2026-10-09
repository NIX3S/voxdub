use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Instant;

use eframe::egui;

use crate::persistence;
use crate::pipeline;
use crate::types::{BatchMode, FileJob, OutputMode, PipelineMsg, Settings, Stage, ValidateMsg, VoiceSource};

pub struct VoxDubApp {
    jobs: Vec<FileJob>,
    selected: usize,
    settings: Settings,
    scripts_dir: PathBuf,
    tx: Sender<PipelineMsg>,
    rx: Receiver<PipelineMsg>,

    /// Batch en cours (None = mode manuel, fichier par fichier). Le batch
    /// tourne sur son propre thread (voir `pipeline::run_batch`), indépendant
    /// de la boucle de rendu : il continue même fenêtre minimisée/en arrière-plan.
    batch: Option<BatchMode>,
    batch_error: Option<String>,
    /// Envoie les validations (segments édités) au thread coordinateur du batch.
    validate_tx: Option<Sender<ValidateMsg>>,
    /// Demande d'arrêt au thread coordinateur (plus rien de nouveau ne démarre).
    batch_stop: Option<Arc<AtomicBool>>,
    /// Pause le thread coordinateur sans l'arrêter (reprend où il en était).
    batch_pause: Option<Arc<AtomicBool>>,
    paused: bool,
    /// Index des fichiers inclus dans le batch en cours (pour détecter la fin).
    batch_job_indices: Vec<usize>,

    /// Bannière affichée si une session précédente a été restaurée au démarrage.
    resume_banner: Option<String>,

    /// Sauvegarde automatique de la session (jobs + réglages) pour pouvoir
    /// reprendre après fermeture de l'app / redémarrage du PC.
    dirty: bool,
    last_save: Instant,

    // Audio preview player (before/after comparison page).
    audio_out: Option<(rodio::OutputStream, rodio::OutputStreamHandle)>,
    current_sink: Option<rodio::Sink>,
    now_playing: Option<String>,
    player_error: Option<String>,
    /// Cache of on-demand extracted "original" segment clips: (job_idx, seg_id) -> wav path.
    orig_clip_cache: HashMap<(usize, usize), PathBuf>,
}

impl VoxDubApp {
    pub fn new(scripts_dir: PathBuf) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let audio_out = rodio::OutputStream::try_default().ok();

        let mut jobs = Vec::new();
        let mut settings = Settings::default();
        let mut resume_banner = None;

        if let Some(saved) = persistence::load() {
            settings = saved.settings;
            jobs = saved.jobs;
            let mut restarted = 0usize;
            for job in jobs.iter_mut() {
                let before = job.stage.clone();
                job.stage = job.stage.clone().normalized_for_resume();
                if job.stage != before {
                    restarted += 1;
                }
                job.synth_progress = None;
            }
            let unfinished = jobs.iter().filter(|j| !j.stage.is_terminal()).count();
            resume_banner = Some(format!(
                "Session précédente restaurée : {} fichier(s), dont {} à reprendre \
                 ({} remis à l'étape précédente car interrompus en plein traitement). \
                 Relance un batch (⚡ ou 📝) pour continuer.",
                jobs.len(),
                unfinished,
                restarted
            ));
        }

        Self {
            jobs,
            selected: 0,
            settings,
            scripts_dir,
            tx,
            rx,
            batch: None,
            batch_error: None,
            validate_tx: None,
            batch_stop: None,
            batch_pause: None,
            paused: false,
            batch_job_indices: Vec::new(),
            resume_banner,
            dirty: false,
            last_save: Instant::now(),
            audio_out,
            current_sink: None,
            now_playing: None,
            player_error: None,
            orig_clip_cache: HashMap::new(),
        }
    }

    /// Persists jobs + settings to disk (throttled to avoid hammering the
    /// disk every frame during an active batch). This is what lets the user
    /// close the app entirely — or shut the PC down — and pick the batch
    /// back up later, not just pause while the app stays open.
    fn maybe_save(&mut self) {
        if !self.dirty {
            return;
        }
        if self.last_save.elapsed().as_secs_f32() < 2.0 {
            return;
        }
        let state = persistence::SavedState {
            settings: self.settings.clone(),
            jobs: self.jobs.clone(),
            batch_mode: self.batch,
        };
        let _ = persistence::save(&state);
        self.dirty = false;
        self.last_save = Instant::now();
    }

    /// Plays a file (wav/mp3/ogg/flac/...) through the default output device,
    /// stopping whatever was playing before. Fire-and-forget: playback runs
    /// on rodio's own audio thread, no blocking here.
    fn play(&mut self, path: &Path, label: &str) {
        self.player_error = None;
        let Some((_, handle)) = &self.audio_out else {
            self.player_error = Some("Aucun périphérique audio disponible.".into());
            return;
        };
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                self.player_error = Some(format!("Impossible d'ouvrir {path:?} : {e}"));
                return;
            }
        };
        let source = match rodio::Decoder::new(std::io::BufReader::new(file)) {
            Ok(s) => s,
            Err(e) => {
                self.player_error = Some(format!("Décodage audio impossible : {e}"));
                return;
            }
        };
        match rodio::Sink::try_new(handle) {
            Ok(sink) => {
                sink.append(source);
                self.current_sink = Some(sink);
                self.now_playing = Some(label.to_string());
            }
            Err(e) => {
                self.player_error = Some(format!("Lecture impossible : {e}"));
            }
        }
    }

    fn stop_playback(&mut self) {
        self.current_sink = None;
        self.now_playing = None;
    }

    /// Extracts (once, then cached) the original-audio slice matching a
    /// transcript segment, so it can be played back for comparison.
    fn get_or_extract_orig_clip(&mut self, job_idx: usize, seg_index: usize) -> anyhow::Result<PathBuf> {
        let job = &self.jobs[job_idx];
        let seg = &job.segments[seg_index];
        let key = (job_idx, seg.id);
        if let Some(p) = self.orig_clip_cache.get(&key) {
            if p.exists() {
                return Ok(p.clone());
            }
        }
        let dir = pipeline::workdir_for(&job.input_path)?;
        let out = dir.join(format!("orig_seg_{:04}.wav", seg.id));
        crate::audio::extract_clip_to_wav(&job.input_path, &out, seg.start, seg.duration(), false)?;
        self.orig_clip_cache.insert(key, out.clone());
        Ok(out)
    }

    fn drain_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            self.dirty = true;
            match msg {
                PipelineMsg::Stage(idx, stage) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        // Nouvelle étape de review => l'humain doit (re)valider.
                        if matches!(stage, Stage::ReviewTranscript | Stage::ReviewTranslation) {
                            j.validated = false;
                        }
                        j.stage = stage;
                    }
                }
                PipelineMsg::TranscriptReady(idx, segs) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        j.segments = segs;
                    }
                }
                PipelineMsg::TranslationReady(idx, translations) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        for (seg, t) in j.segments.iter_mut().zip(translations.into_iter()) {
                            seg.translated_text = t;
                        }
                    }
                }
                PipelineMsg::SynthesisProgress(idx, done, total) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        j.synth_progress = Some((done, total));
                    }
                }
                PipelineMsg::SegmentSynthesized(idx, seg_idx, path, dur) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        if let Some(seg) = j.segments.get_mut(seg_idx) {
                            seg.synth_path = Some(path);
                            seg.synth_duration = Some(dur);
                        }
                    }
                }
                PipelineMsg::FileDurations(idx, orig, out) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        j.original_duration = Some(orig);
                        j.output_duration = Some(out);
                    }
                }
                PipelineMsg::Error(idx, msg) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        j.stage = Stage::Error(msg);
                    }
                }
                PipelineMsg::Log(idx, msg) => {
                    if let Some(j) = self.jobs.get_mut(idx) {
                        j.last_log = Some(msg);
                    }
                }
            }
        }
    }

    fn add_files(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus"])
            .pick_files()
        {
            for p in paths {
                self.jobs.push(FileJob::new(p));
            }
            self.dirty = true;
        }
    }

    /// Wipes the file list and the on-disk autosave, for starting a fresh
    /// project without the "session précédente restaurée" banner coming
    /// back on next launch. Refuses while a batch is running so it can't
    /// erase the state out from under the background coordinator thread.
    fn new_project(&mut self) {
        if self.batch.is_some() {
            self.batch_error =
                Some("Arrête d'abord le batch en cours avant de repartir à zéro.".into());
            return;
        }
        self.jobs.clear();
        self.selected = 0;
        self.resume_banner = None;
        self.batch_error = None;
        self.orig_clip_cache.clear();
        self.stop_playback();
        persistence::clear();
        self.dirty = false;
    }

    fn start_transcription(&mut self, idx: usize) {
        let job = &self.jobs[idx];
        let input_path = job.input_path.clone();
        let voice_override = job.voice_override.clone();
        let settings = self.settings.clone();
        let tx = self.tx.clone();
        self.jobs[idx].stage = Stage::Converting;
        self.jobs[idx].last_log = None;
        self.dirty = true;
        std::thread::spawn(move || {
            pipeline::run_transcription(idx, input_path, voice_override, settings, tx);
        });
    }

    fn start_translation(&mut self, idx: usize) {
        let segments = self.jobs[idx].segments.clone();
        let settings = self.settings.clone();
        let scripts_dir = self.scripts_dir.clone();
        let tx = self.tx.clone();
        self.jobs[idx].stage = Stage::Translating;
        self.dirty = true;
        std::thread::spawn(move || {
            pipeline::run_translation(idx, segments, settings, scripts_dir, tx);
        });
    }

    fn start_synthesis(&mut self, idx: usize) {
        let input_path = self.jobs[idx].input_path.clone();
        let segments = self.jobs[idx].segments.clone();
        let settings = self.settings.clone();
        let scripts_dir = self.scripts_dir.clone();
        let tx = self.tx.clone();
        self.jobs[idx].stage = Stage::Synthesizing;
        self.dirty = true;
        std::thread::spawn(move || {
            pipeline::run_synthesis(idx, input_path, segments, settings, scripts_dir, tx);
        });
    }

    /// Starts a batch over all not-yet-finished files (fresh `Queued` ones,
    /// and — when resuming a restored session — files already mid-pipeline
    /// at `ReviewTranscript`/`ReviewTranslation`). Checks the required
    /// settings up front so the user gets one clear message instead of N
    /// identical errors, then hands the whole batch to a dedicated
    /// background thread (`pipeline::run_batch`) that runs independently of
    /// the GUI's frame loop — so it keeps going even if the window is
    /// minimized, unfocused, or not being redrawn.
    fn start_batch(&mut self, mode: BatchMode) {
        self.batch_error = None;
        self.resume_banner = None;
        if self.jobs.is_empty() {
            self.batch_error = Some("Ajoute d'abord des fichiers audio.".into());
            return;
        }
        let mut missing = Vec::new();
        if self.settings.whisper_exe_path.is_none() {
            missing.push("exécutable whisper-cli");
        }
        if self.settings.whisper_model_path.is_none() {
            missing.push("modèle Whisper");
        }
        if self.settings.output_dir.is_none() {
            missing.push("dossier de sortie");
        }
        if !missing.is_empty() {
            self.batch_error = Some(format!("Réglages manquants : {}.", missing.join(", ")));
            return;
        }

        let seeds: Vec<pipeline::BatchSeed> = self
            .jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| !j.stage.is_terminal())
            .map(|(i, j)| pipeline::BatchSeed {
                idx: i,
                path: j.input_path.clone(),
                stage: j.stage.clone(),
                segments: j.segments.clone(),
                validated: j.validated,
                voice_override: j.voice_override.clone(),
            })
            .collect();
        if seeds.is_empty() {
            self.batch_error = Some("Tous les fichiers sont déjà traités.".into());
            return;
        }

        let settings = self.settings.clone();
        let scripts_dir = self.scripts_dir.clone();
        let tx = self.tx.clone();
        let (validate_tx, validate_rx) = std::sync::mpsc::channel::<ValidateMsg>();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));

        self.validate_tx = Some(validate_tx);
        self.batch_stop = Some(stop_flag.clone());
        self.batch_pause = Some(pause_flag.clone());
        self.paused = false;
        self.batch_job_indices = seeds.iter().map(|s| s.idx).collect();
        self.batch = Some(mode);
        self.dirty = true;

        std::thread::spawn(move || {
            pipeline::run_batch(mode, seeds, settings, scripts_dir, tx, validate_rx, stop_flag, pause_flag);
        });
    }

    fn stop_batch(&mut self) {
        if let Some(flag) = &self.batch_stop {
            flag.store(true, Ordering::Relaxed);
        }
        self.batch = None;
        self.validate_tx = None;
        self.batch_stop = None;
        self.batch_pause = None;
        self.paused = false;
        self.batch_job_indices.clear();
        self.dirty = true;
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        if let Some(flag) = &self.batch_pause {
            flag.store(self.paused, Ordering::Relaxed);
        }
    }

    /// Cosmetic only: the coordinator thread already finished the actual
    /// work regardless of whether this runs. This just clears the "batch in
    /// progress" UI state once every job the batch covered is Done/Error.
    fn check_batch_finished(&mut self) {
        if self.batch.is_none() || self.batch_job_indices.is_empty() {
            return;
        }
        let finished = self
            .batch_job_indices
            .iter()
            .all(|&i| self.jobs.get(i).map(|j| j.stage.is_terminal()).unwrap_or(true));
        if finished {
            self.batch = None;
            self.validate_tx = None;
            self.batch_stop = None;
            self.batch_pause = None;
            self.paused = false;
            self.batch_job_indices.clear();
            self.dirty = true;
        }
    }

    /// Validation controls shown under a review screen. Returns true when the
    /// caller should launch the next stage right now (manual mode only).
    fn validation_controls(
        &mut self,
        ui: &mut egui::Ui,
        idx: usize,
        manual_label: &str,
        next_step: &str,
    ) -> bool {
        match self.batch {
            None => ui.button(manual_label).clicked(),
            Some(BatchMode::Auto) => {
                ui.label(format!(
                    "Batch automatique : {next_step} démarre dès que son tour arrive."
                ));
                false
            }
            Some(BatchMode::WithReview) => {
                if self.jobs[idx].validated {
                    ui.colored_label(
                        egui::Color32::GREEN,
                        format!("Validé ✔ — {next_step} démarrera dans le batch."),
                    );
                    ui.small(
                        "(déjà transmis au batch, qui tourne en tâche de fond \
                         même si tu changes d'onglet ou minimises la fenêtre)",
                    );
                } else if ui.button("✔ Valider (le batch enchaîne la suite)").clicked() {
                    self.jobs[idx].validated = true;
                    self.dirty = true;
                    if let Some(vtx) = &self.validate_tx {
                        let segments = self.jobs[idx].segments.clone();
                        vtx.send(ValidateMsg { job_idx: idx, segments }).ok();
                    }
                }
                false
            }
        }
    }

    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Réglages");
        ui.separator();

        ui.label("Exécutable whisper.cpp (whisper-cli.exe) :");
        ui.horizontal(|ui| {
            let label = self
                .settings
                .whisper_exe_path
                .as_ref()
                .map(|p| truncate_path(p))
                .unwrap_or_else(|| "(non défini)".to_string());
            let full = self
                .settings
                .whisper_exe_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            ui.label(label).on_hover_text(full);
            if ui.button("Choisir…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_file() {
                    self.settings.whisper_exe_path = Some(p);
                    self.dirty = true;
                }
            }
        });

        ui.label("Modèle Whisper (.bin ggml, ex: ggml-medium.bin) :");
        ui.horizontal(|ui| {
            let label = self
                .settings
                .whisper_model_path
                .as_ref()
                .map(|p| truncate_path(p))
                .unwrap_or_else(|| "(non défini)".to_string());
            let full = self
                .settings
                .whisper_model_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            ui.label(label).on_hover_text(full);
            if ui.button("Choisir…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_file() {
                    self.settings.whisper_model_path = Some(p);
                    self.dirty = true;
                }
            }
        });

        ui.label("Interpréteur Python (avec argostranslate + TTS installés) :");
        if ui.text_edit_singleline(&mut self.settings.python_bin).changed() {
            self.dirty = true;
        }

        ui.horizontal(|ui| {
            ui.label("Langue source :");
            if ui.text_edit_singleline(&mut self.settings.source_lang).changed() {
                self.dirty = true;
            }
            ui.label("Langue cible :");
            if ui.text_edit_singleline(&mut self.settings.target_lang).changed() {
                self.dirty = true;
            }
        });
        ui.small("Codes ISO 639-1, ex: fr, en, es, de, it…");

        ui.separator();
        ui.label("Mode de sortie :");
        if ui
            .radio_value(
                &mut self.settings.output_mode,
                OutputMode::FitOriginalDuration,
                "Même durée que l'original (fit / dubbing synchronisé)",
            )
            .clicked()
        {
            self.dirty = true;
        }
        if ui
            .radio_value(
                &mut self.settings.output_mode,
                OutputMode::JustTranslated,
                "Juste traduit (rythme naturel de la voix de synthèse)",
            )
            .clicked()
        {
            self.dirty = true;
        }

        ui.separator();
        ui.label("Dossier de sortie :");
        ui.horizontal(|ui| {
            let label = self
                .settings
                .output_dir
                .as_ref()
                .map(|p| truncate_path(p))
                .unwrap_or_else(|| "(non défini)".to_string());
            let full = self
                .settings
                .output_dir
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            ui.label(label).on_hover_text(full);
            if ui.button("Choisir…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.settings.output_dir = Some(p);
                    self.dirty = true;
                }
            }
        });

        ui.separator();
        ui.strong("Voix pour le clonage (réglage par défaut du batch)");
        ui.label(match &self.settings.voice_source {
            VoiceSource::FromInputFile => {
                "Mode actuel : chaque fichier est doublé avec SA PROPRE voix".to_string()
            }
            VoiceSource::CustomFile(p) => format!("Mode actuel : voix dédiée — {}", truncate_path(p)),
        });
        ui.horizontal(|ui| {
            if ui.button("Voix de chaque fichier (par défaut)").clicked() {
                self.settings.voice_source = VoiceSource::FromInputFile;
                self.dirty = true;
            }
            if ui.button("Choisir une voix dédiée…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("Audio", &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus"])
                    .pick_file()
                {
                    self.settings.voice_source = VoiceSource::CustomFile(p);
                    self.dirty = true;
                }
            }
        });
        ui.small(
            "Par défaut, la voix clonée pour chaque fichier vient de SON PROPRE audio \
             (comportement d'origine). Choisis un fichier vocal dédié (ex: un acteur voix \
             qui a accepté d'être cloné) pour appliquer LA MÊME voix à tous les fichiers du \
             batch. Remplaçable au cas par cas pour un fichier précis dans son propre onglet.",
        );

        ui.separator();
        ui.strong("Performance / gros batchs");
        ui.horizontal(|ui| {
            ui.label("Fichiers en parallèle :");
            if ui
                .add(egui::DragValue::new(&mut self.settings.max_parallel).range(1..=32))
                .changed()
            {
                self.dirty = true;
            }
        });
        ui.small(
            "1 = comme avant (un fichier à la fois). Chaque fichier en plus lance \
             son propre whisper-cli / processus Python : la vitesse augmente, mais \
             RAM et CPU (et VRAM si GPU) aussi — monte progressivement (2, puis 4…) \
             en surveillant le Gestionnaire des tâches.",
        );
        if ui
            .checkbox(
                &mut self.settings.low_priority,
                "Priorité CPU réduite (laisse le PC réactif pour autre chose)",
            )
            .changed()
        {
            self.dirty = true;
        }
        if ui
            .checkbox(
                &mut self.settings.use_gpu,
                "Utiliser le GPU (CUDA) pour la voix (XTTS)",
            )
            .changed()
        {
            self.dirty = true;
        }
        ui.small(
            "Nécessite un torch installé avec support CUDA (voir README) ; sinon \
             retombe automatiquement sur le CPU. Pour que la TRANSCRIPTION utilise \
             aussi le GPU, c'est le binaire whisper-cli téléchargé qui compte (build \
             cuBLAS/Vulkan), pas une case à cocher ici — voir le README.",
        );
    }

    fn job_list(&mut self, ui: &mut egui::Ui) {
        ui.heading("Fichiers");
        ui.horizontal(|ui| {
            if ui.button("+ Ajouter des fichiers audio…").clicked() {
                self.add_files();
            }
            if ui
                .button("🗑 Nouveau projet")
                .on_hover_text("Vide la liste et efface la sauvegarde (voxdub_state.json)")
                .clicked()
            {
                self.new_project();
            }
        });
        ui.separator();

        if let Some(banner) = self.resume_banner.clone() {
            ui.colored_label(egui::Color32::LIGHT_BLUE, banner);
            ui.separator();
        }

        // --- Batch : tout lancer d'un coup ---
        match self.batch {
            None => {
                ui.strong("Traiter tous les fichiers :");
                if ui.button("⚡ Batch sans review").clicked() {
                    self.start_batch(BatchMode::Auto);
                }
                ui.small("Tout est enchaîné automatiquement, sans validation humaine.");
                if ui.button("📝 Batch avec review").clicked() {
                    self.start_batch(BatchMode::WithReview);
                }
                ui.small(
                    "Transcription de tous les fichiers, puis attente de ta validation ; \
                     ensuite traduction, nouvelle validation, puis voix.",
                );
            }
            Some(mode) => {
                let total = self.batch_job_indices.len().max(1);
                let done = self
                    .batch_job_indices
                    .iter()
                    .filter(|&&i| matches!(self.jobs.get(i).map(|j| &j.stage), Some(Stage::Done(_))))
                    .count();
                let failed = self
                    .batch_job_indices
                    .iter()
                    .filter(|&&i| matches!(self.jobs.get(i).map(|j| &j.stage), Some(Stage::Error(_))))
                    .count();
                ui.strong(match mode {
                    BatchMode::Auto => "⚡ Batch automatique en cours",
                    BatchMode::WithReview => "📝 Batch avec review en cours",
                });
                ui.label(format!("{done}/{total} terminés, {failed} en erreur"));

                if mode == BatchMode::WithReview {
                    let pending: Vec<usize> = self
                        .batch_job_indices
                        .iter()
                        .copied()
                        .filter(|&i| {
                            self.jobs
                                .get(i)
                                .map(|j| {
                                    matches!(j.stage, Stage::ReviewTranscript | Stage::ReviewTranslation)
                                        && !j.validated
                                })
                                .unwrap_or(false)
                        })
                        .collect();
                    if !pending.is_empty() {
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!("{} fichier(s) à valider", pending.len()),
                        );
                        if ui.button("→ Prochain à valider").clicked() {
                            self.selected = pending[0];
                        }
                        if ui.button("✔ Tout valider (sans relire)").clicked() {
                            for i in pending {
                                self.jobs[i].validated = true;
                                if let Some(vtx) = &self.validate_tx {
                                    let segments = self.jobs[i].segments.clone();
                                    vtx.send(ValidateMsg { job_idx: i, segments }).ok();
                                }
                            }
                            self.dirty = true;
                        }
                    }
                }

                ui.horizontal(|ui| {
                    let pause_label = if self.paused { "▶ Reprendre" } else { "⏸ Pause" };
                    if ui.button(pause_label).clicked() {
                        self.toggle_pause();
                    }
                    if ui.button("⏹ Arrêter le batch").clicked() {
                        self.stop_batch();
                    }
                });
                if self.paused {
                    ui.colored_label(
                        egui::Color32::YELLOW,
                        "⏸ En pause : le fichier en cours termine son étape actuelle, \
                         puis plus rien ne démarre jusqu'à Reprendre.",
                    );
                } else {
                    ui.small("L'arrêt laisse finir l'étape en cours, mais n'en démarre plus de nouvelle.");
                }
            }
        }
        if let Some(err) = &self.batch_error {
            ui.colored_label(egui::Color32::RED, err);
        }
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            for (i, job) in self.jobs.iter().enumerate() {
                let mark = if job.validated
                    && matches!(job.stage, Stage::ReviewTranscript | Stage::ReviewTranslation)
                {
                    "✔ "
                } else {
                    ""
                };
                let label = format!(
                    "{mark}{}  [{}]",
                    truncate(&job.name(), 26),
                    stage_label(&job.stage)
                );
                if ui.selectable_label(self.selected == i, label).clicked() {
                    self.selected = i;
                }
            }
        });
    }

    /// Before/after comparison page: total durations, then per-segment
    /// duration table with play buttons for the original vs dubbed clip.
    fn comparison_table(&mut self, ui: &mut egui::Ui, job_idx: usize) {
        ui.heading("📊 Comparaison avant / après");

        let (orig_total, out_total) = {
            let j = &self.jobs[job_idx];
            (j.original_duration, j.output_duration)
        };
        egui::Grid::new("totals_grid").num_columns(2).show(ui, |ui| {
            ui.label("Durée totale originale :");
            ui.label(fmt_dur(orig_total));
            ui.end_row();
            ui.label("Durée totale doublée :");
            ui.label(fmt_dur(out_total));
            ui.end_row();
            if let (Some(o), Some(d)) = (orig_total, out_total) {
                let delta = d - o;
                let pct = if o > 0.01 { delta / o * 100.0 } else { 0.0 };
                ui.label("Écart :");
                ui.label(format!("{delta:+.2} s ({pct:+.1} %)"));
                ui.end_row();
            }
        });

        ui.add_space(8.0);
        ui.label("Détail par segment :");

        let seg_count = self.jobs[job_idx].segments.len();
        egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
            egui::Grid::new("segments_grid")
                .num_columns(6)
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("#");
                    ui.strong("Texte original");
                    ui.strong("Texte traduit");
                    ui.strong("Durée orig.");
                    ui.strong("Durée doublée");
                    ui.strong("Écouter");
                    ui.end_row();

                    for seg_index in 0..seg_count {
                        let (id, orig_text, trans_text, orig_dur, synth_dur, has_synth) = {
                            let seg = &self.jobs[job_idx].segments[seg_index];
                            (
                                seg.id,
                                truncate(&seg.original_text, 60),
                                truncate(&seg.translated_text, 60),
                                seg.duration(),
                                seg.synth_duration,
                                seg.synth_path.is_some(),
                            )
                        };
                        ui.label(format!("{id}"));
                        ui.label(orig_text);
                        ui.label(trans_text);
                        ui.label(format!("{orig_dur:.2}s"));
                        ui.label(
                            synth_dur
                                .map(|d| format!("{d:.2}s"))
                                .unwrap_or_else(|| "—".into()),
                        );
                        ui.horizontal(|ui| {
                            if ui.small_button("▶ orig").clicked() {
                                match self.get_or_extract_orig_clip(job_idx, seg_index) {
                                    Ok(p) => self.play(&p, &format!("original, segment {id}")),
                                    Err(e) => self.player_error = Some(format!("{e:#}")),
                                }
                            }
                            if has_synth {
                                if ui.small_button("▶ doublé").clicked() {
                                    let p = self.jobs[job_idx].segments[seg_index]
                                        .synth_path
                                        .clone()
                                        .unwrap();
                                    self.play(&p, &format!("doublé, segment {id}"));
                                }
                            }
                        });
                        ui.end_row();
                    }
                });
        });
    }

    fn main_panel(&mut self, ui: &mut egui::Ui) {
        if self.jobs.is_empty() {
            ui.label("Ajoute un ou plusieurs fichiers audio pour commencer.");
            return;
        }
        let idx = self.selected.min(self.jobs.len() - 1);
        let stage = self.jobs[idx].stage.clone();

        ui.heading(self.jobs[idx].name());
        ui.label(format!("Étape : {}", stage_label(&stage)));

        ui.horizontal(|ui| {
            ui.label("🎙 Voix clonée :");
            match self.jobs[idx].voice_override.clone() {
                Some(p) => {
                    ui.label(truncate_path(&p)).on_hover_text(p.display().to_string());
                    if ui.small_button("✕ Revenir au réglage par défaut").clicked() {
                        self.jobs[idx].voice_override = None;
                        self.dirty = true;
                    }
                }
                None => {
                    let default_label = match &self.settings.voice_source {
                        VoiceSource::FromInputFile => "ce fichier lui-même (réglage par défaut)".to_string(),
                        VoiceSource::CustomFile(p) => format!("voix dédiée du batch — {}", truncate_path(p)),
                    };
                    ui.label(default_label);
                    if ui.small_button("Choisir une voix pour CE fichier…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Audio", &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus"])
                            .pick_file()
                        {
                            self.jobs[idx].voice_override = Some(p);
                            self.dirty = true;
                        }
                    }
                }
            }
        });
        if matches!(stage, Stage::ReviewTranscript | Stage::Synthesizing | Stage::Done(_)) {
            ui.small(
                "Ce choix doit être fait avant de lancer la synthèse vocale (avant ou pendant \
                 l'étape de validation) ; le changer après que la voix a déjà été générée n'a \
                 d'effet qu'en relançant un traitement depuis le début.",
            );
        }
        if let Some(log) = self.jobs[idx].last_log.clone() {
            let color = if log.starts_with('⚠') {
                egui::Color32::YELLOW
            } else {
                egui::Color32::GRAY
            };
            ui.colored_label(color, log);
        }
        ui.separator();

        ui.horizontal(|ui| {
            if let Some(playing) = &self.now_playing {
                ui.label(format!("🔊 Lecture : {playing}"));
                if ui.small_button("⏹ Stop").clicked() {
                    self.stop_playback();
                }
            }
        });
        if let Some(err) = &self.player_error {
            ui.colored_label(egui::Color32::RED, err);
        }
        ui.separator();

        match stage {
            Stage::Queued => {
                if self.batch.is_some() {
                    ui.label("En attente dans le batch…");
                } else if ui.button("▶ Lancer la transcription").clicked() {
                    self.start_transcription(idx);
                }
            }
            Stage::Converting | Stage::Transcribing | Stage::Translating | Stage::Stitching => {
                ui.spinner();
                ui.label("Traitement en cours…");
            }
            Stage::Synthesizing => {
                ui.spinner();
                match self.jobs[idx].synth_progress {
                    Some((done, total)) => {
                        ui.label(format!("Synthèse vocale… segment {done}/{total}"));
                        ui.add(egui::ProgressBar::new(done as f32 / total.max(1) as f32));
                    }
                    None => {
                        ui.label("Synthèse vocale… (chargement du modèle XTTS — peut prendre une minute)");
                    }
                }
            }
            Stage::ReviewTranscript => {
                ui.label("Vérifie / corrige la transcription, puis valide :");
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    for seg in self.jobs[idx].segments.iter_mut() {
                        ui.horizontal(|ui| {
                            ui.monospace(format!("[{:>6.1}s - {:>6.1}s]", seg.start, seg.end));
                        });
                        if ui.text_edit_multiline(&mut seg.original_text).changed() {
                            self.dirty = true;
                        }
                        ui.add_space(6.0);
                    }
                });
                if self.validation_controls(
                    ui,
                    idx,
                    "✔ Valider la transcription et traduire",
                    "la traduction",
                ) {
                    self.start_translation(idx);
                }
            }
            Stage::ReviewTranslation => {
                ui.label("Vérifie / corrige la traduction, puis valide :");
                egui::ScrollArea::vertical().max_height(400.0).show(ui, |ui| {
                    for seg in self.jobs[idx].segments.iter_mut() {
                        ui.monospace(format!("[{:>6.1}s - {:>6.1}s]", seg.start, seg.end));
                        ui.label(format!("Original : {}", seg.original_text));
                        if ui.text_edit_multiline(&mut seg.translated_text).changed() {
                            self.dirty = true;
                        }
                        ui.add_space(6.0);
                    }
                });
                if self.validation_controls(
                    ui,
                    idx,
                    "✔ Valider la traduction et générer la voix",
                    "la synthèse vocale",
                ) {
                    self.start_synthesis(idx);
                }
            }
            Stage::Done(ref path) => {
                let path = path.clone();
                ui.colored_label(egui::Color32::GREEN, "Terminé ✔");
                ui.label(format!("Fichier généré : {}", path.display()));
                ui.horizontal(|ui| {
                    if ui.button("Ouvrir le dossier de sortie").clicked() {
                        if let Some(dir) = path.parent() {
                            let _ = open_folder(dir);
                        }
                    }
                    if ui.button("▶ Écouter le résultat complet").clicked() {
                        self.play(&path, "résultat final");
                    }
                    if ui.button("▶ Écouter l'original complet").clicked() {
                        let input = self.jobs[idx].input_path.clone();
                        self.play(&input, "audio original");
                    }
                });

                ui.separator();
                self.comparison_table(ui, idx);
            }
            Stage::Error(ref e) => {
                ui.colored_label(egui::Color32::RED, format!("Erreur : {e}"));
                if ui.button("Réessayer depuis le début").clicked() {
                    self.jobs[idx].stage = Stage::Queued;
                    self.jobs[idx].segments.clear();
                    self.jobs[idx].validated = false;
                    self.jobs[idx].synth_progress = None;
                    self.dirty = true;
                }
            }
        }
    }
}

impl eframe::App for VoxDubApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_messages();
        self.check_batch_finished();
        self.maybe_save();
        ctx.request_repaint_after(std::time::Duration::from_millis(200));

        egui::SidePanel::left("jobs")
            .resizable(true)
            .default_width(240.0)
            .width_range(140.0..=500.0)
            .show(ctx, |ui| {
                self.job_list(ui);
            });
        egui::SidePanel::right("settings")
            .resizable(true)
            .default_width(320.0)
            .width_range(180.0..=600.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.settings_panel(ui);
                });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.main_panel(ui);
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Force a final save on the way out, even if the 2s save-throttle
        // would otherwise have skipped this particular frame.
        let state = persistence::SavedState {
            settings: self.settings.clone(),
            jobs: self.jobs.clone(),
            batch_mode: self.batch,
        };
        let _ = persistence::save(&state);
    }
}

fn fmt_dur(d: Option<f32>) -> String {
    match d {
        Some(v) => format!("{v:.2} s"),
        None => "—".to_string(),
    }
}

fn truncate_path(p: &std::path::Path) -> String {
    let s = p.display().to_string();
    let max = 28;
    if s.chars().count() <= max {
        return s;
    }
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    if name.chars().count() >= max {
        let tail: String = name.chars().rev().take(max - 1).collect();
        format!("…{}", tail.chars().rev().collect::<String>())
    } else {
        format!("…\\{name}")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}

fn stage_label(stage: &Stage) -> &'static str {
    match stage {
        Stage::Queued => "en attente",
        Stage::Converting => "conversion audio",
        Stage::Transcribing => "transcription…",
        Stage::ReviewTranscript => "à valider : transcription",
        Stage::Translating => "traduction…",
        Stage::ReviewTranslation => "à valider : traduction",
        Stage::Synthesizing => "synthèse vocale…",
        Stage::Stitching => "assemblage final…",
        Stage::Done(_) => "terminé",
        Stage::Error(_) => "erreur",
    }
}

#[cfg(target_os = "windows")]
fn open_folder(dir: &std::path::Path) -> std::io::Result<std::process::Child> {
    std::process::Command::new("explorer").arg(dir).spawn()
}
#[cfg(target_os = "macos")]
fn open_folder(dir: &std::path::Path) -> std::io::Result<std::process::Child> {
    std::process::Command::new("open").arg(dir).spawn()
}
#[cfg(all(unix, not(target_os = "macos")))]
fn open_folder(dir: &std::path::Path) -> std::io::Result<std::process::Child> {
    std::process::Command::new("xdg-open").arg(dir).spawn()
}
