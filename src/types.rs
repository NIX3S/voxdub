use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    pub id: usize,
    /// Start time in the ORIGINAL audio, in seconds.
    pub start: f32,
    /// End time in the ORIGINAL audio, in seconds.
    pub end: f32,
    pub original_text: String,
    pub translated_text: String,
    /// Path to the synthesized (TTS) wav for this segment, once generated.
    pub synth_path: Option<PathBuf>,
    /// Actual duration of the synthesized clip (after fit/stretch, if any).
    pub synth_duration: Option<f32>,
}

impl Segment {
    pub fn duration(&self) -> f32 {
        (self.end - self.start).max(0.05)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Stage {
    Queued,
    Converting,
    Transcribing,
    ReviewTranscript,
    Translating,
    ReviewTranslation,
    Synthesizing,
    Stitching,
    Done(PathBuf),
    Error(String),
}

impl Stage {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Stage::Done(_) | Stage::Error(_))
    }

    /// Mid-flight stages (the app was in the middle of a subprocess call)
    /// cannot be trusted after a restart/crash/reboot: the subprocess is
    /// gone and any partial output is unverified. This maps each such stage
    /// back to the nearest safe, restartable checkpoint. Used when reloading
    /// a persisted session.
    pub fn normalized_for_resume(self) -> Stage {
        match self {
            Stage::Converting | Stage::Transcribing => Stage::Queued,
            Stage::Translating => Stage::ReviewTranscript,
            Stage::Synthesizing | Stage::Stitching => Stage::ReviewTranslation,
            other => other,
        }
    }
}

/// How the "run everything" batch behaves.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum BatchMode {
    /// No human review: transcribe -> translate -> synthesize, fully automatic.
    Auto,
    /// Each step runs for all files; after transcription and after translation
    /// the batch waits for the human to validate each file before moving on.
    WithReview,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileJob {
    pub input_path: PathBuf,
    pub stage: Stage,
    pub segments: Vec<Segment>,
    /// Batch "avec review" : l'utilisateur a validé l'étape de review en cours.
    pub validated: bool,
    /// (segments synthétisés, total) — pour afficher une progression.
    pub synth_progress: Option<(usize, usize)>,
    /// Total duration of the original input file, in seconds (set once known).
    pub original_duration: Option<f32>,
    /// Total duration of the final dubbed output file, in seconds (set once done).
    pub output_duration: Option<f32>,
    /// Si défini, remplace `Settings::voice_source` pour CE fichier : la voix
    /// clonée vient de ce fichier audio-là plutôt que du réglage global
    /// (lui-même "depuis l'entrée" par défaut, ou une voix dédiée globale).
    #[serde(default)]
    pub voice_override: Option<PathBuf>,
}

impl FileJob {
    pub fn new(input_path: PathBuf) -> Self {
        Self {
            input_path,
            stage: Stage::Queued,
            segments: Vec::new(),
            validated: false,
            synth_progress: None,
            original_duration: None,
            output_duration: None,
            voice_override: None,
        }
    }

    pub fn name(&self) -> String {
        self.input_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "?".to_string())
    }
}

/// Where the reference clip used for voice cloning (XTTS speaker_wav) comes
/// from. Either each file is dubbed in ITS OWN voice (extracted from its own
/// audio, the original behaviour), or every file in the batch is cloned onto
/// ONE dedicated voice sample provided separately (e.g. a professional voice
/// actor recording) instead of whatever voice happens to be in the source
/// audio. Can be set globally (`Settings::voice_source`) and overridden per
/// file (`FileJob::voice_override`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum VoiceSource {
    FromInputFile,
    CustomFile(PathBuf),
}

impl Default for VoiceSource {
    fn default() -> Self {
        VoiceSource::FromInputFile
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum OutputMode {
    /// Stretch/compress + place each dubbed segment so the result matches the
    /// original timeline and total duration.
    FitOriginalDuration,
    /// Just concatenate the dubbed segments, natural TTS pacing, no timeline sync.
    JustTranslated,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub whisper_exe_path: Option<PathBuf>,
    pub whisper_model_path: Option<PathBuf>,
    pub python_bin: String,
    pub source_lang: String,
    pub target_lang: String,
    pub output_mode: OutputMode,
    pub output_dir: Option<PathBuf>,
    /// Combien de fichiers traiter EN MÊME TEMPS pendant un batch (chacun
    /// lance ses propres processus whisper-cli / python). 1 = comme avant.
    /// Chaque processus XTTS parallèle recharge son propre modèle (~2 Go de
    /// RAM) : monte progressivement, ne mets pas une grosse valeur d'un coup.
    pub max_parallel: usize,
    /// Lance whisper-cli / Python à priorité CPU réduite, pour que la
    /// machine reste utilisable pendant un gros batch.
    pub low_priority: bool,
    /// Demande à XTTS d'utiliser le GPU (CUDA) s'il en détecte un
    /// disponible via torch ; sinon il repasse automatiquement en CPU.
    /// (Pour whisper-cli, le GPU dépend du binaire whisper.cpp téléchargé —
    /// voir le README — ce réglage n'agit que sur la synthèse vocale.)
    pub use_gpu: bool,
    /// Réglage par défaut pour tout le batch : cloner la voix de chaque
    /// fichier depuis lui-même (comportement d'origine), ou cloner UNE voix
    /// dédiée (fichier audio séparé) sur tous les fichiers. Remplaçable au
    /// cas par cas via `FileJob::voice_override`.
    #[serde(default)]
    pub voice_source: VoiceSource,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            whisper_exe_path: None,
            whisper_model_path: None,
            python_bin: "python3".to_string(),
            source_lang: "fr".to_string(),
            target_lang: "en".to_string(),
            output_mode: OutputMode::FitOriginalDuration,
            output_dir: None,
            max_parallel: 1,
            low_priority: true,
            use_gpu: false,
            voice_source: VoiceSource::FromInputFile,
        }
    }
}

/// Sent from the UI thread to the batch coordinator thread when the user
/// validates a review screen (transcript or translation) for a given job.
/// Carries the edited segments so the coordinator uses the corrected text,
/// not whatever the engine originally produced.
pub struct ValidateMsg {
    pub job_idx: usize,
    pub segments: Vec<Segment>,
}

/// Messages sent from worker threads back to the UI thread.
pub enum PipelineMsg {
    Stage(usize, Stage),
    TranscriptReady(usize, Vec<Segment>),
    TranslationReady(usize, Vec<String>),
    SynthesisProgress(usize, usize, usize), // job_idx, done, total
    /// job_idx, segment_index, clip path, clip duration (s) — for the before/after table.
    SegmentSynthesized(usize, usize, PathBuf, f32),
    /// job_idx, original_duration_s, output_duration_s
    FileDurations(usize, f32, f32),
    Error(usize, String),
}
