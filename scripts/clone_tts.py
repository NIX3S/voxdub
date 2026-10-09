#!/usr/bin/env python3
"""
Synthese vocale BATCH avec clonage de voix, offline, via Coqui TTS (XTTS-v2).

Charge le modele XTTS UNE SEULE FOIS pour tout un fichier, puis synthetise
tous ses segments a la suite - au lieu de relancer un processus (et de
RECHARGER le modele, ~2Go, 10-30s+) pour chaque segment individuellement.
C'etait le principal goulot d'etranglement de performance de la version
precedente : avec des dizaines de segments par fichier x des milliers de
fichiers, le rechargement du modele dominait largement le temps total.

Entree, un seul JSON sur stdin:
    {"lang": "en", "speaker": "/chemin/ref.wav",
     "items": [{"text": "...", "out": "/chemin/seg0.wav"}, ...]}

Pour chaque segment termine, imprime UNE ligne JSON sur stdout (streaming,
pas attendre la fin du fichier) :
    {"index": 0, "ok": true, "out": "/chemin/seg0.wav"}
    {"index": 1, "ok": false, "error": "..."}
afin que l'appelant (Rust) puisse suivre la progression segment par segment
et ne pas attendre la fin du fichier entier pour afficher quoi que ce soit.

Usage:
    <stdin JSON> | python clone_tts.py [--gpu]

Installation (necessite Python 3.9-3.11, et idealement un GPU mais marche
aussi sur CPU, plus lentement):
    pip install TTS
Le modele (~2GB) est telecharge automatiquement au premier lancement et mis
en cache localement ensuite (fonctionnement offline par la suite).

GPU (--gpu) : necessite un torch installe avec support CUDA (voir le
README : pip install torch --index-url https://download.pytorch.org/whl/cu121
ou la variante correspondant a ton GPU/driver). Si --gpu est demande mais
qu'aucun CUDA n'est detecte, on retombe automatiquement sur le CPU.
"""
import argparse
import json
import sys

# Sur Windows, stdin/stdout/stderr utilisent par defaut l'encodage console
# local (cp1252 typiquement), pas UTF-8. Des lors qu'un texte transcrit ou
# traduit contient un accent ou un caractere combinant, un print() (y
# compris ceux internes a la lib TTS) peut planter avec un
# UnicodeEncodeError, ou pire, ecrire des octets corrompus. On force UTF-8
# partout, avec remplacement plutot que crash en dernier recours.
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch

# PyTorch >= 2.6 changed torch.load's default to weights_only=True, which
# breaks loading the XTTS checkpoint (it's a pickled config object, not just
# tensors). We trust the official Coqui weights, so force the old behaviour.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Recent torchaudio versions route torchaudio.load through torchcodec, which
# needs FFmpeg "full-shared" DLLs on Windows. XTTS only uses it to read our
# reference clip, which is a plain PCM16 wav produced by ffmpeg, so we swap
# in a tiny stdlib-based loader and avoid the torchcodec dependency entirely.
import wave  # noqa: E402

import numpy as np  # noqa: E402
import torchaudio  # noqa: E402


def _load_wav_pcm16(path, *args, **kwargs):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sample_width != 2:
        raise RuntimeError(
            f"Only 16-bit PCM wav supported for the speaker reference (got {sample_width * 8}-bit)"
        )
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    data = data.reshape(-1, channels).T  # (channels, frames), like torchaudio.load
    return torch.from_numpy(data.copy()), sr


torchaudio.load = _load_wav_pcm16

from TTS.api import TTS  # noqa: E402  (must import after the patches above)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"


def write_silence(out_path: str, seconds: float = 0.1, sample_rate: int = 24000) -> None:
    """Minimal silent wav, used for segments whose translated text is empty."""
    n_frames = int(seconds * sample_rate)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Utiliser le GPU (CUDA) si disponible")
    args = parser.parse_args()

    request = json.loads(sys.stdin.read())
    lang = request["lang"]
    speaker = request["speaker"]
    items = request["items"]  # [{"text": ..., "out": ...}, ...]

    use_gpu = bool(args.gpu) and torch.cuda.is_available()
    if args.gpu and not use_gpu:
        print(
            "GPU demande mais aucun CUDA detecte par torch (driver/torch-cuda "
            "manquant ?) : retour au CPU.",
            file=sys.stderr,
        )

    tts = TTS(MODEL_NAME, gpu=use_gpu)

    for i, item in enumerate(items):
        text = item["text"]
        out = item["out"]
        try:
            if text.strip():
                tts.tts_to_file(text=text, speaker_wav=speaker, language=lang, file_path=out)
            else:
                write_silence(out)
            emit({"index": i, "ok": True, "out": out})
        except Exception as e:
            emit({"index": i, "ok": False, "error": str(e)})


if __name__ == "__main__":
    main()
