#!/usr/bin/env python3
"""
Synthese vocale avec clonage de voix, offline, via Coqui TTS (XTTS-v2) —
PROCESS PERSISTANT : charge le modele XTTS UNE SEULE FOIS au demarrage, puis
reste vivant en traitant un FICHIER (plusieurs segments) a la fois, autant
de fois que le process Rust appelant lui en envoie, jusqu'a ce que stdin se
ferme (EOF).

Ca remplace un design precedent qui rechargeait deja le modele une fois par
FICHIER plutot qu'une fois par SEGMENT (un premier gain), mais relancait
quand meme un nouveau process Python (et donc un nouveau chargement du
modele, ~2Go, 10-30s+) pour chaque fichier. Avec des milliers de fichiers
dans un batch, ce cout de rechargement repete dominait encore largement le
temps de synthese reel. Ici, un seul process reste ouvert et traite tous
les fichiers qui lui sont confies a la suite : le modele n'est charge
qu'UNE fois pour toute la duree de vie du process (un process par
"emplacement" de parallelisme cote app, pas un par fichier).

Protocole (JSON Lines sur stdin/stdout, un objet par ligne) :

  - Au demarrage, une fois le modele charge, une ligne annonce le device :
        {"device": "cuda"}  (ou "cpu")

  - Pour CHAQUE fichier a traiter, le process Rust ecrit UNE ligne requete :
        {"req_id": 0, "lang": "en", "speaker": "/chemin/ref.wav",
         "items": [{"text": "...", "out": "/chemin/seg0.wav"}, ...]}

  - Pour chaque segment termine, ce script repond par UNE ligne (streaming,
    pas besoin d'attendre la fin du fichier) :
        {"req_id": 0, "index": 0, "ok": true, "out": "/chemin/seg0.wav"}
        {"req_id": 0, "index": 1, "ok": false, "error": "..."}
    puis, une fois TOUS les segments de ce fichier traites :
        {"req_id": 0, "done": true}
    Le process attend alors la prochaine ligne requete sur stdin (fichier
    suivant), et ainsi de suite jusqu'a fermeture de stdin par l'appelant.

Usage :
    python clone_tts.py [--gpu]   # puis une ligne JSON par fichier sur stdin

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
import os
import sys

# Le modele XTTS-v2 est sous licence non-commerciale (CPML) : au tout
# premier chargement, Coqui TTS demande une confirmation interactive
# (input() sur la console) avant de telecharger/charger le modele. Notre
# stdin est un pipe (pas un vrai terminal), donc ce prompt resterait bloque
# indefiniment a attendre une frappe qui n'arrivera jamais. Cette variable
# d'environnement contourne le prompt (la meme licence est quand meme
# rappelee ci-dessous, pour que l'utilisateur sache a quoi il consent).
# setdefault : si le cote Rust l'a deja positionnee, on ne l'ecrase pas.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# Filet de securite : si malgre COQUI_TOS_AGREED une version de la lib
# appelle quand meme input() (autre prompt, version differente, etc.), on
# repond automatiquement "y" au lieu de rester bloque indefiniment sur un
# stdin qui n'est pas un vrai terminal. On logue sur stderr pour que ce ne
# soit jamais silencieux.
import builtins  # noqa: E402

_orig_input = builtins.input


def _auto_yes_input(prompt: str = "") -> str:
    print(
        f"[voxdub] prompt interactif intercepte (repondu automatiquement 'y') : {prompt!r}",
        file=sys.stderr,
    )
    return "y"


builtins.input = _auto_yes_input

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

    cuda_ok = torch.cuda.is_available()
    print(
        f"[voxdub] torch={torch.__version__}  torch_cuda_build={torch.version.cuda}  "
        f"cuda_available={cuda_ok}",
        file=sys.stderr,
    )
    if cuda_ok:
        try:
            print(f"[voxdub] GPU detecte : {torch.cuda.get_device_name(0)}", file=sys.stderr)
        except Exception as e:  # pragma: no cover - purely informational
            print(f"[voxdub] GPU detecte mais nom illisible : {e}", file=sys.stderr)
    elif args.gpu:
        print(
            "[voxdub] --gpu demande mais torch.cuda.is_available() = False. Cause la plus "
            "frequente : le torch installe est une build CPU-ONLY (c'est ce que "
            "'pip install torch' installe par defaut sur Windows, meme si 'TTS' l'a deja "
            "installe comme dependance). Verifie avec : "
            "python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\" "
            "-- si torch.version.cuda affiche None, reinstalle avec : "
            "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121 "
            "(adapte cu121 a ta version CUDA). Voir le README.",
            file=sys.stderr,
        )

    use_gpu = bool(args.gpu) and cuda_ok
    device = "cuda" if use_gpu else "cpu"

    print(
        "[voxdub] chargement du modele XTTS-v2 (licence non-commerciale CPML, "
        "confirmation auto via COQUI_TOS_AGREED=1 - voir "
        "https://coqui.ai/cpml.txt) ... peut prendre 10-30s, ou plusieurs "
        "minutes au tout premier lancement le temps du telechargement (~2 Go).",
        file=sys.stderr,
    )

    # NOTE: le kwarg historique `TTS(model, gpu=True)` est peu fiable selon
    # la version de coqui-tts/TTS installee (silencieusement ignore sur
    # certaines versions) ; `.to(device)` explicite est la facon documentee
    # et fiable de placer le modele sur le GPU.
    tts = TTS(MODEL_NAME)
    try:
        tts = tts.to(device)
    except Exception as e:
        print(f"[voxdub] echec de chargement sur {device} ({e}) : repli sur CPU.", file=sys.stderr)
        device = "cpu"
        tts = tts.to(device)

    print(f"[voxdub] XTTS charge sur : {device}", file=sys.stderr)
    # Ligne structuree (stdout, pas stderr) que le cote Rust peut parser pour
    # afficher dans l'app quel device a vraiment ete utilise, meme quand tout
    # se passe bien (le stderr n'est normalement affiche qu'en cas d'echec).
    emit({"device": device})

    # Boucle persistante : une ligne JSON sur stdin = un fichier entier a
    # traiter. `for line in sys.stdin` bloque (sans consommer de CPU) tant
    # qu'aucune ligne n'est disponible, et se termine proprement des que
    # l'appelant ferme stdin (EOF) - c'est le signal "plus rien a faire,
    # tu peux t'arreter" ; aucune commande d'arret explicite n'est donc
    # necessaire.
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as e:
            emit({"fatal": f"ligne requete non-JSON recue : {e}"})
            continue

        req_id = request.get("req_id")
        lang = request["lang"]
        speaker = request["speaker"]
        items = request["items"]  # [{"text": ..., "out": ...}, ...]

        for i, item in enumerate(items):
            text = item["text"]
            out = item["out"]
            try:
                if text.strip():
                    tts.tts_to_file(text=text, speaker_wav=speaker, language=lang, file_path=out)
                else:
                    write_silence(out)
                emit({"req_id": req_id, "index": i, "ok": True, "out": out})
            except Exception as e:
                emit({"req_id": req_id, "index": i, "ok": False, "error": str(e)})

        emit({"req_id": req_id, "done": True})


if __name__ == "__main__":
    main()
