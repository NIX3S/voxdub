#!/usr/bin/env python3
"""
Traduction offline via Argos Translate.
Lit un JSON sur stdin: {"texts": [...], "src": "fr", "tgt": "en"}
Ecrit un JSON sur stdout: {"translations": [...]}

Si le pack de langue (src -> tgt) n'est pas installe, il est telecharge
automatiquement (internet necessaire UNE fois), avec pivot via l'anglais
si la paire directe n'existe pas (ex: fr -> es = fr->en + en->es).
Ensuite tout fonctionne offline.
"""
import sys
import time

# Sur Windows, stdin/stdout/stderr utilisent par defaut l'encodage console
# local (cp1252 typiquement), pas UTF-8. Le JSON lu sur stdin et ecrit sur
# stdout doit etre en UTF-8 des deux cotes (c'est ce que Rust envoie et
# attend), sinon les accents sont corrompus.
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def find_translation(src: str, tgt: str):
    import argostranslate.translate as at

    if hasattr(at.get_installed_languages, "cache_clear"):
        at.get_installed_languages.cache_clear()
    langs = at.get_installed_languages()
    from_lang = next((l for l in langs if l.code == src), None)
    to_lang = next((l for l in langs if l.code == tgt), None)
    if from_lang is None or to_lang is None:
        return None
    return from_lang.get_translation(to_lang)


def install_pair(src: str, tgt: str) -> bool:
    import argostranslate.package as ap

    available = ap.get_available_packages()
    pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
    if pkg is None:
        return False
    log(f"Telechargement du pack Argos {src} -> {tgt}...")
    ap.install_from_path(pkg.download())
    return True


def ensure_translation(src: str, tgt: str):
    t = find_translation(src, tgt)
    if t is not None:
        return t

    log(f"Pack {src} -> {tgt} absent, tentative d'installation automatique...")
    try:
        import argostranslate.package as ap

        ap.update_package_index()
        if not install_pair(src, tgt) and src != "en" and tgt != "en":
            install_pair(src, "en")
            install_pair("en", tgt)
    except Exception as e:  # reseau coupe, index inaccessible, etc.
        log(f"Installation automatique impossible: {e}")

    return find_translation(src, tgt)


def translate_with_retry(translation, text: str, attempts: int = 8, base_delay: float = 0.4) -> str:
    """
    Argos Translate decoupe les phrases via un pipeline Stanza qui, a
    chaque NOUVEAU processus Python, revalide/retelecharge son fichier
    'resources.json' local. Sur Windows, un antivirus ou un autre acces
    concurrent au fichier peut faire echouer ce renommage avec une
    PermissionError (WinError 5) transitoire. Comme ce chargement n'arrive
    qu'UNE fois par processus (mis en cache ensuite en memoire), un simple
    nouvel essai avec un petit delai suffit presque toujours.
    """
    last_err = None
    for attempt in range(attempts):
        try:
            return translation.translate(text) if text.strip() else ""
        except PermissionError as e:
            last_err = e
            log(
                f"Erreur de permission transitoire (tentative {attempt + 1}/{attempts}), "
                f"nouvel essai dans {base_delay * (2 ** attempt):.1f}s..."
            )
            time.sleep(base_delay * (2 ** attempt))
    raise last_err


def main() -> None:
    req = json.loads(sys.stdin.read())
    texts = req["texts"]
    src = req["src"].strip().lower()
    tgt = req["tgt"].strip().lower()

    if src == "auto":
        log("La langue source ne peut pas etre 'auto' pour la traduction: "
            "indique un code precis (fr, en, es, de...) dans les Reglages.")
        sys.exit(1)

    translation = ensure_translation(src, tgt)
    if translation is None:
        import argostranslate.translate as at

        installed = sorted(l.code for l in at.get_installed_languages())
        log(f"Impossible de traduire {src} -> {tgt}: aucun pack Argos disponible.")
        log(f"Langues installees: {installed}")
        log("Verifie les codes de langue (ISO 639-1) et ta connexion pour le "
            "telechargement initial du pack.")
        sys.exit(1)

    translations = []
    for t in texts:
        translations.append(translate_with_retry(translation, t))

    sys.stdout.write(json.dumps({"translations": translations}))


if __name__ == "__main__":
    main()
