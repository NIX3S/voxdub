# VoxDub

App Rust (GUI, `eframe`/`egui`) pour doubler des fichiers audio, 100% offline :

**Audio(s) d'entrée → transcription (Whisper local) → validation/édition →
traduction (Argos Translate local) → validation/édition → clonage vocal
(Coqui XTTS-v2 local) → fichier de sortie**, en mode « même durée que
l'original » (synchro façon doublage) ou « juste traduit » (rythme naturel).

Aucune lib Rust mature ne fait de la transcription + traduction + clonage
vocal nativement, donc l'app **orchestre 3 moteurs** :

| Étape          | Moteur                              | Où il tourne              |
|----------------|--------------------------------------|----------------------------|
| Transcription  | `whisper-cli` (binaire whisper.cpp)  | subprocess externe          |
| Traduction     | Argos Translate                      | subprocess Python           |
| Clonage vocal  | Coqui TTS — XTTS-v2                  | subprocess Python           |
| Conversion/mix | `ffmpeg` / `ffprobe`                 | subprocess externe           |

## ⚠️ Avant de lancer l'app — à télécharger toi-même

L'app ne fait QUE l'orchestration : elle ne contient ni Whisper, ni les
modèles, ni Python. Sans ces deux téléchargements, la transcription ne
fonctionnera pas :

1. **`whisper-cli`** (l'exécutable whisper.cpp) — release précompilée :
   👉 https://github.com/ggml-org/whisper.cpp/releases
   Prends l'archive de ton OS (`whisper-cli.exe` sur Windows, `whisper-cli`
   sur macOS/Linux), et **garde tous les fichiers de l'archive ensemble**
   dans le même dossier (DLLs/libs associées, pas que l'exécutable seul).
   Pour la transcription sur GPU, prends la variante **cuBLAS** (NVIDIA) ou
   **Vulkan** si l'une est proposée pour ton OS — le choix du binaire est
   ce qui détermine si whisper-cli utilise le GPU, il n'y a rien à cocher
   dans l'app pour ça. Si aucune release précompilée ne correspond à ta
   config CUDA (carte récente, version de CUDA Toolkit trop neuve pour le
   build GitHub), compile `whisper-cli` toi-même avec CUDA — voir
   **« Compiler whisper.cpp avec CUDA sur Windows (GPU NVIDIA) »** ci-dessous.

2. **Un modèle Whisper** (`.bin`, format ggml) — par ex. `ggml-medium.bin` :
   👉 https://huggingface.co/ggerganov/whisper.cpp (dossier `resolve/main`)
   Plus le modèle est gros, plus la transcription est précise mais lente
   (`tiny`/`base` = rapide et léger, `medium`/`large-v3` = plus fidèle).

Tu indiqueras les chemins des **deux** (exécutable + modèle) dans le panneau
Réglages de l'app, au premier lancement.

Voir aussi §3 ci-dessous pour l'environnement Python (traduction + voix),
également requis.

### Compiler whisper.cpp avec CUDA sur Windows (GPU NVIDIA)

À utiliser seulement si aucune release précompilée cuBLAS ne convient (carte
trop récente, version de CUDA Toolkit différente de celle utilisée par les
binaires GitHub, etc.). Dans tous les autres cas, la release précompilée du
§1 ci-dessus suffit et évite tout ça.

**Prérequis (si tu pars de zéro) :**
1. **Visual Studio** (Community Edition) avec la charge de travail
   **« Développement Desktop en C++ »** cochée à l'installation.
2. **NVIDIA CUDA Toolkit** (version correspondant à ton driver/ta carte) —
   fournit `nvcc`, les DLLs `cudart64_*`, etc.
3. **Git pour Windows** (pour cloner le dépôt).

**Build rapide (cas standard) :**

Ouvre **Developer Command Prompt for VS** (pas un terminal classique), puis :
```cmd
cd %USERPROFILE%
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp

:: -DGGML_CUDA=1 active le support GPU NVIDIA
cmake -B build -DGGML_CUDA=1
cmake --build build --config Release
```
`whisper-cli.exe` se retrouve dans `%USERPROFILE%\whisper.cpp\build\bin\Release\`.

Télécharge ensuite un modèle et teste :
```cmd
build\bin\Release\whisper-download-models.exe base
build\bin\Release\whisper-cli.exe -m models\ggml-base.bin -f ton_fichier_16k.wav
```
Au lancement, les premières lignes doivent mentionner ta carte NVIDIA
(`BLAS = 1` / mention CUDA) : c'est la confirmation que le GPU est utilisé.

**Si ça ne compile pas (plusieurs versions de CUDA installées, ou CUDA trop
récent pour ta version de MSVC/Visual Studio) :**

Les erreurs typiques viennent d'un **mélange de versions CUDA** (ex. CUDA
12.6 et 13.4 installés tous les deux, et CMake qui pioche dans le mauvais)
ou d'un **CUDA qui refuse ton compilateur MSVC** (`host_config.h` rejette la
version de Visual Studio). Dans ce cas, repars d'un build complètement
vierge en forçant explicitement la version de CUDA et l'architecture GPU
visée (`sm_89` pour une RTX 4000 Laptop par ex. — adapte à ta carte) :

```cmd
cd /d <chemin>\whisper.cpp

:: 1. Nettoie tout build précédent
rmdir /s /q build

:: 2. Force CUDA X.Y dans l'environnement de CE terminal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y"
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib\x64;%PATH%"

:: 3. Vérifie que c'est bien CE nvcc qui est trouvé en premier
where nvcc
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin\nvcc.exe" --version
nvidia-smi

:: 4. Configure CMake en pointant explicitement vers ce CUDA
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vX.Y" ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DGGML_CUDA=ON

:: 5. Vérifie AVANT de compiler que la config a pris le bon CUDA
findstr /I "CUDA CUDAToolkit CMAKE_CUDA_ARCHITECTURES" build\CMakeCache.txt
:: -> tu dois voir CUDA\vX.Y partout, et CMAKE_CUDA_ARCHITECTURES=89
:: -> tu NE dois PLUS voir une autre version de CUDA traîner (ex. v12.6\include)

:: 6. Compile (ajoute --parallel 8 pour limiter la charge CPU pendant le build)
cmake --build build --config Release --parallel
```
- Remplace `vX.Y` par ta version de CUDA Toolkit installée, et `89` par
  l'architecture CUDA de ta carte (`89` = RTX 4000 Laptop/Desktop série ;
  vérifie la tienne si besoin, ex. `86` pour une RTX 3000).
- N'ajoute **pas** `-allow-unsupported-compiler` tant que tu n'as pas
  confirmé que le problème vient bien d'un mélange de versions CUDA plutôt
  que d'une vraie incompatibilité CUDA ↔ Visual Studio — ce flag contourne
  la vérification au lieu de régler la cause, et peut produire un binaire
  qui compile mais plante à l'exécution.
- Si l'étape 6 échoue encore, l'erreur utile est généralement la première
  ligne commençant par `fatal error` ou `error C...` dans la sortie —
  c'est elle qui indique si le souci est CUDA↔VS ou autre chose.

## Obtenir l'app (Windows / macOS / Linux)

### Option A — build automatique via GitHub Actions (recommandé)
Le dépôt inclut `.github/workflows/build.yml`, qui compile nativement sur
Windows, macOS (Intel + Apple Silicon) et Linux à chaque push, et publie les
binaires en artifacts téléchargeables :

1. Pousse ce dossier sur un dépôt GitHub (`git init && git add . && git commit -m init`,
   crée un repo sur GitHub, `git push`).
2. Onglet **Actions** → workflow **build** → laisse-le tourner (automatique
   au push, ou bouton **Run workflow**).
3. Une fois le run vert, ouvre-le et télécharge dans **Artifacts** :
   `voxdub-windows-x64`, `voxdub-macos-x64`, `voxdub-macos-arm64` ou
   `voxdub-linux-x64` (chacun contient l'exécutable + `scripts/` + le README).

Ça évite d'installer un toolchain Rust sur chaque machine cible : chaque OS
compile le sien nativement chez GitHub.

### Option B — build local (une seule plateforme, celle où tu es)
```bash
cargo run --release
```
Nécessite Rust (https://rustup.rs) installé sur la machine. Plus besoin de
CMake/LLVM/MSVC pour whisper.cpp (il est appelé en binaire externe) ; ce qui
reste à compiler (eframe/rodio/cpal, etc.) ne demande qu'un toolchain Rust
standard.

### Après téléchargement/build : par OS
- **Windows** : double-clique `voxdub.exe` (ou `cargo run --release`).
  Windows Defender/SmartScreen peut avertir pour un exe non signé — « Plus
  d'infos » → « Exécuter quand même ».
- **macOS** : `chmod +x voxdub && ./voxdub` dans un Terminal (ou double-clic).
  Gatekeeper bloquera un binaire non signé au premier lancement — clic droit
  sur `voxdub` → **Ouvrir** (au lieu du double-clic), puis confirme.
- **Linux** : `chmod +x voxdub && ./voxdub`. Il faut les libs GTK/ALSA
  installées (voir §1) même pour un binaire déjà compilé.

## Installation

### 1. Dépendances système
- [ffmpeg](https://ffmpeg.org/) (fournit aussi `ffprobe`), sur le `PATH`.
- Linux : paquets de dev pour la fenêtre GUI (`libgtk-3-dev`,
  `libxkbcommon-dev`, `libssl-dev` selon distro).

### 2. Whisper : exécutable + modèle
Voir la section **⚠️ Avant de lancer l'app** plus haut.

### 3. Environnement Python (traduction + clonage vocal)
`TTS` (Coqui) ne supporte que **Python 3.9 à 3.11**. Vérifie/choisis la bonne
version avant de créer le venv :
```powershell
py -0                     # Windows : liste les versions Python installées
py -3.11 -m venv .venv    # crée le venv avec Python 3.11 (installe-le si absent)
.venv\Scripts\activate
```
```bash
python3.11 -m venv .venv  # macOS / Linux
source .venv/bin/activate
```
Puis :
```bash
python -m pip install --upgrade pip
pip install -r scripts/requirements.txt
```

> Si tu ne veux pas installer un Python plus ancien à côté de ta version
> actuelle, utilise plutôt le fork communautaire maintenu, compatible
> jusqu'à Python 3.12 : `pip install coqui-tts` au lieu de `TTS` (même
> import `from TTS.api import TTS` dans `scripts/clone_tts.py`, rien à
> changer côté code).

Puis télécharge le pack de langue Argos Translate pour ta paire
source→cible (exemple fr→en) :
```bash
python -c "
import argostranslate.package as p
p.update_package_index()
pkgs = p.get_available_packages()
pkg = next(x for x in pkgs if x.from_code=='fr' and x.to_code=='en')
p.install_from_path(pkg.download())
"
```
Le modèle XTTS-v2 (~2 Go) se télécharge automatiquement au premier appel de
`scripts/clone_tts.py`, puis reste en cache localement.

> **Licence du modèle XTTS-v2** : non-commerciale (CPML —
> https://coqui.ai/cpml.txt). Coqui TTS demande normalement une confirmation
> interactive au tout premier chargement ; comme VoxDub pilote Python en
> pipe (pas un vrai terminal), `clone_tts.py` positionne automatiquement
> `COQUI_TOS_AGREED=1` pour éviter que ce prompt reste bloqué indéfiniment
> à attendre une frappe qui n'arrivera jamais (symptôme observé : l'étape
> « Synthèse vocale… chargement du modèle » qui ne bouge plus pendant
> plusieurs minutes, aussi bien en mode batch qu'en single-file). Utiliser
> l'app vaut acceptation de cette licence.

**Pour utiliser le GPU côté synthèse vocale** (case « Utiliser le GPU » dans
les Réglages), installe en plus un torch avec support CUDA, par exemple
(adapter à ta version CUDA) :
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Voir https://pytorch.org/get-started/locally/ pour la commande exacte selon
ton GPU/driver. Sans ce torch-CUDA, la case GPU retombe automatiquement sur
le CPU (pas d'erreur, juste pas d'accélération).

Dans l'app, indique le chemin de l'interpréteur Python du venv (ex.
`/chemin/vers/.venv/bin/python`) dans le panneau Réglages.

### 4. Build & lancement
```bash
cargo run --release
```

## Utilisation

1. **Réglages** (panneau de droite) : chemin de `whisper-cli.exe`, chemin du
   modèle Whisper, interpréteur Python, langue source/cible, mode de sortie,
   dossier de sortie, et les réglages de performance (voir plus bas).
2. **+ Ajouter des fichiers audio…** : ogg, wav, mp3, flac, m4a, aac, opus.
3. Sélectionne un fichier dans la liste, clique **Lancer la transcription**.
4. Relis/corrige le texte segment par segment, **Valider et traduire**.
5. Relis/corrige la traduction, **Valider et générer la voix**.
6. Le fichier doublé est écrit dans le dossier de sortie sous
   `<nom>_<langue_cible>.<extension>` — **la sortie est ré-encodée dans le
   même format que l'entrée** (ogg → ogg, mp3 → mp3, wav → wav, ...) via
   `ffmpeg`. Le mix interne se fait toujours en wav, puis un dernier passage
   ffmpeg convertit vers le format d'origine.
7. Une fois terminé, un écran **« Comparaison avant / après »** apparaît :
   durée totale originale vs doublée (et l'écart en % ), plus un tableau par
   segment (texte original/traduit, durée avant/après) avec un bouton
   **▶ orig** / **▶ doublé** pour écouter chaque extrait directement dans
   l'app (lecture via `rodio`, pas besoin de lecteur externe) et juger de la
   qualité avant de garder le fichier.

### Mode batch (tous les fichiers d'un coup)
Dans le panneau de gauche, deux boutons traitent **tous** les fichiers ajoutés :

- **⚡ Batch sans review** : transcription → traduction → synthèse vocale de
  tous les fichiers, entièrement automatique, aucune validation humaine.
- **📝 Batch avec review** : le batch transcrit tous les fichiers, puis
  **attend ta validation** fichier par fichier (bouton « ✔ Valider », ou
  « Tout valider » si tu ne relis pas). Dès qu'un fichier est validé, l'étape
  suivante (traduction) est enchaînée pour lui pendant que tu relis les
  autres ; même principe après la traduction, avant la synthèse vocale.
  « → Prochain à valider » t'amène directement au prochain fichier en
  attente.

Le batch tourne sur un thread dédié, indépendant de l'affichage : il continue
même si tu minimises la fenêtre, changes d'onglet, ou que le PC verrouille
l'écran (il ne continue PAS si le PC s'éteint ou hiberne — voir « Reprendre
après fermeture » plus bas pour ce cas).

- **⏸ Pause / ▶ Reprendre** : met le batch en pause (le fichier en cours
  termine son étape actuelle — un fichier déjà lancé en synthèse n'est pas
  coupé en plein milieu — puis plus rien de nouveau ne démarre) et le
  relance exactement où il en était, **sans fermer l'app**.
- **⏹ Arrêter le batch** : comme Pause, mais referme aussi le thread du
  batch. Pour reprendre après ça, reclique sur ⚡/📝 : les fichiers déjà
  terminés ne sont pas refaits, seuls les fichiers non terminés repartent
  (voir ci-dessous).

Les fichiers en erreur sont sautés (le batch continue avec les autres) et
peuvent être relancés via « Réessayer depuis le début ».

### Reprendre après fermeture de l'app / redémarrage du PC
L'app sauvegarde automatiquement la liste des fichiers (et leur étape
actuelle) dans `voxdub_state.json`, à côté de l'exécutable, à chaque
changement notable (toutes les ~2 secondes pendant un batch actif, et à la
fermeture de l'app). Au prochain lancement, si ce fichier existe, une
bannière apparaît : « Session précédente restaurée : N fichier(s)... ».
Il suffit de recliquer sur **⚡ Batch sans review** ou **📝 Batch avec
review** pour reprendre : les fichiers déjà **Terminé** ou **Erreur** ne
sont pas retraités, seuls les autres repartent.

Limite à connaître : un fichier qui était **en plein milieu** d'une étape
(conversion, transcription, traduction ou synthèse) au moment de la
fermeture/coupure ne peut pas reprendre à l'octet où il s'est arrêté — cette
étape-là est entièrement refaite depuis son dernier point sûr (par exemple,
un fichier interrompu en pleine synthèse vocale repart depuis la traduction
déjà validée, pas depuis zéro). Un fichier déjà **Terminé** avant la
fermeture n'est, lui, jamais retraité.

### Performance sur un gros volume (des milliers de fichiers)
Trois réglages dans le panneau Réglages, sous « Performance / gros batchs » :

- **Fichiers en parallèle** (1 à 32) : combien de fichiers l'app traite EN
  MÊME TEMPS à chaque étape (plusieurs `whisper-cli` ou process Python
  tournant à la fois, au lieu d'un seul). C'est le levier le plus direct
  pour accélérer un batch de plusieurs milliers de fichiers — mais chaque
  processus Python de synthèse qui tourne en parallèle recharge son propre
  modèle XTTS (~2 Go de RAM chacun), donc monte progressivement (2, puis 4…)
  en surveillant le Gestionnaire des tâches plutôt que de passer direct à
  32.
- **Priorité CPU réduite** : whisper-cli et les scripts Python tournent à
  priorité système réduite (`BELOW_NORMAL` sous Windows, `nice` sous
  Linux/macOS), pour que le PC reste réactif si tu veux continuer à
  travailler pendant que le batch tourne en arrière-plan. Ça ne limite pas
  à un pourcentage de CPU fixe : l'OS donne quand même tout le CPU libre au
  batch dès que tu n'en as pas besoin, il cède juste la priorité plus vite
  dès qu'une app au premier plan en demande.
- **Utiliser le GPU (CUDA) pour la voix** : accélère XTTS si un torch-CUDA
  est installé (voir §3) — sinon repasse automatiquement sur CPU, sans
  erreur. Pour que la **transcription** utilise aussi le GPU, c'est le choix
  du binaire whisper-cli téléchargé (cuBLAS/Vulkan) qui compte, pas un
  réglage ici — voir §⚠️ en haut.

  Depuis la synthèse, l'app affiche maintenant dans l'onglet du fichier
  concerné une ligne indiquant le device réellement utilisé (🟢 GPU, ou ⚠
  CPU si le GPU était demandé mais pas obtenu). **La cause la plus
  fréquente d'un XTTS qui reste bloqué sur CPU malgré la case cochée** :
  `pip install TTS` installe `torch` comme dépendance, et sur Windows ça
  installe par défaut une build **CPU-only**, même si tu as déjà fait
  `pip install torch --index-url .../cu121` plus tôt (une réinstallation ou
  un ordre d'installation différent peut l'écraser avec la version CPU).
  Vérifie avec :
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
  ```
  Si `torch.version.cuda` affiche `None` ou si `torch.cuda.is_available()`
  est `False`, réinstalle en forçant la build CUDA (adapte `cu121` à ta
  version de CUDA Toolkit, et fais-le **en dernier**, après `pip install -r
  requirements.txt`, pour être sûr que rien ne la réécrase ensuite) :
  ```bash
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
  ```

**Le changement qui compte le plus** pour un très gros volume n'est même pas
dans ces réglages. Il y a eu deux étapes :
1. D'abord, arrêter de recharger le modèle XTTS (~2 Go, 10-30s+) **à chaque
   segment** (un fichier de plusieurs dizaines de segments rechargeait le
   modèle autant de fois) — en le rechargeant une seule fois **par fichier**
   à la place.
2. Maintenant, un cran plus loin : le process `clone_tts.py` reste **vivant
   pour toute la durée du batch** et traite les fichiers les uns après les
   autres sans jamais redémarrer. Le modèle n'est donc chargé **qu'une
   seule fois par emplacement de parallélisme** (voir « Fichiers en
   parallèle » ci-dessus), pas une fois par fichier : pour 5000 fichiers
   avec `max_parallel = 1`, c'est 1 chargement du modèle au lieu de 5000.
   Si un worker plante en cours de route, l'app en relance un neuf
   automatiquement pour le fichier suivant (auto-guérison, aucune action
   nécessaire).

Autrement dit, un batch de plusieurs milliers de fichiers ne paie quasiment
plus **aucun** rechargement de modèle une fois lancé — tout le temps passé
est du temps de synthèse réel, pas de l'attente de chargement.

### Voix pour le clonage : depuis le fichier, ou une voix dédiée
Par défaut, chaque fichier est doublé **avec sa propre voix**, extraite
automatiquement des 12 premières secondes de son propre audio — c'est le
comportement d'origine, rien à faire pour l'avoir.

Deux façons de changer ça :

- **Réglages → « Voix pour le clonage (réglage par défaut du batch) »** :
  bouton **« Choisir une voix dédiée… »** pour sélectionner UN fichier audio
  (ogg/wav/mp3/flac/m4a/aac/opus) dont la voix sera clonée sur **tous les
  fichiers** du batch, à la place de la voix de chacun. Utile par exemple
  pour doubler tout un lot avec la voix d'un acteur voix qui a accepté
  d'être cloné, plutôt que de garder la voix (souvent de moins bonne
  qualité d'enregistrement) de chaque source. Bouton **« Voix de chaque
  fichier (par défaut) »** pour revenir au comportement d'origine.
- **Dans l'onglet d'un fichier précis** (en haut, sous son nom) : **« Choisir
  une voix pour CE fichier… »** remplace, uniquement pour ce fichier-là, le
  réglage par défaut ci-dessus (qu'il soit « chaque fichier » ou « voix
  dédiée »). Un bouton **« ✕ Revenir au réglage par défaut »** apparaît une
  fois une voix spécifique choisie, pour l'annuler.

Dans les deux cas, ce choix doit être fait **avant que la synthèse vocale ne
démarre** pour ce fichier (donc avant ou pendant l'étape de validation de la
traduction) : le changer après que la voix a déjà été générée n'a d'effet
qu'en relançant le traitement de ce fichier depuis le début.

### Modes de sortie
- **Même durée que l'original** : chaque segment doublé est étiré/compressé
  (`ffmpeg atempo`) pour tenir dans la fenêtre temporelle du segment
  d'origine, puis tous les segments sont mixés sur une piste silencieuse à
  leur timing d'origine (façon doublage synchronisé).
- **Juste traduit** : les segments doublés sont simplement mis bout à bout,
  au rythme naturel de la synthèse vocale (pas de contrainte de durée).

## Limites connues / pistes d'amélioration
- Le clonage vocal utilise un seul extrait de référence de 12 secondes
  (depuis le fichier source, ou depuis la voix dédiée choisie — voir
  « Voix pour le clonage » plus haut) ; pour des fichiers multi-locuteurs,
  il faudrait détecter les locuteurs (diarization) et cloner chaque voix
  séparément plutôt qu'une seule voix pour tout le fichier.
- Le format de sortie correspond déjà au format d'entrée (voir section
  Utilisation) ; seul `.wav` est utilisé comme format intermédiaire interne.
- La pause (⏸) agit entre fichiers/étapes, pas en plein milieu d'un appel
  à whisper-cli ou à clone_tts.py sur un très gros fichier : un fichier déjà
  lancé sur une étape la termine avant que la pause prenne effet.
