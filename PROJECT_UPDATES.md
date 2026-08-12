# Project Updates

> Living status doc. Read this first when resuming work in a new session.
> Last updated: 2026-08-12 (corrected stale Phase 4 status — diarization was
> already scaffolded and verified on 2026-08-06 but this doc never said so;
> confirmed via a fresh dry-run that all 199 files are ready for the real
> batch run; wrote `notebooks/colab_diarization.ipynb` to actually run it —
> not yet executed)

## What this project is

`persona-clone` — a text-to-text persona/style replication pipeline for a
public figure, grounded in their real public statements. Two-tier design:

- **RAG tier**: retrieve grounded facts/quotes from the person's real
  statements at inference time.
- **Fine-tuning tier**: LoRA/QLoRA fine-tune a base LLM to imitate the
  person's speaking style.

Full pipeline (11 stages, `src/<stage>/`):

1. **acquisition** — download source video/audio via yt-dlp
2. **audio** — extract audio tracks from raw video via ffmpeg
3. **transcription** — transcribe audio with faster-whisper
4. **diarization** — separate speakers with pyannote.audio, isolate target speaker
5. **cleaning** — normalize/clean diarized transcripts
6. **chunking** — split cleaned text into semantic chunks (LlamaIndex)
7. **embeddings** — embed chunks (BGE-M3), index in Qdrant for RAG
8. **dataset_gen** — synthesize instruction-response pairs for fine-tuning
9. **finetuning** — LoRA/QLoRA fine-tune a base LLM on generated pairs
10. **rag** — retrieve relevant context from Qdrant at inference time
11. **serving** — serve the persona clone via FastAPI (+ vLLM)

Evaluation (BERTScore, ROUGE, BLEU, RAGAS) lives in `eval/`.

## Current status: Phase 1–3 done. Phase 4 (diarization) is code-complete and verified on 1 file; the full 199-file batch run is the next real step.

Stages 5–11 are still empty scaffold stubs (`src/<stage>/*.py` are ~6-line
placeholder files). Stage 4 (diarization) is **not** a stub — it was fully
scaffolded and verified end-to-end on a real file back on 2026-08-06
(commit `9607e5b`), but this file was never updated to say so until now
(2026-08-12). See "What's built — Phase 4" below. Real work so far is in
`src/acquisition/`, `src/audio/`, `src/transcription/`, and `src/diarization/`.

All 199 audio files in `data/raw/audio/` now have a matching transcript in
`data/transcripts/` (verified 2026-08-06) — see "Phase 3 completion" below
for how the run was actually done (split across Colab T4 and the local
T1000 to work around a Colab free-tier daily GPU quota cutoff).

**⚠️ Hardware discrepancy, still not explicitly resolved**: the GPU machine
actually used since 2026-08-04 is an **NVIDIA T1000, 4GB VRAM** (Windows 11,
driver 595.95, CUDA 13.2, compute capability 7.5) — confirmed via
`nvidia-smi`. This does **not** match the machine described below in the
original Phase 3 hardware note (128GB RAM / RTX A4000 16GB VRAM). Unclear
whether the plan changed to this T1000 box, or the A4000 machine is a
separate/different machine still to be used. The T1000 has now proven
*capable* of running Phase 3 end-to-end (see below), so this is no longer a
hard blocker for transcription — but it's still an open question for stages
4 (diarization), 7 (embeddings), 9 (fine-tuning), 11 (serving): a 4GB card
is a much tighter constraint than 16GB, especially for fine-tuning/serving,
and Phase 3 only worked on the T1000 because faster-whisper/ctranslate2 has
unusually low VRAM requirements relative to those later stages.

### What's built — Phase 1 (`src/acquisition/`)

CLI: `python -m src.acquisition.download <url> [options]`

Given a single video, playlist, or channel URL, it:
- Flat-resolves the URL into a list of video entries via yt-dlp
- Downloads each video at a configurable max quality:
  - video+audio → `data/raw/videos/<id>.<ext>`
  - `--audio-only` → `data/raw/audio/<id>.<ext>` (separate directory, so the
    two modes never collide on disk)
- Extracts and saves per-video metadata JSON to `data/raw/metadata/<id>.json`
  (title, description, channel, duration, view/like counts, resolution, etc.)
- Logs every attempt (success/failed) to `data/raw/download_history.csv`,
  including which `mode` (`video`/`audio`) it was
- Skips videos already downloaded **per mode** (checks history CSV + disk) —
  reruns are idempotent, and downloading audio-only doesn't skip a later
  full video+audio download of the same video, or vice versa
- Retries failed downloads with exponential backoff
- Supports `--dry-run` to preview without downloading
- Supports `--max-items` (cap playlist/channel size), `--quality`,
  `--output-dir`, `--log-level`, `--config` overrides
- `--quality` also controls audio selection in `--audio-only` mode:
  `lowest`/`worst` → `worstaudio/worst`, anything else → `bestaudio/best`
  (the default, numeric height values like `"1080"` only apply to video mode)

File map: `download.py` (CLI) → `downloader.py` (`VideoDownloader` engine)
→ `config.py` (`configs/acquisition.yaml` → `AcquisitionConfig`) +
`metadata.py` (per-video JSON sidecars) + `history.py` (`DownloadHistory`,
CSV ledger keyed on `(video_id, mode)`) + `logging_setup.py`.

**Note on history schema**: `history.py`'s `DownloadHistory` auto-migrates
old-schema CSVs (pre-`mode` column) to the current schema on load, rather
than corrupting them via a naive positional append — this was a real bug
hit and fixed during Phase 1 hardening (2026-07-31).

### What's built — Phase 2 (`src/audio/`)

CLI: `python -m src.audio.extract [options]`

- Scans `data/raw/videos/` for video files (no video present for an ID = no
  extraction to do — audio-only downloads never land there in the first
  place)
- For each video, extracts its audio track via `ffmpeg` (shells out via
  `subprocess`, not the `ffmpeg-python` package in requirements.txt) into
  `data/raw/audio/<id>.<format>` (default `wav`, 16kHz mono — matches
  faster-whisper's expected input)
- Skips an ID if **any** audio file already exists for it in
  `data/raw/audio/` (any extension) or is recorded in
  `data/raw/audio_extract_history.csv` — so a video whose audio was already
  fetched directly via `--audio-only` is never redundantly re-extracted
- Supports `--dry-run`, `--input-dir`, `--output-dir`, `--format`
  (`wav`/`flac`/`mp3`), `--log-level`, `--config` overrides
- Raises immediately (before scanning) if `ffmpeg` isn't on PATH

File map: `extract.py` (CLI) → `extractor.py` (`AudioExtractor` engine) →
`config.py` (`configs/audio.yaml` → `AudioConfig`) + `history.py`
(`ExtractHistory`, simpler CSV ledger — no mode dimension needed) +
`logging_setup.py` (self-contained copy, not shared with acquisition, to
keep stages independently runnable).

**Known gap**: there's no `--history-file` CLI override for either stage —
overriding `--output-dir` alone still writes to the *default* history file
location. Hit this during testing (a scratch/test run's history row landed
in the real `data/raw/*_history.csv` and had to be manually cleaned up).
Worth adding a `--history-file` override if ad-hoc testing against real
config paths becomes routine.

### What's built — Phase 3 (`src/transcription/`) — scaffolded 2026-08-03, NOT yet run on real data

CLI: `python -m src.transcription.transcribe [options]`

Written so the code could be committed on this (GPU-less) dev machine and
pulled onto a separate GPU box to actually run — see hardware note below.

- Scans `data/raw/audio/` for **any** audio file extension (webm/m4a/wav/
  flac/mp3 — audio-only downloads and ffmpeg extractions don't share one
  format), transcribes each with faster-whisper
- The `WhisperModel` is loaded **lazily**, only on the first real
  transcription — not at construction — so `--dry-run` works without the
  model weights downloaded or a GPU present. Verified: `--dry-run` correctly
  listed all 199 files in `data/raw/audio/` on this CPU-only laptop with
  `faster-whisper` not even installed.
- Per-file output: `data/transcripts/<id>.json` — language, duration,
  segment list (id/start/end/text), and word-level timestamps (needed for
  diarization alignment in stage 4)
- Skips an ID if a transcript JSON already exists for it or is recorded in
  `data/transcripts/transcription_history.csv`
- Supports `--dry-run`, `--input-dir`, `--output-dir`, `--history-file`
  (added proactively here — Phase 1/2 are missing this override, see known
  gap above), `--model-size`, `--device`, `--compute-type`, `--language`,
  `--log-level`, `--config` overrides

File map: `transcribe.py` (CLI) → `transcriber.py` (`Transcriber` engine) →
`config.py` (`configs/transcription.yaml` → `TranscriptionConfig`) +
`history.py` (`TranscriptionHistory`) + `logging_setup.py` (self-contained
copy, same convention as audio).

**Model choice**: `configs/transcription.yaml` uses faster-whisper
`large-v3`. Chosen over smaller/distilled models because transcript fidelity
directly feeds fine-tuning and RAG quality — decided explicitly with the
user rather than defaulting to something lighter for speed.

**Updated 2026-08-04** (on the T1000 GPU machine, see hardware note below):
`device` changed from `auto` → `cuda`, and `compute_type` changed from
`default` → `int8_float16`.
- `device: cuda` (not `auto`) so a broken CUDA/cuDNN setup fails loudly at
  model load instead of silently falling back to CPU (which would make a
  `large-v3` run take days without any obvious error).
- `compute_type: int8_float16` because `large-v3` in `float16` needs ~4.7GB
  VRAM, which does not fit the T1000's 4GB. `int8_float16` needs roughly
  3.0–3.2GB, leaving headroom for the VAD filter and word-timestamp
  alignment. Revisit both settings if this pipeline ends up running on
  different/bigger GPU hardware (see hardware discrepancy note above).

**Hardware note**: this was scaffolded on a Dell Latitude 7490 (16GB RAM,
Intel UHD 620 integrated graphics, no CUDA) — not viable for running
`large-v3` at any real scale (CPU-only, rough estimate well under real-time
throughput; 199 files at Sadhguru-talk lengths could mean days of runtime).
User is planning to run the real batch on a separate machine (128GB RAM,
2TB SSD, RTX A4000 16GB VRAM) — confirmed sufficient for `large-v3`
transcription, pyannote diarization (stage 4), and BGE-M3 embeddings
(stage 7). That machine's 16GB VRAM will be the binding constraint later,
at fine-tuning (stage 9) and serving (stage 11): comfortable for a 7B-class
base model via QLoRA, tighter for 13B, not realistic beyond that without
multi-GPU or heavier quantization. Base model choice for fine-tuning/serving
is still undecided — `configs/finetuning.yaml` and `configs/serving.yaml`
are still empty stubs.

**Phase 3 is now complete** (2026-08-06) — all 199 files in `data/raw/audio/`
have a transcript in `data/transcripts/`. See "Phase 3 completion" section
below for the full story (Colab T4 run + local T1000 run, split due to a
Colab quota cutoff) and the real RTF numbers measured on both.

### What's built — Phase 4 (`src/diarization/`) — scaffolded 2026-08-06, verified on 1 file, full batch NOT yet run

CLI: `python -m src.diarization.diarize [options]`

Follows the same CLI/config/history/logging pattern as Phase 3. Reads each
`data/raw/audio/<id>.<ext>` plus its matching `data/transcripts/<id>.json`
(needs the word-level timestamps for alignment), runs
`pyannote/speaker-diarization-3.1`, then isolates the target speaker's words
into a new segment list.

- **Target-speaker identification (v1)**: whichever diarized speaker label
  has the most total speaking time in a file is assumed to be the target
  speaker (Sadhguru) — `target_speaker.strategy: longest_total_duration` in
  `configs/diarization.yaml`. Only strategy implemented so far; flagged in
  the config as worth revisiting with a voice-embedding-match strategy if a
  file turns up where an interviewer/host talks more than Sadhguru does.
- Word-to-speaker assignment: a word counts as the target speaker's if its
  timestamp midpoint falls inside one of that speaker's turns; consecutive
  kept words are regrouped into new segments, splitting wherever a
  non-target word breaks continuity (`_regroup_target_segments` in
  `diarizer.py`).
- The pyannote `Pipeline` is loaded **lazily** (same convention as
  faster-whisper in Phase 3) — `--dry-run` works without `HF_TOKEN`, a GPU,
  or model weights present.
- Per-file output: `data/diarized/<id>.json` — per-speaker total speaking
  time/turn count, the identified `target_speaker` label, all raw turns, the
  isolated `target_speaker_segments`, and the convenience field
  `target_speaker_text` (isolated segments joined into one string).
- Skips an ID if a diarized JSON already exists for it or is recorded in
  `data/diarized/diarization_history.csv`; skips (with a warning, counted
  separately as `missing_transcript`) any audio file that has no matching
  Phase 3 transcript yet.
- Supports `--dry-run`, `--audio-dir`, `--transcripts-dir`, `--output-dir`,
  `--history-file`, `--device`, `--log-level`, `--config` overrides.

**Audio decoding deliberately bypasses pyannote's default loader**: pyannote
normally decodes audio via `torchcodec`, but on the T1000 machine
`torchcodec`'s bundled native libraries failed to load regardless of the
FFmpeg version installed — apparently a mismatch with the
`torch==2.9.1`+CUDA build pinned there (itself pinned to match the last
Windows CUDA `torchaudio` release). Worked around by decoding via a direct
`ffmpeg` subprocess call + the stdlib `wave` module instead, then handing
pyannote a preloaded waveform dict. `ffmpeg` must be on `PATH`, same
requirement as Phase 2's extractor. **Not yet verified whether this
workaround is even needed on Linux/Colab** — the torchcodec failure was
Windows-specific; the ffmpeg+wave path should work regardless of OS since it
never touches torchcodec, but this hasn't actually been run on Colab yet.

**Verified 2026-08-06 (on the T1000)**: ran end-to-end on one real file —
correctly excluded the interviewer's question and isolated the target
speaker's (Sadhguru's) answer. Batch run against all 199 files was
intentionally paused there and never done.

**Re-verified 2026-08-12 (on the Dell laptop, file-discovery only)**: a
fresh `--dry-run` (after installing just `python-dotenv`, the one
lightweight import `diarize.py` needs even for `--dry-run`) reports
`succeeded=0 failed=0 skipped=0 missing_transcript=0 would_diarize=199` —
confirms every one of the 199 audio files still has a matching transcript
and none are diarized yet on this machine. `data/diarized/` is currently
empty here (the one verified file from 2026-08-06 lived only on the T1000;
gitignored, so it never traveled anywhere).

**Not yet done for the real batch run**:
- No `.env` exists yet anywhere confirmed — `HF_TOKEN` needs to be set, and
  the HF account it belongs to needs to have accepted the gated model terms
  for `pyannote/speaker-diarization-3.1` (and its dependency
  `pyannote/segmentation-3.0`) on huggingface.co. Not yet confirmed done.
- `torch` + `pyannote.audio` are not installed anywhere but the T1000 (by
  design — same "scaffold on the CPU laptop, install heavy deps only on the
  machine that'll run it" pattern as Phase 3).
- **Decided 2026-08-12: Colab** (matches the now-stated default heavy-compute
  plan). Added `notebooks/colab_diarization.ipynb`, mirroring
  `colab_transcription.ipynb`'s structure (confirm GPU → clone repo → mount
  Drive → configure paths → install deps → dry run → single-file RTF
  benchmark + corpus-time extrapolation → full batch run → sanity check),
  plus diarization-specific steps: gated-model prerequisites (accept terms
  for `pyannote/speaker-diarization-3.1` **and** its dependency
  `pyannote/segmentation-3.0` on huggingface.co, since only the top-level
  pipeline is obviously gated but the segmentation model underneath is
  gated too), an `HF_TOKEN` cell that reads it from a Colab secret
  (`google.colab.userdata`) with a hidden-`getpass` fallback so the token is
  never pasted in plaintext, and an explicit pipeline-load sanity check
  before the dry run so a bad token/un-accepted terms fails fast with a
  clear message instead of partway through the batch. Also proactively
  passes `--history-file` into the Drive-mounted output dir on the full-run
  cell — Phase 3's notebook skipped this on its full-run cell and lost the
  history CSV to the ephemeral Colab VM disk on disconnect (harmless there
  since file-existence is also checked, but no reason to repeat it here).
  **Not yet run** — next session should open it in Colab and work through it.

### GPU machine session log (2026-08-04)

Repo was cloned onto a Windows 11 machine with an **NVIDIA T1000 (4GB
VRAM)** to actually run Phase 3. Progress this session:

- Confirmed via `nvidia-smi` the GPU is idle and visible: driver 595.95,
  CUDA 13.2, compute capability 7.5.
- **Python was not installed on this machine at all** (no `python`/`py`/
  `pip`/`conda` on PATH, no venv). Installed Python 3.11.9 per-user (no
  admin/UAC available in this shell — `winget`'s default install path tries
  to write a machine-wide launcher to `C:\ProgramData\Package Cache` and
  fails with access denied at user scope too). Worked around it by running
  the cached python.org installer directly with
  `/quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0`, targeting
  `%LOCALAPPDATA%\Programs\Python\Python311`. Confirmed the per-user `PATH`
  registry entry persists for new shells.
- Created a venv at `.venv/` (already gitignored). Installed **only**
  Phase 1–3 deps (`pyyaml`, `tqdm`, `ffmpeg-python`, `yt-dlp`,
  `faster-whisper` — which pulls in `ctranslate2`) rather than the full
  `requirements.txt`. **`requirements.txt` as it stands will not fully
  install on this Windows machine** — `vllm` (stage 11) is Linux-only, and
  `bitsandbytes`/`pyannote.audio`/etc. are for stages not needed yet. Install
  those incrementally, stage by stage, rather than all at once.
- `ctranslate2.get_cuda_device_count()` returns `1` — the T1000 is correctly
  visible to the library faster-whisper actually uses for GPU inference.
  Note: `faster-whisper`/`ctranslate2` does **not** depend on `torch` at
  all — no need to fight PyTorch CUDA-build/wheel matching for this stage.
- Updated `configs/transcription.yaml` (`device: cuda`,
  `compute_type: int8_float16`) — see Phase 3 section above for why.

**Blocking issue found**: `data/raw/` (the 199 downloaded audio files +
metadata + history CSVs from Phase 1) is gitignored, so it never came over
with `git clone`. It only exists on the original Dell laptop where
acquisition ran. **This machine's `data/` only has the empty scaffold
subdirectories** — there is nothing to transcribe here yet. User is
manually transferring `data/raw/` from the laptop to this machine
(USB/network share/cloud — outside git). No real transcription benchmark or
full run has happened yet as a result.

### Phase 3 completion (2026-08-05 – 2026-08-06)

`data/raw/` (199 audio files + metadata) was transferred from the Dell
laptop to the T1000 machine manually via a Google Drive zip download (not
git — `data/raw/` is still gitignored). Landed as
`audio-20260805T104651Z-1-001.zip` in `Downloads`, unzipped into
`data/raw/audio/`.

**T1000 blocker found and fixed — missing CUDA runtime, not just a slow
GPU**: the first real (non-benchmark) transcription attempt on the T1000
failed with `Library cublas64_12.dll is not found or cannot be loaded`, even
though `nvidia-smi` and `ctranslate2.get_cuda_device_count()` both showed
the GPU as available. Root cause: this machine has the NVIDIA *display
driver* but never got the CUDA Toolkit (or its redistributable runtime
libraries) installed — `nvidia-smi` only proves the driver is there, not
that cuBLAS/cuDNN exist. Since there's no admin/UAC access on this machine
(same constraint hit installing Python originally), a normal CUDA Toolkit
installer wasn't an option. Fixed instead with the pip-installable runtime
packages, which need no admin rights:
```
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```
These land under `.venv/Lib/site-packages/nvidia/{cublas,cudnn,cuda_nvrtc}/bin/`.
Those directories have to be prepended to `PATH` *before* invoking the
transcribe CLI — they are not picked up automatically, and this is not
persisted anywhere, so **every fresh shell/session on this machine needs**:
```
export PATH="$(pwd)/.venv/Lib/site-packages/nvidia/cublas/bin:$(pwd)/.venv/Lib/site-packages/nvidia/cudnn/bin:$(pwd)/.venv/Lib/site-packages/nvidia/cuda_nvrtc/bin:$PATH"
```
before running `python -m src.transcription.transcribe` with `--device cuda`.
Worth turning into a small activation script or a note in the venv's
activate script if this machine keeps getting used — currently a manual,
easy-to-forget step.

**Real RTF measured** (single ~4:43 / 283s audio file, both after the model
was already cached locally):
- **T1000**, `compute_type: int8_float16`: 70s processing time → RTF ≈
  0.247, ~4.0x real-time.
- **Colab T4**, `compute_type: float16` (via `notebooks/colab_transcription.ipynb`):
  RTF = 0.227, ~4.4x real-time — only ~9% faster than the T1000, much closer
  than expected. Likely explanation: the T1000 and T4 are both Turing
  architecture (compute capability 7.5) with native INT8 tensor-core
  support, so `int8_float16` on the T1000 plays directly to that shared
  architectural strength, while the T4 running plain `float16` isn't
  exploiting its extra compute the same way. A same-compute-type comparison
  (both on `int8_float16`) would likely show a bigger T4 lead.

**Colab run**: uploaded `data/raw/audio/` to Google Drive, ran
`notebooks/colab_transcription.ipynb` against a T4. Got through **155 of
199 files** before the Colab session disconnected and the account's free
daily GPU quota was cut — no more Colab GPU time available that day. The
155 completed transcripts were downloaded from Drive back down to this
machine into `data/transcripts/`.

**Local T1000 finished the rest**: the transcription pipeline's built-in
skip-duplicate logic (`Transcriber._already_transcribed` in
`src/transcription/transcriber.py` — checks both the history CSV *and*
whether `data/transcripts/<id>.json` already exists on disk) meant no code
changes were needed to resume locally without redoing Colab's work. Colab's
own history CSV was lost (it defaulted to the Colab VM's ephemeral disk,
not the Drive-mounted output dir — a gap in the current notebook, since only
the benchmark cell passes `--history-file`, not the full-run cell), but that
didn't matter: the file-existence check alone was sufficient. A `--dry-run`
correctly reported `skipped=155, would_transcribe=44, failed=0` before the
real run. Ran:
```
python -m src.transcription.transcribe --device cuda --compute-type int8_float16
```
(with the cuBLAS/cuDNN `PATH` fix above applied first). **Result: 44/44
succeeded, 0 failed**, wall clock 1:54:29 for that batch. Final state:
`data/transcripts/` has all 199 `.json` files;
`transcription_history.csv` has the 44 rows from this local run (the 155
Colab-completed ones were never re-recorded there, since they were skipped
via the file-existence path, not the history path — harmless, since nothing
downstream reads the history CSV as a source of truth, only as a skip
optimization).

**Known gap to fix if this notebook is reused**: add `--history-file`
pointing into the Drive-mounted output dir to the full-run cell in
`notebooks/colab_transcription.ipynb`, matching what the benchmark cell
already does — otherwise every Colab session loses its history CSV to the
ephemeral VM disk on disconnect (harmless so far since file-existence is
also checked, but wasteful if `data/transcripts/` itself isn't Drive-backed
in some future run).

### Dev environment switch (2026-08-12)

Back to working on the Dell Latitude 7490 (16GB RAM, 512GB SSD, no CUDA) for
day-to-day/CPU-only work; **Colab** is the plan for heavy/GPU code going
forward. The **T1000 (4GB VRAM) machine is still available on request** —
if a future task needs a real GPU run, ask before assuming Colab is the only
option, since the user can switch back to the T1000 machine.

Also generated `data/transcripts_combined.txt` (gitignored, not tracked): a
plain-text concatenation of just the top-level `"text"` field from all 199
files in `data/transcripts/*.json` (segments/words/timestamps/metadata
dropped), transcripts separated by a blank line, in filename-sorted order.
This is an ad-hoc utility artifact for quickly reading/reviewing the raw
transcript content — it is **not** a pipeline stage output; stage 4
(diarization) and stage 5 (cleaning) are still the real next steps to
produce the actual per-speaker, cleaned transcripts the pipeline needs.

### Git state as of last check (2026-08-12)

On `main` (renamed from `master` on 2026-08-01 — see below), up to date with
`origin/main` as of the last push. Latest commits, newest first:
- `9607e5b Scaffold Phase 4 diarization and add transcript quality
  evaluation` — adds `src/diarization/` (see "What's built — Phase 4"
  above), plus reference-free transcript quality auditing
  (`src/transcription/quality.py` + `evaluate.py`) and real ASR
  accuracy scoring against YouTube's manual subtitles where available
  (`eval/asr/`) — 122/199 files had manual subtitles, mean WER 7.2%.
- `b20afcb Add executed Colab notebook as a run record for Phase 3` —
  commits `notebooks/colab_transcription_v2.ipynb` (the executed copy with
  cell outputs, previously untracked — see below).
- `c31bf7f Mark Phase 3 (transcription) complete in PROJECT_UPDATES.md`.
- `ebb2a4c Add Colab T4 notebook for Phase 3 transcription` — adds
  `notebooks/colab_transcription.ipynb` (see "Phase 3 completion" above).
- `51182a7 Set up GPU machine for Phase 3, tune transcription config for
  T1000 VRAM` — commits the `device: cuda` / `compute_type: int8_float16`
  change to `configs/transcription.yaml` (this was the previously-uncommitted
  change noted in earlier versions of this doc — now committed).
- `4ffaff9 Scaffold Phase 3 transcription stage (faster-whisper)` — adds
  `src/transcription/`.
- `9bf96de Harden Phase 1 acquisition and implement Phase 2 audio extraction`
  — Phase 1 hardening (mode-aware dedup, history migration, audio-only
  quality selector) and the full Phase 2 implementation.

`data/raw/` and `data/transcripts/` are both gitignored, so none of the
downloaded media/metadata/history/transcript files are tracked in git —
only source code, configs, and the notebook.

`notebooks/colab_transcription_v2.ipynb` (the executed copy downloaded back
from Colab, with cell outputs embedded) is now committed too (`b20afcb`) —
kept alongside the original as a run record rather than merged into it.

This session's changes (this file, `.gitignore`'s
`data/transcripts_combined.txt` entry, and `notebooks/colab_diarization.ipynb`
— see "What's built — Phase 4" above) are committed and pushed as `54c4f54`,
merged with 2 commits this laptop hadn't pulled yet (`ac5f196` PDF export of
the Phase 3 eval report, `e48bbcd` adds `src/transcription/export_text.py` —
a per-file plain-text transcript exporter to `data/transcripts_txt/<id>.txt`,
complementary to but distinct from this session's single-file
`data/transcripts_combined.txt`) via merge commit `f0442d0`. Working tree is
clean, `origin/main` is up to date as of this push.

**Branch rename (2026-08-01)**: the GitHub repo
(`DAG-21/PureBillion-Cloner`) had two unrelated branches — `main`
(a single disconnected "Initial commit", GitHub's default HEAD branch) and
`master` (the actual project history, what local work was based on).
Local `master` was renamed to `main` and force-pushed, replacing the
throwaway initial commit so `origin/main` now has the full real history and
matches what GitHub already treated as default. `origin/master` still
exists on GitHub with the old (pre-force-push) history and was **not**
deleted — it's an orphaned leftover, safe to delete whenever, just hasn't
been asked for yet.

**Test suite is currently broken and intentionally left that way**:
`tests/acquisition/test_download.py`'s `make_config()` fixture builds
`OutputConfig(...)` without the `audio_dir` field added during Phase 1
hardening, so every test in that file fails at construction
(`TypeError: missing required argument 'audio_dir'`). User explicitly said
not to maintain the test suite for now ("Make the changes in the main
file") — don't fix this without being asked.

### Real data collected so far

**Note**: this was originally acquired on the Dell laptop; as of 2026-08-06,
`data/raw/audio/` (199 files) is also present on the T1000 machine (manually
transferred via a Google Drive zip — see "Phase 3 completion" above), along
with `data/transcripts/` (199 transcript JSONs, now fully populated). Both
directories are still gitignored, so this is only guaranteed to exist on
whichever machine(s) it's been manually copied to, not automatically synced
anywhere — confirm before assuming availability on a new/different machine.

- 1 video+audio download: `lNw1Ts_vO9A.mp4` (~15MB) in `data/raw/videos/`
- 199 audio-only downloads in `data/raw/audio/`, across 6 playlists/videos:
  - 1 standalone test video (`lNw1Ts_vO9A`)
  - "Sadhguru on Mysticism & Occult" playlist (`PLU4wqwok6puw`) — 15 videos
  - An anxiety/stress/sleep playlist (`PL3uDtbb3OvDOJlFBIo5JfBYYth9S1WT6W`)
    — 7 videos
  - A relationships/love playlist (`PLbStFAipRy-A`) — 33 videos
  - "Sadhguru on Food & Health" playlist (`PLPEQGwS9Hkso`) — 13 videos
    (downloaded 2026-08-03)
  - "Sadhguru on Mental Health & Mind" playlist (`PLDj3qLjlJpYY`) — 15 videos
    (downloaded 2026-08-03)
  - "Thoughts and the Mind" playlist (`PL3uDtbb3OvDPBGzSYKBeEFlrG48_0DBC4`)
    — 127 videos total; 115 succeeded, 8 already-downloaded skipped, 4
    permanently failed (private/access-restricted videos:
    `xSFtKjdhBTk`, `eBAVAro8t2g`, `q1lfjC1eH9Y`, `YVLxIhCHhpg` — not
    recoverable via retry) (downloaded 2026-08-03)
- All successfully-downloaded audio files have a matching metadata JSON in
  `data/raw/metadata/`
- No ffmpeg-based extraction has run for real yet (the one real video
  already has audio available directly, so extraction correctly skips it
  — verified via a scratch-directory test, not against real data)

## What's NOT done yet

- Test suite is stale/broken (see above) — not being maintained right now
  per explicit user instruction
- No transcript/caption fetching — Phase 1 is video+metadata only
- Stages 5–11 (cleaning through serving) are unimplemented stubs. Phase 4
  (diarization) is code-complete and verified on 1 file (see "What's built —
  Phase 4" above) — the full 199-file batch run is what's actually left.
- No `.env` filled in yet (only `.env.example` exists)
- Target public figure so far is Sadhguru (based on videos downloaded), but
  this hasn't been explicitly confirmed as *the* project target — worth
  double-checking before scaling up acquisition
- `notebooks/colab_transcription.ipynb`'s full-run cell doesn't pass
  `--history-file` (unlike its benchmark cell) — see "Known gap" note in
  Phase 3 completion above.

## Likely next steps

1. **Run the Phase 4 (diarization) full batch on Colab**: code is done and
   verified on 1 file, `notebooks/colab_diarization.ipynb` is written (see
   "What's built — Phase 4" above) but not yet executed. Needs: an HF
   account with the gated model terms accepted (2 separate model pages, see
   the notebook's prerequisites), a read-scoped HF access token added as a
   Colab secret, and `data/transcripts/` (the full 199, not the
   155-before-the-quota-cut partial set) uploaded to Drive alongside
   `data/raw/audio/`. Then: dry run → single-file benchmark → decide if the
   estimated full-batch time fits one Colab session → full run → sync
   `data/diarized/` back down.
2. **Resolve the hardware discrepancy** (see flag near the top): is the
   T1000 (4GB VRAM) the real long-term GPU machine, or is there still a
   separate 128GB RAM / A4000 (16GB VRAM) machine this should run on
   instead? This materially changes what's feasible for stages 4/7/9/11 —
   Phase 3 working on the T1000 doesn't mean the heavier later stages will.
3. Run a real ffmpeg extraction against a video that doesn't already have
   audio downloaded separately, to validate Phase 2 end-to-end on real
   (non-scratch) data — currently only verified via a scratch-directory test.
4. Confirm the target public figure / source channel(s) before acquiring
   more data at scale (everything downloaded so far is Sadhguru content).
5. Decide whether to delete the orphaned `origin/master` branch on GitHub.
6. Decide on a base model for fine-tuning/serving (stages 9 & 11) — nothing
   picked yet, and now depends on resolving the hardware question above:
   16GB VRAM (A4000) comfortably fits a 7B-class model via QLoRA, tighter
   for 13B; 4GB (T1000) would not be viable for fine-tuning or serving any
   realistic base model at all.
7. Fix the `--history-file` gap in `notebooks/colab_transcription.ipynb`'s
   full-run cell (see "What's NOT done yet" above) if that notebook gets
   reused — `notebooks/colab_diarization.ipynb` already does this correctly.

---
*Update this file whenever a phase is completed, the architecture changes,
or there's context a future session would need but can't get from reading
the code alone (e.g. decisions, blockers, why something was done a certain
way).*
