# Project Updates

> Living status doc. Read this first when resuming work in a new session.
> Last updated: 2026-08-04 (GPU machine set up, Phase 3 real run still blocked on data transfer)

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

## Current status: Phase 1 (acquisition) + Phase 2 (audio extraction) done; Phase 3 (transcription) scaffolded, GPU machine now set up, real run still blocked on getting data onto that machine

Stages 4–11 are still empty scaffold stubs (`src/<stage>/*.py` are ~6-line
placeholder files). Real work so far is in `src/acquisition/`, `src/audio/`,
and (as of 2026-08-03) `src/transcription/`.

**⚠️ Hardware discrepancy, unresolved**: the GPU machine actually used on
2026-08-04 is an **NVIDIA T1000, 4GB VRAM** (Windows 11, driver 595.95, CUDA
13.2, compute capability 7.5) — confirmed via `nvidia-smi`. This does **not**
match the machine described below in the original Phase 3 hardware note
(128GB RAM / RTX A4000 16GB VRAM). Unclear whether the plan changed to this
T1000 box, or the A4000 machine is a separate/different machine still to be
used. **Confirm which machine is actually the long-term GPU box before
assuming stages 4 (diarization), 7 (embeddings), 9 (fine-tuning), 11
(serving) can run on whatever hardware is at hand** — a 4GB card is a much
tighter constraint than 16GB, especially for fine-tuning/serving.

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

**Not yet done for Phase 3**: no real transcription run has happened
anywhere (dry-run only, on the Dell laptop). Needs to be run for real on the
GPU machine before Phase 4 (diarization) can start — see session log below
for exactly what's blocking that.

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

**Next steps once `data/raw/` lands on this machine**:
1. Run one real audio file through `Transcriber` (`device: cuda` is now
   forced in config) and time it against `nvidia-smi -l 1` running in
   parallel, to confirm GPU utilization visually and get a real
   wall-clock/audio-duration ratio (RTF) on this exact T1000.
2. Sum the `duration` field across `data/raw/metadata/*.json` to get total
   corpus hours, then multiply by the measured RTF for a real full-199-file
   time estimate (no reliable estimate exists yet — T1000 is far weaker
   than the T4/A100-class hardware faster-whisper's published benchmarks
   use, so those numbers don't transfer directly).
3. Run the actual `python -m src.transcription.transcribe` batch job.

### Git state as of last check

On `main` (renamed from `master` on 2026-08-01 — see below). As of the
2026-08-04 GPU-machine session, working tree has one **uncommitted** change:
`configs/transcription.yaml` (`device`/`compute_type` update, see Phase 3
section above) — not yet committed, do that first in the next session if it
hasn't already been done. Otherwise up to date with `origin/main` (pushed
2026-08-04). Latest commit:
`4ffaff9 Scaffold Phase 3 transcription stage (faster-whisper)` — adds
`src/transcription/` (see Phase 3 section above). Prior commit
`9bf96de Harden Phase 1 acquisition and implement Phase 2 audio extraction`
covers all the Phase 1 hardening (mode-aware dedup, history migration,
audio-only quality selector) and the full Phase 2 implementation.
`data/raw/` is gitignored, so none of the downloaded media/metadata/history
CSVs are tracked in git — only source code and configs.

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

**Note**: all of this exists only on the original Dell laptop as of
2026-08-04 (`data/raw/` is gitignored, never synced via git). User is
manually transferring it to the T1000 GPU machine — see session log above.
Confirm it actually landed there before assuming any of this is available
for the next transcription run.

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
- Stages 4–11 (diarization through serving) are unimplemented stubs; Phase 3
  (transcription) is scaffolded but has never been run for real (see above)
- No `.env` filled in yet (only `.env.example` exists)
- Target public figure so far is Sadhguru (based on videos downloaded), but
  this hasn't been explicitly confirmed as *the* project target — worth
  double-checking before scaling up acquisition

## Likely next steps

1. **Resolve the hardware discrepancy** (see flag near the top): is the
   T1000 (4GB VRAM) the real long-term GPU machine, or is there still a
   separate 128GB RAM / A4000 (16GB VRAM) machine this should run on
   instead? This materially changes what's feasible for stages 4/7/9/11.
2. Confirm `data/raw/` finished transferring onto the T1000 machine, then
   run the Phase 3 benchmark (single file → RTF → extrapolate) and the real
   `python -m src.transcription.transcribe` batch — see GPU machine session
   log above for the exact plan. This is the first real (non-dry-run) use
   of Phase 3.
3. Commit the pending `configs/transcription.yaml` change (see Git state
   above) once back in a session with git access on that machine.
4. Run a real ffmpeg extraction against a video that doesn't already have
   audio downloaded separately, to validate Phase 2 end-to-end on real
   (non-scratch) data — currently only verified via a scratch-directory test.
5. Confirm the target public figure / source channel(s) before acquiring
   more data at scale (everything downloaded so far is Sadhguru content).
6. Decide whether to delete the orphaned `origin/master` branch on GitHub.
7. Decide on a base model for fine-tuning/serving (stages 9 & 11) — nothing
   picked yet, and now depends on resolving the hardware question above:
   16GB VRAM (A4000) comfortably fits a 7B-class model via QLoRA, tighter
   for 13B; 4GB (T1000) would not be viable for fine-tuning or serving any
   realistic base model at all.

---
*Update this file whenever a phase is completed, the architecture changes,
or there's context a future session would need but can't get from reading
the code alone (e.g. decisions, blockers, why something was done a certain
way).*
