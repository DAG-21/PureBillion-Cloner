# Project Updates

> Living status doc. Read this first when resuming work in a new session.
> Last updated: 2026-07-31

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

## Current status: Phase 1 (acquisition) + Phase 2 (audio extraction) — implemented, uncommitted

Stages 3–11 are still empty scaffold stubs (`src/<stage>/*.py` are ~6-line
placeholder files). Real work so far is in `src/acquisition/` and
`src/audio/`.

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

### Git state as of last check

On `master`. Working tree has **uncommitted changes**:

- Modified: `configs/acquisition.yaml`, `configs/audio.yaml`,
  `src/acquisition/config.py`, `downloader.py`, `history.py`,
  `src/audio/extract.py`
- New/untracked: `src/audio/config.py`, `extractor.py`, `history.py`,
  `logging_setup.py`, `tests/acquisition/` (has a `test_download.py` written
  earlier, now **stale** — see below)
- Deleted: `tests/.gitkeep`

**Not yet committed** — first thing to check on resume is whether this
should be committed, and if so with what message.

**Test suite is currently broken and intentionally left that way**:
`tests/acquisition/test_download.py`'s `make_config()` fixture builds
`OutputConfig(...)` without the `audio_dir` field added during Phase 1
hardening, so every test in that file fails at construction
(`TypeError: missing required argument 'audio_dir'`). User explicitly said
not to maintain the test suite for now ("Make the changes in the main
file") — don't fix this without being asked.

### Real data collected so far

- 1 video+audio download: `lNw1Ts_vO9A.mp4` (~15MB)
- 16 audio-only downloads in `data/raw/audio/`: the same video plus the
  full 15-video "Sadhguru on Mysticism & Occult" playlist
  (`PLU4wqwok6puw`)
- No ffmpeg-based extraction has run for real yet (the one real video
  already has audio available directly, so extraction correctly skips it
  — verified via a scratch-directory test, not against real data)

## What's NOT done yet

- Test suite is stale/broken (see above) — not being maintained right now
  per explicit user instruction
- No transcript/caption fetching — Phase 1 is video+metadata only
- Stages 3–11 (transcription through serving) are unimplemented stubs
- No `.env` filled in yet (only `.env.example` exists)
- Target public figure so far is Sadhguru (based on videos downloaded), but
  this hasn't been explicitly confirmed as *the* project target — worth
  double-checking before scaling up acquisition

## Likely next steps

1. Decide whether to commit the Phase 1 + Phase 2 work as-is.
2. Run a real ffmpeg extraction against a video that doesn't already have
   audio downloaded separately, to validate Phase 2 end-to-end on real
   (non-scratch) data.
3. Move on to Phase 3 (transcription via faster-whisper) —
   `src/transcription/transcribe.py` is currently a stub.
4. Confirm the target public figure / source channel(s) before acquiring
   more data at scale.

---
*Update this file whenever a phase is completed, the architecture changes,
or there's context a future session would need but can't get from reading
the code alone (e.g. decisions, blockers, why something was done a certain
way).*
