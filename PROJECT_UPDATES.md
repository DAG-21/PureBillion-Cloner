# Project Updates

> Living status doc. Read this first when resuming work in a new session.
> Last updated: 2026-08-03 (Phase 3 scaffolded)

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

## Current status: Phase 1 (acquisition) + Phase 2 (audio extraction) done; Phase 3 (transcription) scaffolded, not yet run for real

Stages 4–11 are still empty scaffold stubs (`src/<stage>/*.py` are ~6-line
placeholder files). Real work so far is in `src/acquisition/`, `src/audio/`,
and (as of 2026-08-03) `src/transcription/`.

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

**Model choice**: `configs/transcription.yaml` defaults to faster-whisper
`large-v3`, `device: auto`, `compute_type: default` (ctranslate2 picks the
best type for whatever device it resolves to). Chosen over smaller/distilled
models because transcript fidelity directly feeds fine-tuning and RAG
quality — decided explicitly with the user rather than defaulting to
something lighter for speed.

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
anywhere (dry-run only, on this machine). Needs to be pulled onto the GPU
machine and run for real before Phase 4 (diarization) can start.

### Git state as of last check

On `main` (renamed from `master` on 2026-08-01 — see below), working tree
clean, up to date with `origin/main`. Latest commit:
`9bf96de Harden Phase 1 acquisition and implement Phase 2 audio extraction`
— covers all the Phase 1 hardening (mode-aware dedup, history migration,
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

1. Pull `src/transcription/` onto the GPU machine (128GB RAM / A4000) and
   run `python -m src.transcription.transcribe` for real against all 199
   audio files — this is the first real (non-dry-run) use of Phase 3.
2. Run a real ffmpeg extraction against a video that doesn't already have
   audio downloaded separately, to validate Phase 2 end-to-end on real
   (non-scratch) data — currently only verified via a scratch-directory test.
3. Confirm the target public figure / source channel(s) before acquiring
   more data at scale (everything downloaded so far is Sadhguru content).
4. Decide whether to delete the orphaned `origin/master` branch on GitHub.
5. Decide on a base model for fine-tuning/serving (stages 9 & 11) — nothing
   picked yet; A4000's 16GB VRAM comfortably fits a 7B-class model via
   QLoRA, tighter for 13B, not realistic beyond that on a single card.

---
*Update this file whenever a phase is completed, the architecture changes,
or there's context a future session would need but can't get from reading
the code alone (e.g. decisions, blockers, why something was done a certain
way).*
