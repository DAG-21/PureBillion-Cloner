# Project Updates

> Living status doc. Read this first when resuming work in a new session.
> Last updated: 2026-07-28

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

## Current status: Phase 1 (acquisition) — implemented, uncommitted

Everything below Phase 1 (stages 2–11) is still an empty scaffold stub
(`src/<stage>/*.py` are ~6-line placeholder files). All real work so far is
in `src/acquisition/`.

### What's built

CLI: `python -m src.acquisition.download <url> [options]`

Given a single video, playlist, or channel URL, it:
- Flat-resolves the URL into a list of video entries via yt-dlp
- Downloads each video (video+audio, or `--audio-only`) at a configurable
  max quality, into `data/raw/videos/<id>.<ext>`
- Extracts and saves per-video metadata JSON to `data/raw/metadata/<id>.json`
  (title, description, channel, duration, view/like counts, resolution, etc.)
- Logs every attempt (success/failed) to `data/raw/download_history.csv`
- Skips videos already downloaded (checks history CSV + checks disk) —
  reruns are idempotent
- Retries failed downloads with exponential backoff
- Supports `--dry-run` to preview without downloading
- Supports `--max-items` (cap playlist/channel size), `--quality`,
  `--output-dir`, `--log-level`, `--config` overrides

### File map (`src/acquisition/`)

| File | Role |
|---|---|
| `download.py` | CLI entry point — arg parsing, config/logging setup, calls `VideoDownloader`. No download logic itself. |
| `downloader.py` | `VideoDownloader` class — the actual engine: URL resolution, yt-dlp options, retry/backoff loop, progress bars, wires metadata + history together. Importable independent of the CLI. |
| `config.py` | Loads `configs/acquisition.yaml` into typed dataclasses (`AcquisitionConfig`). |
| `metadata.py` | Extracts a stable field set from yt-dlp's info dict, saves JSON sidecar per video. |
| `history.py` | CSV-backed download ledger (`DownloadHistory`) — tracks completed video IDs so reruns skip them. |
| `logging_setup.py` | Console + optional file logging; quiets noisy yt-dlp/urllib3 loggers below DEBUG. |

Config: `configs/acquisition.yaml` — output dirs, download quality/format/
rate-limit/concurrency, retry policy (3 attempts, exponential backoff),
network settings (timeout, optional cookies file for age-restricted videos),
playlist handling (ignore-errors, optional max-items cap), logging level/file.

### Git state as of last check

On `master`, 1 commit ahead of nothing (`341b56a Scaffold persona-clone
project structure`). Working tree has **uncommitted changes** implementing
Phase 1:

- Modified: `.gitignore`, `configs/acquisition.yaml`, `requirements.txt`,
  `src/acquisition/download.py`
- New/untracked: `src/acquisition/config.py`, `downloader.py`, `history.py`,
  `logging_setup.py`, `metadata.py`

**Not yet committed** — first thing to check on resume is whether this
should be committed, and if so with what message.

## What's NOT done yet

- No tests for `src/acquisition/` (or anywhere — `tests/` dir exists but
  appears empty/unpopulated)
- No transcript/caption fetching — Phase 1 is video+metadata only
- Stages 2–11 (audio extraction through serving) are unimplemented stubs
- No `.env` filled in yet (only `.env.example` exists)
- No actual data collected yet (need to confirm: has anyone run
  `download.py` against a real channel/playlist to seed `data/raw/`?)

## Likely next steps

1. Decide whether to commit the Phase 1 work as-is.
2. Either write tests for acquisition, or move on to Phase 2 (audio
   extraction via ffmpeg) — `src/audio/extract.py` is currently a stub.
3. Identify the actual target public figure / source channel(s) to run
   acquisition against for real.

---
*Update this file whenever a phase is completed, the architecture changes,
or there's context a future session would need but can't get from reading
the code alone (e.g. decisions, blockers, why something was done a certain
way).*
