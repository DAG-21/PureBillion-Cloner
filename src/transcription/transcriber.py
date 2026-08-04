"""Core faster-whisper wrapper: scans raw audio files and transcribes each one.

The whisper model itself is loaded lazily, on the first audio file actually
transcribed -- not at construction time. That keeps ``--dry-run`` (and any
config/CLI validation) usable on a machine that doesn't have the model
weights downloaded or a GPU available, since only a real transcription run
needs the model loaded into memory.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.transcription.config import TranscriptionConfig
from src.transcription.history import TranscriptionHistory, TranscriptionRecord, now_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranscriptionSummary:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    would_transcribe: int = 0

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped + self.would_transcribe


class Transcriber:
    """Transcribes every raw audio file not already transcribed, via faster-whisper."""

    def __init__(self, config: TranscriptionConfig) -> None:
        self.config = config
        self.config.output.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.history = TranscriptionHistory(config.output.history_file)
        self._model: Any = None  # lazily-loaded faster_whisper.WhisperModel

    # -- discovery ------------------------------------------------------

    def find_audio_files(self) -> List[Path]:
        """Return every file directly under ``audio_dir`` (any extension -- audio-only
        downloads and ffmpeg extractions land in a mix of webm/m4a/wav/flac/mp3)."""
        audio_dir = self.config.input.audio_dir
        if not audio_dir.exists():
            return []
        return sorted(p for p in audio_dir.iterdir() if p.is_file())

    def _output_path(self, video_id: str) -> Path:
        return self.config.output.transcripts_dir / f"{video_id}.json"

    def _already_transcribed(self, video_id: str) -> bool:
        if self.history.is_transcribed(video_id):
            return True
        return self._output_path(video_id).exists()

    # -- model ------------------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel  # deferred: heavy import, needs no GPU for dry-run

            model_cfg = self.config.model
            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)...",
                model_cfg.size,
                model_cfg.device,
                model_cfg.compute_type,
            )
            self._model = WhisperModel(
                model_cfg.size,
                device=model_cfg.device,
                compute_type=model_cfg.compute_type,
            )
        return self._model

    # -- transcription ------------------------------------------------------

    def _transcribe_one(self, audio_path: Path) -> Dict[str, Any]:
        model = self._load_model()
        model_cfg = self.config.model

        segments_iter, info = model.transcribe(
            str(audio_path),
            language=model_cfg.language,
            beam_size=model_cfg.beam_size,
            vad_filter=model_cfg.vad_filter,
            word_timestamps=model_cfg.word_timestamps,
        )

        segments: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        for segment in segments_iter:
            words: Optional[List[Dict[str, Any]]] = None
            if segment.words:
                words = [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in segment.words
                ]
            segments.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": words,
                }
            )
            text_parts.append(segment.text.strip())

        return {
            "video_id": audio_path.stem,
            "source_audio": str(audio_path),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration,
            "model": {
                "size": model_cfg.size,
                "device": model_cfg.device,
                "compute_type": model_cfg.compute_type,
            },
            "segments": segments,
            "text": " ".join(text_parts),
            "transcribed_at": now_iso(),
        }

    def transcribe_all(self, dry_run: bool = False) -> TranscriptionSummary:
        """Transcribe every audio file not already transcribed."""
        audio_files = self.find_audio_files()
        logger.info("Found %d audio file(s) in %s", len(audio_files), self.config.input.audio_dir)

        summary = TranscriptionSummary()
        for audio_path in tqdm(audio_files, desc="Audio files", unit="file"):
            video_id = audio_path.stem

            if self._already_transcribed(video_id):
                logger.info("Skipping already-transcribed audio: %s", video_id)
                summary.skipped += 1
                continue

            if dry_run:
                logger.info("[dry-run] Would transcribe: %s (%s)", video_id, audio_path.name)
                summary.would_transcribe += 1
                continue

            output_path = self._output_path(video_id)
            try:
                transcript = self._transcribe_one(audio_path)
                output_path.write_text(
                    json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self.history.record(
                    TranscriptionRecord(
                        video_id=video_id,
                        source_path=str(audio_path),
                        status="success",
                        output_path=str(output_path),
                        timestamp=now_iso(),
                    )
                )
                summary.succeeded += 1
                logger.info("Transcribed %s -> %s", audio_path.name, output_path)
            except Exception as exc:
                logger.error("Failed to transcribe %s: %s", audio_path.name, exc)
                self.history.record(
                    TranscriptionRecord(
                        video_id=video_id,
                        source_path=str(audio_path),
                        status="failed",
                        error=str(exc),
                        timestamp=now_iso(),
                    )
                )
                summary.failed += 1

        return summary
