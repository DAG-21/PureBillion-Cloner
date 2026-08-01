"""Core ffmpeg wrapper: scans raw video files and extracts their audio tracks.

Only video files under ``input.videos_dir`` are ever processed -- videos
that were never downloaded (e.g. because an audio-only download went
straight to ``output.audio_dir`` instead) simply aren't there to scan, so
they're skipped implicitly. A video whose audio has already been extracted
(or was fetched directly via ``--audio-only``) is skipped explicitly via the
on-disk / history check, so reruns are idempotent.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.audio.config import AudioConfig
from src.audio.history import ExtractHistory, ExtractRecord, now_iso

logger = logging.getLogger(__name__)

# Codec to use for each supported output format.
_FORMAT_CODECS = {
    "wav": "pcm_s16le",
    "flac": "flac",
    "mp3": "libmp3lame",
}


@dataclass(slots=True)
class ExtractSummary:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    would_extract: int = 0

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped + self.would_extract


class AudioExtractor:
    """Extracts audio tracks from every raw video file not already extracted."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.config.output.audio_dir.mkdir(parents=True, exist_ok=True)
        self.history = ExtractHistory(config.output.history_file)

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg was not found on PATH. Install ffmpeg and ensure it's "
                "callable as 'ffmpeg' before running audio extraction."
            )

        fmt = self.config.extract.format
        if fmt not in _FORMAT_CODECS:
            raise ValueError(
                f"Unsupported extract format '{fmt}'. Supported: {sorted(_FORMAT_CODECS)}"
            )

    # -- discovery ------------------------------------------------------

    def find_video_files(self) -> List[Path]:
        """Return every file directly under ``videos_dir`` (Phase 1's video+audio downloads)."""
        videos_dir = self.config.input.videos_dir
        if not videos_dir.exists():
            return []
        return sorted(p for p in videos_dir.iterdir() if p.is_file())

    def _output_path(self, video_id: str) -> Path:
        return self.config.output.audio_dir / f"{video_id}.{self.config.extract.format}"

    def _already_extracted(self, video_id: str) -> bool:
        # Any audio file for this ID counts as "already have it" -- whether
        # it's a prior extraction in the configured format, or an audio-only
        # download in whatever format yt-dlp picked (e.g. webm/m4a).
        if self.history.is_extracted(video_id):
            return True
        return any(self.config.output.audio_dir.glob(f"{video_id}.*"))

    # -- ffmpeg -----------------------------------------------------------

    def _run_ffmpeg(self, video_path: Path, output_path: Path) -> None:
        codec = _FORMAT_CODECS[self.config.extract.format]
        cmd = [
            "ffmpeg",
            "-y",  # overwrite output if present
            "-i", str(video_path),
            "-vn",  # drop the video stream
            "-acodec", codec,
            "-ar", str(self.config.extract.sample_rate),
            "-ac", str(self.config.extract.channels),
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg exited {result.returncode}: {result.stderr.strip()[-500:]}")

    # -- extraction loop --------------------------------------------------

    def extract_all(self, dry_run: bool = False) -> ExtractSummary:
        """Extract audio for every video not already extracted."""
        video_files = self.find_video_files()
        logger.info("Found %d video file(s) in %s", len(video_files), self.config.input.videos_dir)

        summary = ExtractSummary()
        for video_path in tqdm(video_files, desc="Videos", unit="video"):
            video_id = video_path.stem

            if self._already_extracted(video_id):
                logger.info("Skipping already-extracted video: %s", video_id)
                summary.skipped += 1
                continue

            if dry_run:
                logger.info("[dry-run] Would extract: %s (%s)", video_id, video_path.name)
                summary.would_extract += 1
                continue

            output_path = self._output_path(video_id)
            try:
                self._run_ffmpeg(video_path, output_path)
                self.history.record(
                    ExtractRecord(
                        video_id=video_id,
                        source_path=str(video_path),
                        status="success",
                        output_path=str(output_path),
                        timestamp=now_iso(),
                    )
                )
                summary.succeeded += 1
                logger.info("Extracted %s -> %s", video_path.name, output_path)
            except Exception as exc:
                logger.error("Failed to extract audio from %s: %s", video_path.name, exc)
                self.history.record(
                    ExtractRecord(
                        video_id=video_id,
                        source_path=str(video_path),
                        status="failed",
                        error=str(exc),
                        timestamp=now_iso(),
                    )
                )
                summary.failed += 1

        return summary
