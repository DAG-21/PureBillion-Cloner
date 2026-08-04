"""Transcription history tracking (``transcription_history.csv``) for the transcription stage."""
from __future__ import annotations

import csv
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ["video_id", "source_path", "status", "output_path", "error", "timestamp"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class TranscriptionRecord:
    video_id: str
    source_path: str
    status: str  # "success" | "failed"
    output_path: str = ""
    error: str = ""
    timestamp: str = field(default_factory=now_iso)


class TranscriptionHistory:
    """CSV-backed record of transcribed audio files, used to skip already-transcribed ones."""

    def __init__(self, history_file: Path) -> None:
        self._history_file = history_file
        self._lock = threading.Lock()
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._completed_ids: Set[str] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._history_file.exists():
            return
        with self._history_file.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "success" and row.get("video_id"):
                    self._completed_ids.add(row["video_id"])
        logger.debug("Loaded %d previously transcribed audio file(s) from history", len(self._completed_ids))

    def is_transcribed(self, video_id: str) -> bool:
        return video_id in self._completed_ids

    def record(self, entry: TranscriptionRecord) -> None:
        """Append ``entry`` to the history CSV, writing a header if the file is new."""
        with self._lock:
            is_new = not self._history_file.exists()
            with self._history_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(asdict(entry))
            if entry.status == "success":
                self._completed_ids.add(entry.video_id)
