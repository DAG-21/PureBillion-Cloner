"""Download history tracking (``download_history.csv``) for the acquisition stage."""
from __future__ import annotations

import csv
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ["video_id", "title", "url", "mode", "status", "output_path", "error", "timestamp"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class HistoryRecord:
    video_id: str
    title: str
    url: str
    status: str  # "success" | "failed" | "skipped"
    mode: str = "video"  # "video" | "audio" -- which download mode produced this record
    output_path: str = ""
    error: str = ""
    timestamp: str = field(default_factory=now_iso)


class DownloadHistory:
    """CSV-backed record of processed videos, used to skip already-downloaded ones.

    A video is tracked per ``(video_id, mode)`` pair, since ``--audio-only``
    and video+audio downloads of the same video are distinct outputs and
    completing one should not cause the other to be skipped.
    """

    def __init__(self, history_file: Path) -> None:
        self._history_file = history_file
        self._lock = threading.Lock()
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._completed: Set[Tuple[str, str]] = set()
        self._migrate_if_needed()
        self._load_existing()

    def _migrate_if_needed(self) -> None:
        """Rewrite the CSV onto the current ``HISTORY_FIELDS`` schema if its header is stale.

        ``csv.DictWriter`` writes rows positionally according to
        ``fieldnames``, not by matching an existing header -- appending a row
        with today's field order onto a file whose header predates a schema
        change (e.g. the "mode" column) silently misaligns every column.
        Rewriting the whole file up front keeps append-only writes safe.
        """
        if not self._history_file.exists():
            return
        with self._history_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == HISTORY_FIELDS:
                return
            rows = list(reader)

        with self._lock:
            with self._history_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                for row in rows:
                    row["mode"] = row.get("mode") or "video"
                    writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})
        logger.info(
            "Migrated %s to the current history schema (%d existing row(s))",
            self._history_file,
            len(rows),
        )

    def _load_existing(self) -> None:
        if not self._history_file.exists():
            return
        with self._history_file.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "success" and row.get("video_id"):
                    # Rows written before the "mode" column existed are all
                    # video+audio downloads (audio-only support came later).
                    mode = row.get("mode") or "video"
                    self._completed.add((row["video_id"], mode))
        logger.debug("Loaded %d previously completed download(s) from history", len(self._completed))

    def is_downloaded(self, video_id: str, mode: str = "video") -> bool:
        return (video_id, mode) in self._completed

    def record(self, entry: HistoryRecord) -> None:
        """Append ``entry`` to the history CSV, writing a header if the file is new."""
        with self._lock:
            is_new = not self._history_file.exists()
            with self._history_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
                if is_new:
                    writer.writeheader()
                writer.writerow(asdict(entry))
            if entry.status == "success":
                self._completed.add((entry.video_id, entry.mode))
