"""Metadata extraction and persistence for downloaded videos."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Fields pulled directly from a yt-dlp info dict. Missing fields are stored as None
# rather than omitted, so every metadata JSON file has a stable schema.
_INFO_FIELDS = (
    "id",
    "title",
    "description",
    "upload_date",
    "duration",
    "duration_string",
    "channel",
    "channel_id",
    "channel_url",
    "uploader",
    "uploader_id",
    "uploader_url",
    "tags",
    "categories",
    "view_count",
    "like_count",
    "comment_count",
    "thumbnail",
    "width",
    "height",
    "fps",
    "resolution",
    "ext",
    "language",
    "availability",
    "age_limit",
    "was_live",
    "live_status",
)


def extract_metadata(info: Dict[str, Any]) -> Dict[str, Any]:
    """Build a stable metadata dict from a yt-dlp ``extract_info`` result."""
    metadata: Dict[str, Any] = {field: info.get(field) for field in _INFO_FIELDS}
    metadata["url"] = info.get("webpage_url") or info.get("original_url") or info.get("url")
    metadata["downloaded_at"] = datetime.now(timezone.utc).isoformat()
    return metadata


def save_metadata(metadata: Dict[str, Any], metadata_dir: Path) -> Path:
    """Write ``metadata`` to ``<metadata_dir>/<video_id>.json`` and return the path."""
    video_id = metadata.get("id")
    if not video_id:
        raise ValueError("Metadata is missing a video id; cannot save.")

    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f"{video_id}.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Saved metadata for %s to %s", video_id, path)
    return path
