"""Phase 5-b (Genre Classification) engine.

Adds a broad subject-matter ``genre`` field to each of Phase 5's cleaned
per-video entries (``data/cleaned/sadhguru_cleaned.jsonl``), using the
static id -> genre mapping in ``genres.py``. See that module's docstring for
how the mapping was produced.

Every entry's id must be present in the mapping -- a missing id is a hard
error, not a silent "Other"/None default, since guessing would misrepresent
the corpus's actual genre distribution.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.genre_classification.genres import GENRE_BY_ID

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GenreClassificationSummary:
    total: int
    genre_counts: Dict[str, int]


def classify_all(
    cleaned_file: Path, output_file: Path, dry_run: bool = False
) -> GenreClassificationSummary:
    if not cleaned_file.exists():
        raise FileNotFoundError(f"Cleaned file not found: {cleaned_file}")

    entries: List[dict] = []
    with cleaned_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    logger.info("Loaded %d cleaned entries from %s", len(entries), cleaned_file)

    missing = [e["id"] for e in entries if e["id"] not in GENRE_BY_ID]
    if missing:
        raise KeyError(
            f"{len(missing)} entry id(s) have no genre mapping in genres.py: {missing}"
        )

    for entry in entries:
        entry["genre"] = GENRE_BY_ID[entry["id"]]

    genre_counts = Counter(e["genre"] for e in entries)

    if dry_run:
        logger.info(
            "[dry-run] would write %d genre-tagged entries -> %s", len(entries), output_file
        )
        return GenreClassificationSummary(total=len(entries), genre_counts=dict(genre_counts))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Wrote %d genre-tagged entries to %s", len(entries), output_file)
    return GenreClassificationSummary(total=len(entries), genre_counts=dict(genre_counts))
