"""Phase 5 (Cleaning) engine.

Parses the single combined export file produced by
``src/diarization/export_target_text.py``
(``data/sadhguru_topics_and_text.txt``) into per-video (topic, text) entries
and applies mechanical text cleanup: stray-character removal, whitespace/
punctuation normalization, standalone ASR-disfluency removal, and sentence
casing.

This deliberately does NOT touch Sadhguru's actual verbal tics ("you know",
"isn't it", "right?", "dhani") -- those are style signal the fine-tuning
tier wants to learn, not noise to strip. It also does not attempt to repair
truncated sentence openings caused by diarization turn-boundary imprecision
(e.g. a dropped leading pronoun) -- that would require re-processing audio/
alignment, not text-level cleanup.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DIVIDER = "\n" + ("-" * 80) + "\n\n"
_FFFD_RE = re.compile("�")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")
_MISSING_SPACE_AFTER_PUNCT_RE = re.compile(r"([.!?])([A-Za-z])")
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_BOUNDARY_RE = re.compile(r"([.!?]\s+)([a-z])")


@dataclass(slots=True)
class RawEntry:
    topic: str
    text: str


@dataclass(slots=True)
class CleanedEntry:
    id: str
    topic: str
    text: str
    raw_word_count: int
    cleaned_word_count: int


@dataclass(slots=True)
class CleaningSummary:
    total: int
    cleaned: int
    would_clean: int


def parse_export_file(export_file: Path) -> List[RawEntry]:
    """Parse the combined ``Topic:``/``Text:`` export file into raw entries."""
    raw = export_file.read_text(encoding="utf-8")
    blocks = raw.split(_DIVIDER)
    entries: List[RawEntry] = []
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        if not lines or not lines[0].startswith("Topic: "):
            raise ValueError(f"Malformed entry #{i} in {export_file}: missing 'Topic: ' line")
        topic = lines[0][len("Topic: "):]
        text_line = next((l for l in lines[1:] if l.startswith("Text: ")), None)
        if text_line is None:
            raise ValueError(f"Malformed entry #{i} in {export_file}: missing 'Text: ' line")
        text = text_line[len("Text: "):]
        entries.append(RawEntry(topic=topic, text=text))
    return entries


def _strip_disfluencies(text: str, disfluencies: List[str]) -> str:
    if not disfluencies:
        return text
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in disfluencies) + r")\b"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _sentence_case(text: str) -> str:
    if not text:
        return text
    text = text[0].upper() + text[1:] if text[0].isalpha() else text

    def _cap(m: re.Match) -> str:
        return m.group(1) + m.group(2).upper()

    return _SENTENCE_BOUNDARY_RE.sub(_cap, text)


def clean_text(text: str, disfluencies: List[str], sentence_case: bool) -> str:
    text = _FFFD_RE.sub("", text)
    text = _strip_disfluencies(text, disfluencies)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _MISSING_SPACE_AFTER_PUNCT_RE.sub(r"\1 \2", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if sentence_case:
        text = _sentence_case(text)
    return text


class Cleaner:
    def __init__(self, config) -> None:  # config: CleaningConfig
        self.config = config

    def clean_all(self, dry_run: bool = False) -> CleaningSummary:
        export_file = self.config.input.export_file
        if not export_file.exists():
            raise FileNotFoundError(f"Export file not found: {export_file}")

        entries = parse_export_file(export_file)
        logger.info("Parsed %d entries from %s", len(entries), export_file)

        if dry_run:
            logger.info(
                "[dry-run] would clean %d entries -> %s",
                len(entries), self.config.output.cleaned_file,
            )
            return CleaningSummary(
                total=len(entries), cleaned=0, would_clean=len(entries),
            )

        cleaned_entries: List[CleanedEntry] = []
        for i, entry in enumerate(entries):
            cleaned_text = clean_text(
                entry.text,
                self.config.cleaning.disfluencies,
                self.config.cleaning.sentence_case,
            )
            cleaned_entries.append(
                CleanedEntry(
                    id=f"{i:04d}",
                    topic=entry.topic,
                    text=cleaned_text,
                    raw_word_count=len(entry.text.split()),
                    cleaned_word_count=len(cleaned_text.split()),
                )
            )

        output_file = self.config.output.cleaned_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for e in cleaned_entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

        logger.info("Wrote %d cleaned entries to %s", len(cleaned_entries), output_file)
        return CleaningSummary(
            total=len(entries), cleaned=len(cleaned_entries), would_clean=0,
        )
