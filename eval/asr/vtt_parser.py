"""Parses a WebVTT caption file into plain reference text for WER/CER scoring.

YouTube's manually-uploaded ("subtitles", as opposed to auto-generated
"automatic captions") tracks for this corpus turn out to be clean,
human-transcribed text with inline speaker labels (``Sadhguru: ...``) and
parenthetical stage directions (``(Laughs)``). Both are stripped here since
Whisper's output is plain speech-only text with no speaker attribution or
non-speech annotations.
"""
from __future__ import annotations

import re
from pathlib import Path

_TIMESTAMP_LINE = re.compile(r"-->")
_INLINE_TAG = re.compile(r"<[^>]+>")
_SPEAKER_PREFIX = re.compile(r"^[A-Za-z][A-Za-z .'\-]{0,40}:\s+")
_PARENTHETICAL = re.compile(r"\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


def parse_vtt(path: Path) -> str:
    """Return the plain spoken-text content of a WebVTT file at ``path``."""
    lines = path.read_text(encoding="utf-8").splitlines()

    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if line.startswith("NOTE") or line.startswith("STYLE"):
            continue
        if _TIMESTAMP_LINE.search(line):
            continue
        if line.isdigit():  # stray SRT-style cue index, if present
            continue
        line = _INLINE_TAG.sub("", line)
        # Each cue line starts fresh with "Speaker: " only when the speaker
        # actually changes -- continuation lines of the same speaker's turn
        # don't repeat it, so this must be stripped per-line, not once on
        # the fully joined text.
        line = _SPEAKER_PREFIX.sub("", line)
        text_lines.append(line)

    text = " ".join(text_lines)
    text = _PARENTHETICAL.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
