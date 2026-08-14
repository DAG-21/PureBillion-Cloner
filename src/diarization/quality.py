"""Reference-free quality audit for Phase 4 diarization output.

There's no ground-truth speaker-labeled reference for this corpus (no
manual RTTM annotations exist), so the standard diarization metric --
Diarization Error Rate (DER), which requires exactly that -- doesn't apply
here. Same situation as ``src/transcription/quality.py`` for Phase 3: no
reference, so this uses structural signals already present in each
``data/diarized/<id>.json`` (speaker count, per-speaker time share, turn
fragmentation) plus a cross-check against the Phase 3 transcript's total
word count, to flag files worth a manual listen before Phase 5 (cleaning)
builds on top of this output.

Duplicated (not imported) from ``src/transcription/quality.py``'s
repetition-score logic -- same convention as this project's per-stage
self-contained modules (e.g. each stage's own ``logging_setup.py``), so
Phase 4 stays independently runnable without reaching into Phase 3 internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

REPETITION_NGRAM_SIZE = 4

# A turn shorter than this is more likely a diarization artifact (a brief
# misattributed word, cross-talk blip) than a genuine short utterance.
SHORT_TURN_SECONDS = 0.3
HIGH_SHORT_TURN_RATIO = 0.3

# Same plausible-speech-rate bounds as src/transcription/quality.py --
# isolated target-speaker text should still read as normal spoken English.
MIN_PLAUSIBLE_WORDS_PER_SECOND = 1.0
MAX_PLAUSIBLE_WORDS_PER_SECOND = 4.5

LOW_TARGET_SHARE = 0.15
MANY_SPEAKERS_THRESHOLD = 5
LOW_TEXT_RETENTION_RATIO = 0.05
HIGH_REPETITION_SCORE = 0.3


@dataclass(slots=True)
class DiarizationQuality:
    video_id: str
    duration: float
    num_speakers: int
    target_speaker: Optional[str]
    target_speaker_share: float
    target_turn_count: int
    num_target_segments: int
    avg_target_segment_seconds: float
    short_turn_ratio: float
    target_word_count: int
    target_words_per_second: float
    full_transcript_word_count: Optional[int]
    target_text_retention_ratio: Optional[float]
    repetition_score: float
    flags: List[str] = field(default_factory=list)


def _repetition_score(text: str, n: int = REPETITION_NGRAM_SIZE) -> float:
    """Fraction of duplicate n-grams -- proxy for the isolated segments
    stitching together into an accidentally repetitive/looping string
    (e.g. if word-to-speaker misalignment causes the same phrase to be
    re-emitted across adjacent regrouped segments)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return 1.0 - unique_ratio


def evaluate_diarization(
    diarized: Dict[str, Any], full_transcript_word_count: Optional[int] = None
) -> DiarizationQuality:
    """Compute reference-free quality metrics for one diarized-file dict."""
    video_id = diarized.get("video_id", "")
    duration = float(diarized.get("duration") or 0.0)
    speakers: Dict[str, Any] = diarized.get("speakers", {})
    target_speaker = diarized.get("target_speaker")
    turns: List[Dict[str, Any]] = diarized.get("turns", [])
    target_segments: List[Dict[str, Any]] = diarized.get("target_speaker_segments", [])
    target_text = diarized.get("target_speaker_text", "")

    num_speakers = len(speakers)
    target_stats = speakers.get(target_speaker, {}) if target_speaker else {}
    target_total_seconds = float(target_stats.get("total_seconds") or 0.0)
    target_turn_count = int(target_stats.get("turn_count") or 0)
    target_speaker_share = target_total_seconds / duration if duration > 0 else 0.0

    num_target_segments = len(target_segments)
    segment_durations = [max(0.0, s["end"] - s["start"]) for s in target_segments]
    avg_target_segment_seconds = (
        sum(segment_durations) / len(segment_durations) if segment_durations else 0.0
    )

    target_turns = [t for t in turns if t.get("speaker") == target_speaker]
    short_turns = [
        t for t in target_turns if (t["end"] - t["start"]) < SHORT_TURN_SECONDS
    ]
    short_turn_ratio = len(short_turns) / len(target_turns) if target_turns else 0.0

    target_word_count = len(target_text.split())
    target_words_per_second = (
        target_word_count / target_total_seconds if target_total_seconds > 0 else 0.0
    )
    repetition_score = _repetition_score(target_text)

    retention_ratio: Optional[float] = None
    if full_transcript_word_count:
        retention_ratio = target_word_count / full_transcript_word_count

    flags: List[str] = []
    if target_word_count == 0:
        flags.append("empty_target_text")
    if num_speakers <= 1:
        flags.append("single_speaker_detected")
    if num_speakers > MANY_SPEAKERS_THRESHOLD:
        flags.append("many_speakers_detected")
    if duration > 0 and target_speaker_share < LOW_TARGET_SHARE:
        flags.append("low_target_speaker_share")
    if short_turn_ratio > HIGH_SHORT_TURN_RATIO:
        flags.append("fragmented_turns")
    if target_words_per_second > 0 and target_words_per_second < MIN_PLAUSIBLE_WORDS_PER_SECOND:
        flags.append("implausibly_slow")
    if target_words_per_second > MAX_PLAUSIBLE_WORDS_PER_SECOND:
        flags.append("implausibly_fast")
    if retention_ratio is not None and retention_ratio < LOW_TEXT_RETENTION_RATIO:
        flags.append("low_text_retention")
    if repetition_score > HIGH_REPETITION_SCORE:
        flags.append("possible_repetition_artifact")

    return DiarizationQuality(
        video_id=video_id,
        duration=duration,
        num_speakers=num_speakers,
        target_speaker=target_speaker,
        target_speaker_share=target_speaker_share,
        target_turn_count=target_turn_count,
        num_target_segments=num_target_segments,
        avg_target_segment_seconds=avg_target_segment_seconds,
        short_turn_ratio=short_turn_ratio,
        target_word_count=target_word_count,
        target_words_per_second=target_words_per_second,
        full_transcript_word_count=full_transcript_word_count,
        target_text_retention_ratio=retention_ratio,
        repetition_score=repetition_score,
        flags=flags,
    )
