"""Reference-free quality audit for Phase 3 transcripts.

There's no ground-truth reference transcript for these talks (Phase 1 is
video+metadata only -- no captions were fetched), so standard reference-
based metrics (WER/CER, or the BERTScore/ROUGE/BLEU in ``eval/``, which
compare generated persona responses against reference statements at a much
later pipeline stage) don't apply to raw ASR output here -- there's nothing
to compare against.

Instead, this uses signals faster-whisper already produces (per-word
confidence) plus structural heuristics (repetition, speech coverage,
speaking rate) to flag transcripts worth a manual listen before they feed
diarization, cleaning, and eventually fine-tuning data.

Note: every transcript's ``language_probability`` is exactly ``1`` (Phase 3
forced ``language: en`` rather than auto-detecting -- see
``configs/transcription.yaml``), so it carries no real signal and is
deliberately not used as a quality metric here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOW_CONFIDENCE_THRESHOLD = 0.5
REPETITION_NGRAM_SIZE = 4

# Rough bounds for plausible spoken-English pace. Outside this range more
# likely reflects a transcription problem (dropped audio, hallucinated
# filler) than genuine variation in speaking speed.
MIN_PLAUSIBLE_WORDS_PER_SECOND = 1.0
MAX_PLAUSIBLE_WORDS_PER_SECOND = 4.5

MIN_PLAUSIBLE_COVERAGE_RATIO = 0.15


@dataclass(slots=True)
class TranscriptQuality:
    video_id: str
    duration: float
    word_count: int
    avg_word_confidence: Optional[float]
    low_confidence_word_ratio: Optional[float]
    words_per_second: float
    speech_coverage_ratio: float
    repetition_score: float
    flags: List[str] = field(default_factory=list)


def _word_confidences(transcript: Dict[str, Any]) -> List[float]:
    confidences: List[float] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words") or []:
            probability = word.get("probability")
            if probability is not None:
                confidences.append(probability)
    return confidences


def _speech_coverage_ratio(transcript: Dict[str, Any]) -> float:
    duration = transcript.get("duration") or 0.0
    if duration <= 0:
        return 0.0
    spoken = sum(
        max(0.0, segment["end"] - segment["start"]) for segment in transcript.get("segments", [])
    )
    return min(spoken / duration, 1.0)


def _repetition_score(text: str, n: int = REPETITION_NGRAM_SIZE) -> float:
    """Fraction of duplicate n-grams in the full transcript text -- a proxy
    for the "stuck in a loop" hallucination Whisper-family models sometimes
    produce on silence/noise. 0 = no repeated n-grams, 1 = maximally
    repetitive."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return 1.0 - unique_ratio


def evaluate_transcript(transcript: Dict[str, Any]) -> TranscriptQuality:
    """Compute reference-free quality metrics for one transcript dict."""
    video_id = transcript.get("video_id", "")
    duration = float(transcript.get("duration") or 0.0)
    text = transcript.get("text", "")
    word_count = len(text.split())

    confidences = _word_confidences(transcript)
    avg_confidence = sum(confidences) / len(confidences) if confidences else None
    low_confidence_ratio = (
        sum(1 for c in confidences if c < LOW_CONFIDENCE_THRESHOLD) / len(confidences)
        if confidences
        else None
    )

    words_per_second = word_count / duration if duration > 0 else 0.0
    coverage_ratio = _speech_coverage_ratio(transcript)
    repetition_score = _repetition_score(text)

    flags: List[str] = []
    if avg_confidence is not None and avg_confidence < 0.7:
        flags.append("low_avg_confidence")
    if low_confidence_ratio is not None and low_confidence_ratio > 0.15:
        flags.append("many_low_confidence_words")
    if words_per_second > 0 and words_per_second < MIN_PLAUSIBLE_WORDS_PER_SECOND:
        flags.append("implausibly_slow")
    if words_per_second > MAX_PLAUSIBLE_WORDS_PER_SECOND:
        flags.append("implausibly_fast")
    if coverage_ratio < MIN_PLAUSIBLE_COVERAGE_RATIO:
        flags.append("low_speech_coverage")
    if repetition_score > 0.3:
        flags.append("possible_repetition_hallucination")
    if word_count == 0:
        flags.append("empty_transcript")

    return TranscriptQuality(
        video_id=video_id,
        duration=duration,
        word_count=word_count,
        avg_word_confidence=avg_confidence,
        low_confidence_word_ratio=low_confidence_ratio,
        words_per_second=words_per_second,
        speech_coverage_ratio=coverage_ratio,
        repetition_score=repetition_score,
        flags=flags,
    )
