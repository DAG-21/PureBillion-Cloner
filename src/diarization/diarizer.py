"""Core pyannote.audio wrapper: diarizes speakers and isolates the target speaker.

The pyannote pipeline itself is loaded lazily, on the first audio file
actually diarized -- not at construction time. That keeps ``--dry-run``
usable on a machine without ``HF_TOKEN``/GPU/model weights present, same
convention as ``src/transcription/transcriber.py``.

Target-speaker identification (v1): whichever diarized speaker label has
the most total speaking time in a given file is assumed to be the target
speaker (Sadhguru). See ``configs/diarization.yaml`` for the rationale and
the config knob (``target_speaker.strategy``) this is read from.

Audio is decoded via a direct ``ffmpeg`` subprocess call + the stdlib
``wave`` module, then handed to the pipeline as a preloaded waveform dict
rather than a file path. This deliberately avoids pyannote's default
file-loading path, which goes through ``torchcodec`` -- on this machine
``torchcodec``'s bundled native libraries fail to load regardless of the
installed FFmpeg version, apparently a version mismatch with this
``torch==2.9.1`` build (itself pinned to match the last Windows CUDA
``torchaudio`` release, 2.9.1). ``ffmpeg`` must be on ``PATH`` -- same
requirement as ``src/audio/extractor.py`` (Phase 2).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.diarization.config import DiarizationConfig
from src.diarization.history import DiarizationHistory, DiarizationRecord, now_iso

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiarizationSummary:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    missing_transcript: int = 0
    would_diarize: int = 0

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped + self.missing_transcript + self.would_diarize


def _flatten_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten every segment's words into one chronological list.

    Falls back to treating a whole segment as a single pseudo-word if it has
    no word-level timestamps (defensive -- the transcript schema allows
    ``words: null``, even though every real Phase 3 output has them since
    ``word_timestamps: true`` was used for the full run).
    """
    words: List[Dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        seg_words = segment.get("words")
        if seg_words:
            words.extend(seg_words)
        else:
            words.append({"start": segment["start"], "end": segment["end"], "word": segment.get("text", "")})
    return words


def _speaker_stats(turns: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for turn in turns:
        entry = stats.setdefault(turn["speaker"], {"total_seconds": 0.0, "turn_count": 0})
        entry["total_seconds"] += turn["end"] - turn["start"]
        entry["turn_count"] += 1
    return stats


def _in_turns(midpoint: float, turns: List[Dict[str, Any]]) -> bool:
    return any(turn["start"] <= midpoint <= turn["end"] for turn in turns)


def _regroup_target_segments(
    words: List[Dict[str, Any]], target_turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Keep only words whose midpoint falls inside a target-speaker turn, then
    regroup consecutive kept words into new segments -- splitting wherever a
    dropped (non-target) word breaks continuity."""
    segments: List[Dict[str, Any]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = "".join(w["word"] for w in current).strip()
        segments.append(
            {
                "id": len(segments),
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": text,
            }
        )
        current.clear()

    for word in words:
        midpoint = (word["start"] + word["end"]) / 2
        if _in_turns(midpoint, target_turns):
            current.append(word)
        else:
            flush()
    flush()
    return segments


class Diarizer:
    """Diarizes every raw audio file not already diarized, via pyannote.audio."""

    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config
        self.config.output.diarized_dir.mkdir(parents=True, exist_ok=True)
        self.history = DiarizationHistory(config.output.history_file)
        self._pipeline: Any = None  # lazily-loaded pyannote.audio.Pipeline

    # -- discovery ------------------------------------------------------

    def find_audio_files(self) -> List[Path]:
        """Return every file directly under ``audio_dir`` (any extension, same
        convention as the transcription stage)."""
        audio_dir = self.config.input.audio_dir
        if not audio_dir.exists():
            return []
        return sorted(p for p in audio_dir.iterdir() if p.is_file())

    def _transcript_path(self, video_id: str) -> Path:
        return self.config.input.transcripts_dir / f"{video_id}.json"

    def _output_path(self, video_id: str) -> Path:
        return self.config.output.diarized_dir / f"{video_id}.json"

    def _already_diarized(self, video_id: str) -> bool:
        if self.history.is_diarized(video_id):
            return True
        return self._output_path(video_id).exists()

    # -- model ------------------------------------------------------------

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            from pyannote.audio import Pipeline  # deferred: heavy import, needs no GPU/token for dry-run
            import torch

            token = os.environ.get("HF_TOKEN")
            if not token:
                raise RuntimeError(
                    "HF_TOKEN not set -- required to download the gated pyannote model "
                    f"'{self.config.model.pipeline}'. Set it in .env (see .env.example)."
                )

            logger.info("Loading diarization pipeline '%s'...", self.config.model.pipeline)
            pipeline = Pipeline.from_pretrained(self.config.model.pipeline, token=token)

            device = self.config.model.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                pipeline.to(torch.device("cuda"))

            self._pipeline = pipeline
        return self._pipeline

    # -- audio decoding -----------------------------------------------------

    def _load_waveform(self, audio_path: Path) -> Dict[str, Any]:
        """Decode ``audio_path`` to a mono 16kHz waveform via ffmpeg + the stdlib
        ``wave`` module, bypassing pyannote's default (torchcodec-based) loader --
        see module docstring for why."""
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_wav = Path(tmp_dir) / "audio.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp_wav)],
                    check=True,
                    capture_output=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("ffmpeg not found on PATH -- required to decode audio for diarization") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"ffmpeg failed to decode {audio_path}: {exc.stderr.decode(errors='replace')}") from exc

            with wave.open(str(tmp_wav), "rb") as wf:
                sample_rate = wf.getframerate()
                raw = wf.readframes(wf.getnframes())

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(samples).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    # -- diarization ------------------------------------------------------

    def _diarize_one(self, audio_path: Path, transcript: Dict[str, Any]) -> Dict[str, Any]:
        pipeline = self._load_pipeline()
        audio_input = self._load_waveform(audio_path)
        output = pipeline(audio_input)
        # exclusive_speaker_diarization has no overlapping turns, which keeps
        # the word-midpoint-in-turn check below unambiguous.
        annotation = output.exclusive_speaker_diarization

        turns: List[Dict[str, Any]] = sorted(
            (
                {"speaker": label, "start": turn.start, "end": turn.end}
                for turn, _, label in annotation.itertracks(yield_label=True)
            ),
            key=lambda t: t["start"],
        )
        if not turns:
            raise RuntimeError("Diarization produced no speaker turns")

        speaker_stats = _speaker_stats(turns)
        target_speaker = max(speaker_stats, key=lambda label: speaker_stats[label]["total_seconds"])
        target_turns = [t for t in turns if t["speaker"] == target_speaker]

        words = _flatten_words(transcript)
        target_segments = _regroup_target_segments(words, target_turns)

        return {
            "video_id": audio_path.stem,
            "source_audio": str(audio_path),
            "duration": transcript.get("duration"),
            "speakers": speaker_stats,
            "target_speaker": target_speaker,
            "target_speaker_strategy": self.config.target_speaker.strategy,
            "turns": turns,
            "target_speaker_segments": target_segments,
            "target_speaker_text": " ".join(seg["text"] for seg in target_segments),
            "diarized_at": now_iso(),
        }

    def diarize_all(self, dry_run: bool = False) -> DiarizationSummary:
        """Diarize every audio file not already diarized, skipping any without a
        matching Phase 3 transcript."""
        audio_files = self.find_audio_files()
        logger.info("Found %d audio file(s) in %s", len(audio_files), self.config.input.audio_dir)

        summary = DiarizationSummary()
        for audio_path in tqdm(audio_files, desc="Audio files", unit="file"):
            video_id = audio_path.stem

            if self._already_diarized(video_id):
                logger.info("Skipping already-diarized audio: %s", video_id)
                summary.skipped += 1
                continue

            transcript_path = self._transcript_path(video_id)
            if not transcript_path.exists():
                logger.warning(
                    "No transcript found for %s (%s) -- run Phase 3 transcription first, skipping",
                    video_id,
                    transcript_path,
                )
                summary.missing_transcript += 1
                continue

            if dry_run:
                logger.info("[dry-run] Would diarize: %s (%s)", video_id, audio_path.name)
                summary.would_diarize += 1
                continue

            output_path = self._output_path(video_id)
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
                result = self._diarize_one(audio_path, transcript)
                output_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self.history.record(
                    DiarizationRecord(
                        video_id=video_id,
                        source_audio=str(audio_path),
                        status="success",
                        output_path=str(output_path),
                        timestamp=now_iso(),
                    )
                )
                summary.succeeded += 1
                logger.info(
                    "Diarized %s -> %s (target_speaker=%s)",
                    audio_path.name,
                    output_path,
                    result["target_speaker"],
                )
            except Exception as exc:
                logger.error("Failed to diarize %s: %s", audio_path.name, exc)
                self.history.record(
                    DiarizationRecord(
                        video_id=video_id,
                        source_audio=str(audio_path),
                        status="failed",
                        error=str(exc),
                        timestamp=now_iso(),
                    )
                )
                summary.failed += 1

        return summary
