"""
Computes WER, CER, MER, and WIL between Phase 3 Whisper transcripts and
YouTube manual-subtitle reference text (see fetch_references.py).

Unlike the reference-free metrics in src/transcription/quality.py, these
measure actual transcription accuracy against real (human-transcribed)
text -- not just the model's internal confidence. Both hypothesis
(Whisper) and reference (YouTube subtitle) text are lowercased and
stripped of punctuation before comparison, since jiwer's bare default
transform does neither and would otherwise inflate error rates purely from
style differences (curly vs. straight quotes, capitalization, etc.).

Only video IDs with both a transcript and a fetched reference are scored;
everything else (no manual subtitle available) is reported as uncovered,
not treated as an error.

Usage:
    python -m eval.asr.compute_wer [options]

Input:  data/transcripts/<id>.json, data/references/<id>.txt
Output: eval/results/asr_accuracy_report.json (+ printed summary)
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import jiwer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_REFERENCES_DIR = PROJECT_ROOT / "data" / "references"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "results" / "asr_accuracy_report.json"

NORMALIZE = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
# cer() aligns at the character level, so it needs its own transform ending
# in ReduceToListOfListOfChars -- reusing NORMALIZE's word-level reduction
# would break it.
NORMALIZE_CER = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)

_METRIC_FIELDS = ("wer", "cer", "mer", "wil")


@dataclass(slots=True)
class AccuracyResult:
    video_id: str
    reference_word_count: int
    hypothesis_word_count: int
    wer: float
    cer: float
    mer: float
    wil: float


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compute_wer.py",
        description="Compute WER/CER/MER/WIL between transcripts and reference captions.",
    )
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--references-dir", type=Path, default=DEFAULT_REFERENCES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=20, help="Worst-WER files to print (default: %(default)s).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _score_one(video_id: str, reference: str, hypothesis: str) -> AccuracyResult:
    return AccuracyResult(
        video_id=video_id,
        reference_word_count=len(reference.split()),
        hypothesis_word_count=len(hypothesis.split()),
        wer=jiwer.wer(reference, hypothesis, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE),
        cer=jiwer.cer(reference, hypothesis, reference_transform=NORMALIZE_CER, hypothesis_transform=NORMALIZE_CER),
        mer=jiwer.mer(reference, hypothesis, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE),
        wil=jiwer.wil(reference, hypothesis, reference_transform=NORMALIZE, hypothesis_transform=NORMALIZE),
    )


def _corpus_stats(results: List[AccuracyResult]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field_name in _METRIC_FIELDS:
        values = sorted(getattr(r, field_name) for r in results)
        if not values:
            continue
        stats[field_name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
            "p10": statistics.quantiles(values, n=10)[0] if len(values) >= 2 else values[0],
            "p90": statistics.quantiles(values, n=10)[-1] if len(values) >= 2 else values[-1],
        }
    return stats


def run(transcripts_dir: Path, references_dir: Path, output_path: Path, top_n: int) -> Dict[str, Any]:
    reference_ids = sorted(p.stem for p in references_dir.glob("*.txt"))
    logger.info("Found %d reference file(s) in %s", len(reference_ids), references_dir)

    results: List[AccuracyResult] = []
    missing_transcript = 0
    empty_reference = 0

    for video_id in reference_ids:
        transcript_path = transcripts_dir / f"{video_id}.json"
        if not transcript_path.exists():
            logger.warning("No transcript for %s, skipping", video_id)
            missing_transcript += 1
            continue

        reference = (references_dir / f"{video_id}.txt").read_text(encoding="utf-8").strip()
        if not reference:
            empty_reference += 1
            continue

        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        hypothesis = transcript.get("text", "")

        results.append(_score_one(video_id, reference, hypothesis))

    stats = _corpus_stats(results)
    worst_first = sorted(results, key=lambda r: -r.wer)

    report = {
        "num_scored": len(results),
        "missing_transcript": missing_transcript,
        "empty_reference": empty_reference,
        "corpus_stats": stats,
        "files": [asdict(r) for r in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote accuracy report to %s", output_path)

    logger.info("=== Corpus-wide accuracy (%d files scored) ===", len(results))
    for field_name, field_stats in stats.items():
        logger.info(
            "%-5s mean=%.4f median=%.4f p10=%.4f p90=%.4f min=%.4f max=%.4f",
            field_name.upper(),
            field_stats["mean"],
            field_stats["median"],
            field_stats["p10"],
            field_stats["p90"],
            field_stats["min"],
            field_stats["max"],
        )

    logger.info("=== Worst %d files by WER ===", min(top_n, len(worst_first)))
    for r in worst_first[:top_n]:
        logger.info(
            "%s | wer=%.4f cer=%.4f mer=%.4f wil=%.4f | ref_words=%d hyp_words=%d",
            r.video_id,
            r.wer,
            r.cer,
            r.mer,
            r.wil,
            r.reference_word_count,
            r.hypothesis_word_count,
        )

    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level.upper())
    run(args.transcripts_dir, args.references_dir, args.output, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
