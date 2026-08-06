"""
Phase 3 quality audit CLI: runs reference-free quality metrics over every
transcript in data/transcripts/ and flags files worth a manual listen.

See src/transcription/quality.py for why this is reference-free (no ground-
truth transcripts exist for this corpus) and what each metric means.

Usage:
    python -m src.transcription.evaluate [options]

Input:  data/transcripts/<id>.json
Output: eval/results/transcription_quality_report.json (+ printed summary)
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.transcription.logging_setup import configure_logging
from src.transcription.quality import TranscriptQuality, evaluate_transcript

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "results" / "transcription_quality_report.json"

_NUMERIC_FIELDS = (
    "avg_word_confidence",
    "low_confidence_word_ratio",
    "words_per_second",
    "speech_coverage_ratio",
    "repetition_score",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Audit Phase 3 transcript quality with reference-free metrics.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=DEFAULT_TRANSCRIPTS_DIR,
        help="Directory of transcript JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the JSON report to (default: %(default)s).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many flagged files to print in the summary (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _corpus_stats(results: List[TranscriptQuality]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field_name in _NUMERIC_FIELDS:
        values = [getattr(r, field_name) for r in results if getattr(r, field_name) is not None]
        if not values:
            continue
        values_sorted = sorted(values)
        stats[field_name] = {
            "mean": statistics.mean(values_sorted),
            "median": statistics.median(values_sorted),
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "p10": statistics.quantiles(values_sorted, n=10)[0] if len(values_sorted) >= 2 else values_sorted[0],
            "p90": statistics.quantiles(values_sorted, n=10)[-1] if len(values_sorted) >= 2 else values_sorted[-1],
        }
    return stats


def _flagged_sorted(results: List[TranscriptQuality]) -> List[TranscriptQuality]:
    flagged = [r for r in results if r.flags]
    # Worst first: most flags, then lowest average word confidence.
    return sorted(
        flagged,
        key=lambda r: (-len(r.flags), r.avg_word_confidence if r.avg_word_confidence is not None else 1.0),
    )


def run(transcripts_dir: Path, output_path: Path, top_n: int) -> Dict[str, Any]:
    transcript_files = sorted(transcripts_dir.glob("*.json"))
    logger.info("Found %d transcript file(s) in %s", len(transcript_files), transcripts_dir)

    results: List[TranscriptQuality] = []
    for path in transcript_files:
        transcript = json.loads(path.read_text(encoding="utf-8"))
        results.append(evaluate_transcript(transcript))

    stats = _corpus_stats(results)
    flagged = _flagged_sorted(results)

    report = {
        "num_files": len(results),
        "num_flagged": len(flagged),
        "corpus_stats": stats,
        "files": [asdict(r) for r in results],
        "flagged_files": [asdict(r) for r in flagged],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote quality report to %s", output_path)

    logger.info("=== Corpus-wide stats (%d files) ===", len(results))
    for field_name, field_stats in stats.items():
        logger.info(
            "%-28s mean=%.3f median=%.3f p10=%.3f p90=%.3f min=%.3f max=%.3f",
            field_name,
            field_stats["mean"],
            field_stats["median"],
            field_stats["p10"],
            field_stats["p90"],
            field_stats["min"],
            field_stats["max"],
        )

    logger.info("=== %d/%d files flagged for manual review (worst %d shown) ===", len(flagged), len(results), min(top_n, len(flagged)))
    for r in flagged[:top_n]:
        conf = f"{r.avg_word_confidence:.2f}" if r.avg_word_confidence is not None else "n/a"
        logger.info(
            "%s | conf=%s wps=%.2f coverage=%.2f repetition=%.2f | %s",
            r.video_id,
            conf,
            r.words_per_second,
            r.speech_coverage_ratio,
            r.repetition_score,
            ", ".join(r.flags),
        )

    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    run(args.transcripts_dir, args.output, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
