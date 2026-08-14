"""
Phase 4 quality audit CLI: runs reference-free quality metrics over every
diarized file in data/diarized/ and flags files worth a manual listen.

See src/diarization/quality.py for why this is reference-free (no ground-
truth speaker-labeled data exists for this corpus, so DER can't be
computed) and what each metric/flag means.

Usage:
    python -m src.diarization.evaluate [options]

Input:  data/diarized/<id>.json (+ matching data/transcripts/<id>.json,
        used only for the text-retention cross-check -- missing transcripts
        don't block evaluation, retention_ratio is just left null)
Output: eval/results/diarization_quality_report.json (+ printed summary)
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

from src.diarization.quality import DiarizationQuality, evaluate_diarization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIARIZED_DIR = PROJECT_ROOT / "data" / "diarized"
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "results" / "diarization_quality_report.json"

_NUMERIC_FIELDS = (
    "num_speakers",
    "target_speaker_share",
    "target_turn_count",
    "num_target_segments",
    "avg_target_segment_seconds",
    "short_turn_ratio",
    "target_word_count",
    "target_words_per_second",
    "target_text_retention_ratio",
    "repetition_score",
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Audit Phase 4 diarization quality with reference-free metrics.",
    )
    parser.add_argument("--diarized-dir", type=Path, default=DEFAULT_DIARIZED_DIR)
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=25,
                         help="How many flagged files to print in the summary (default: %(default)s).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _transcript_word_count(transcripts_dir: Path, video_id: str) -> Optional[int]:
    path = transcripts_dir / f"{video_id}.json"
    if not path.exists():
        return None
    transcript = json.loads(path.read_text(encoding="utf-8"))
    text = transcript.get("text", "")
    return len(text.split())


def _corpus_stats(results: List[DiarizationQuality]) -> Dict[str, Dict[str, float]]:
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


def _flag_counts(results: List[DiarizationQuality]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for flag in r.flags:
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _flagged_sorted(results: List[DiarizationQuality]) -> List[DiarizationQuality]:
    flagged = [r for r in results if r.flags]
    return sorted(flagged, key=lambda r: (-len(r.flags), r.target_speaker_share))


def run(diarized_dir: Path, transcripts_dir: Path, output_path: Path, top_n: int) -> Dict[str, Any]:
    diarized_files = sorted(diarized_dir.glob("*.json"))
    logger.info("Found %d diarized file(s) in %s", len(diarized_files), diarized_dir)

    results: List[DiarizationQuality] = []
    for path in diarized_files:
        diarized = json.loads(path.read_text(encoding="utf-8"))
        video_id = diarized.get("video_id", path.stem)
        transcript_wc = _transcript_word_count(transcripts_dir, video_id)
        results.append(evaluate_diarization(diarized, transcript_wc))

    stats = _corpus_stats(results)
    flag_counts = _flag_counts(results)
    flagged = _flagged_sorted(results)

    report = {
        "num_files": len(results),
        "num_flagged": len(flagged),
        "flag_counts": flag_counts,
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
            "%-30s mean=%.3f median=%.3f p10=%.3f p90=%.3f min=%.3f max=%.3f",
            field_name,
            field_stats["mean"],
            field_stats["median"],
            field_stats["p10"],
            field_stats["p90"],
            field_stats["min"],
            field_stats["max"],
        )

    logger.info("=== Flag counts (%d files flagged of %d total) ===", len(flagged), len(results))
    for flag, count in flag_counts.items():
        logger.info("%-30s %d", flag, count)

    logger.info("=== Worst %d flagged files ===", min(top_n, len(flagged)))
    for r in flagged[:top_n]:
        retention = f"{r.target_text_retention_ratio:.2f}" if r.target_text_retention_ratio is not None else "n/a"
        logger.info(
            "%s | speakers=%d share=%.2f retention=%s frag=%.2f rep=%.2f | %s",
            r.video_id,
            r.num_speakers,
            r.target_speaker_share,
            retention,
            r.short_turn_ratio,
            r.repetition_score,
            ", ".join(r.flags),
        )

    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level)
    run(args.diarized_dir, args.transcripts_dir, args.output, args.top_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
