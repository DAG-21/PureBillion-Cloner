"""
Exports each transcript's plain text into its own .txt file, stripped of
the segment/word/timestamp structure -- a quick-to-read companion to
data/transcripts/, not a pipeline stage in its own right (no config/history
tracking; cheap enough to just regenerate).

Usage:
    python -m src.transcription.export_text [options]

Input:  data/transcripts/<id>.json
Output: data/transcripts_txt/<id>.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "transcripts_txt"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_text.py",
        description="Export transcript JSON 'text' fields as plain .txt files.",
    )
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if the .txt already exists.")
    return parser.parse_args(argv)


def run(transcripts_dir: Path, output_dir: Path, overwrite: bool) -> None:
    transcript_files = sorted(transcripts_dir.glob("*.json"))
    logger.info("Found %d transcript(s) in %s", len(transcript_files), transcripts_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for path in transcript_files:
        out_path = output_dir / f"{path.stem}.txt"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        transcript = json.loads(path.read_text(encoding="utf-8"))
        text = transcript.get("text", "").strip()
        out_path.write_text(text + "\n", encoding="utf-8")
        written += 1

    logger.info("Done. written=%d skipped_existing=%d (total=%d)", written, skipped, len(transcript_files))


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run(args.transcripts_dir, args.output_dir, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
