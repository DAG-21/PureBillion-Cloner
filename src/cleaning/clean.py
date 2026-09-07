"""
Phase 5 (Cleaning) CLI: normalizes the isolated target-speaker text produced
by Phase 4.

Input is the single combined export file (data/sadhguru_topics_and_text.txt,
produced by src/diarization/export_target_text.py), not the per-file
data/diarized/<id>.json outputs -- confirmed by the user as the correct
source file to build this stage on (2026-09-07), superseding this stage's
original per-file design. See src/cleaning/cleaner.py's module docstring
for exactly what cleanup is (and deliberately isn't) applied.

Usage:
    python -m src.cleaning.clean [options]

Input:  data/sadhguru_topics_and_text.txt
Output: data/cleaned/sadhguru_cleaned.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.cleaning.cleaner import Cleaner
from src.cleaning.config import DEFAULT_CONFIG_PATH, load_config
from src.cleaning.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clean.py",
        description="Clean and normalize isolated target-speaker text for Phase 5.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to the cleaning config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--export-file", type=Path, default=None,
        help="Override the input combined export file.",
    )
    parser.add_argument(
        "--diarized-dir", type=Path, default=None,
        help="Override the diarized dir used for best-effort video_id recovery.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Override the output cleaned JSONL file.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many entries would be cleaned, without writing output.",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override the configured logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.export_file is not None:
        config.input.export_file = args.export_file
    if args.diarized_dir is not None:
        config.input.diarized_dir = args.diarized_dir
    if args.output_file is not None:
        config.output.cleaned_file = args.output_file
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting cleaning from: %s", config.input.export_file)

    cleaner = Cleaner(config)
    summary = cleaner.clean_all(dry_run=args.dry_run)

    logger.info(
        "Done. total=%d cleaned=%d would_clean=%d video_ids_recovered=%s",
        summary.total, summary.cleaned, summary.would_clean, summary.video_ids_recovered,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
