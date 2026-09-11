"""
Phase 5-b (Genre Classification) CLI: adds a broad subject-matter `genre`
field to each of Phase 5's cleaned entries.

Usage:
    python -m src.genre_classification.classify [options]

Input:  data/cleaned/sadhguru_cleaned.jsonl (Phase 5 output)
Output: data/cleaned/sadhguru_cleaned.jsonl (same file by default, `genre`
        field added to every entry)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.genre_classification.classifier import classify_all
from src.genre_classification.config import DEFAULT_CONFIG_PATH, load_config
from src.genre_classification.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="classify.py",
        description="Add a genre field to Phase 5's cleaned entries.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to the genre classification config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--cleaned-file", type=Path, default=None,
        help="Override the input cleaned JSONL file.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Override the output JSONL file (default: same as input, in place).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the genre distribution without writing output.",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Override the configured logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.cleaned_file is not None:
        config.input.cleaned_file = args.cleaned_file
    if args.output_file is not None:
        config.output.cleaned_file = args.output_file
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting genre classification from: %s", config.input.cleaned_file)

    summary = classify_all(
        config.input.cleaned_file, config.output.cleaned_file, dry_run=args.dry_run
    )

    logger.info("Done. total=%d distinct_genres=%d", summary.total, len(summary.genre_counts))
    for genre, count in sorted(summary.genre_counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-45s %d", genre, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
