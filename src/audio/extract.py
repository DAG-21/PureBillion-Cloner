"""
Phase 2 (Audio Extraction) CLI: extracts audio tracks from raw videos via ffmpeg.

Scans data/raw/videos/ for video files and extracts each one's audio track
into data/raw/audio/. A video whose audio already exists there (whether
extracted previously or downloaded directly via
`python -m src.acquisition.download --audio-only`) is skipped, so reruns
are idempotent.

Usage:
    python -m src.audio.extract [options]

Input:  data/raw/videos/<id>.<ext>
Output: data/raw/audio/<id>.<format>, data/raw/audio_extract_history.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.audio.config import DEFAULT_CONFIG_PATH, load_config
from src.audio.extractor import AudioExtractor
from src.audio.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="extract.py",
        description="Extract audio tracks from raw video files for Phase 2.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the audio config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the input videos directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output audio directory.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["wav", "flac", "mp3"],
        help="Override the output audio format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List videos that would be extracted, without extracting them.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override the configured logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)

    if args.input_dir is not None:
        config.input.videos_dir = args.input_dir
    if args.output_dir is not None:
        config.output.audio_dir = args.output_dir
    if args.format is not None:
        config.extract.format = args.format
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting audio extraction from: %s", config.input.videos_dir)

    extractor = AudioExtractor(config)
    summary = extractor.extract_all(dry_run=args.dry_run)

    logger.info(
        "Done. succeeded=%d failed=%d skipped=%d would_extract=%d (total=%d)",
        summary.succeeded,
        summary.failed,
        summary.skipped,
        summary.would_extract,
        summary.total,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
