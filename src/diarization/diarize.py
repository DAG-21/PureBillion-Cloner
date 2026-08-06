"""
Phase 4 (Diarization) CLI: separates speakers with pyannote.audio and
isolates the target speaker.

Scans data/raw/audio/ for audio files with a matching Phase 3 transcript in
data/transcripts/ and diarizes each one not already diarized into
data/diarized/. The pyannote pipeline is loaded lazily, so --dry-run works
without HF_TOKEN, a GPU, or the model weights present -- useful for
validating the file list before running the real batch job.

Usage:
    python -m src.diarization.diarize [options]

Input:  data/raw/audio/<id>.<ext>, data/transcripts/<id>.json
Output: data/diarized/<id>.json, data/diarized/diarization_history.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.diarization.config import DEFAULT_CONFIG_PATH, load_config
from src.diarization.diarizer import Diarizer
from src.diarization.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="diarize.py",
        description="Diarize speakers and isolate the target speaker for Phase 4.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the diarization config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Override the input audio directory.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Override the input transcripts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output diarized directory.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Override the history CSV file (independent of --output-dir).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Override the inference device.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List audio files that would be diarized, without diarizing them.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override the configured logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()  # populates HF_TOKEN (and friends) from .env, if present

    args = parse_args(argv)
    config = load_config(args.config)

    if args.audio_dir is not None:
        config.input.audio_dir = args.audio_dir
    if args.transcripts_dir is not None:
        config.input.transcripts_dir = args.transcripts_dir
    if args.output_dir is not None:
        config.output.diarized_dir = args.output_dir
    if args.history_file is not None:
        config.output.history_file = args.history_file
    if args.device is not None:
        config.model.device = args.device
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting diarization from: %s", config.input.audio_dir)

    diarizer = Diarizer(config)
    summary = diarizer.diarize_all(dry_run=args.dry_run)

    logger.info(
        "Done. succeeded=%d failed=%d skipped=%d missing_transcript=%d would_diarize=%d (total=%d)",
        summary.succeeded,
        summary.failed,
        summary.skipped,
        summary.missing_transcript,
        summary.would_diarize,
        summary.total,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
