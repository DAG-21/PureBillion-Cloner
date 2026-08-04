"""
Phase 3 (Transcription) CLI: transcribes raw audio with faster-whisper.

Scans data/raw/audio/ for audio files and transcribes each one not already
transcribed into data/transcripts/. The whisper model is loaded lazily, so
--dry-run works without the model weights or a GPU present -- useful for
validating the file list on one machine before running the real batch job
on another (e.g. a GPU box).

Usage:
    python -m src.transcription.transcribe [options]

Input:  data/raw/audio/<id>.<ext>
Output: data/transcripts/<id>.json, data/transcripts/transcription_history.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.transcription.config import DEFAULT_CONFIG_PATH, _resolve_language, load_config
from src.transcription.logging_setup import configure_logging
from src.transcription.transcriber import Transcriber

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe.py",
        description="Transcribe raw audio files with faster-whisper for Phase 3.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the transcription config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the input audio directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output transcripts directory.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Override the history CSV file (independent of --output-dir).",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Override the faster-whisper model size (e.g. large-v3, distil-large-v3, medium).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Override the inference device.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Override the ctranslate2 compute type (default, float16, int8, int8_float16, float32).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override the forced language code (e.g. en), or 'auto' to detect it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List audio files that would be transcribed, without transcribing them.",
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
        config.input.audio_dir = args.input_dir
    if args.output_dir is not None:
        config.output.transcripts_dir = args.output_dir
    if args.history_file is not None:
        config.output.history_file = args.history_file
    if args.model_size is not None:
        config.model.size = args.model_size
    if args.device is not None:
        config.model.device = args.device
    if args.compute_type is not None:
        config.model.compute_type = args.compute_type
    if args.language is not None:
        config.model.language = _resolve_language(args.language)
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting transcription from: %s", config.input.audio_dir)

    transcriber = Transcriber(config)
    summary = transcriber.transcribe_all(dry_run=args.dry_run)

    logger.info(
        "Done. succeeded=%d failed=%d skipped=%d would_transcribe=%d (total=%d)",
        summary.succeeded,
        summary.failed,
        summary.skipped,
        summary.would_transcribe,
        summary.total,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
