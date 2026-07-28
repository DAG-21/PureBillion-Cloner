"""
Phase 1 (Data Collection) CLI: downloads YouTube videos and metadata via yt-dlp.

Accepts a single video, playlist, or channel URL, saves videos to
data/raw/videos/, saves per-video metadata JSON to data/raw/metadata/, and
tracks every attempt in data/raw/download_history.csv. Already-downloaded
videos are skipped automatically.

Usage:
    python -m src.acquisition.download <url> [options]

Input: a single video, playlist, or channel URL (positional CLI argument)
Output: data/raw/videos/<id>.<ext>, data/raw/metadata/<id>.json,
        data/raw/download_history.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.acquisition.config import DEFAULT_CONFIG_PATH, load_config
from src.acquisition.downloader import VideoDownloader
from src.acquisition.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download.py",
        description="Download YouTube video(s) and metadata for Phase 1 data collection.",
    )
    parser.add_argument("url", help="Video, playlist, or channel URL to download.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the acquisition config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        help="Override the max video height, e.g. 720, 1080.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        default=False,
        help="Download audio only instead of video+audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the videos output directory.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Limit the number of videos processed from a playlist/channel.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List videos that would be downloaded, without downloading them.",
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

    if args.quality is not None:
        config.download.quality = args.quality
    if args.audio_only:
        config.download.audio_only = True
    if args.output_dir is not None:
        config.output.videos_dir = args.output_dir
    if args.max_items is not None:
        config.playlist.max_items = args.max_items
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting data collection for URL: %s", args.url)

    downloader = VideoDownloader(config)
    summary = downloader.download_url(args.url, dry_run=args.dry_run)

    logger.info(
        "Done. succeeded=%d failed=%d skipped=%d would_download=%d (total=%d)",
        summary.succeeded,
        summary.failed,
        summary.skipped,
        summary.would_download,
        summary.total,
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    sys.exit(main())
