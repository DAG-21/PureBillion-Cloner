"""
Phase 6 (Chunking) CLI: splits Phase 5's cleaned per-video text into
semantically coherent chunks (LlamaIndex's SemanticSplitterNodeParser,
boundary detection via HuggingFace's free Inference API) for the RAG vector
store.

Usage:
    python -m src.chunking.chunk [options]

Input:  data/cleaned/sadhguru_cleaned.jsonl
Output: data/chunks/sadhguru_chunks.jsonl

Requires HF_TOKEN in .env (not needed for --dry-run).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.chunking.chunker import Chunker
from src.chunking.config import DEFAULT_CONFIG_PATH, load_config
from src.chunking.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chunk.py",
        description="Split cleaned per-video text into semantic chunks for Phase 6.",
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to the chunking config YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--cleaned-file", type=Path, default=None,
        help="Override the input cleaned JSONL file.",
    )
    parser.add_argument(
        "--output-file", type=Path, default=None,
        help="Override the output chunks JSONL file.",
    )
    parser.add_argument(
        "--embedding-model", type=str, default=None,
        help="Override the HF model used for semantic chunk-boundary detection.",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None,
        help="Override the number of sentences grouped together when evaluating similarity.",
    )
    parser.add_argument(
        "--breakpoint-percentile-threshold", type=int, default=None,
        help="Override the dissimilarity percentile that triggers a chunk split.",
    )
    parser.add_argument(
        "--min-chunk-words", type=int, default=None,
        help="Override the minimum words per chunk (smaller chunks get merged into a neighbor).",
    )
    parser.add_argument(
        "--max-chunk-words", type=int, default=None,
        help="Override the maximum words per chunk (larger chunks get split back down).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many videos would be chunked, without writing output.",
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
        config.output.chunks_file = args.output_file
    if args.embedding_model is not None:
        config.chunking.embedding_model = args.embedding_model
    if args.buffer_size is not None:
        config.chunking.buffer_size = args.buffer_size
    if args.breakpoint_percentile_threshold is not None:
        config.chunking.breakpoint_percentile_threshold = args.breakpoint_percentile_threshold
    if args.min_chunk_words is not None:
        config.chunking.min_chunk_words = args.min_chunk_words
    if args.max_chunk_words is not None:
        config.chunking.max_chunk_words = args.max_chunk_words
    if args.log_level is not None:
        config.logging.level = args.log_level

    configure_logging(config.logging.level, config.logging.log_file)
    logger.info("Starting chunking from: %s", config.input.cleaned_file)

    chunker = Chunker(config)
    summary = chunker.chunk_all(dry_run=args.dry_run)

    logger.info(
        "Done. total_videos=%d total_chunks=%d would_chunk_videos=%d skipped_empty=%d",
        summary.total_videos, summary.total_chunks,
        summary.would_chunk_videos, summary.skipped_empty,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
