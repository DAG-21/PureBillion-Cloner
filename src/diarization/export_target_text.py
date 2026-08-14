"""
Combines every file's isolated target-speaker (Sadhguru) text with a topic
line into one plain-text deliverable -- an ad-hoc export for handing the
corpus to someone outside the pipeline, not a pipeline stage in its own
right (no config wiring, cheap enough to just regenerate).

Topic comes from the real YouTube title (looked up via yt-dlp against the
video_id, metadata-only -- no download). If a lookup fails (deleted/private
video, network hiccup after retries), falls back to a topic line derived
from the first sentence of the isolated speech itself, so every entry still
gets a topic even if the lookup can't succeed.

Looked-up titles are cached to disk (--cache-file) so a rerun after an
interruption doesn't re-fetch videos already resolved.

Usage:
    python -m src.diarization.export_target_text [options]

Input:  data/diarized/<id>.json  (needs target_speaker_text)
Output: data/sadhguru_topics_and_text.txt (single combined file)
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIARIZED_DIR = PROJECT_ROOT / "data" / "diarized"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "sadhguru_topics_and_text.txt"
DEFAULT_CACHE_FILE = PROJECT_ROOT / "data" / "video_title_cache.json"

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def derive_topic_from_text(text: str, max_words: int = 15) -> str:
    text = text.strip()
    if not text:
        return "(no speech isolated)"
    first_sentence = SENTENCE_SPLIT_RE.split(text, maxsplit=1)[0]
    words = first_sentence.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return first_sentence


def fetch_title(video_id: str, retries: int = 2, timeout: int = 30) -> Optional[str]:
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download", "--no-warnings",
        "--print", "%(title)s",
        url,
    ]
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Title fetch timed out for %s (attempt %d/%d)", video_id, attempt, retries)
            continue
        if result.returncode == 0 and result.stdout.strip():
            lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
            if lines:
                return lines[-1].strip()
        logger.warning(
            "Title fetch failed for %s (attempt %d/%d): %s",
            video_id, attempt, retries, result.stderr.strip().splitlines()[-1:] or "unknown error",
        )
        time.sleep(1.0)
    return None


def load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return {}


def save_cache(cache_file: Path, cache: dict) -> None:
    cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_target_text.py",
        description="Combine all diarized target-speaker text + topic lines into one file.",
    )
    parser.add_argument("--diarized-dir", type=Path, default=DEFAULT_DIARIZED_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--no-title-lookup", action="store_true",
                         help="Skip yt-dlp title lookups entirely; always derive topic from text.")
    parser.add_argument("--request-delay", type=float, default=0.5,
                         help="Seconds to sleep between yt-dlp lookups (politeness/rate-limit avoidance).")
    return parser.parse_args(argv)


def run(diarized_dir: Path, output_file: Path, cache_file: Path,
        no_title_lookup: bool, request_delay: float) -> None:
    diarized_files = sorted(diarized_dir.glob("*.json"))
    logger.info("Found %d diarized file(s) in %s", len(diarized_files), diarized_dir)

    cache = {} if no_title_lookup else load_cache(cache_file)
    entries = []
    fetched = 0
    from_cache = 0
    derived = 0

    for i, path in enumerate(diarized_files, start=1):
        data = json.loads(path.read_text(encoding="utf-8"))
        video_id = data.get("video_id", path.stem)
        text = data.get("target_speaker_text", "").strip()

        title = None
        if not no_title_lookup:
            if video_id in cache:
                title = cache[video_id]
                from_cache += 1
            else:
                title = fetch_title(video_id)
                cache[video_id] = title
                fetched += 1
                time.sleep(request_delay)

        if title:
            topic = title
        else:
            topic = derive_topic_from_text(text)
            derived += 1

        entries.append((video_id, topic, text))
        logger.info("[%d/%d] %s -> topic: %s", i, len(diarized_files), video_id, topic)

        if not no_title_lookup and i % 20 == 0:
            save_cache(cache_file, cache)

    if not no_title_lookup:
        save_cache(cache_file, cache)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for i, (video_id, topic, text) in enumerate(entries):
            if i > 0:
                f.write("\n" + ("-" * 80) + "\n\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Text: {text}\n")

    logger.info(
        "Done. entries=%d titles_fetched=%d titles_from_cache=%d topics_derived_from_text=%d -> %s",
        len(entries), fetched, from_cache, derived, output_file,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run(args.diarized_dir, args.output_file, args.cache_file, args.no_title_lookup, args.request_delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
