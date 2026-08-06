"""
Fetches YouTube's manually-uploaded (non-auto-generated) English subtitles
for every video that has a Phase 3 transcript, as reference text for WER/CER
scoring in compute_wer.py.

Deliberately uses ``writesubtitles`` (creator-uploaded "subtitles"), not
``writeautomaticsub`` (YouTube's own ASR-generated "automatic captions") --
comparing Whisper against another ASR system's output wouldn't measure
accuracy, just inter-ASR agreement. Not every video has a manual subtitle
track; those are skipped and counted, not treated as failures.

Usage:
    python -m eval.asr.fetch_references [options]

Input:  data/transcripts/<id>.json (for the video_id list)
Output: data/references/<id>.vtt (raw), data/references/<id>.txt (parsed)
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from yt_dlp import YoutubeDL

from eval.asr.vtt_parser import parse_vtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "transcripts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "references"
DEFAULT_SUB_LANG = "en"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch_references.py",
        description="Fetch manual YouTube subtitles as WER reference text.",
    )
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sub-lang", type=str, default=DEFAULT_SUB_LANG)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _fetch_one(video_id: str, sub_lang: str, tmp_dir: Path) -> Optional[Path]:
    """Download the manual subtitle track for one video, if it has one.
    Returns the downloaded .vtt path, or None if no manual subtitle exists."""
    outtmpl = str(tmp_dir / f"{video_id}.%(ext)s")
    opts = {
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": [sub_lang],
        "subtitlesformat": "vtt",
        "skip_download": True,
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "logtostderr": True,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

    vtt_path = tmp_dir / f"{video_id}.{sub_lang}.vtt"
    return vtt_path if vtt_path.exists() else None


def run(transcripts_dir: Path, output_dir: Path, sub_lang: str) -> None:
    video_ids = sorted(p.stem for p in transcripts_dir.glob("*.json"))
    logger.info("Found %d transcript(s) to fetch references for", len(video_ids))

    output_dir.mkdir(parents=True, exist_ok=True)

    fetched = 0
    skipped_existing = 0
    no_reference = 0
    failed = 0

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for video_id in video_ids:
            out_txt = output_dir / f"{video_id}.txt"
            if out_txt.exists():
                skipped_existing += 1
                continue

            try:
                vtt_path = _fetch_one(video_id, sub_lang, tmp_dir)
            except Exception as exc:
                logger.error("Failed to fetch subtitles for %s: %s", video_id, exc)
                failed += 1
                continue

            if vtt_path is None:
                logger.info("No manual '%s' subtitle available for %s", sub_lang, video_id)
                no_reference += 1
                continue

            text = parse_vtt(vtt_path)
            out_txt.write_text(text, encoding="utf-8")
            (output_dir / f"{video_id}.vtt").write_text(
                vtt_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            vtt_path.unlink()
            fetched += 1
            logger.info("Fetched reference for %s (%d words)", video_id, len(text.split()))

    logger.info(
        "Done. fetched=%d skipped_existing=%d no_reference=%d failed=%d (total=%d)",
        fetched,
        skipped_existing,
        no_reference,
        failed,
        len(video_ids),
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(args.log_level.upper())
    run(args.transcripts_dir, args.output_dir, args.sub_lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
