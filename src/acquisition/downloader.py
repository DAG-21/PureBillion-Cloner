"""Core yt-dlp wrapper: resolves URLs, downloads videos, and records results.

Handles a single video URL, a playlist URL, or a channel URL identically -
each is flat-resolved into a list of video entries, and every entry not
already downloaded is fetched with retry/backoff.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from yt_dlp import YoutubeDL

from src.acquisition.config import AcquisitionConfig
from src.acquisition.history import DownloadHistory, HistoryRecord, now_iso
from src.acquisition.metadata import extract_metadata, save_metadata

logger = logging.getLogger(__name__)

_RATE_UNIT_MULTIPLIERS = {"K": 1024, "M": 1024**2, "G": 1024**3}


def _parse_rate_limit(value: str) -> int:
    """Parse a rate limit like ``"5M"`` or ``"800K"`` into bytes/second."""
    value = value.strip().upper()
    unit = value[-1] if value and value[-1] in _RATE_UNIT_MULTIPLIERS else None
    if unit:
        return int(float(value[:-1]) * _RATE_UNIT_MULTIPLIERS[unit])
    return int(float(value))


class _YtDlpLogger:
    """Routes yt-dlp's internal logging through the standard ``logging`` module."""

    def __init__(self, target: logging.Logger) -> None:
        self._target = target

    def debug(self, msg: str) -> None:
        if msg.startswith("[debug] "):
            self._target.debug(msg)
        else:
            self._target.info(msg)

    def info(self, msg: str) -> None:
        self._target.info(msg)

    def warning(self, msg: str) -> None:
        self._target.warning(msg)

    def error(self, msg: str) -> None:
        self._target.error(msg)


@dataclass(slots=True)
class DownloadSummary:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    would_download: int = 0

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.skipped + self.would_download


class VideoDownloader:
    """Downloads videos and metadata for a single video, playlist, or channel URL."""

    def __init__(self, config: AcquisitionConfig) -> None:
        self.config = config
        self.config.output.videos_dir.mkdir(parents=True, exist_ok=True)
        self.config.output.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config.output.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.history = DownloadHistory(config.output.history_file)
        self._progress_bars: Dict[str, tqdm] = {}

    # -- URL resolution ---------------------------------------------------

    def resolve_entries(self, url: str) -> List[Dict[str, Any]]:
        """Flat-extract ``url`` into a list of video entries.

        Works uniformly for a single video, a playlist, or a channel URL:
        yt-dlp always returns either a single info dict or one with an
        ``entries`` list.
        """
        opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "ignoreerrors": True,
            "logger": _YtDlpLogger(logger),
        }
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if info is None:
            logger.error("Could not resolve any videos from URL: %s", url)
            return []

        entries = info["entries"] if "entries" in info else [info]
        entries = [e for e in entries if e]  # drop failed/unavailable entries

        max_items = self.config.playlist.max_items
        if max_items is not None:
            entries = entries[:max_items]

        return entries

    # -- yt-dlp options -----------------------------------------------------

    def _mode(self) -> str:
        """Return "audio" or "video" -- the download's identity for dedup/history."""
        return "audio" if self.config.download.audio_only else "video"

    def _output_dir(self) -> Path:
        """Audio-only downloads go to audio_dir; video+audio downloads go to videos_dir."""
        return self.config.output.audio_dir if self.config.download.audio_only else self.config.output.videos_dir

    def _format_selector(self) -> str:
        if self.config.download.audio_only:
            if self.config.download.quality in ("lowest", "worst"):
                return "worstaudio/worst"
            return "bestaudio/best"
        height = self.config.download.quality
        return f"bestvideo[height<={height}]+bestaudio/best[height<={height}]"

    def _base_ydl_opts(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            "format": self._format_selector(),
            "merge_output_format": self.config.download.container,
            "outtmpl": str(self._output_dir() / "%(id)s.%(ext)s"),
            "noplaylist": True,  # we resolve playlists ourselves and download one video at a time
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": self.config.network.socket_timeout,
            "retries": self.config.retry.max_retries,
            "fragment_retries": self.config.retry.max_retries,
            "concurrent_fragment_downloads": self.config.download.concurrent_fragments,
            "logger": _YtDlpLogger(logger),
            "progress_hooks": [self._progress_hook],
        }
        if self.config.download.rate_limit:
            opts["ratelimit"] = _parse_rate_limit(self.config.download.rate_limit)
        if self.config.network.cookies_file:
            opts["cookiefile"] = str(self.config.network.cookies_file)
        return opts

    def _progress_hook(self, d: Dict[str, Any]) -> None:
        status = d.get("status")
        filename = d.get("filename", "")

        if status == "downloading":
            bar = self._progress_bars.get(filename)
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if bar is None:
                bar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=Path(filename).name if filename else "downloading",
                    leave=False,
                )
                self._progress_bars[filename] = bar
            if total and bar.total != total:
                bar.total = total
            bar.n = downloaded
            bar.refresh()
        elif status in ("finished", "error"):
            bar = self._progress_bars.pop(filename, None)
            if bar is not None:
                bar.close()

    # -- output path resolution ---------------------------------------------

    def _resolve_output_path(self, info: Dict[str, Any]) -> Path:
        requested = info.get("requested_downloads")
        if requested:
            return Path(requested[0]["filepath"])
        ext = info.get("ext", self.config.download.container)
        return self._output_dir() / f"{info['id']}.{ext}"

    def _already_on_disk(self, video_id: str, mode: str) -> bool:
        output_dir = self.config.output.audio_dir if mode == "audio" else self.config.output.videos_dir
        return any(output_dir.glob(f"{video_id}.*"))

    # -- download loop --------------------------------------------------

    def download_url(self, url: str, dry_run: bool = False) -> DownloadSummary:
        """Download every video resolved from ``url``, skipping ones already fetched."""
        entries = self.resolve_entries(url)
        logger.info("Resolved %d video(s) from %s", len(entries), url)

        mode = self._mode()
        summary = DownloadSummary()
        for entry in tqdm(entries, desc="Videos", unit="video"):
            video_id = entry.get("id")
            if not video_id:
                logger.warning("Skipping entry with no video id: %s", entry.get("title"))
                continue

            title = entry.get("title") or video_id
            video_url = (
                entry.get("webpage_url")
                or entry.get("url")
                or f"https://www.youtube.com/watch?v={video_id}"
            )

            if self.history.is_downloaded(video_id, mode) or self._already_on_disk(video_id, mode):
                logger.info("Skipping already-downloaded video: %s (%s, mode=%s)", video_id, title, mode)
                summary.skipped += 1
                continue

            if dry_run:
                logger.info("[dry-run] Would download: %s (%s)", video_id, title)
                summary.would_download += 1
                continue

            self._download_single(video_id, video_url, title, mode, summary)

        return summary

    def _download_single(
        self, video_id: str, video_url: str, title: str, mode: str, summary: DownloadSummary
    ) -> None:
        max_retries = self.config.retry.max_retries
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                with YoutubeDL(self._base_ydl_opts()) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                if info is None:
                    raise RuntimeError("yt-dlp returned no info for this video")

                metadata = extract_metadata(info)
                save_metadata(metadata, self.config.output.metadata_dir)
                output_path = self._resolve_output_path(info)

                self.history.record(
                    HistoryRecord(
                        video_id=video_id,
                        title=title,
                        url=video_url,
                        status="success",
                        mode=mode,
                        output_path=str(output_path),
                        timestamp=now_iso(),
                    )
                )
                summary.succeeded += 1
                logger.info("Downloaded %s (%s) -> %s", video_id, title, output_path)
                return
            except Exception as exc:  # yt-dlp raises several exception types across versions
                last_error = exc
                logger.warning(
                    "Attempt %d/%d failed for %s (%s): %s",
                    attempt,
                    max_retries,
                    video_id,
                    title,
                    exc,
                )
                if attempt < max_retries:
                    delay = self.config.retry.retry_delay_seconds * (
                        self.config.retry.backoff_factor ** (attempt - 1)
                    )
                    logger.info("Retrying %s in %.1fs...", video_id, delay)
                    time.sleep(delay)

        logger.error("Giving up on %s (%s) after %d attempt(s)", video_id, title, max_retries)
        self.history.record(
            HistoryRecord(
                video_id=video_id,
                title=title,
                url=video_url,
                status="failed",
                mode=mode,
                error=str(last_error) if last_error else "unknown error",
                timestamp=now_iso(),
            )
        )
        summary.failed += 1
