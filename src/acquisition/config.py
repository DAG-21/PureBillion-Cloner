"""Configuration loading for the data acquisition stage.

Reads ``configs/acquisition.yaml`` into typed, validated dataclasses so the
rest of the acquisition package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/acquisition/config.py -> src/acquisition -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "acquisition.yaml"


@dataclass(slots=True)
class OutputConfig:
    videos_dir: Path
    audio_dir: Path
    metadata_dir: Path
    history_file: Path


@dataclass(slots=True)
class DownloadConfig:
    quality: str = "1080"
    container: str = "mp4"
    audio_only: bool = False
    rate_limit: Optional[str] = None
    concurrent_fragments: int = 4


@dataclass(slots=True)
class RetryConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    backoff_factor: float = 2.0


@dataclass(slots=True)
class NetworkConfig:
    socket_timeout: int = 30
    cookies_file: Optional[Path] = None


@dataclass(slots=True)
class PlaylistConfig:
    ignore_errors: bool = True
    max_items: Optional[int] = None


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class AcquisitionConfig:
    output: OutputConfig
    download: DownloadConfig
    retry: RetryConfig
    network: NetworkConfig
    playlist: PlaylistConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> AcquisitionConfig:
    """Load and validate the acquisition config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Acquisition config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    output_raw = raw.get("output") or {}
    download_raw = raw.get("download") or {}
    retry_raw = raw.get("retry") or {}
    network_raw = raw.get("network") or {}
    playlist_raw = raw.get("playlist") or {}
    logging_raw = raw.get("logging") or {}

    output = OutputConfig(
        videos_dir=_resolve_path(output_raw.get("videos_dir", "data/raw/videos")),  # type: ignore[arg-type]
        audio_dir=_resolve_path(output_raw.get("audio_dir", "data/raw/audio")),  # type: ignore[arg-type]
        metadata_dir=_resolve_path(output_raw.get("metadata_dir", "data/raw/metadata")),  # type: ignore[arg-type]
        history_file=_resolve_path(
            output_raw.get("history_file", "data/raw/download_history.csv")
        ),  # type: ignore[arg-type]
    )
    download = DownloadConfig(
        quality=str(download_raw.get("quality", "1080")),
        container=str(download_raw.get("format", "mp4")),
        audio_only=bool(download_raw.get("audio_only", False)),
        rate_limit=download_raw.get("rate_limit"),
        concurrent_fragments=int(download_raw.get("concurrent_fragments", 4)),
    )
    retry = RetryConfig(
        max_retries=int(retry_raw.get("max_retries", 3)),
        retry_delay_seconds=float(retry_raw.get("retry_delay_seconds", 5.0)),
        backoff_factor=float(retry_raw.get("backoff_factor", 2.0)),
    )
    network = NetworkConfig(
        socket_timeout=int(network_raw.get("socket_timeout", 30)),
        cookies_file=_resolve_path(network_raw.get("cookies_file")),
    )
    playlist = PlaylistConfig(
        ignore_errors=bool(playlist_raw.get("ignore_errors", True)),
        max_items=playlist_raw.get("max_items"),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return AcquisitionConfig(
        output=output,
        download=download,
        retry=retry,
        network=network,
        playlist=playlist,
        logging=logging_cfg,
    )
