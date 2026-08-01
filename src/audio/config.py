"""Configuration loading for the audio extraction stage.

Reads ``configs/audio.yaml`` into typed, validated dataclasses so the rest
of the audio package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/audio/config.py -> src/audio -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "audio.yaml"


@dataclass(slots=True)
class InputConfig:
    videos_dir: Path


@dataclass(slots=True)
class OutputConfig:
    audio_dir: Path
    history_file: Path


@dataclass(slots=True)
class ExtractConfig:
    format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class AudioConfig:
    input: InputConfig
    output: OutputConfig
    extract: ExtractConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> AudioConfig:
    """Load and validate the audio extraction config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Audio extraction config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    extract_raw = raw.get("extract") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        videos_dir=_resolve_path(input_raw.get("videos_dir", "data/raw/videos")),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        audio_dir=_resolve_path(output_raw.get("audio_dir", "data/raw/audio")),  # type: ignore[arg-type]
        history_file=_resolve_path(
            output_raw.get("history_file", "data/raw/audio_extract_history.csv")
        ),  # type: ignore[arg-type]
    )
    extract = ExtractConfig(
        format=str(extract_raw.get("format", "wav")),
        sample_rate=int(extract_raw.get("sample_rate", 16000)),
        channels=int(extract_raw.get("channels", 1)),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return AudioConfig(
        input=input_cfg,
        output=output,
        extract=extract,
        logging=logging_cfg,
    )
