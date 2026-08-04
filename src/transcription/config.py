"""Configuration loading for the transcription stage.

Reads ``configs/transcription.yaml`` into typed, validated dataclasses so the
rest of the transcription package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/transcription/config.py -> src/transcription -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "transcription.yaml"


@dataclass(slots=True)
class InputConfig:
    audio_dir: Path


@dataclass(slots=True)
class OutputConfig:
    transcripts_dir: Path
    history_file: Path


@dataclass(slots=True)
class ModelConfig:
    size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "default"
    language: Optional[str] = "en"
    vad_filter: bool = True
    word_timestamps: bool = True
    beam_size: int = 5


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class TranscriptionConfig:
    input: InputConfig
    output: OutputConfig
    model: ModelConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _resolve_language(value: Optional[str]) -> Optional[str]:
    """"auto" (or empty) means let whisper detect the language -> None."""
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return value


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> TranscriptionConfig:
    """Load and validate the transcription config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Transcription config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    model_raw = raw.get("model") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        audio_dir=_resolve_path(input_raw.get("audio_dir", "data/raw/audio")),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        transcripts_dir=_resolve_path(output_raw.get("transcripts_dir", "data/transcripts")),  # type: ignore[arg-type]
        history_file=_resolve_path(
            output_raw.get("history_file", "data/transcripts/transcription_history.csv")
        ),  # type: ignore[arg-type]
    )
    model = ModelConfig(
        size=str(model_raw.get("size", "large-v3")),
        device=str(model_raw.get("device", "auto")),
        compute_type=str(model_raw.get("compute_type", "default")),
        language=_resolve_language(model_raw.get("language", "en")),
        vad_filter=bool(model_raw.get("vad_filter", True)),
        word_timestamps=bool(model_raw.get("word_timestamps", True)),
        beam_size=int(model_raw.get("beam_size", 5)),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return TranscriptionConfig(
        input=input_cfg,
        output=output,
        model=model,
        logging=logging_cfg,
    )
