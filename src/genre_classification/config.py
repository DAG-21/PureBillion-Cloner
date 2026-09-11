"""Configuration loading for the genre classification stage.

Reads ``configs/genre_classification.yaml`` into typed, validated
dataclasses so the rest of the package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/genre_classification/config.py -> src/genre_classification -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "genre_classification.yaml"


@dataclass(slots=True)
class InputConfig:
    cleaned_file: Path


@dataclass(slots=True)
class OutputConfig:
    cleaned_file: Path


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class GenreClassificationConfig:
    input: InputConfig
    output: OutputConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(
    config_path: Union[Path, str] = DEFAULT_CONFIG_PATH,
) -> GenreClassificationConfig:
    """Load and validate the genre classification config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Genre classification config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        cleaned_file=_resolve_path(
            input_raw.get("cleaned_file", "data/cleaned/sadhguru_cleaned.jsonl")
        ),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        cleaned_file=_resolve_path(
            output_raw.get("cleaned_file", "data/cleaned/sadhguru_cleaned.jsonl")
        ),  # type: ignore[arg-type]
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return GenreClassificationConfig(input=input_cfg, output=output, logging=logging_cfg)
