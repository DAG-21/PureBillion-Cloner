"""Configuration loading for the cleaning stage.

Reads ``configs/cleaning.yaml`` into typed, validated dataclasses so the
rest of the cleaning package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# src/cleaning/config.py -> src/cleaning -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cleaning.yaml"


@dataclass(slots=True)
class InputConfig:
    export_file: Path
    diarized_dir: Path


@dataclass(slots=True)
class OutputConfig:
    cleaned_file: Path


@dataclass(slots=True)
class CleaningRulesConfig:
    disfluencies: List[str] = field(default_factory=lambda: ["um", "uh", "erm"])
    sentence_case: bool = True


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class CleaningConfig:
    input: InputConfig
    output: OutputConfig
    cleaning: CleaningRulesConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> CleaningConfig:
    """Load and validate the cleaning config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Cleaning config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    cleaning_raw = raw.get("cleaning") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        export_file=_resolve_path(
            input_raw.get("export_file", "data/sadhguru_topics_and_text.txt")
        ),  # type: ignore[arg-type]
        diarized_dir=_resolve_path(
            input_raw.get("diarized_dir", "data/diarized")
        ),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        cleaned_file=_resolve_path(
            output_raw.get("cleaned_file", "data/cleaned/sadhguru_cleaned.jsonl")
        ),  # type: ignore[arg-type]
    )
    cleaning = CleaningRulesConfig(
        disfluencies=list(cleaning_raw.get("disfluencies", ["um", "uh", "erm"])),
        sentence_case=bool(cleaning_raw.get("sentence_case", True)),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return CleaningConfig(
        input=input_cfg,
        output=output,
        cleaning=cleaning,
        logging=logging_cfg,
    )
