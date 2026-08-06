"""Configuration loading for the diarization stage.

Reads ``configs/diarization.yaml`` into typed, validated dataclasses so the
rest of the diarization package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/diarization/config.py -> src/diarization -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "diarization.yaml"


@dataclass(slots=True)
class InputConfig:
    audio_dir: Path
    transcripts_dir: Path


@dataclass(slots=True)
class OutputConfig:
    diarized_dir: Path
    history_file: Path


@dataclass(slots=True)
class ModelConfig:
    pipeline: str = "pyannote/speaker-diarization-3.1"
    device: str = "auto"


@dataclass(slots=True)
class TargetSpeakerConfig:
    strategy: str = "longest_total_duration"


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class DiarizationConfig:
    input: InputConfig
    output: OutputConfig
    model: ModelConfig
    target_speaker: TargetSpeakerConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> DiarizationConfig:
    """Load and validate the diarization config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Diarization config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    model_raw = raw.get("model") or {}
    target_speaker_raw = raw.get("target_speaker") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        audio_dir=_resolve_path(input_raw.get("audio_dir", "data/raw/audio")),  # type: ignore[arg-type]
        transcripts_dir=_resolve_path(
            input_raw.get("transcripts_dir", "data/transcripts")
        ),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        diarized_dir=_resolve_path(output_raw.get("diarized_dir", "data/diarized")),  # type: ignore[arg-type]
        history_file=_resolve_path(
            output_raw.get("history_file", "data/diarized/diarization_history.csv")
        ),  # type: ignore[arg-type]
    )
    model = ModelConfig(
        pipeline=str(model_raw.get("pipeline", "pyannote/speaker-diarization-3.1")),
        device=str(model_raw.get("device", "auto")),
    )
    target_speaker = TargetSpeakerConfig(
        strategy=str(target_speaker_raw.get("strategy", "longest_total_duration")),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return DiarizationConfig(
        input=input_cfg,
        output=output,
        model=model,
        target_speaker=target_speaker,
        logging=logging_cfg,
    )
