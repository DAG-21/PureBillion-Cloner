"""Configuration loading for the chunking stage.

Reads ``configs/chunking.yaml`` into typed, validated dataclasses so the
rest of the chunking package never touches raw dicts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# src/chunking/config.py -> src/chunking -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "chunking.yaml"


@dataclass(slots=True)
class InputConfig:
    cleaned_file: Path


@dataclass(slots=True)
class OutputConfig:
    chunks_file: Path


@dataclass(slots=True)
class ChunkingParamsConfig:
    embedding_model: str = "BAAI/bge-m3"
    buffer_size: int = 1
    breakpoint_percentile_threshold: int = 95
    embed_batch_size: int = 32
    min_chunk_words: int = 10
    max_chunk_words: int = 512


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None


@dataclass(slots=True)
class ChunkingConfig:
    input: InputConfig
    output: OutputConfig
    chunking: ChunkingParamsConfig
    logging: LoggingConfig


def _resolve_path(value: Optional[str]) -> Optional[Path]:
    """Resolve a config path relative to the project root, unless already absolute."""
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(config_path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> ChunkingConfig:
    """Load and validate the chunking config YAML at ``config_path``."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Chunking config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    input_raw = raw.get("input") or {}
    output_raw = raw.get("output") or {}
    chunking_raw = raw.get("chunking") or {}
    logging_raw = raw.get("logging") or {}

    input_cfg = InputConfig(
        cleaned_file=_resolve_path(
            input_raw.get("cleaned_file", "data/cleaned/sadhguru_cleaned.jsonl")
        ),  # type: ignore[arg-type]
    )
    output = OutputConfig(
        chunks_file=_resolve_path(
            output_raw.get("chunks_file", "data/chunks/sadhguru_chunks.jsonl")
        ),  # type: ignore[arg-type]
    )
    chunking = ChunkingParamsConfig(
        embedding_model=str(
            chunking_raw.get("embedding_model", "BAAI/bge-m3")
        ),
        buffer_size=int(chunking_raw.get("buffer_size", 1)),
        breakpoint_percentile_threshold=int(
            chunking_raw.get("breakpoint_percentile_threshold", 95)
        ),
        embed_batch_size=int(chunking_raw.get("embed_batch_size", 32)),
        min_chunk_words=int(chunking_raw.get("min_chunk_words", 10)),
        max_chunk_words=int(chunking_raw.get("max_chunk_words", 512)),
    )
    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        log_file=_resolve_path(logging_raw.get("log_file")),
    )

    return ChunkingConfig(
        input=input_cfg,
        output=output,
        chunking=chunking,
        logging=logging_cfg,
    )
