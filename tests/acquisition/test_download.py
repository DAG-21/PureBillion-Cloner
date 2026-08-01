"""Unit tests for src/acquisition/download.py (the CLI layer only).

VideoDownloader and load_config are mocked out everywhere, so
downloader.py's real logic (yt-dlp, network I/O) never executes here.
See tests/acquisition/test_downloader.py (not yet written) for testing
the actual download behavior.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.acquisition import download
from src.acquisition.config import (
    DEFAULT_CONFIG_PATH,
    AcquisitionConfig,
    DownloadConfig,
    LoggingConfig,
    NetworkConfig,
    OutputConfig,
    PlaylistConfig,
    RetryConfig,
)
from src.acquisition.downloader import DownloadSummary


def make_config() -> AcquisitionConfig:
    return AcquisitionConfig(
        output=OutputConfig(
            videos_dir=Path("data/raw/videos"),
            metadata_dir=Path("data/raw/metadata"),
            history_file=Path("data/raw/download_history.csv"),
        ),
        download=DownloadConfig(),
        retry=RetryConfig(),
        network=NetworkConfig(),
        playlist=PlaylistConfig(),
        logging=LoggingConfig(),
    )


# -- parse_args: pure argument parsing, no mocking needed -------------------


def test_parse_args_defaults():
    args = download.parse_args(["https://youtu.be/abc123"])
    assert args.url == "https://youtu.be/abc123"
    assert args.quality is None
    assert args.audio_only is False
    assert args.output_dir is None
    assert args.max_items is None
    assert args.dry_run is False
    assert args.log_level is None


def test_parse_args_overrides():
    args = download.parse_args(
        [
            "https://youtu.be/abc123",
            "--quality",
            "720",
            "--audio-only",
            "--output-dir",
            "custom/videos",
            "--max-items",
            "5",
            "--dry-run",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.quality == "720"
    assert args.audio_only is True
    assert args.output_dir == Path("custom/videos")
    assert args.max_items == 5
    assert args.dry_run is True
    assert args.log_level == "DEBUG"


def test_parse_args_requires_url():
    with pytest.raises(SystemExit):
        download.parse_args([])


# -- main(): does download.py build the right AcquisitionConfig object? ----
# No config *file* is written -- load_config() reads the YAML into memory,
# main() applies CLI overrides to that object, then passes it directly (by
# reference) to VideoDownloader(config). These tests assert that object is
# built correctly, without ever running downloader.py's real logic.


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_uses_default_config_path_when_not_specified(
    mock_load_config, mock_downloader_cls
):
    mock_load_config.return_value = make_config()
    mock_downloader_cls.return_value.download_url.return_value = DownloadSummary(succeeded=1)

    download.main(["https://youtu.be/abc123"])

    mock_load_config.assert_called_once_with(DEFAULT_CONFIG_PATH)


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_uses_specified_config_path(mock_load_config, mock_downloader_cls):
    mock_load_config.return_value = make_config()
    mock_downloader_cls.return_value.download_url.return_value = DownloadSummary(succeeded=1)

    download.main(["https://youtu.be/abc123", "--config", "custom/acquisition.yaml"])

    mock_load_config.assert_called_once_with(Path("custom/acquisition.yaml"))


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_leaves_unspecified_config_fields_untouched(
    mock_load_config, mock_downloader_cls
):
    """No CLI overrides given -> the config object reaches VideoDownloader as-is."""
    config = make_config()
    mock_load_config.return_value = config
    mock_downloader_cls.return_value.download_url.return_value = DownloadSummary(succeeded=1)

    download.main(["https://youtu.be/abc123"])

    assert config.download.quality == "1080"  # DownloadConfig default, unmodified
    assert config.download.audio_only is False
    assert config.output.videos_dir == Path("data/raw/videos")
    assert config.playlist.max_items is None
    mock_downloader_cls.assert_called_once_with(config)


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_success_returns_zero(mock_load_config, mock_downloader_cls):
    mock_load_config.return_value = make_config()
    mock_downloader = MagicMock()
    mock_downloader.download_url.return_value = DownloadSummary(succeeded=2, skipped=1)
    mock_downloader_cls.return_value = mock_downloader

    exit_code = download.main(["https://youtu.be/abc123"])

    assert exit_code == 0
    mock_downloader.download_url.assert_called_once_with(
        "https://youtu.be/abc123", dry_run=False
    )


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_failure_returns_one(mock_load_config, mock_downloader_cls):
    mock_load_config.return_value = make_config()
    mock_downloader = MagicMock()
    mock_downloader.download_url.return_value = DownloadSummary(failed=1)
    mock_downloader_cls.return_value = mock_downloader

    exit_code = download.main(["https://youtu.be/abc123"])

    assert exit_code == 1


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_applies_cli_overrides_to_config(mock_load_config, mock_downloader_cls):
    config = make_config()
    mock_load_config.return_value = config
    mock_downloader = MagicMock()
    mock_downloader.download_url.return_value = DownloadSummary(succeeded=1)
    mock_downloader_cls.return_value = mock_downloader

    download.main(
        [
            "https://youtu.be/abc123",
            "--quality",
            "480",
            "--audio-only",
            "--output-dir",
            "custom/videos",
            "--max-items",
            "3",
        ]
    )

    # download.main() mutates the loaded config in place before constructing
    # VideoDownloader(config) -- assert those overrides landed.
    assert config.download.quality == "480"
    assert config.download.audio_only is True
    assert config.output.videos_dir == Path("custom/videos")
    assert config.playlist.max_items == 3
    mock_downloader_cls.assert_called_once_with(config)


@patch("src.acquisition.download.VideoDownloader")
@patch("src.acquisition.download.load_config")
def test_main_dry_run_passes_through(mock_load_config, mock_downloader_cls):
    mock_load_config.return_value = make_config()
    mock_downloader = MagicMock()
    mock_downloader.download_url.return_value = DownloadSummary(would_download=4)
    mock_downloader_cls.return_value = mock_downloader

    exit_code = download.main(["https://youtu.be/abc123", "--dry-run"])

    assert exit_code == 0
    mock_downloader.download_url.assert_called_once_with(
        "https://youtu.be/abc123", dry_run=True
    )
