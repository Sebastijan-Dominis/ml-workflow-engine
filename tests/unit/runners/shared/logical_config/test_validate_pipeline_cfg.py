"""Unit tests for pipeline-config hash validation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError, PipelineContractError
from ml.runners.shared.logical_config.validate_pipeline_cfg import validate_pipeline_cfg

pytestmark = pytest.mark.unit


def _make_model_cfg(pipeline_path: Path) -> TrainModelConfig:
    """Create a minimal typed model-cfg stub exposing `pipeline.path` only."""
    return cast(
        TrainModelConfig,
        SimpleNamespace(pipeline=SimpleNamespace(path=str(pipeline_path))),
    )


def test_validate_pipeline_cfg_accepts_metadata_pipeline_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use `metadata.pipeline_hash` when present and return the validated active hash."""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps: []\n", encoding="utf-8")
    model_cfg = _make_model_cfg(pipeline_file)

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_json",
        lambda _path: {"metadata": {"pipeline_hash": "abc123"}},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_yaml",
        lambda _path: {"steps": []},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.compute_model_config_hash",
        lambda _cfg: "abc123",
    )

    result = validate_pipeline_cfg(tmp_path / "metadata.json", model_cfg)

    assert result == "abc123"


def test_validate_pipeline_cfg_falls_back_to_config_fingerprint_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read expected hash from legacy `config_fingerprint.pipeline_cfg_hash` when needed."""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps: []\n", encoding="utf-8")
    model_cfg = _make_model_cfg(pipeline_file)

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_json",
        lambda _path: {"config_fingerprint": {"pipeline_cfg_hash": "legacy-hash"}},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_yaml",
        lambda _path: {"steps": ["SchemaValidator"]},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.compute_model_config_hash",
        lambda _cfg: "legacy-hash",
    )

    result = validate_pipeline_cfg(tmp_path / "metadata.json", model_cfg)

    assert result == "legacy-hash"


def test_validate_pipeline_cfg_raises_when_expected_hash_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when metadata does not expose any supported pipeline hash key."""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps: []\n", encoding="utf-8")
    model_cfg = _make_model_cfg(pipeline_file)

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_json",
        lambda _path: {"metadata": {}, "config_fingerprint": {}},
    )

    with pytest.raises(ConfigError, match="Pipeline hash not found in metadata file"):
        validate_pipeline_cfg(tmp_path / "metadata.json", model_cfg)


def test_validate_pipeline_cfg_raises_when_pipeline_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when configured pipeline file path does not exist."""
    missing_pipeline_file = tmp_path / "missing_pipeline.yaml"
    model_cfg = _make_model_cfg(missing_pipeline_file)

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_json",
        lambda _path: {"metadata": {"pipeline_hash": "abc"}},
    )

    with pytest.raises(ConfigError, match="Pipeline configuration file not found"):
        validate_pipeline_cfg(tmp_path / "metadata.json", model_cfg)


def test_validate_pipeline_cfg_raises_when_hashes_do_not_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `PipelineContractError` when active pipeline hash differs from metadata."""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps: []\n", encoding="utf-8")
    model_cfg = _make_model_cfg(pipeline_file)

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_json",
        lambda _path: {"metadata": {"pipeline_hash": "expected"}},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.load_yaml",
        lambda _path: {"steps": []},
    )
    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_pipeline_cfg.compute_model_config_hash",
        lambda _cfg: "actual",
    )

    with pytest.raises(PipelineContractError, match="Pipeline configuration hash mismatch"):
        validate_pipeline_cfg(tmp_path / "metadata.json", model_cfg)
