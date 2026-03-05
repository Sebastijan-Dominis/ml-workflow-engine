"""Unit tests for shared lineage integrity validation helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import PipelineContractError
from ml.runners.shared.lineage.validate_lineage_integrity import (
    validate_lineage_integrity,
)
from ml.runners.shared.lineage.validations.base import validate_base_lineage_integrity
from ml.runners.shared.lineage.validations.configs_match import validate_configs_match

pytestmark = pytest.mark.unit


def test_validate_base_lineage_integrity_passes_when_metadata_and_runtime_exist(tmp_path: Path) -> None:
    """Accept lineage source directory when both required metadata and runtime files exist."""
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runtime.json").write_text("{}", encoding="utf-8")

    validate_base_lineage_integrity(tmp_path)


@pytest.mark.parametrize("missing_name", ["metadata.json", "runtime.json"])
def test_validate_base_lineage_integrity_raises_when_required_file_missing(tmp_path: Path, missing_name: str) -> None:
    """Raise PipelineContractError naming the missing required lineage file."""
    if missing_name != "metadata.json":
        (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    if missing_name != "runtime.json":
        (tmp_path / "runtime.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PipelineContractError, match=f"{missing_name} does not exist"):
        validate_base_lineage_integrity(tmp_path)


def test_validate_configs_match_passes_when_config_hash_matches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pass config lineage check when computed hash equals persisted training metadata hash."""
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    cfg = cast(Any, SimpleNamespace(model_dump=lambda **kwargs: {"alpha": 1}))
    training_metadata = SimpleNamespace(config_fingerprint=SimpleNamespace(config_hash="hash-123"))

    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.load_json", lambda _: {"metadata": "raw"})
    monkeypatch.setattr(
        "ml.runners.shared.lineage.validations.configs_match.validate_training_metadata",
        lambda _: training_metadata,
    )
    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.compute_model_config_hash", lambda _: "hash-123")

    validate_configs_match(tmp_path, cfg)


def test_validate_configs_match_raises_when_metadata_file_missing(tmp_path: Path) -> None:
    """Raise PipelineContractError when required training metadata file does not exist."""
    cfg = cast(Any, SimpleNamespace(model_dump=lambda **kwargs: {"alpha": 1}))

    with pytest.raises(PipelineContractError, match="metadata.json does not exist"):
        validate_configs_match(tmp_path, cfg)


def test_validate_configs_match_raises_when_expected_config_hash_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise PipelineContractError when persisted metadata does not contain config hash."""
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    cfg = cast(Any, SimpleNamespace(model_dump=lambda **kwargs: {"alpha": 1}))
    training_metadata = SimpleNamespace(config_fingerprint=SimpleNamespace(config_hash=""))

    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.load_json", lambda _: {"metadata": "raw"})
    monkeypatch.setattr(
        "ml.runners.shared.lineage.validations.configs_match.validate_training_metadata",
        lambda _: training_metadata,
    )
    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.compute_model_config_hash", lambda _: "hash-actual")

    with pytest.raises(PipelineContractError, match="config hash not found in train metadata"):
        validate_configs_match(tmp_path, cfg)


def test_validate_configs_match_raises_when_hashes_do_not_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise PipelineContractError when computed config hash differs from persisted hash."""
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    cfg = cast(Any, SimpleNamespace(model_dump=lambda **kwargs: {"alpha": 1}))
    training_metadata = SimpleNamespace(config_fingerprint=SimpleNamespace(config_hash="hash-expected"))

    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.load_json", lambda _: {"metadata": "raw"})
    monkeypatch.setattr(
        "ml.runners.shared.lineage.validations.configs_match.validate_training_metadata",
        lambda _: training_metadata,
    )
    monkeypatch.setattr("ml.runners.shared.lineage.validations.configs_match.compute_model_config_hash", lambda _: "hash-actual")

    with pytest.raises(PipelineContractError, match="config hash mismatch"):
        validate_configs_match(tmp_path, cfg)


def test_validate_lineage_integrity_runs_base_only_when_cfg_not_provided(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Invoke only base lineage checks when no config object is passed to orchestrator."""
    calls: list[str] = []

    monkeypatch.setattr(
        "ml.runners.shared.lineage.validate_lineage_integrity.validate_base_lineage_integrity",
        lambda source_dir: calls.append(f"base:{source_dir}"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.lineage.validate_lineage_integrity.validate_configs_match",
        lambda source_dir, cfg: calls.append("configs"),
    )

    validate_lineage_integrity(tmp_path, cfg=None)

    assert calls == [f"base:{tmp_path}"]


def test_validate_lineage_integrity_runs_base_and_configs_when_cfg_provided(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Invoke both base and config-hash checks when config is provided to orchestrator."""
    calls: list[str] = []
    cfg = cast(Any, SimpleNamespace(name="cfg"))

    monkeypatch.setattr(
        "ml.runners.shared.lineage.validate_lineage_integrity.validate_base_lineage_integrity",
        lambda source_dir: calls.append("base"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.lineage.validate_lineage_integrity.validate_configs_match",
        lambda source_dir, got_cfg: calls.append(f"configs:{got_cfg is cfg}"),
    )

    validate_lineage_integrity(tmp_path, cfg=cfg)

    assert calls == ["base", "configs:True"]
