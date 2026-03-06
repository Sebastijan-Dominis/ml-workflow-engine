"""Unit tests for threshold compatibility and value validation helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from ml.config.schemas.model_specs import TaskConfig, TaskType
from ml.exceptions import ConfigError
from ml.runners.shared.logical_config.validate_threshold import validate_threshold

pytestmark = pytest.mark.unit


def test_validate_threshold_returns_none_for_unsupported_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip threshold loading entirely when task/subtype is not threshold-enabled."""
    task = TaskConfig(type=TaskType.regression, subtype=None)

    def _should_not_be_called(_path: Path) -> dict:
        raise AssertionError("load_json should not be called for unsupported tasks")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        _should_not_be_called,
    )

    result = validate_threshold(task, tmp_path / "metrics.json")

    assert result is None


def test_validate_threshold_defaults_to_point_five_when_value_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return default threshold 0.5 when metrics do not contain threshold value."""
    task = TaskConfig(type=TaskType.classification, subtype="binary")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        lambda _path: {"metrics": {}},
    )

    assert validate_threshold(task, tmp_path / "metrics.json") == 0.5


def test_validate_threshold_returns_numeric_value_as_float(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return valid numeric threshold values cast to float."""
    task = TaskConfig(type=TaskType.classification, subtype="binary")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        lambda _path: {"metrics": {"threshold": {"value": 1}}},
    )

    result = validate_threshold(task, tmp_path / "metrics.json")

    assert result == 1.0
    assert isinstance(result, float)


@pytest.mark.parametrize("value", [0.0, 1.0, 0.42])
def test_validate_threshold_accepts_inclusive_valid_range(
    value: float,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accept inclusive [0, 1] threshold bounds for supported task types."""
    task = TaskConfig(type=TaskType.classification, subtype="binary")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        lambda _path: {"metrics": {"threshold": {"value": value}}},
    )

    assert validate_threshold(task, tmp_path / "metrics.json") == value


def test_validate_threshold_raises_for_non_numeric_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when threshold value is not numeric."""
    task = TaskConfig(type=TaskType.classification, subtype="binary")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        lambda _path: {"metrics": {"threshold": {"value": "0.9"}}},
    )

    with pytest.raises(ConfigError, match="Threshold value must be a number"):
        validate_threshold(task, tmp_path / "metrics.json")


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_validate_threshold_raises_for_out_of_range_value(
    value: float,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Raise `ConfigError` when threshold falls outside inclusive [0, 1] bounds."""
    task = TaskConfig(type=TaskType.classification, subtype="binary")

    monkeypatch.setattr(
        "ml.runners.shared.logical_config.validate_threshold.load_json",
        lambda _path: {"metrics": {"threshold": {"value": value}}},
    )

    with pytest.raises(ConfigError, match="must be between 0 and 1"):
        validate_threshold(task, tmp_path / "metrics.json")
