"""Unit tests for target validation constraints and task-specific checks."""

from dataclasses import dataclass
from typing import cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError, DataError
from ml.features.validation.validate_target import (
    validate_min_class_count,
    validate_target,
)

pytestmark = pytest.mark.unit

ModelConfig = SearchModelConfig | TrainModelConfig


@dataclass
class _TaskStub:
    """Minimal task stub exposing task type consumed by target validators."""

    type: str


@dataclass
class _ClassesStub:
    """Minimal class metadata used for classification-specific constraints."""

    positive_class: int | str
    min_class_count: int


@dataclass
class _ConstraintsStub:
    """Minimal numeric constraint stub used in regression target validation."""

    min_value: float | None = None
    max_value: float | None = None


@dataclass
class _TargetStub:
    """Minimal target config stub exposing fields read by validate_target."""

    name: str
    allowed_dtypes: list[str]
    classes: _ClassesStub | None
    constraints: _ConstraintsStub


@dataclass
class _ModelCfgStub:
    """Minimal model config stub exposing task and target attributes."""

    task: _TaskStub
    target: _TargetStub


def _as_model_config(cfg: _ModelCfgStub) -> ModelConfig:
    """Cast lightweight stub to validator function's expected model union."""
    return cast(ModelConfig, cfg)


def test_validate_min_class_count_raises_when_target_has_single_class() -> None:
    """Reject classification targets that contain fewer than two unique classes."""
    y = pd.Series([1, 1, 1], name="is_canceled")

    with pytest.raises(DataError, match="at least two classes"):
        validate_min_class_count(y, min_class_count=1)


def test_validate_min_class_count_raises_when_class_is_below_required_minimum() -> None:
    """Reject targets where any class count violates minimum class frequency."""
    y = pd.Series([0, 0, 1], name="is_canceled")

    with pytest.raises(DataError, match="less than the minimum required"):
        validate_min_class_count(y, min_class_count=2)


def test_validate_target_raises_when_target_contains_null_values() -> None:
    """Fail fast when target column includes null values."""
    y = pd.Series([1, None, 0], name="is_canceled")
    data = pd.DataFrame({"is_canceled": [1, None, 0]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="classification"),
        target=_TargetStub(
            name="is_canceled",
            allowed_dtypes=["int64"],
            classes=_ClassesStub(positive_class=1, min_class_count=1),
            constraints=_ConstraintsStub(),
        ),
    )

    with pytest.raises(DataError, match="contains null values"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_raises_when_dtype_not_in_allowed_list() -> None:
    """Reject target dtype values that violate configured allowed dtype contract."""
    y = pd.Series([1, 0, 1], dtype="int64", name="is_canceled")
    data = pd.DataFrame({"is_canceled": [1, 0, 1]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="classification"),
        target=_TargetStub(
            name="is_canceled",
            allowed_dtypes=["float64"],
            classes=_ClassesStub(positive_class=1, min_class_count=1),
            constraints=_ConstraintsStub(),
        ),
    )

    with pytest.raises(DataError, match="expected one of"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_raises_config_error_when_classes_missing_for_classification() -> None:
    """Require classes config block for classification tasks."""
    y = pd.Series([1, 0, 1], dtype="int64", name="is_canceled")
    data = pd.DataFrame({"is_canceled": [1, 0, 1]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="classification"),
        target=_TargetStub(
            name="is_canceled",
            allowed_dtypes=["int64"],
            classes=None,
            constraints=_ConstraintsStub(),
        ),
    )

    with pytest.raises(ConfigError, match="Classes configuration must be provided"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_raises_when_positive_class_not_present() -> None:
    """Reject classification targets that do not include configured positive class."""
    y = pd.Series([0, 0, 0], dtype="int64", name="is_canceled")
    data = pd.DataFrame({"is_canceled": [0, 0, 0]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="classification"),
        target=_TargetStub(
            name="is_canceled",
            allowed_dtypes=["int64"],
            classes=_ClassesStub(positive_class=1, min_class_count=1),
            constraints=_ConstraintsStub(),
        ),
    )

    with pytest.raises(DataError, match="Positive class 1 not found"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_raises_when_regression_value_below_min_constraint() -> None:
    """Enforce regression minimum bound when min_value is configured."""
    y = pd.Series([0.1, 0.5, 1.2], dtype="float64", name="adr")
    data = pd.DataFrame({"adr": [0.1, 0.5, 1.2]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="regression"),
        target=_TargetStub(
            name="adr",
            allowed_dtypes=["float64"],
            classes=None,
            constraints=_ConstraintsStub(min_value=0.2, max_value=2.0),
        ),
    )

    with pytest.raises(DataError, match="Target min"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_raises_when_regression_value_above_max_constraint() -> None:
    """Enforce regression maximum bound when max_value is configured."""
    y = pd.Series([0.3, 1.0, 2.5], dtype="float64", name="adr")
    data = pd.DataFrame({"adr": [0.3, 1.0, 2.5]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="regression"),
        target=_TargetStub(
            name="adr",
            allowed_dtypes=["float64"],
            classes=None,
            constraints=_ConstraintsStub(min_value=0.2, max_value=2.0),
        ),
    )

    with pytest.raises(DataError, match="Target max"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)


def test_validate_target_logs_warning_when_regression_constraints_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log warning, but pass, when regression min/max constraints are both unset."""
    y = pd.Series([0.3, 1.0, 2.5], dtype="float64", name="adr")
    data = pd.DataFrame({"adr": [0.3, 1.0, 2.5]})
    cfg = _ModelCfgStub(
        task=_TaskStub(type="regression"),
        target=_TargetStub(
            name="adr",
            allowed_dtypes=["float64"],
            classes=None,
            constraints=_ConstraintsStub(min_value=None, max_value=None),
        ),
    )

    with caplog.at_level("WARNING"):
        validate_target(y=y, model_cfg=_as_model_config(cfg), data=data)

    assert "Min and max value constraints are not set" in caplog.text
