"""Unit tests for training logical-configuration check helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import ConfigError
from ml.runners.training.utils.logical_config_checks import (
    validate_logical_config as orchestrator_module,
)
from ml.runners.training.utils.logical_config_checks.validations import (
    validate_allowed_params as allowed_params_module,
)
from ml.runners.training.utils.logical_config_checks.validations import (
    validate_training_behavior_consistency as behavior_module,
)

pytestmark = pytest.mark.unit


def _cfg_for_allowed_params(*, algorithm: str = "CatBoost") -> Any:
    """Build minimal config stub required by allowed-params validator."""
    return SimpleNamespace(algorithm=SimpleNamespace(value=algorithm))


def _cfg_for_behavior(*, cv: int, early_stopping_rounds: int | None, seed: int | None) -> Any:
    """Build minimal config stub required by behavior-consistency validator."""
    return SimpleNamespace(
        cv=cv,
        seed=seed,
        training=SimpleNamespace(early_stopping_rounds=early_stopping_rounds),
    )


def test_validate_allowed_params_accepts_known_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow metadata best params when every key is in algorithm registry."""
    monkeypatch.setitem(allowed_params_module.MODEL_PARAM_REGISTRY, "catboost", ["depth", "learning_rate"])
    monkeypatch.setattr(
        allowed_params_module,
        "load_json",
        lambda path: {"best_model_params": {"depth": 6, "learning_rate": 0.1}},
    )

    allowed_params_module.validate_allowed_params(
        _cfg_for_allowed_params(),  # type: ignore[arg-type]
        Path("experiments") / "search" / "run-1",
    )


def test_validate_allowed_params_rejects_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raise ConfigError and log unknown best-param keys for selected algorithm."""
    monkeypatch.setitem(allowed_params_module.MODEL_PARAM_REGISTRY, "catboost", ["depth"])
    monkeypatch.setattr(
        allowed_params_module,
        "load_json",
        lambda path: {"best_model_params": {"depth": 6, "unknown_knob": 42}},
    )

    with caplog.at_level("ERROR", logger=allowed_params_module.__name__), pytest.raises(
        ConfigError,
        match="not allowed for algorithm CatBoost",
    ):
        allowed_params_module.validate_allowed_params(
            _cfg_for_allowed_params(),  # type: ignore[arg-type]
            Path("experiments") / "search" / "run-2",
        )

    assert "unknown_knob" in caplog.text


def test_validate_training_behavior_consistency_rejects_early_stopping_when_cv_too_low() -> None:
    """Require cv > 1 whenever early stopping is enabled."""
    with pytest.raises(ConfigError, match="Early stopping is enabled but cv is set to 1 or less"):
        behavior_module.validate_training_behavior_consistency(
            _cfg_for_behavior(cv=1, early_stopping_rounds=30, seed=123),  # type: ignore[arg-type]
        )


def test_validate_training_behavior_consistency_requires_seed() -> None:
    """Require explicit random seed for reproducible training runs."""
    with pytest.raises(ConfigError, match="A random seed must be specified"):
        behavior_module.validate_training_behavior_consistency(
            _cfg_for_behavior(cv=3, early_stopping_rounds=None, seed=None),  # type: ignore[arg-type]
        )


def test_validate_training_behavior_consistency_accepts_valid_configuration() -> None:
    """Pass when cv/early-stopping and seed settings are logically consistent."""
    behavior_module.validate_training_behavior_consistency(
        _cfg_for_behavior(cv=3, early_stopping_rounds=20, seed=42),  # type: ignore[arg-type]
    )


def test_validate_logical_config_runs_checks_in_expected_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invoke both logical checks with the same inputs in deterministic order."""
    calls: list[str] = []
    cfg = SimpleNamespace(name="cfg")
    search_dir = Path("experiments") / "search" / "run-3"

    monkeypatch.setattr(
        orchestrator_module,
        "validate_allowed_params",
        lambda model_cfg, search_dir_arg: calls.append(
            f"allowed:{model_cfg is cfg}:{search_dir_arg == search_dir}"
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "validate_training_behavior_consistency",
        lambda model_cfg: calls.append(f"behavior:{model_cfg is cfg}"),
    )

    orchestrator_module.validate_logical_config(cfg, search_dir)  # type: ignore[arg-type]

    assert calls == ["allowed:True:True", "behavior:True"]
