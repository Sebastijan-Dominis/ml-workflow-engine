"""Unit tests for shared metrics persistence helper."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import PersistenceError
from ml.runners.shared.persistence import save_metrics as save_metrics_module

pytestmark = pytest.mark.unit


class _MetricsModelStub:
    """Validation-model stub exposing ``model_dump`` for persisted payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_kwargs: dict[str, Any] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return payload and capture model_dump options for assertions."""
        self.model_dump_kwargs = kwargs
        return self.payload


def _model_cfg() -> Any:
    """Build minimal model config stub required by save_metrics helper."""
    return SimpleNamespace(
        task=SimpleNamespace(type="classification"),
        algorithm=SimpleNamespace(value="catboost"),
    )


def test_save_metrics_training_validates_payload_and_writes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate training metrics branch and persist normalized JSON payload."""
    model_stub = _MetricsModelStub(payload={"task_type": "classification", "algorithm": "catboost", "metrics": {"train_auc": 0.9}})
    captured: dict[str, Any] = {}

    def _validate_training(payload: dict[str, Any]) -> _MetricsModelStub:
        captured.update(payload)
        return model_stub

    monkeypatch.setattr(save_metrics_module, "validate_training_metrics", _validate_training)

    result_path = save_metrics_module.save_metrics(
        {"train_auc": 0.9},
        model_cfg=_model_cfg(),  # type: ignore[arg-type]
        target_run_id="run-1",
        experiment_dir=tmp_path,
        stage="training",
    )

    expected_path = tmp_path / "training" / "run-1" / "metrics.json"
    assert Path(result_path) == expected_path
    assert expected_path.exists()
    assert captured == {
        "task_type": "classification",
        "algorithm": "catboost",
        "metrics": {"train_auc": 0.9},
    }
    assert model_stub.model_dump_kwargs == {"exclude_none": True}
    assert json.loads(expected_path.read_text(encoding="utf-8")) == {
        "task_type": "classification",
        "algorithm": "catboost",
        "metrics": {"train_auc": 0.9},
    }


def test_save_metrics_evaluation_uses_evaluation_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route evaluation stage through evaluation validator and persist output."""
    eval_model = _MetricsModelStub(payload={"task_type": "classification", "algorithm": "catboost", "metrics": {"val_auc": 0.81}})
    called: list[str] = []

    def _validate_evaluation(payload: dict[str, Any]) -> _MetricsModelStub:
        called.append("evaluation")
        assert payload["metrics"] == {"val_auc": 0.81}
        return eval_model

    monkeypatch.setattr(save_metrics_module, "validate_evaluation_metrics", _validate_evaluation)

    result_path = save_metrics_module.save_metrics(
        {"val_auc": 0.81},
        model_cfg=_model_cfg(),  # type: ignore[arg-type]
        target_run_id="eval-1",
        experiment_dir=tmp_path,
        stage="evaluation",
    )

    assert called == ["evaluation"]
    assert Path(result_path) == tmp_path / "evaluation" / "eval-1" / "metrics.json"
    assert eval_model.model_dump_kwargs == {"exclude_none": True}


def test_save_metrics_wraps_write_errors_as_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap file-write failures in PersistenceError with stage/run specific path."""
    monkeypatch.setattr(
        save_metrics_module,
        "validate_training_metrics",
        lambda payload: _MetricsModelStub(payload=payload),
    )

    def _failing_open(*args: Any, **kwargs: Any) -> Any:
        _ = (args, kwargs)
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", _failing_open)

    with pytest.raises(PersistenceError, match="Failed to save metrics") as exc_info:
        save_metrics_module.save_metrics(
            {"train_auc": 0.9},
            model_cfg=_model_cfg(),  # type: ignore[arg-type]
            target_run_id="run-err",
            experiment_dir=tmp_path,
            stage="training",
        )

    assert "training" in str(exc_info.value)
    assert "run-err" in str(exc_info.value)
