"""Unit tests for promotion context builder path and identity wiring."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
from ml.promotion.context import build_context

pytestmark = pytest.mark.unit


def _args() -> Namespace:
    """Return minimal promotion CLI args needed by context builder."""
    return Namespace(
        problem="cancellation",
        segment="city_hotel",
        version="v1",
        experiment_id="exp_123",
        train_run_id="train_123",
        eval_run_id="eval_123",
        explain_run_id="explain_123",
    )


def test_build_context_constructs_expected_run_id_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build deterministic run identity and all stage-specific filesystem paths."""
    mkdir_calls: list[tuple[object, bool, bool]] = []

    monkeypatch.setattr("ml.promotion.context.iso_no_colon", lambda _dt: "20260306T220000")
    monkeypatch.setattr("ml.promotion.context.uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))

    def _record_mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        mkdir_calls.append((self, parents, exist_ok))

    monkeypatch.setattr("pathlib.Path.mkdir", _record_mkdir)

    context = build_context(_args())

    assert context.timestamp == "20260306T220000"
    assert context.run_id == "20260306T220000_abcdef01"
    assert context.runners_metadata is None

    assert context.paths.model_registry_dir.as_posix() == "model_registry"
    assert context.paths.run_dir.as_posix() == "model_registry/runs/20260306T220000_abcdef01"
    assert context.paths.promotion_configs_dir.as_posix() == "configs/promotion"
    assert context.paths.registry_path.as_posix() == "model_registry/models.yaml"
    assert context.paths.archive_path.as_posix() == "model_registry/archive.yaml"

    assert context.paths.train_run_dir.as_posix() == (
        "experiments/cancellation/city_hotel/v1/exp_123/training/train_123"
    )
    assert context.paths.eval_run_dir.as_posix() == (
        "experiments/cancellation/city_hotel/v1/exp_123/evaluation/eval_123"
    )
    assert context.paths.explain_run_dir.as_posix() == (
        "experiments/cancellation/city_hotel/v1/exp_123/explainability/explain_123"
    )

    assert len(mkdir_calls) == 1
    _, parents, exist_ok = mkdir_calls[0]
    assert parents is True
    assert exist_ok is False


def test_build_context_propagates_run_dir_creation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface filesystem collisions/errors from run-dir creation without masking."""
    monkeypatch.setattr("ml.promotion.context.iso_no_colon", lambda _dt: "20260306T220500")
    monkeypatch.setattr("ml.promotion.context.uuid4", lambda: SimpleNamespace(hex="0011223344556677"))

    def _raise_mkdir(self, parents: bool = False, exist_ok: bool = False) -> None:
        raise FileExistsError("run dir already exists")

    monkeypatch.setattr("pathlib.Path.mkdir", _raise_mkdir)

    with pytest.raises(FileExistsError, match="run dir already exists"):
        build_context(_args())
