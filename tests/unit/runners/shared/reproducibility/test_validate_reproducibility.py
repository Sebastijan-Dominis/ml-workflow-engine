"""Unit tests for reproducibility validation orchestration entrypoint."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import RuntimeMLError
from ml.runners.shared.reproducibility.validate_reproducibility import (
    validate_reproducibility,
)

pytestmark = pytest.mark.unit


def test_validate_reproducibility_runs_all_checks_in_expected_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load runtime payload once and invoke validation checks in deterministic sequence."""
    runtime_info_path = Path("runtime.json")
    raw_payload = {"runtime": "raw"}
    runtime_info = SimpleNamespace(name="runtime-info")
    calls: list[str] = []

    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.load_json",
        lambda path: calls.append(f"load_json:{path}") or raw_payload,
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_runtime_info",
        lambda payload: calls.append(f"validate_runtime_info:{payload is raw_payload}") or runtime_info,
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_git_commits_match",
        lambda obj: calls.append(f"validate_git:{obj is runtime_info}"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_conda_envs_match",
        lambda obj: calls.append(f"validate_conda:{obj is runtime_info}"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_runtime_matches",
        lambda obj: calls.append(f"validate_runtime:{obj is runtime_info}"),
    )

    validate_reproducibility(runtime_info_path)

    assert calls == [
        f"load_json:{runtime_info_path}",
        "validate_runtime_info:True",
        "validate_git:True",
        "validate_conda:True",
        "validate_runtime:True",
    ]


def test_validate_reproducibility_stops_when_runtime_info_validation_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Propagate RuntimeMLError from runtime-info validation and skip downstream checks."""
    calls: list[str] = []

    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.load_json",
        lambda path: {"invalid": "payload"},
    )

    def _failing_validate_runtime_info(payload: dict[str, Any]) -> Any:
        calls.append("validate_runtime_info")
        raise RuntimeMLError("Runtime info validation failed.")

    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_runtime_info",
        _failing_validate_runtime_info,
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_git_commits_match",
        lambda obj: calls.append("validate_git"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_conda_envs_match",
        lambda obj: calls.append("validate_conda"),
    )
    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.validate_runtime_matches",
        lambda obj: calls.append("validate_runtime"),
    )

    with pytest.raises(RuntimeMLError, match="Runtime info validation failed"):
        validate_reproducibility(Path("runtime.json"))

    assert calls == ["validate_runtime_info"]


def test_validate_reproducibility_propagates_load_json_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface loader exceptions directly when runtime JSON cannot be loaded."""
    def _failing_loader(path: Path) -> dict[str, Any]:
        raise FileNotFoundError(f"missing: {path}")

    monkeypatch.setattr(
        "ml.runners.shared.reproducibility.validate_reproducibility.load_json",
        _failing_loader,
    )

    with pytest.raises(FileNotFoundError, match="missing"):
        validate_reproducibility(Path("runtime.json"))
