"""Unit tests for shared reproducibility validation helpers."""

import logging
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import RuntimeMLError
from ml.modeling.models.runtime_info import RuntimeInfo
from ml.runners.shared.reproducibility.validations.conda_envs_match import validate_conda_envs_match
from ml.runners.shared.reproducibility.validations.git_commits_match import (
    validate_git_commits_match,
)
from ml.runners.shared.reproducibility.validations.runtime_comparison import (
    validate_runtime_matches,
)
from ml.runners.shared.reproducibility.validations.validate_runtime_info import (
    validate_runtime_info,
)

pytestmark = pytest.mark.unit


def _runtime_info_stub(
    *,
    git_commit: str = "expected-commit",
    conda_env_hash: str = "expected-env-hash",
    python_version: str = "3.11.0",
    os_name: str = "Linux",
    processor: str = "x86_64",
    os_release: str = "6.1.0",
) -> RuntimeInfo:
    """Build a lightweight RuntimeInfo-like object for helper-level validation tests."""
    return cast(
        RuntimeInfo,
        SimpleNamespace(
            execution=SimpleNamespace(git_commit=git_commit),
            environment=SimpleNamespace(conda_env_hash=conda_env_hash),
            runtime=SimpleNamespace(
                python_version=python_version,
                os=os_name,
                processor=processor,
                os_release=os_release,
            ),
        ),
    )


def test_validate_git_commits_match_logs_debug_when_commit_matches(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Emit debug log when current commit exactly matches expected runtime commit."""
    caplog.set_level(logging.DEBUG)
    runtime_info = _runtime_info_stub(git_commit="abc123")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.git_commits_match.get_git_commit", lambda: "abc123")

    validate_git_commits_match(runtime_info)

    assert "Git commit matches expected" in caplog.text


def test_validate_git_commits_match_logs_descendant_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Warn when current commit is a descendant of the expected commit."""
    runtime_info = _runtime_info_stub(git_commit="base-commit")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.git_commits_match.get_git_commit", lambda: "head-commit")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.git_commits_match.is_descendant_commit", lambda a, b: True)

    validate_git_commits_match(runtime_info)

    assert "is a descendant of expected" in caplog.text


def test_validate_git_commits_match_logs_branch_divergence_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Warn strongly when neither commit is ancestor/descendant of the other."""
    runtime_info = _runtime_info_stub(git_commit="expected")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.git_commits_match.get_git_commit", lambda: "current")

    calls: list[tuple[str, str]] = []

    def _is_descendant(a: str, b: str) -> bool:
        calls.append((a, b))
        return False

    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.git_commits_match.is_descendant_commit", _is_descendant)

    validate_git_commits_match(runtime_info)

    assert calls == [("current", "expected"), ("expected", "current")]
    assert "different branch than expected" in caplog.text


def test_validate_conda_envs_match_logs_debug_when_hash_matches(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Emit debug log when current conda environment hash matches expected runtime hash."""
    caplog.set_level(logging.DEBUG)
    runtime_info = _runtime_info_stub(conda_env_hash="hash-1")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.conda_envs_match.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.conda_envs_match.hash_environment", lambda _: "hash-1")

    validate_conda_envs_match(runtime_info)

    assert "Conda environment hash matches expected" in caplog.text


def test_validate_conda_envs_match_logs_warning_when_hash_differs(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Warn when current conda environment hash differs from expected runtime hash."""
    runtime_info = _runtime_info_stub(conda_env_hash="expected-hash")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.conda_envs_match.get_conda_env_export", lambda: "env export")
    monkeypatch.setattr("ml.runners.shared.reproducibility.validations.conda_envs_match.hash_environment", lambda _: "current-hash")

    validate_conda_envs_match(runtime_info)

    assert "does not match expected" in caplog.text


def test_validate_runtime_matches_logs_debug_when_all_runtime_fields_match(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Emit debug logs for each runtime field when all observed values match expected metadata."""
    caplog.set_level(logging.DEBUG)
    runtime_info = _runtime_info_stub(
        python_version="3.11.8",
        os_name="Windows",
        processor="Intel64",
        os_release="11",
    )

    monkeypatch.setattr("platform.python_version", lambda: "3.11.8")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.processor", lambda: "Intel64")
    monkeypatch.setattr("platform.release", lambda: "11")

    validate_runtime_matches(runtime_info)

    assert "Python version matches expected" in caplog.text
    assert "Operating system matches expected" in caplog.text
    assert "CPU matches expected" in caplog.text
    assert "OS release matches expected" in caplog.text


def test_validate_runtime_matches_logs_warnings_for_mismatched_runtime_fields(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Warn independently for Python/OS/CPU/release mismatches against expected runtime metadata."""
    runtime_info = _runtime_info_stub(
        python_version="3.10.0",
        os_name="Linux",
        processor="x86_64",
        os_release="5.10",
    )

    monkeypatch.setattr("platform.python_version", lambda: "3.11.8")
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr("platform.processor", lambda: "Intel64")
    monkeypatch.setattr("platform.release", lambda: "11")

    validate_runtime_matches(runtime_info)

    assert "does not match expected" in caplog.text
    assert caplog.text.count("Reproducibility may be affected") == 4


def test_validate_runtime_info_returns_typed_model_for_valid_payload() -> None:
    """Return RuntimeInfo model when runtime payload satisfies schema requirements."""
    payload = {
        "environment": {
            "conda_env_export": "name: env",
            "conda_env_hash": "hash-1",
        },
        "execution": {
            "created_at": "2026-03-05T00:00:00",
            "duration_seconds": 1.25,
            "git_commit": "abc123",
            "python_executable": "python",
        },
        "gpu_info": {
            "cuda_version": "12.0",
            "gpu_count": 0,
            "gpu_devices_available": [],
            "gpu_devices_used": [],
            "gpu_driver_version": "none",
            "gpu_memories_gb": [],
            "gpu_names": [],
            "task_type": "CPU",
        },
        "runtime": {
            "os": "Windows",
            "os_release": "11",
            "architecture": "x86_64",
            "processor": "Intel64",
            "ram_total_gb": 32.0,
            "platform_string": "Windows-11",
            "hostname": "host",
            "python_version": "3.11.8",
            "python_impl": "CPython",
            "python_build": ["main", "Mar  5 2026"],
        },
    }

    result = validate_runtime_info(payload)

    assert isinstance(result, RuntimeInfo)
    assert result.execution.git_commit == "abc123"


def test_validate_runtime_info_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Raise RuntimeMLError when payload is missing required RuntimeInfo fields."""
    with pytest.raises(RuntimeMLError, match="Runtime info validation failed"):
        validate_runtime_info({"environment": {"conda_env_hash": "x"}})
