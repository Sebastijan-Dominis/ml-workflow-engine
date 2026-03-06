"""Unit tests for runtime platform information helpers."""

from types import SimpleNamespace

import pytest
from ml.exceptions import RuntimeMLError
from ml.utils.runtime.runtime_info import get_runtime_info

pytestmark = pytest.mark.unit


def test_get_runtime_info_collects_expected_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect expected runtime metadata fields from platform and psutil."""
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.system", lambda: "Linux")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.release", lambda: "6.8")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.machine", lambda: "x86_64")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.processor", lambda: "Intel")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.platform", lambda: "Linux-6.8")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.node", lambda: "test-host")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.python_version", lambda: "3.11.0")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.python_implementation", lambda: "CPython")
    monkeypatch.setattr("ml.utils.runtime.runtime_info.platform.python_build", lambda: ("main", "date"))
    monkeypatch.setattr(
        "ml.utils.runtime.runtime_info.psutil.virtual_memory",
        lambda: SimpleNamespace(total=16_000_000_000),
    )

    result = get_runtime_info()

    assert result["os"] == "Linux"
    assert result["hostname"] == "test-host"
    assert result["python_version"] == "3.11.0"
    assert result["ram_total_gb"] == 16.0


def test_get_runtime_info_wraps_failures_in_runtime_ml_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wrap runtime collection failures in `RuntimeMLError` with context."""
    def _raise() -> SimpleNamespace:
        raise OSError("psutil failed")

    monkeypatch.setattr("ml.utils.runtime.runtime_info.psutil.virtual_memory", _raise)

    with pytest.raises(RuntimeMLError, match="Failed to get runtime info"):
        get_runtime_info()
