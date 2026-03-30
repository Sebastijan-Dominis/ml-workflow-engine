"""Integration tests for runtime information helpers."""

from __future__ import annotations

from typing import Any

import psutil
import pytest
from ml.exceptions import RuntimeMLError
from ml.utils.runtime.runtime_info import get_runtime_info


def test_get_runtime_info_returns_expected_keys(monkeypatch: Any) -> None:
    class FakeMem:
        total = 16_000_000_000

    monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeMem())

    info = get_runtime_info()

    assert isinstance(info, dict)
    for key in (
        "os",
        "os_release",
        "architecture",
        "processor",
        "ram_total_gb",
        "platform_string",
        "hostname",
        "python_version",
        "python_impl",
        "python_build",
    ):
        assert key in info

    # RAM calculation should match our fake memory total
    assert info["ram_total_gb"] == round(FakeMem.total / 1e9, 2)


def test_get_runtime_info_raises_runtime_mle_error_on_failure(monkeypatch: Any) -> None:
    def raise_error():
        raise RuntimeError("boom")

    monkeypatch.setattr(psutil, "virtual_memory", raise_error)

    with pytest.raises(RuntimeMLError):
        get_runtime_info()
