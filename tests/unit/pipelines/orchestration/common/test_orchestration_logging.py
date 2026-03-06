"""Unit tests for shared orchestration completion logging helpers."""

from __future__ import annotations

import pytest
from pipelines.orchestration.common import orchestration_logging

pytestmark = pytest.mark.unit


def test_log_completion_formats_duration_in_seconds(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Render durations under one minute in seconds with two decimal places."""
    monkeypatch.setattr(orchestration_logging.time, "perf_counter", lambda: 110.5)
    monkeypatch.setattr(orchestration_logging, "iso_no_colon", lambda _dt: "20260306T133000")

    with caplog.at_level("INFO", logger=orchestration_logging.__name__):
        orchestration_logging.log_completion(100.0, "Completed")

    assert "Completed at 20260306T133000 after 10.50 seconds." in caplog.text


def test_log_completion_formats_duration_in_minutes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Render durations between one minute and one hour in minutes."""
    monkeypatch.setattr(orchestration_logging.time, "perf_counter", lambda: 280.0)
    monkeypatch.setattr(orchestration_logging, "iso_no_colon", lambda _dt: "20260306T133500")

    with caplog.at_level("INFO", logger=orchestration_logging.__name__):
        orchestration_logging.log_completion(100.0, "Completed")

    assert "Completed at 20260306T133500 after 3.00 minutes." in caplog.text


def test_log_completion_formats_duration_in_hours(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Render durations of one hour or more in hours."""
    monkeypatch.setattr(orchestration_logging.time, "perf_counter", lambda: 7300.0)
    monkeypatch.setattr(orchestration_logging, "iso_no_colon", lambda _dt: "20260306T134000")

    with caplog.at_level("INFO", logger=orchestration_logging.__name__):
        orchestration_logging.log_completion(100.0, "Completed")

    assert "Completed at 20260306T134000 after 2.00 hours." in caplog.text
