"""Unit tests for frontend utility functions in ``ml_service.frontend``.

These tests mock external HTTP calls to keep them fast and reliable.
"""
from __future__ import annotations

from typing import Any

import requests
from ml_service.frontend.pipelines.utils import call_pipeline
from ml_service.frontend.scripts.utils import call_script


def test_call_script_success(mock_requests: dict[str, Any]) -> None:
    """`call_script` returns parsed JSON when the backend responds successfully."""

    MockResponse = mock_requests["MockResponse"]

    def fake_post(url, json=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"ok": True})

    mock_requests["patch_post"](fake_post)

    res = call_script("scripts/check_import_layers", {"foo": "bar"})
    assert res == {"ok": True}


def test_call_script_error(mock_requests: dict[str, Any]) -> None:
    """`call_script` returns an error dict when the HTTP client raises an exception."""

    def fake_post(url, json=None, **kwargs):
        raise requests.RequestException("connection failed")

    mock_requests["patch_post"](fake_post)

    res = call_script("scripts/check_import_layers", {"foo": "bar"})
    assert "error" in res
    assert "connection failed" in res["error"]


def test_call_pipeline_success(mock_requests: dict[str, Any]) -> None:
    """`call_pipeline` returns parsed JSON on successful response."""

    MockResponse = mock_requests["MockResponse"]

    def fake_post(url, json=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "started"})

    mock_requests["patch_post"](fake_post)

    res = call_pipeline("pipelines/run", {"x": 1})
    assert res == {"status": "started"}


def test_call_pipeline_error(mock_requests: dict[str, Any]) -> None:
    """`call_pipeline` returns an error dict when the HTTP client raises an exception."""

    def fake_post(url, json=None, **kwargs):
        raise requests.RequestException("timeout")

    mock_requests["patch_post"](fake_post)

    res = call_pipeline("pipelines/run", {"x": 1})
    assert "error" in res
    assert "timeout" in res["error"]
