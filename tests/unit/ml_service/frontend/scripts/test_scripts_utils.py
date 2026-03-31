"""Tests for script utils (call_script)."""

from __future__ import annotations

from typing import Any

import requests
from ml_service.frontend.scripts.utils import call_script


def test_call_script_success(mock_requests: dict[str, Any]) -> None:
    MockResponse = mock_requests["MockResponse"]

    def fake_post(url, json=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"ok": True})

    mock_requests["patch_post"](fake_post)

    res = call_script("scripts/check_import_layers", {"foo": "bar"})
    assert res == {"ok": True}


def test_call_script_error(mock_requests: dict[str, Any]) -> None:
    def fake_post(url, json=None, **kwargs):
        raise requests.RequestException("connection failed")

    mock_requests["patch_post"](fake_post)

    res = call_script("scripts/check_import_layers", {"foo": "bar"})
    assert "error" in res
    assert "connection failed" in res["error"]
