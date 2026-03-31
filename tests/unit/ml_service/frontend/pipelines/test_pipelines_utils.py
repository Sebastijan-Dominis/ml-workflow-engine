"""Tests for pipeline utils (call_pipeline)."""

from __future__ import annotations

from typing import Any

import requests
from ml_service.frontend.pipelines.utils import call_pipeline


def test_call_pipeline_success(mock_requests: dict[str, Any]) -> None:
    MockResponse = mock_requests["MockResponse"]

    def fake_post(url, json=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "started"})

    mock_requests["patch_post"](fake_post)

    res = call_pipeline("pipelines/run", {"x": 1})
    assert res == {"status": "started"}


def test_call_pipeline_error(mock_requests: dict[str, Any]) -> None:
    def fake_post(url, json=None, **kwargs):
        raise requests.RequestException("timeout")

    mock_requests["patch_post"](fake_post)

    res = call_pipeline("pipelines/run", {"x": 1})
    assert "error" in res
    assert "timeout" in res["error"]
