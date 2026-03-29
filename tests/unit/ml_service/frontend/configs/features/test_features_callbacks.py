"""Tests for `ml_service.frontend.configs.features.callbacks`.

Reuses `dummy_dash_app` and `mock_requests` fixtures used across frontend tests.
"""
from __future__ import annotations

from typing import Any

import yaml
from ml_service.frontend.configs.features.callbacks import register_callbacks


def _find_callback_by_name(app_callbacks: list[dict[str, Any]], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_validate_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_yaml")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # backend not ok
    def fake_not_ok(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=500, text="oops")

    reqs["patch_post"](fake_not_ok)
    alert, is_open, val = cb["func"](1, "fname", "v1", "content")
    assert "Backend error 500" in str(alert)
    assert is_open is False
    assert val == "content"

    # invalid response
    def fake_invalid(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": False, "error": "bad cfg"})

    reqs["patch_post"](fake_invalid)
    alert2, is_open2, val2 = cb["func"](1, "fname", "v1", "content")
    assert "bad cfg" in str(alert2)
    assert is_open2 is False

    # exists response
    def fake_exists(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "exists": True})

    reqs["patch_post"](fake_exists)
    alert3, is_open3, val3 = cb["func"](1, "fname", "v1", "content")
    assert "fname/v1 already exists" in str(alert3)
    assert is_open3 is False

    # success returns normalized YAML
    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "normalized": {"a": 1}})

    reqs["patch_post"](fake_success)
    alert4, is_open4, norm = cb["func"](1, "fname", "v1", "content")
    assert "Config valid" in str(alert4) or "Config valid." in str(alert4)
    assert is_open4 is True
    assert yaml.safe_load(norm)["a"] == 1


def test_write_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "write_yaml")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # backend not ok
    def fake_not_ok(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=502, text="bad")

    reqs["patch_post"](fake_not_ok)
    alert, is_open = cb["func"](1, "fname", "v1", "content")
    assert "Backend error 502" in str(alert)
    assert is_open is False

    # exists status
    def fake_exists(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "exists", "message": "already"})

    reqs["patch_post"](fake_exists)
    alert2, is_open2 = cb["func"](1, "fname", "v1", "content")
    assert "already" in str(alert2)
    assert is_open2 is False

    # success
    def fake_written(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"path": "/x"})

    reqs["patch_post"](fake_written)
    alert3, is_open3 = cb["func"](1, "fname", "v1", "content")
    assert "/x" in str(alert3)
    assert is_open3 is False
