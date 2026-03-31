"""Tests for `ml_service.frontend.configs.modeling.callbacks` (renamed to avoid basename collisions)."""
from __future__ import annotations

from typing import Any

import yaml
from ml_service.frontend.configs.modeling.callbacks import register_callbacks


def _find_callback_by_name(app_callbacks: list[dict[str, Any]], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_validate_yaml_backend_error(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_yaml")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    def fake_err(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=500, text="server error")

    reqs["patch_post"](fake_err)

    alert, is_open, v1, v2, v3 = cb["func"](1, "a: 1", "b: 2", "c: 3")
    assert "Backend error 500" in str(alert)
    assert is_open is False
    assert v1 == "a: 1"


def test_validate_yaml_invalid_and_success(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_yaml")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # invalid result from backend
    def fake_invalid(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": False, "error": "bad config"})

    reqs["patch_post"](fake_invalid)
    alert, is_open, v1, v2, v3 = cb["func"](1, "a:1", "b:2", "c:3")
    assert "bad config" in str(alert)
    assert is_open is False

    # success path returns normalized YAML strings
    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={
            "valid": True,
            "normalized": {"model_specs": {"foo": 1}, "search": {}, "training": {}},
        })

    reqs["patch_post"](fake_success)
    alert2, is_open2, n1, n2, n3 = cb["func"](1, "a:1", "b:2", "c:3")
    assert "Config is valid" in str(alert2) or "Config is valid." in str(alert2)
    assert is_open2 is True
    assert yaml.safe_load(n1)["foo"] == 1


def test_write_yaml_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "write_yaml")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # backend not ok
    def fake_not_ok(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=502, text="bad gateway")

    reqs["patch_post"](fake_not_ok)
    alert, is_open = cb["func"](1, "a:1", "b:2", "c:3")
    assert "Backend error 502" in str(alert)
    assert is_open is False

    # success -> check that returned paths appear in Alert text
    def fake_written(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"paths": {"model_specs": "p1", "search": "p2", "training": "p3"}})

    reqs["patch_post"](fake_written)
    alert2, is_open2 = cb["func"](1, "a:1", "b:2", "c:3")
    assert "p1" in str(alert2)
    assert is_open2 is False
