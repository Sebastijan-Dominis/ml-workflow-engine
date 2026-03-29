"""Tests for `ml_service.frontend.configs.pipeline_cfg.callbacks`.

These tests register callbacks on the `dummy_dash_app` fixture and invoke
the registered functions directly, patching `requests.post` via
`mock_requests` for deterministic behavior.
"""
from __future__ import annotations

from typing import Any

import yaml
from ml_service.frontend.configs.pipeline_cfg.callbacks import register_callbacks


def _find_callback_by_name(app_callbacks: list[dict[str, Any]], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_validate_config_requires_fields(dummy_dash_app):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_config")[0]

    # missing data_type/algorithm should return validation Alert and original text
    alert, is_open, value = cb["func"](1, None, None, "version: v1")
    assert "Data type and algorithm are required." in str(alert)
    assert is_open is False
    assert value == "version: v1"


def test_validate_config_yaml_parse_error(dummy_dash_app):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_config")[0]

    # Provide YAML that parses but lacks 'version' to trigger the missing-version branch
    alert, is_open, _ = cb["func"](1, "dt", "alg", "foo: bar")
    assert "YAML parsing error" in str(alert)
    assert is_open is False


def test_validate_config_backend_invalid_and_success(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_config")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # Backend returns invalid result
    def fake_invalid(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": False, "error": "bad config"})

    reqs["patch_post"](fake_invalid)
    alert, is_open, _ = cb["func"](1, "dt", "alg", "version: v1")
    assert "bad config" in str(alert)
    assert is_open is False

    # Backend returns success with normalized payload
    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "exists": False, "normalized": {"a": 1, "version": "v1"}})

    reqs["patch_post"](fake_success)
    alert2, is_open2, normalized = cb["func"](1, "dt", "alg", "version: v1")
    assert "Config valid" in str(alert2) or "Config valid." in str(alert2)
    assert is_open2 is True
    # normalized is YAML dump of the returned normalized dict
    assert yaml.safe_load(normalized)["a"] == 1


def test_write_config_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "write_config")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    # missing fields
    alert, is_open = cb["func"](1, None, None, "version: v1")
    assert "Data type and algorithm are required." in str(alert)
    assert is_open is False

    # YAML parsing error: provide YAML without `version` to hit the missing-version branch
    alert2, is_open2 = cb["func"](1, "dt", "alg", "foo: bar")
    assert "YAML parsing error" in str(alert2)
    assert is_open2 is False

    # backend reports exists
    def fake_exists(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "exists", "message": "already"})

    reqs["patch_post"](fake_exists)
    alert3, is_open3 = cb["func"](1, "dt", "alg", "version: v1")
    assert "already" in str(alert3)
    assert is_open3 is False

    # backend reports written
    def fake_written(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "written", "path": "/x/y"})

    reqs["patch_post"](fake_written)
    alert4, is_open4 = cb["func"](1, "dt", "alg", "version: v1")
    assert "Config written successfully" in str(alert4)
    assert is_open4 is False
