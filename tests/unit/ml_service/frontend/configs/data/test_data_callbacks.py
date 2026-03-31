"""Tests for `ml_service.frontend.configs.data.callbacks`.

Cover editor tab behavior, validate and write branches similar to other frontend callback tests.
"""
from __future__ import annotations

from typing import Any

import yaml

from ml_service.frontend.configs.data.callbacks import register_callbacks


def _find_callback_by_name(app_callbacks: list[dict[str, Any]], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_update_editor_on_tab_change(dummy_dash_app):
    register_callbacks(dummy_dash_app)
    cb = _find_callback_by_name(dummy_dash_app.callbacks, "update_editor_on_tab_change")[0]

    from ml_service.frontend.configs.data.examples.interim import INTERIM_EXAMPLE
    from ml_service.frontend.configs.data.examples.processed import PROCESSED_EXAMPLE
    from ml_service.frontend.configs.data.layout import PAGE_PREFIX

    # interim tab
    v = cb["func"](f"{PAGE_PREFIX}-interim-tab")
    assert v == INTERIM_EXAMPLE

    # processed tab
    v2 = cb["func"](f"{PAGE_PREFIX}-processed-tab")
    assert v2 == PROCESSED_EXAMPLE

    # unknown tab
    v3 = cb["func"]("something-else")
    assert v3 == ""


def test_validate_and_write_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    vcb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_config")[0]
    wcb = _find_callback_by_name(dummy_dash_app.callbacks, "write_config")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    from ml_service.frontend.configs.data.layout import PAGE_PREFIX

    active_tab = f"{PAGE_PREFIX}-interim-tab"

    # YAML parse error (missing data keys)
    alert, is_open, val = vcb["func"](1, active_tab, "no-data-here")
    assert "YAML parsing error" in str(alert)
    assert is_open is False

    # backend not ok
    def fake_not_ok(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=500, text="bad")

    valid_yaml = "data:\n  name: x\n  version: v"

    reqs["patch_post"](fake_not_ok)
    alert2, is_open2, val2 = vcb["func"](1, active_tab, valid_yaml)
    assert "Backend error 500" in str(alert2)
    assert is_open2 is False

    # invalid response
    def fake_invalid(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": False, "error": "bad cfg"})

    reqs["patch_post"](fake_invalid)
    alert3, is_open3, val3 = vcb["func"](1, active_tab, valid_yaml)
    assert "bad cfg" in str(alert3)
    assert is_open3 is False

    # exists response
    def fake_exists(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "exists": True, "normalized": {}})

    reqs["patch_post"](fake_exists)
    alert4, is_open4, val4 = vcb["func"](1, active_tab, valid_yaml)
    assert "already exists" in str(alert4)
    assert is_open4 is False

    # success
    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "normalized": {"data": {"name": "n", "version": "v"}}})

    reqs["patch_post"](fake_success)
    alert5, is_open5, normalized = vcb["func"](1, active_tab, valid_yaml)
    assert "Config valid" in str(alert5)
    assert is_open5 is True
    assert yaml.safe_load(normalized)["data"]["name"] == "n"

    # write: backend not ok
    reqs["patch_post"](fake_not_ok)
    w_alert, w_open = wcb["func"](1, active_tab, valid_yaml)
    assert "Backend error 500" in str(w_alert)

    # write: exists
    def fake_write_exists(url, json=None, timeout=None, **kwargs):
        payload = json or {}
        msg = f"{payload.get('name')}/{payload.get('version')} already exists."
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "exists", "message": msg})

    reqs["patch_post"](fake_write_exists)
    w_alert2, w_open2 = wcb["func"](1, active_tab, valid_yaml)
    assert "already exists" in str(w_alert2)

    # write: success
    def fake_written(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "written", "path": "/x/y"})

    reqs["patch_post"](fake_written)
    w_alert3, w_open3 = wcb["func"](1, active_tab, valid_yaml)
    assert "/x/y" in str(w_alert3)
