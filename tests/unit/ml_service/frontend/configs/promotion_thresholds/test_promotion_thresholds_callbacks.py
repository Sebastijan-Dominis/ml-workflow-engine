"""Tests for `ml_service.frontend.configs.promotion_thresholds.callbacks`.

Exercise validate and write branches mirroring other frontend config tests.
"""
from __future__ import annotations

from typing import Any

import yaml
from ml_service.frontend.configs.promotion_thresholds.callbacks import register_callbacks


def _find_callback_by_name(app_callbacks: list[dict[str, Any]], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_validate_and_write_branches(dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    vcb = _find_callback_by_name(dummy_dash_app.callbacks, "validate_config")[0]
    wcb = _find_callback_by_name(dummy_dash_app.callbacks, "write_config")[0]

    reqs = mock_requests
    MockResponse = reqs["MockResponse"]

    problem = "no_show"
    segment = "city_hotel"

    # missing required inputs
    alert, is_open, val = vcb["func"](1, None, None, "cfg")
    assert "Problem type and segment are required" in str(alert)
    assert is_open is False

    # backend not ok
    def fake_not_ok(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=False, status_code=500, text="bad")

    cfg_yaml = "thresholds:\n  x: 1"

    reqs["patch_post"](fake_not_ok)
    alert2, is_open2, val2 = vcb["func"](1, problem, segment, cfg_yaml)
    assert "Backend error 500" in str(alert2)
    assert is_open2 is False

    # invalid response
    def fake_invalid(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": False, "error": "bad cfg"})

    reqs["patch_post"](fake_invalid)
    alert3, is_open3, val3 = vcb["func"](1, problem, segment, cfg_yaml)
    assert "bad cfg" in str(alert3)
    assert is_open3 is False

    # exists response
    def fake_exists(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "exists": True, "normalized": {}})

    reqs["patch_post"](fake_exists)
    alert4, is_open4, val4 = vcb["func"](1, problem, segment, cfg_yaml)
    assert "already exists" in str(alert4)
    assert is_open4 is False

    # success
    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "normalized": {"thresholds": {"x": 1}}})

    reqs["patch_post"](fake_success)
    alert5, is_open5, normalized = vcb["func"](1, problem, segment, cfg_yaml)
    assert "Config valid" in str(alert5)
    assert is_open5 is True
    assert yaml.safe_load(normalized)["thresholds"]["x"] == 1

    # write: backend not ok
    reqs["patch_post"](fake_not_ok)
    w_alert, w_open = wcb["func"](1, problem, segment, cfg_yaml)
    assert "Backend error 500" in str(w_alert)

    # write: exists
    def fake_write_exists(url, json=None, timeout=None, **kwargs):
        msg = f"Thresholds for {json.get('problem_type')}/{json.get('segment')} already exist." if json else "already exist"
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "exists", "message": msg})

    reqs["patch_post"](fake_write_exists)
    w_alert2, w_open2 = wcb["func"](1, problem, segment, cfg_yaml)
    assert "already exist" in str(w_alert2)

    # write: success
    def fake_written(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"status": "written", "path": "/x/y"})

    reqs["patch_post"](fake_written)
    w_alert3, w_open3 = wcb["func"](1, problem, segment, cfg_yaml)
    assert "/x/y" in str(w_alert3)
