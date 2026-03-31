"""Extra tests for promotion thresholds layout and callbacks.

These complement existing callback tests by ensuring the layout builder
is executed and the success branch in `validate_config` is covered via
the `normalized` YAML dump path.
"""
from __future__ import annotations

from importlib import import_module, reload

import yaml

promo_pkg = import_module("ml_service.frontend.configs.promotion_thresholds")
register_callbacks = import_module("ml_service.frontend.configs.promotion_thresholds.callbacks").register_callbacks


def test_promotion_thresholds_layout_builds():
    layout_mod = reload(import_module("ml_service.frontend.configs.promotion_thresholds.layout"))
    layout = layout_mod.build_layout()
    assert layout is not None


def test_promotion_thresholds_callbacks_success_branch(monkeypatch, dummy_dash_app, mock_requests):
    register_callbacks(dummy_dash_app)
    vcb = [c for c in dummy_dash_app.callbacks if c["func"].__name__ == "validate_config"][0]

    MockResponse = mock_requests["MockResponse"]

    def fake_success(url, json=None, timeout=None, **kwargs):
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"valid": True, "normalized": {"thresholds": {"x": 2}}})

    mock_requests["patch_post"](fake_success)
    alert, is_open, normalized = vcb["func"](1, "no_show", "city", "cfg")
    assert "Config valid" in str(alert)
    assert is_open is True
    assert yaml.safe_load(normalized)["thresholds"]["x"] == 2
