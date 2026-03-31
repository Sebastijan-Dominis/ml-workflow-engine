"""Targeted branches for data config callbacks."""

from __future__ import annotations

from ml_service.frontend.configs.data.callbacks import register_callbacks
from ml_service.frontend.configs.data.layout import PAGE_PREFIX


def _find_callback(app_callbacks: list[dict], name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name][0]["func"]


def test_write_config_parsing_error(dummy_dash_app):
    register_callbacks(dummy_dash_app)
    write_fn = _find_callback(dummy_dash_app.callbacks, "write_config")

    # invalid YAML should hit the parse-exception branch
    alert, is_open = write_fn(1, f"{PAGE_PREFIX}-interim-tab", "not: [yaml")
    assert "YAML parsing error" in str(alert)
    assert is_open is False
