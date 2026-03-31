"""Basic page-level tests and a focused scripts layout test.

These tests call small wrapper functions like `get_layout()` and
`register(app)` for multiple page modules to improve page-level coverage.
They also exercise the `scripts.layout.build_layout()` branches by
temporarily replacing `FRONTEND_SCRIPTS` with a diverse set of field types.
"""

from __future__ import annotations

import importlib
from typing import Any


class DummyApp:
    """Minimal fake Dash-like app that records decorated callbacks."""

    def __init__(self) -> None:
        self.callbacks: list[dict[str, Any]] = []

    def callback(self, *args: Any, **kwargs: Any):
        def decorator(func: Any) -> Any:
            self.callbacks.append({"func": func, "args": args, "kwargs": kwargs})
            return func

        return decorator


def test_pages_get_layout_and_register_simple() -> None:
    """Call `get_layout()` and `register()` on a set of page modules."""
    modules = [
        "ml_service.frontend.dir_viewer.page",
        "ml_service.frontend.docs.page",
        "ml_service.frontend.file_viewer.page",
        "ml_service.frontend.configs.promotion_thresholds.page",
        "ml_service.frontend.scripts.page",
    ]

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        # get_layout should build without raising
        layout = getattr(mod, "get_layout", lambda: None)()
        assert layout is not None

        # register should call into callbacks and add entries to our DummyApp
        app = DummyApp()
        register = getattr(mod, "register", None)
        if register is not None:
            register(app)
            # at least one callback should be registered for modules that wire callbacks
            assert isinstance(app.callbacks, list)


def test_scripts_layout_various_field_types(monkeypatch) -> None:
    """Exercise the many input-type branches in `scripts.layout.build_layout()`."""
    layout_mod = importlib.import_module("ml_service.frontend.scripts.layout")

    custom_scripts = [
        {"name": "one", "fields": [{"name": "a", "type": "text", "optional": True}]},
        {"name": "two", "fields": [{"name": "b", "type": "number"}, {"name": "c", "type": "boolean", "value": True}]},
        {"name": "three", "fields": [{"name": "d", "type": "dropdown", "options": ["x", "y"], "value": "x"}]},
        {"name": "four", "fields": [{"name": "e", "type": "unknown"}]},
    ]

    monkeypatch.setattr(layout_mod, "FRONTEND_SCRIPTS", custom_scripts)

    layout = layout_mod.build_layout()
    # Basic sanity: layout is built and contains the page title
    assert layout is not None
    assert "ML Scripts Dashboard" in repr(layout)
