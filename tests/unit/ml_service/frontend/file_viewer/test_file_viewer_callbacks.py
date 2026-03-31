"""Tests for `ml_service.frontend.file_viewer.callbacks`.

Covers path-required, backend-unreachable, non-OK response, and success branches.
"""

from __future__ import annotations

import importlib
from typing import Any


class DummyApp:
    def __init__(self) -> None:
        self.callbacks: list[dict[str, Any]] = []

    def callback(self, *a: Any, **k: Any):
        def dec(f: Any) -> Any:
            self.callbacks.append({"func": f, "args": a, "kwargs": k})
            return f

        return dec


class FakeResp:
    def __init__(self, ok: bool = True, json_data: dict | None = None, status_code: int = 200, text: str = "") -> None:
        self.ok = ok
        self._json = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._json


def _find(app: DummyApp, name: str):
    for c in app.callbacks:
        if c["func"].__name__ == name:
            return c["func"]
    raise AssertionError(name)


def test_load_file_no_path() -> None:
    mod = importlib.import_module("ml_service.frontend.file_viewer.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    fn = _find(app, "load_file")

    value, mode, alert = fn(None, "")
    assert value == ""
    assert mode == "yaml"
    assert "path required" in str(alert).lower()


def test_load_file_backend_unreachable(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.file_viewer.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    fn = _find(app, "load_file")

    def bad_post(*a: Any, **k: Any):
        raise RuntimeError("boom")

    monkeypatch.setattr("ml_service.frontend.file_viewer.callbacks.requests.post", bad_post)
    value, mode, alert = fn(None, "/some/path")
    assert value == ""
    assert mode == "yaml"
    assert "backend unreachable" in str(alert).lower()


def test_load_file_non_ok(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.file_viewer.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    fn = _find(app, "load_file")

    monkeypatch.setattr(
        "ml_service.frontend.file_viewer.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=False, status_code=404, text="not found"),
    )

    value, mode, alert = fn(None, "/x")
    assert value == ""
    assert mode == "yaml"
    assert "404" in str(alert) or "not found" in str(alert).lower()


def test_load_file_success(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.file_viewer.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    fn = _find(app, "load_file")

    monkeypatch.setattr(
        "ml_service.frontend.file_viewer.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"content": "abc", "mode": "text", "path": "/x/y"}),
    )

    value, mode, alert = fn(None, "/x/y")
    assert value == "abc"
    assert mode == "text"
    assert "loaded /x/y" in str(alert).lower()
