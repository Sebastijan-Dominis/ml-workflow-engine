"""Branch-targeted tests for `ml_service.frontend.scripts.callbacks`."""

from __future__ import annotations

import importlib
import types


def test_toggle_modal_else_returns_current(monkeypatch) -> None:
    """When a non-matching button triggers the modal callback, it should return current state."""
    mod = importlib.import_module("ml_service.frontend.scripts.callbacks")

    script = {"name": "s1", "endpoint": "ep", "fields": [{"name": "f1", "type": "text"}]}
    monkeypatch.setattr(mod, "FRONTEND_SCRIPTS", [script])

    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(func):
                self._callbacks.append(func)
                return func

            return _decorator

    fake_app = FakeApp()
    mod.register_callbacks(fake_app)
    funcs = {f.__name__: f for f in fake_app._callbacks}
    toggle = funcs["toggle_modal"]

    # triggered by unknown button id -> should return current state
    monkeypatch.setattr(mod.dash, "callback_context", types.SimpleNamespace(triggered=[{"prop_id": "unknown.n_clicks"}]))
    assert toggle(None, None, None, True) is True
    assert toggle(None, None, None, False) is False


def test_run_pipeline_number_none_and_float(monkeypatch) -> None:
    """Cover `number` field branches: None and float conversions."""
    mod = importlib.import_module("ml_service.frontend.scripts.callbacks")

    script = {"name": "num", "endpoint": "ep", "fields": [{"name": "n", "type": "number"}]}
    monkeypatch.setattr(mod, "FRONTEND_SCRIPTS", [script])

    class FakeApp:
        def __init__(self):
            self._callbacks = []

        def callback(self, *args, **kwargs):
            def _decorator(func):
                self._callbacks.append(func)
                return func

            return _decorator

    fake_app = FakeApp()
    mod.register_callbacks(fake_app)
    funcs = {f.__name__: f for f in fake_app._callbacks}
    run_pipeline = funcs["run_pipeline"]

    captured = {}

    def fake_call(endpoint, payload):
        captured["payload"] = payload
        return {"status": "SUCCESS"}

    monkeypatch.setattr(mod, "call_script", fake_call)

    # None value -> payload 'n' should be None
    _ = run_pipeline(1, None)
    assert captured["payload"]["n"] is None

    # float value -> payload 'n' should be float
    _ = run_pipeline(1, "3.14")
    assert isinstance(captured["payload"]["n"], float)
    assert abs(captured["payload"]["n"] - 3.14) < 1e-8
