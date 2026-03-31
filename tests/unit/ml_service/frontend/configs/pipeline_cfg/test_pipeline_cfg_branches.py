"""Tests for branches in ``frontend.configs.pipeline_cfg.callbacks``.

These tests exercise backend-response branches (non-OK responses,
validation-failure payloads, exists/written paths) by monkeypatching
``requests.post`` used by the callbacks module.
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


class FakeResp:
    """Small fake response object compatible with ``requests`` usage."""

    def __init__(self, ok: bool = True, json_data: dict | None = None, status_code: int = 200, text: str = "") -> None:
        self.ok = ok
        self._json = json_data or {}
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._json


def _find_callback(app: DummyApp, name: str):
    for c in app.callbacks:
        if c["func"].__name__ == name:
            return c["func"]
    raise AssertionError(f"callback {name} not found")


def test_validate_config_handles_non_ok(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    # Backend returns non-OK HTTP response
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=False, status_code=500, text="srv err"),
    )
    alert, is_open, _ = validate_fn(None, "dt", "alg", "version: 1")
    assert is_open is False
    assert "Backend error" in str(alert) or "500" in str(alert)


def test_validate_config_invalid_result(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    # Backend returns JSON saying config invalid
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"valid": False, "error": "bad config"}),
    )
    alert, is_open, _ = validate_fn(None, "dt", "alg", "version: 1")
    assert is_open is False
    assert "bad config" in str(alert)


def test_write_config_exists_and_written(monkeypatch: Any) -> None:
    """Test the write callback for both exists and written backend responses."""

    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    write_fn = _find_callback(app, "write_config")

    # Backend reports exists
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"status": "exists", "message": "already"}),
    )
    alert1, is_open1 = write_fn(None, "dt", "alg", "version: 1")
    assert is_open1 is False
    assert "already" in str(alert1)

    # Backend reports written
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"status": "written", "path": "/x/y"}),
    )
    alert2, is_open2 = write_fn(None, "dt", "alg", "version: 1")
    assert is_open2 is False
    assert "Config written successfully" in str(alert2)


def test_validate_config_missing_inputs(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    alert, is_open, value = validate_fn(None, "", "alg", "version: 1")
    assert is_open is False
    assert "required" in str(alert).lower()
    assert value == "version: 1"


def test_validate_config_yaml_parsing_error(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    # invalid YAML (no version) should trigger parse/validation error
    alert, is_open, value = validate_fn(None, "dt", "alg", "not: [unbalanced")
    assert is_open is False
    assert "yaml" in str(alert).lower() or "parsing" in str(alert).lower()


def test_validate_config_success_normalized(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    normalized = {"data": {"name": "n", "version": "v1"}}
    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"valid": True, "normalized": normalized}),
    )
    alert, is_open, val = validate_fn(None, "dt", "alg", "version: 1")
    assert is_open is True
    # returned value should be YAML dumped form of normalized
    assert "name: n" in val


def test_write_config_backend_error(monkeypatch: Any) -> None:
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    write_fn = _find_callback(app, "write_config")

    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=False, status_code=500, text="oops"),
    )
    alert, is_open = write_fn(None, "dt", "alg", "version: 1")
    assert is_open is False
    assert "backend error" in str(alert).lower() or "500" in str(alert)


def test_validate_config_missing_version(monkeypatch: Any) -> None:
    """A YAML payload missing the `version` key should trigger the parsing/validation error branch."""
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    # valid YAML but missing 'version'
    alert, is_open, value = validate_fn(None, "dt", "alg", "name: nover")
    assert is_open is False
    assert "yaml parsing error" in str(alert).lower() or "missing 'version'" in str(alert).lower()


def test_validate_config_exists_branch(monkeypatch: Any) -> None:
    """When backend reports the config already exists, the exists branch should run."""
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    validate_fn = _find_callback(app, "validate_config")

    monkeypatch.setattr(
        "ml_service.frontend.configs.pipeline_cfg.callbacks.requests.post",
        lambda *a, **k: FakeResp(ok=True, json_data={"valid": True, "exists": True, "normalized": {}}),
    )
    alert, is_open, _ = validate_fn(None, "dt", "alg", "version: 1")
    assert is_open is False
    assert "already exists" in str(alert).lower() or "exists" in str(alert).lower()


def test_write_config_missing_inputs(monkeypatch: Any) -> None:
    """Missing data/algorithm for write should return the required-inputs alert."""
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    write_fn = _find_callback(app, "write_config")

    alert, is_open = write_fn(None, "", "alg", "version: 1")
    assert is_open is False
    assert "required" in str(alert).lower()


def test_write_config_missing_version(monkeypatch: Any) -> None:
    """A write with YAML missing version should hit the YAML parsing/validation error branch."""
    mod = importlib.import_module("ml_service.frontend.configs.pipeline_cfg.callbacks")
    app = DummyApp()
    mod.register_callbacks(app)
    write_fn = _find_callback(app, "write_config")

    alert, is_open = write_fn(None, "dt", "alg", "name: nover")
    assert is_open is False
    assert "yaml parsing error" in str(alert).lower() or "missing 'version'" in str(alert).lower()
