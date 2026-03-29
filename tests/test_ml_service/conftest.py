"""Pytest fixtures for testing the ml_service package.

The fixtures are intentionally lightweight and platform agnostic so tests
run consistently on Windows and Linux.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest


class DummyDashApp:
    """Minimal stand-in for a Dash app capturing callback registration.

    Instances collect registered callbacks as dicts with keys
    ``'args'``, ``'kwargs'`` and ``'func'`` so tests can inspect what was
    registered without importing the real `dash` package.
    """

    def __init__(self) -> None:
        self.callbacks: list[dict[str, Any]] = []

    def callback(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return a decorator that records the wrapped function and metadata."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.callbacks.append({"args": args, "kwargs": kwargs, "func": func})
            return func

        return decorator


@pytest.fixture
def dummy_dash_app() -> DummyDashApp:
    """Provide a dummy Dash-like app for registering callbacks in frontend pages."""

    return DummyDashApp()


@pytest.fixture
def mock_requests(monkeypatch) -> dict[str, Any]:
    """Helpers to patch the `requests` module during tests.

    Returns a small factory dict with a `MockResponse` class and
    `patch_post` / `patch_get` helpers that tests can use to inject
    deterministic responses.
    """

    import requests as _requests


    class MockResponse:
        def __init__(self, ok: bool = True, status_code: int = 200, text: str = "", json_data: Any = None) -> None:
            self.ok = ok
            self.status_code = status_code
            self.text = text
            self._json = json_data if json_data is not None else {}

        def json(self) -> Any:  # pragma: no cover - trivial helper
            return self._json

        def raise_for_status(self) -> None:
            if not self.ok:
                raise _requests.HTTPError(f"{self.status_code}: {self.text}")


    def patch_post(func: Callable[..., Any]) -> None:
        monkeypatch.setattr(_requests, "post", func)


    def patch_get(func: Callable[..., Any]) -> None:
        monkeypatch.setattr(_requests, "get", func)


    return {"MockResponse": MockResponse, "patch_post": patch_post, "patch_get": patch_get}


@pytest.fixture
def patch_subprocess(monkeypatch) -> Callable[[int, str, str], None]:
    """Helper to patch ``subprocess.run`` with a controllable result.

    Usage:

        patch_subprocess(returncode=0, stdout="ok", stderr="")

    After calling the helper, any call to ``subprocess.run`` will return
    an object with ``returncode``, ``stdout`` and ``stderr`` attributes.
    """

    import subprocess as _subprocess


    class Result:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr


    def _patch(returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        result = Result(returncode=returncode, stdout=stdout, stderr=stderr)

        def fake_run(*args: Any, **kwargs: Any) -> Result:
            return result

        monkeypatch.setattr(_subprocess, "run", fake_run)


    return _patch


@pytest.fixture
def fastapi_client() -> Any:
    """Provide a `TestClient` for the ml_service FastAPI app.

    Tests that need to exercise the HTTP layer can use this fixture.
    """

    from fastapi.testclient import TestClient
    from ml_service.backend.main import app as _app

    return TestClient(_app)
