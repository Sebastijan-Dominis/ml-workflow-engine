"""Pytest fixtures for `tests/unit/ml_service`.

Provide a lightweight `dummy_dash_app` that captures callback registration
and a `mock_requests` helper to patch `requests.post` in frontend tests.
"""
from __future__ import annotations

from typing import Any

import pytest
import requests


@pytest.fixture
def dummy_dash_app() -> object:
    """A minimal fake Dash app used to capture callback registration.

    Tests expect an object with a `.callbacks` list and a `.callback` method
    that acts as a decorator. When used, the decorator appends a record
    describing the registration to `.callbacks`.
    """

    class DummyApp:
        def __init__(self) -> None:
            self.callbacks: list[dict[str, Any]] = []

        def callback(self, *args: Any, **kwargs: Any):
            def decorator(f):
                self.callbacks.append({"args": args, "kwargs": kwargs, "func": f})
                return f

            return decorator

    return DummyApp()


@pytest.fixture
def mock_requests(monkeypatch) -> dict[str, Any]:
    """Provide utilities to mock `requests.post` and a small `MockResponse`.

    Usage in tests:
    reqs = mock_requests
    MockResponse = reqs["MockResponse"]
    reqs["patch_post"](fake_post)
    """

    class MockResponse:
        def __init__(self, ok: bool = True, status_code: int = 200, text: str = "", json_data: dict[str, Any] | None = None) -> None:
            self.ok = ok
            self.status_code = status_code
            self.text = text
            self._json = json_data or {}

        def json(self) -> dict[str, Any]:
            return self._json

        def raise_for_status(self) -> None:
            """Emulate requests.Response.raise_for_status."""
            if not self.ok or not (200 <= int(self.status_code) < 300):
                raise requests.HTTPError(f"HTTP {self.status_code}")

    def patch_post(fn):
        monkeypatch.setattr(requests, "post", fn)

    return {"MockResponse": MockResponse, "patch_post": patch_post}
