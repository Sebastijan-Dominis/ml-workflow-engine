"""Tests for `ml_service.frontend.scripts.scripts_metadata` helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

# Use PEP 604 `X | Y` union syntax for compatibility with ruff UP007


def test_is_boolean_and_number_field_helpers() -> None:
    mod = importlib.import_module("ml_service.frontend.scripts.scripts_metadata")

    # boolean detection
    assert mod.is_boolean_field(SimpleNamespace(annotation=bool)) is True
    assert mod.is_boolean_field(SimpleNamespace(annotation=bool | None)) is True

    # number detection
    assert mod.is_number_field(SimpleNamespace(annotation=int)) is True
    assert mod.is_number_field(SimpleNamespace(annotation=int | float)) is True


def test_frontend_scripts_populated() -> None:
    mod = importlib.import_module("ml_service.frontend.scripts.scripts_metadata")
    assert isinstance(mod.FRONTEND_SCRIPTS, list)
    # ensure each entry has expected keys
    for s in mod.FRONTEND_SCRIPTS:
        assert "name" in s and "endpoint" in s and "fields" in s
