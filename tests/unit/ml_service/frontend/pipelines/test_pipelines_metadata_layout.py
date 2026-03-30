"""Tests to exercise remaining branches in pipelines metadata and layout.

These tests patch the frontend registry and reload modules so the
module-level construction logic in `pipelines_metadata` runs and the
layout builder in `layout` is exercised (including the fallback else
branch for unknown field types).
"""
from __future__ import annotations

from importlib import import_module, reload
from types import SimpleNamespace
from typing import Any

registry = import_module("ml_service.frontend.pipelines.pipelines_registry")
pipelines_pkg = import_module("ml_service.frontend.pipelines")


def _mk_field(annotation: Any, default: Any = None) -> Any:
    return SimpleNamespace(annotation=annotation, default=default)


def test_pipelines_metadata_various_field_types(monkeypatch):
    # Build a fake args_schema.model_fields mapping with a range of types
    model_fields = {
        "logging_level": _mk_field(str, default=None),
        "env": _mk_field(str, default=None),
        "stage": _mk_field(str, default=None),
        "is_enabled": _mk_field(bool, default=True),
        "count": _mk_field(int, default=None),
        "name": _mk_field(str, default=None),
    }

    fake_args_schema = SimpleNamespace(model_fields=model_fields)
    fake_registry = [
        {
            "name": "TestPipeline",
            "endpoint": "pipelines/test",
            "args_schema": fake_args_schema,
            "field_metadata": {
                "is_enabled": {"optional": True, "label": "Enabled?"},
                "count": {"optional": True, "placeholder": "count"},
            },
        }
    ]

    monkeypatch.setattr(registry, "FRONTEND_PIPELINES_REGISTRY", fake_registry, raising=False)

    # Reload the pipelines_metadata module so it rebuilds FRONTEND_PIPELINES
    metadata = reload(import_module("ml_service.frontend.pipelines.pipelines_metadata"))

    assert any(p["name"] == "TestPipeline" for p in metadata.FRONTEND_PIPELINES)
    p = next(p for p in metadata.FRONTEND_PIPELINES if p["name"] == "TestPipeline")
    types_by_name = {f["name"]: f["type"] for f in p["fields"]}

    assert types_by_name["logging_level"] == "dropdown"
    assert types_by_name["env"] == "dropdown"
    assert types_by_name["stage"] == "dropdown"
    assert types_by_name["is_enabled"] == "boolean"
    assert types_by_name["count"] == "number"
    assert types_by_name["name"] == "text"


def test_pipelines_layout_else_branch(monkeypatch):
    # Create a pipeline with an unknown field type to hit the layout's else branch
    fake_field = {"name": "mystery", "type": "mystery", "placeholder": "x", "optional": False}
    fake_pipeline = {"name": "MysteryPipeline", "endpoint": "pipelines/mystery", "fields": [fake_field]}

    metadata = reload(import_module("ml_service.frontend.pipelines.pipelines_metadata"))
    monkeypatch.setattr(metadata, "FRONTEND_PIPELINES", [fake_pipeline], raising=False)

    layout_mod = reload(import_module("ml_service.frontend.pipelines.layout"))
    layout = layout_mod.build_layout()
    assert layout is not None
