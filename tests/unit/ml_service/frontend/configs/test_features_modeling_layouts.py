"""Tests for feature registry and modeling layout builders."""

from __future__ import annotations

from ml_service.frontend.configs.features.layout import PAGE_PREFIX as FEATURES_PREFIX
from ml_service.frontend.configs.features.layout import build_layout as build_features_layout
from ml_service.frontend.configs.modeling.config_examples import CONFIG_EXAMPLES_REGISTRY
from ml_service.frontend.configs.modeling.layout import PAGE_PREFIX as MODELING_PREFIX
from ml_service.frontend.configs.modeling.layout import build_layout as build_modeling_layout


def _has_children(obj: object) -> bool:
    return hasattr(obj, "children") or hasattr(obj, "props")


def _collect_ids(obj: object) -> set:
    ids: set = set()
    if hasattr(obj, "id"):
        maybe_id = getattr(obj, "id", None)
        if maybe_id:
            ids.add(maybe_id)
    props = getattr(obj, "props", None)
    if isinstance(props, dict):
        pid = props.get("id")
        if pid:
            ids.add(pid)

    children = getattr(obj, "children", None)
    if children is None:
        return ids
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is None:
            continue
        ids.update(_collect_ids(child))
    return ids


def test_features_layout_contains_expected_ids():
    layout = build_features_layout()
    assert _has_children(layout)
    ids = _collect_ids(layout)
    assert f"{FEATURES_PREFIX}-feature-editor" in ids
    assert f"{FEATURES_PREFIX}-validate-btn" in ids
    assert f"{FEATURES_PREFIX}-confirm-modal" in ids


def test_modeling_layout_contains_all_example_editors():
    layout = build_modeling_layout()
    assert _has_children(layout)
    ids = _collect_ids(layout)
    # Validate button present
    assert f"{MODELING_PREFIX}-validate-btn" in ids
    # Each example in the registry should have an editor id
    for name in CONFIG_EXAMPLES_REGISTRY:
        assert f"{MODELING_PREFIX}-{name}" in ids
