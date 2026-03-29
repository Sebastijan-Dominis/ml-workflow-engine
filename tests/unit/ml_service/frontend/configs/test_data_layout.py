"""Tests for the data config page layout builder."""

from __future__ import annotations

from ml_service.frontend.configs.data.layout import PAGE_PREFIX, build_layout


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


def test_build_data_layout_contains_expected_ids():
    layout = build_layout()
    assert _has_children(layout)
    ids = _collect_ids(layout)
    assert f"{PAGE_PREFIX}-config-tabs" in ids
