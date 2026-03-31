"""Simple tests validating that page layout builders return container-like objects."""

from __future__ import annotations

from ml_service.frontend.app import home_layout
from ml_service.frontend.docs.layout import build_layout as build_docs_layout
from ml_service.frontend.file_viewer.layout import build_layout as build_file_viewer_layout
from ml_service.frontend.pipelines.layout import build_layout as build_pipelines_layout
from ml_service.frontend.scripts.layout import build_layout as build_scripts_layout


def _has_children(obj: object) -> bool:
    return hasattr(obj, "children") or hasattr(obj, "props")


def test_build_scripts_layout():
    layout = build_scripts_layout()
    assert _has_children(layout)


def test_build_pipelines_layout():
    layout = build_pipelines_layout()
    assert _has_children(layout)


def test_build_file_viewer_layout():
    layout = build_file_viewer_layout()
    assert _has_children(layout)


def test_build_docs_layout():
    layout = build_docs_layout()
    assert _has_children(layout)


def test_home_layout():
    layout = home_layout()
    assert _has_children(layout)
