"""Tests for frontend callbacks (scripts, pipelines, file+dir viewer, docs).

These tests use `dummy_dash_app` to capture callback registration and
patch network/call helpers to keep tests deterministic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from ml_service.frontend.dir_viewer.callbacks import register_callbacks as register_dir_callbacks
from ml_service.frontend.docs import callbacks as docs_callbacks
from ml_service.frontend.docs.callbacks import register_callbacks as register_docs_callbacks
from ml_service.frontend.docs.callbacks import rewrite_links
from ml_service.frontend.file_viewer.callbacks import register_callbacks as register_file_callbacks
from ml_service.frontend.pipelines.callbacks import (
    register_callbacks as register_pipelines_callbacks,
)
from ml_service.frontend.pipelines.pipelines_metadata import FRONTEND_PIPELINES
from ml_service.frontend.scripts.callbacks import register_callbacks as register_scripts_callbacks
from ml_service.frontend.scripts.scripts_metadata import FRONTEND_SCRIPTS


def test_file_viewer_and_dir_viewer_callbacks(dummy_dash_app, mock_requests: dict[str, Any]):
    # File viewer
    before = len(dummy_dash_app.callbacks)
    register_file_callbacks(dummy_dash_app)
    new = dummy_dash_app.callbacks[before:]
    funcs = [c["func"] for c in new if c["func"].__name__ == "load_file"]
    assert funcs, "load_file not registered"
    load_file = funcs[0]

    reqs = cast(dict[str, Any], mock_requests)
    MockResponse = reqs["MockResponse"]

    def fake_post_file(url, json: dict[str, Any] | None = None, **kwargs):
        assert json is not None
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"content": "hello", "mode": "yaml", "path": json["path"]})

    reqs["patch_post"](fake_post_file)

    content, mode, alert = load_file(None, "/some/path/config.yaml")
    assert content == "hello"
    assert mode == "yaml"
    assert "Loaded" in str(alert)

    # Empty path returns validation Alert and empty content
    content2, mode2, _ = load_file(None, "")
    assert content2 == ""
    assert mode2 == "yaml"

    # Directory viewer
    before2 = len(dummy_dash_app.callbacks)
    register_dir_callbacks(dummy_dash_app)
    new2 = dummy_dash_app.callbacks[before2:]
    funcs2 = [c["func"] for c in new2 if c["func"].__name__ == "load_dir"]
    assert funcs2, "load_dir not registered"
    load_dir = funcs2[0]

    def fake_post_dir(url, json: dict[str, Any] | None = None, **kwargs):
        assert json is not None
        return MockResponse(ok=True, status_code=200, text="ok", json_data={"tree_yaml": "t", "path": json["path"]})

    reqs["patch_post"](fake_post_dir)

    tcontent, tmode, talert = load_dir(None, "configs")
    assert tcontent == "t"
    assert tmode == "yaml"
    assert "Loaded directory" in str(talert)


def _find_callback_by_name(app_callbacks, name: str):
    return [c for c in app_callbacks if c["func"].__name__ == name]


def test_scripts_and_pipelines_run_pipeline_callbacks(dummy_dash_app, monkeypatch):
    # Scripts
    register_scripts_callbacks(dummy_dash_app)
    run_callbacks = _find_callback_by_name(dummy_dash_app.callbacks, "run_pipeline")
    assert run_callbacks, "No run_pipeline callbacks registered for scripts"

    # Patch call_script used inside the scripts callbacks
    import ml_service.frontend.scripts.callbacks as scripts_callbacks

    monkeypatch.setattr(scripts_callbacks, "call_script", lambda endpoint, payload: {"status": "SUCCESS", "result": payload})

    # Take first run_pipeline callback and invoke it with None values for all fields
    cb = run_callbacks[0]
    output_obj = cb["args"][0]
    comp_id = getattr(output_obj, "component_id", getattr(output_obj, "id", ""))
    matching = [s for s in FRONTEND_SCRIPTS if s["name"] in comp_id]
    script = matching[0] if matching else FRONTEND_SCRIPTS[0]
    field_count = len(script["fields"])

    result_comp = cb["func"](1, *([None] * field_count))
    assert "SUCCESS" in str(result_comp)

    # Pipelines
    register_pipelines_callbacks(dummy_dash_app)
    run_callbacks_p = _find_callback_by_name(dummy_dash_app.callbacks, "run_pipeline")
    # There will be multiple run_pipeline functions (scripts + pipelines); pick one that matches pipeline names
    import ml_service.frontend.pipelines.callbacks as pipelines_callbacks
    monkeypatch.setattr(pipelines_callbacks, "call_pipeline", lambda endpoint, payload: {"status": "SUCCESS", "result": payload})

    # find a pipeline callback whose component id contains one of the pipeline names
    cb_pipeline = None
    for c in run_callbacks_p:
        out = c["args"][0]
        cid = getattr(out, "component_id", getattr(out, "id", ""))
        if any(p["name"] in cid for p in FRONTEND_PIPELINES):
            cb_pipeline = c
            break

    assert cb_pipeline is not None
    pipeline_name = next(p for p in FRONTEND_PIPELINES if p["name"] in getattr(cb_pipeline["args"][0], "component_id", ""))
    pf_count = len(pipeline_name["fields"])
    res_comp = cb_pipeline["func"](1, *([None] * pf_count))
    assert "SUCCESS" in str(res_comp)


def test_docs_rewrite_and_loading(tmp_path: Path):
    # Prepare a small docs tree
    docs_root = tmp_path
    (docs_root / "readme.md").write_text("Hello [About](about.md)")
    (docs_root / "about.md").write_text("About page")

    # Monkeypatch the DOCS_ROOT used by callbacks module
    docs_callbacks.DOCS_ROOT = docs_root

    out = rewrite_links("Hello [About](about.md)", "readme.md")
    assert "/Docs?doc=about.md" in out

    # Register callback and call load_doc_from_url
    class Dummy:
        callbacks: list[dict[str, Any]]

        def __init__(self) -> None:
            self.callbacks = []

    dummy = Dummy()

    # emulate the minimal callback decorator storage used in other tests
    def fake_callback(*args, **kwargs):
        def decorator(f):
            dummy.callbacks.append({"args": args, "kwargs": kwargs, "func": f})
            return f

        return decorator

    # Use register function with our fake app object
    class FakeApp:
        def callback(self, *a, **k):
            return fake_callback(*a, **k)

    register_docs_callbacks(FakeApp())

    # Find the load_doc_from_url function and call it
    funcs = [c["func"] for c in dummy.callbacks if c["func"].__name__ == "load_doc_from_url"]
    assert funcs
    load_fn = funcs[0]
    res = load_fn("?doc=readme.md")
    assert "/Docs?doc=about.md" in res
