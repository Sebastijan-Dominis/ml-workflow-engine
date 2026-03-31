import importlib


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


def test_dir_viewer_layout_has_expected_ids():
    mod = importlib.import_module("ml_service.frontend.dir_viewer.layout")
    layout = mod.build_layout()
    ids = _collect_ids(layout)
    assert f"{mod.PAGE_PREFIX}-path-input" in ids
    assert f"{mod.PAGE_PREFIX}-load-btn" in ids
    assert f"{mod.PAGE_PREFIX}-viewer" in ids
    assert f"{mod.PAGE_PREFIX}-manual-path" in ids


def test_dir_viewer_page_registers(monkeypatch):
    mod = importlib.import_module("ml_service.frontend.dir_viewer.page")
    called = []

    def fake_register(app):
        called.append(True)

    monkeypatch.setattr(
        "ml_service.frontend.dir_viewer.page.register_callbacks",
        fake_register,
        raising=False,
    )
    mod.register(object())
    assert called
