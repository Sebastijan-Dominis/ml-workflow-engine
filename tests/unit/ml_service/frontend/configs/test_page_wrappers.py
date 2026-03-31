"""Smoke tests for simple page wrapper modules (get_layout / register)."""

import importlib


def test_page_wrappers_call_register_and_return_layout(monkeypatch):
    pages = [
        ("ml_service.frontend.configs.data.page", "data"),
        ("ml_service.frontend.configs.features.page", "features"),
        ("ml_service.frontend.configs.modeling.page", "modeling"),
        ("ml_service.frontend.configs.pipeline_cfg.page", "pipeline_cfg"),
    ]

    for module_path, marker in pages:
        mod = importlib.import_module(module_path)
        called = []

        def _reg(app, _marker=marker, _called=called):
            _called.append(_marker)

        monkeypatch.setattr(mod, "register_callbacks", _reg, raising=False)

        layout = mod.get_layout()
        assert layout is not None

        # calling register should invoke our monkeypatched register_callbacks
        mod.register(object())
        assert called and called[0] == marker
