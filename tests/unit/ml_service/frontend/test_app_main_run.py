"""Run `ml_service.frontend.app` as __main__ to cover the script-entry branch.

This executes the module with `runpy.run_module(..., run_name='__main__')` while
injecting dummy page modules and a stubbed `dash.Dash` so the call to
`app.run()` executes but does not start a server.
"""

from __future__ import annotations

import runpy
import sys
import types
from typing import Any

import dash


def test_app_run_as_main_executes_run(monkeypatch):
    page_module_names = [
        "ml_service.frontend.configs.data.page",
        "ml_service.frontend.configs.features.page",
        "ml_service.frontend.configs.modeling.page",
        "ml_service.frontend.configs.pipeline_cfg.page",
        "ml_service.frontend.configs.promotion_thresholds.page",
        "ml_service.frontend.dir_viewer.page",
        "ml_service.frontend.docs.page",
        "ml_service.frontend.file_viewer.page",
        "ml_service.frontend.pipelines.page",
        "ml_service.frontend.scripts.page",
    ]

    # Save originals and install dummy page modules
    originals = {name: sys.modules.get(name) for name in page_module_names}
    orig_app = None
    try:
        for name in page_module_names:
            mod: Any = types.ModuleType(name)
            mod.get_layout = lambda name=name: f"{name}_layout"
            mod.register = lambda app: None
            sys.modules[name] = mod

        # Stub dash.Dash so app.run() is safe to call
        class DummyDash:
            ran = False

            def __init__(self, *args, **kwargs):
                self.server = "stub"

            def run(self, *args, **kwargs):
                DummyDash.ran = True

            def callback(self, *args, **kwargs):
                # return a no-op decorator used by @app.callback(...)
                def _decorator(func):
                    return func

                return _decorator

        monkeypatch.setattr(dash, "Dash", DummyDash)

        # Ensure module is executed as __main__
        # Remove any pre-existing 'ml_service.frontend.app' entry to avoid
        # runpy RuntimeWarning about a module being present in sys.modules
        # prior to execution.
        orig_app = sys.modules.pop("ml_service.frontend.app", None)
        runpy.run_module("ml_service.frontend.app", run_name="__main__", alter_sys=True)

        assert DummyDash.ran is True

    finally:
        # Restore ml_service.frontend.app if it existed before the test
        if orig_app is None:
            sys.modules.pop("ml_service.frontend.app", None)
        else:
            sys.modules["ml_service.frontend.app"] = orig_app

        # Restore sys.modules for the dummy page modules
        for name, orig in originals.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
