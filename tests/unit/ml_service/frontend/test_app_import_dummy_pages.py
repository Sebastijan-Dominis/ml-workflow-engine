"""Test re-importing the Dash app with dummy page modules.

This ensures the top-level registration loop in `ml_service.frontend.app`
calls each page `register()` and that the `PAGES` mapping points to the
expected `get_layout` functions.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any


def test_app_import_calls_registers_and_populates_pages():
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

    original_modules: dict[str, types.ModuleType | None] = {}
    register_calls: list[str] = []

    original_app = sys.modules.pop("ml_service.frontend.app", None)
    try:
        # Insert dummy page modules that expose `get_layout` and `register`.
        for module_name in page_module_names:
            original_modules[module_name] = sys.modules.pop(module_name, None)
            dummy: Any = types.ModuleType(module_name)

            def make_get_layout(name: str):
                return lambda: f"{name}_layout"

            def make_register(name: str):
                def register_function(app):
                    register_calls.append(name)

                return register_function

            dummy.get_layout = make_get_layout(module_name)
            dummy.register = make_register(module_name)
            sys.modules[module_name] = dummy

        # Import the app module fresh so it picks up our dummy page modules.
        app_module = importlib.import_module("ml_service.frontend.app")

        # All dummy register functions should have been called.
        assert set(register_calls) == set(page_module_names)

        # The PAGES mapping should call through to our dummy get_layout functions.
        expected_key_to_module = {
            "Data Config": "ml_service.frontend.configs.data.page",
            "Feature Config": "ml_service.frontend.configs.features.page",
            "Pipelines": "ml_service.frontend.pipelines.page",
            "Scripts": "ml_service.frontend.scripts.page",
            "Docs": "ml_service.frontend.docs.page",
            "File Viewer": "ml_service.frontend.file_viewer.page",
            "Directory Viewer": "ml_service.frontend.dir_viewer.page",
        }

        for page_name, module_name in expected_key_to_module.items():
            layout_func = app_module.PAGES.get(page_name)
            assert layout_func is not None
            assert layout_func() == f"{module_name}_layout"

    finally:
        # Clean up: remove the imported app and restore original modules.
        sys.modules.pop("ml_service.frontend.app", None)
        for module_name, original in original_modules.items():
            if original is not None:
                sys.modules[module_name] = original
            else:
                sys.modules.pop(module_name, None)
        if original_app is not None:
            sys.modules["ml_service.frontend.app"] = original_app
