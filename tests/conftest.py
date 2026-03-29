"""Pytest configuration for test imports.

Provide lightweight stubs for heavy optional dependencies (like `catboost`)
and ensure the project root is on `sys.path` for tests.
"""

import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Lightweight TestClient fixture for FastAPI integration-style tests
try:
    import ml_service.backend.main as _backend_main
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - defensive import for environments without FastAPI
    TestClient = None  # type: ignore
    _backend_main = None  # type: ignore

# Global test stub for the optional `catboost` dependency. Many modules import
# `catboost` at import-time; providing a minimal stub prevents import errors
# when running unit tests in environments without the real package installed.
if "catboost" not in sys.modules:
    _cb: Any = types.ModuleType("catboost")

    class _CBBase:
        def __init__(self, *args, **kwargs):
            self._init_args = args
            self._init_kwargs = kwargs

        def fit(self, X=None, y=None, **kwargs):
            return self

        def predict(self, X):
            # return zeros compatible with expected shapes
            try:
                return [0] * len(X)
            except Exception:
                return 0

        def predict_proba(self, X):
            # return uniform probabilities for binary classification
            try:
                return [[0.5, 0.5] for _ in range(len(X))]
            except Exception:
                return [[0.5, 0.5]]

        def get_feature_importance(self, **kwargs):
            return []

        def get_cat_feature_indices(self):
            return []

    class CatBoostClassifier(_CBBase):
        pass

    class CatBoostRegressor(_CBBase):
        pass

    class Pool:
        def __init__(self, *args, **kwargs):
            # accept data and cat_features kwarg as some modules construct Pool(**...)
            self.data = kwargs.get("data") if "data" in kwargs else (args[0] if args else None)
            self.cat_features = kwargs.get("cat_features", [])

    _cb.CatBoostClassifier = CatBoostClassifier
    _cb.CatBoostRegressor = CatBoostRegressor
    _cb.CatBoost = _CBBase
    _cb.Pool = Pool
    sys.modules["catboost"] = _cb

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def fastapi_client():
    """Provide a `TestClient` for the ml_service FastAPI app.

    Tests that exercise ml_service backend routers can use this fixture.
    If FastAPI isn't available in the environment the fixture will raise
    at import-time when a test attempts to use it.
    """
    if TestClient is None or _backend_main is None:
        raise RuntimeError("FastAPI TestClient or ml_service backend not importable in test environment")

    client = TestClient(_backend_main.app)
    try:
        yield client
    finally:
        client.close()
