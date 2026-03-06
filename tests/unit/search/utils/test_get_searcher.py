"""Unit tests for searcher factory resolution helper."""

import importlib
import sys
import types

import pytest
from ml.exceptions import PipelineContractError

pytestmark = pytest.mark.unit


class _DummySearcher:
    """Minimal concrete searcher stub used for registry-resolution tests."""


def _import_get_searcher_with_registry(registry: dict[str, object]):
    """Import `get_searcher` module with a controlled in-memory SEARCHERS registry."""
    module_name = "ml.search.utils.get_searcher"
    registries_name = "ml.registries"

    sys.modules.pop(module_name, None)
    fake_registries = types.ModuleType(registries_name)
    fake_registries.__dict__["SEARCHERS"] = registry
    sys.modules[registries_name] = fake_registries

    return importlib.import_module(module_name)


def test_get_searcher_instantiates_registered_searcher() -> None:
    """Instantiate and return the concrete searcher class for a known key."""
    get_searcher_module = _import_get_searcher_with_registry({"dummy": _DummySearcher})

    searcher = get_searcher_module.get_searcher("dummy")

    assert isinstance(searcher, _DummySearcher)


def test_get_searcher_raises_pipeline_contract_error_for_unknown_key() -> None:
    """Raise a contract error with a clear message when no registry mapping exists."""
    get_searcher_module = _import_get_searcher_with_registry({})

    with pytest.raises(PipelineContractError, match="No searcher registered for algorithm missing"):
        get_searcher_module.get_searcher("missing")


def test_get_searcher_logs_error_for_unknown_key(caplog: pytest.LogCaptureFixture) -> None:
    """Emit an error log that includes the unresolved algorithm key."""
    get_searcher_module = _import_get_searcher_with_registry({})

    with caplog.at_level("ERROR", logger=get_searcher_module.__name__), pytest.raises(
        PipelineContractError
    ):
        get_searcher_module.get_searcher("unknown_algo")

    assert "No searcher registered for algorithm unknown_algo." in caplog.text


def test_get_searcher_logs_selected_searcher_class(caplog: pytest.LogCaptureFixture) -> None:
    """Emit a debug log with searcher class name and requested algorithm key."""
    get_searcher_module = _import_get_searcher_with_registry({"dummy": _DummySearcher})

    with caplog.at_level("DEBUG", logger=get_searcher_module.__name__):
        get_searcher_module.get_searcher("dummy")

    assert "Using searcher _DummySearcher for algorithm=dummy" in caplog.text


def test_get_searcher_propagates_constructor_errors() -> None:
    """Propagate searcher-constructor exceptions without wrapping or masking them."""

    class _BrokenSearcher:
        def __init__(self) -> None:
            raise RuntimeError("constructor failed")

    get_searcher_module = _import_get_searcher_with_registry({"broken": _BrokenSearcher})

    with pytest.raises(RuntimeError, match="constructor failed"):
        get_searcher_module.get_searcher("broken")
