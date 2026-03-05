"""Unit tests for feature-operator factory instantiation behavior."""

import importlib
import sys
import types

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _import_operator_factory_with_registry(registry: dict[str, object]):
    """Import operator_factory after stubbing its registry dependency."""
    module_name = "ml.pipelines.operator_factory"
    catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)
    fake_catalogs = types.ModuleType(catalogs_name)
    fake_catalogs.__dict__["FEATURE_OPERATORS"] = registry
    sys.modules[catalogs_name] = fake_catalogs

    return importlib.import_module(module_name)


def test_build_operators_instantiates_each_unique_operator_once() -> None:
    """Instantiate each distinct source operator exactly once even when schema rows repeat it."""

    class _OpA:
        pass

    class _OpB:
        pass

    operator_factory = _import_operator_factory_with_registry({"op_a": _OpA, "op_b": _OpB})
    derived_schema = pd.DataFrame({"source_operator": ["op_a", "op_a", "op_b"]})

    result = operator_factory.build_operators(derived_schema)

    assert set(result) == {"op_a", "op_b"}
    assert isinstance(result["op_a"], _OpA)
    assert isinstance(result["op_b"], _OpB)


def test_build_operators_raises_key_error_for_unknown_operator() -> None:
    """Surface a `KeyError` when the schema references an unregistered operator name."""
    operator_factory = _import_operator_factory_with_registry({"op_a": object})
    derived_schema = pd.DataFrame({"source_operator": ["op_a", "missing"]})

    with pytest.raises(KeyError, match="missing"):
        operator_factory.build_operators(derived_schema)
