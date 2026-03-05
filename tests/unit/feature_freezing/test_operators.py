"""Unit tests for operator hashing and validation in feature freezing."""

import importlib
import sys
import types
from typing import Any, cast

import pytest
from ml.exceptions import DataError, UserError

pytestmark = pytest.mark.unit


@pytest.fixture()
def operators_module(monkeypatch: pytest.MonkeyPatch):
    """Import operators module with registry dependencies stubbed for isolation."""
    # Ensure a fresh import under stubbed registry modules.
    sys.modules.pop("ml.feature_freezing.utils.operators", None)

    registries_pkg = cast(Any, types.ModuleType("ml.registries"))
    catalogs_stub = cast(Any, types.ModuleType("ml.registries.catalogs"))
    catalogs_stub.FEATURE_OPERATORS = {}
    registries_pkg.catalogs = catalogs_stub

    monkeypatch.setitem(sys.modules, "ml.registries", registries_pkg)
    monkeypatch.setitem(sys.modules, "ml.registries.catalogs", catalogs_stub)

    module = importlib.import_module("ml.feature_freezing.utils.operators")
    return module


def _op_a(x: int) -> int:
    """Simple operator stub for deterministic source hashing in tests."""
    return x + 1


def _op_b(x: int) -> int:
    """Second operator stub for deterministic source hashing in tests."""
    return x * 2


def test_generate_operator_hash_is_order_insensitive_for_operator_names(
    operators_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate same hash regardless of operator name ordering in input list."""
    monkeypatch.setattr(
        operators_module,
        "FEATURE_OPERATORS",
        {"a": _op_a, "b": _op_b},
    )

    hash_ab = operators_module.generate_operator_hash(["a", "b"])
    hash_ba = operators_module.generate_operator_hash(["b", "a"])

    assert hash_ab == hash_ba


def test_validate_operators_raises_for_unknown_operator_name(
    operators_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject operator lists that include names absent from registry."""
    monkeypatch.setattr(
        operators_module,
        "FEATURE_OPERATORS",
        {"a": _op_a},
    )

    with pytest.raises(UserError, match="Unknown operator"):
        operators_module.validate_operators(["a", "unknown"], operator_hash="ignored")


def test_validate_operators_raises_when_hash_mismatch_detected(
    operators_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise DataError when computed operator hash differs from expected value."""
    monkeypatch.setattr(
        operators_module,
        "FEATURE_OPERATORS",
        {"a": _op_a},
    )

    with pytest.raises(DataError, match="Operator hash mismatch"):
        operators_module.validate_operators(["a"], operator_hash="not-the-real-hash")


def test_validate_operators_passes_when_names_and_hash_match(
    operators_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept operator sets when registry names are valid and hash matches."""
    monkeypatch.setattr(
        operators_module,
        "FEATURE_OPERATORS",
        {"a": _op_a, "b": _op_b},
    )
    expected_hash = operators_module.generate_operator_hash(["a", "b"])

    operators_module.validate_operators(["b", "a"], operator_hash=expected_hash)
