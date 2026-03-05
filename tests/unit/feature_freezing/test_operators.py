"""Unit tests for operator hashing and validation in feature freezing."""

import importlib
import sys
import types

import pytest
from ml.exceptions import DataError, UserError

# Avoid importing the full registries stack in unit tests. This prevents unrelated
# optional-runtime import failures (e.g., NumPy alias removals in downstream modules).
if "ml.registries.catalogs" not in sys.modules:
    catalogs_stub = types.ModuleType("ml.registries.catalogs")
    catalogs_stub.FEATURE_OPERATORS = {}
    sys.modules["ml.registries.catalogs"] = catalogs_stub

operators_module = importlib.import_module("ml.feature_freezing.utils.operators")
generate_operator_hash = operators_module.generate_operator_hash
validate_operators = operators_module.validate_operators

pytestmark = pytest.mark.unit


def _op_a(x: int) -> int:
    """Simple operator stub for deterministic source hashing in tests."""
    return x + 1


def _op_b(x: int) -> int:
    """Second operator stub for deterministic source hashing in tests."""
    return x * 2


def test_generate_operator_hash_is_order_insensitive_for_operator_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate same hash regardless of operator name ordering in input list."""
    monkeypatch.setattr(
        "ml.feature_freezing.utils.operators.FEATURE_OPERATORS",
        {"a": _op_a, "b": _op_b},
    )

    hash_ab = generate_operator_hash(["a", "b"])
    hash_ba = generate_operator_hash(["b", "a"])

    assert hash_ab == hash_ba


def test_validate_operators_raises_for_unknown_operator_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject operator lists that include names absent from registry."""
    monkeypatch.setattr(
        "ml.feature_freezing.utils.operators.FEATURE_OPERATORS",
        {"a": _op_a},
    )

    with pytest.raises(UserError, match="Unknown operator"):
        validate_operators(["a", "unknown"], operator_hash="ignored")


def test_validate_operators_raises_when_hash_mismatch_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise DataError when computed operator hash differs from expected value."""
    monkeypatch.setattr(
        "ml.feature_freezing.utils.operators.FEATURE_OPERATORS",
        {"a": _op_a},
    )

    with pytest.raises(DataError, match="Operator hash mismatch"):
        validate_operators(["a"], operator_hash="not-the-real-hash")


def test_validate_operators_passes_when_names_and_hash_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept operator sets when registry names are valid and hash matches."""
    monkeypatch.setattr(
        "ml.feature_freezing.utils.operators.FEATURE_OPERATORS",
        {"a": _op_a, "b": _op_b},
    )
    expected_hash = generate_operator_hash(["a", "b"])

    validate_operators(["b", "a"], operator_hash=expected_hash)
