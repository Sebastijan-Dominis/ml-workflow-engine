import ml.feature_freezing.utils.operators as op_utils
import ml.features.validation.validate_operators as val_ops
import pytest
from ml.exceptions import DataError


def _op_a(x: int) -> int:
    return x + 1

def _op_b(x: int) -> int:
    return x * 2

@pytest.fixture()
def operators_module(monkeypatch):
    """Patch FEATURE_OPERATORS in validate_operators."""
    monkeypatch.setattr(val_ops, "FEATURE_OPERATORS", {"a": _op_a, "b": _op_b})
    return val_ops

def test_generate_operator_hash_is_order_insensitive_for_operator_names(monkeypatch):
    monkeypatch.setattr(op_utils, "FEATURE_OPERATORS", {"a": _op_a, "b": _op_b})
    hash_ab = op_utils.generate_operator_hash(["a", "b"])
    hash_ba = op_utils.generate_operator_hash(["b", "a"])
    assert hash_ab == hash_ba

def test_validate_operators_raises_when_hash_mismatch(monkeypatch, operators_module):
    # Patch op_utils for hashing
    monkeypatch.setattr(op_utils, "FEATURE_OPERATORS", {"a": _op_a})
    # Patch val_ops for validation
    monkeypatch.setattr(operators_module, "FEATURE_OPERATORS", {"a": _op_a})

    # Should raise DataError because expected hash is wrong
    with pytest.raises(DataError, match="Operator hash mismatch"):
        operators_module.validate_operators(["a"], operator_hash="not-the-real-hash")

def test_validate_operators_passes_when_names_and_hash_match(monkeypatch, operators_module):
    # Patch both modules
    monkeypatch.setattr(op_utils, "FEATURE_OPERATORS", {"a": _op_a, "b": _op_b})
    monkeypatch.setattr(operators_module, "FEATURE_OPERATORS", {"a": _op_a, "b": _op_b})

    expected_hash = op_utils.generate_operator_hash(["a", "b"])
    operators_module.validate_operators(["b", "a"], operator_hash=expected_hash)
