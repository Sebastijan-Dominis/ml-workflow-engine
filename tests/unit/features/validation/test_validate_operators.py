import ml.features.validation.validate_operators as vo
import pytest
from ml.exceptions import DataError, UserError


def test_no_hash_with_operators_raises():
    monkey_ops = ["op_a"]
    # ensure registry contains nothing to avoid unknown-operator path
    vo.FEATURE_OPERATORS = set(["op_a"])

    with pytest.raises(DataError):
        vo.validate_operators(monkey_ops, operator_hash=None)


def test_no_hash_and_no_operators_returns_none():
    vo.FEATURE_OPERATORS = set()
    assert vo.validate_operators([], operator_hash=None) is None


def test_unknown_operator_raises_usererror():
    vo.FEATURE_OPERATORS = set(["op_x"])
    with pytest.raises(UserError):
        vo.validate_operators(["not_a_real_op"], operator_hash="hash")


def test_hash_mismatch_raises_dataerror(monkeypatch):
    vo.FEATURE_OPERATORS = set(["a", "b"])
    monkeypatch.setattr(vo, "generate_operator_hash", lambda ops: "computed")

    with pytest.raises(DataError):
        vo.validate_operators(["a", "b"], operator_hash="different")


def test_hash_match_returns_none(monkeypatch):
    vo.FEATURE_OPERATORS = set(["a", "b"])
    monkeypatch.setattr(vo, "generate_operator_hash", lambda ops: "same")

    assert vo.validate_operators(["a", "b"], operator_hash="same") is None
