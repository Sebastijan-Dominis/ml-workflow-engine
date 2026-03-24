"""Unit tests for tabular feature preparation and operator application helpers."""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import DataError, UserError

pytestmark = pytest.mark.unit


class _AddOneOperator:
    """Test operator that adds one derived feature."""

    output_features = ["x_plus_one"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["x_plus_one"] = X["x"] + 1
        return X


class _MultiplyOperator:
    """Test operator that composes with prior derived feature."""

    output_features = ["x_times_two"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["x_times_two"] = X["x_plus_one"] * 2
        return X


@pytest.fixture()
def features_module(monkeypatch: pytest.MonkeyPatch):
    """Import features helper module with registry dependencies stubbed out."""
    sys.modules.pop("ml.feature_freezing.freeze_strategies.tabular.features", None)

    registries_pkg = cast(Any, types.ModuleType("ml.registries"))
    catalogs_stub = cast(Any, types.ModuleType("ml.registries.catalogs"))
    catalogs_stub.FEATURE_OPERATORS = {}
    registries_pkg.catalogs = catalogs_stub

    monkeypatch.setitem(sys.modules, "ml.registries", registries_pkg)
    monkeypatch.setitem(sys.modules, "ml.registries.catalogs", catalogs_stub)

    return importlib.import_module("ml.feature_freezing.freeze_strategies.tabular.features")


def _config_stub(columns: list[str]) -> Any:
    """Create minimal config-like object exposing selected feature columns."""
    return cast(Any, SimpleNamespace(columns=columns, entity_key="entity_key"))


def test_prepare_features_raises_when_row_id_missing(features_module) -> None:
    """Reject source dataframes that do not include required `entity_key` key."""
    data = pd.DataFrame({"x": [1, 2]})

    with pytest.raises(DataError, match="must contain 'entity_key'"):
        features_module.prepare_features(data, _config_stub(columns=["x"]))


def test_prepare_features_raises_when_required_columns_missing(features_module) -> None:
    """Reject source dataframes when configured feature columns are absent."""
    data = pd.DataFrame({"entity_key": [1, 2], "x": [1, 2]})

    with pytest.raises(DataError, match="Missing required columns"):
        features_module.prepare_features(data, _config_stub(columns=["x", "y"]))


def test_prepare_features_returns_row_id_plus_configured_columns(features_module) -> None:
    """Return copied frame containing entity_key and ordered configured columns."""
    data = pd.DataFrame({"entity_key": [1, 2], "x": [10, 20], "z": [0, 0]})

    out = features_module.prepare_features(data, _config_stub(columns=["x"]))

    assert out.columns.tolist() == ["entity_key", "x"]
    assert out.equals(pd.DataFrame({"entity_key": [1, 2], "x": [10, 20]}))


def test_apply_operators_raises_when_required_features_are_missing(features_module) -> None:
    """Fail fast when required operator inputs are not present in feature frame."""
    X = pd.DataFrame({"entity_key": [1], "x": [5]})

    with pytest.raises(DataError, match="Missing required features"):
        features_module.apply_operators(
            X,
            operator_names=["any"],
            required_features={"any": ["not_present"]},
        )


def test_apply_operators_raises_for_unknown_operator_name(features_module) -> None:
    """Reject operator names that are not registered in FEATURE_OPERATORS."""
    X = pd.DataFrame({"entity_key": [1], "x": [5]})

    with pytest.raises(UserError, match="Unknown operator"):
        features_module.apply_operators(
            X,
            operator_names=["unknown"],
            required_features={"unknown": ["x"]},
        )


def test_apply_operators_applies_registered_operators_in_order(features_module) -> None:
    """Compose operators in listed order and preserve derived outputs."""
    monkeypatch_dict = {"op1": _AddOneOperator, "op2": _MultiplyOperator}
    features_module.FEATURE_OPERATORS.clear()
    features_module.FEATURE_OPERATORS.update(monkeypatch_dict)

    X = pd.DataFrame({"entity_key": [1, 2], "x": [2, 3]})

    out = features_module.apply_operators(
        X,
        operator_names=["op1", "op2"],
        required_features={"op1": ["x"], "op2": ["x"]},
    )

    assert out["x_plus_one"].tolist() == [3, 4]
    assert out["x_times_two"].tolist() == [6, 8]
