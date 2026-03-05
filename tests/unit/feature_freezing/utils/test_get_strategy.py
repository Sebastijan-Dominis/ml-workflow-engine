"""Unit tests for freeze strategy resolver factory helper."""

import importlib
import sys
import types
from dataclasses import dataclass
from typing import Any, cast

import pytest
from ml.exceptions import UserError

pytestmark = pytest.mark.unit


@dataclass
class _DummyStrategy:
    """Minimal strategy stub used for resolver tests."""

    name: str = "dummy"


@pytest.fixture()
def get_strategy_module(monkeypatch: pytest.MonkeyPatch):
    """Import strategy resolver with heavy strategy dependencies stubbed out."""
    sys.modules.pop("ml.feature_freezing.utils.get_strategy", None)

    tabular_strategy_module = cast(
        Any,
        types.ModuleType(
        "ml.feature_freezing.freeze_strategies.tabular.strategy"
        ),
    )
    tabular_strategy_module.FreezeTabular = _DummyStrategy

    time_series_strategy_module = cast(
        Any,
        types.ModuleType(
        "ml.feature_freezing.freeze_strategies.time_series"
        ),
    )
    time_series_strategy_module.FreezeTimeSeries = _DummyStrategy

    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.strategy",
        tabular_strategy_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.time_series",
        time_series_strategy_module,
    )

    return importlib.import_module("ml.feature_freezing.utils.get_strategy")


def test_get_strategy_returns_instantiated_strategy_for_registered_key(
    get_strategy_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate and return resolver strategy class for known data type."""
    monkeypatch.setitem(
        get_strategy_module.FREEZE_STRATEGIES,
        "custom",
        _DummyStrategy,
    )

    strategy = get_strategy_module.get_strategy("custom")

    assert isinstance(strategy, _DummyStrategy)


def test_get_strategy_raises_for_unregistered_data_type(get_strategy_module) -> None:
    """Reject unsupported data types that do not map to a freeze strategy."""
    with pytest.raises(UserError, match="No freeze strategy registered"):
        get_strategy_module.get_strategy("nonexistent")
