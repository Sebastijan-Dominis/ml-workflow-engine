"""Unit tests for target strategy resolution and execution helpers."""

from __future__ import annotations

import ml.features.loading.get_target as get_target_module
import pandas as pd
import pytest
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_get_target_with_row_id_uses_registry_strategy_for_requested_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instantiate and execute the strategy class mapped to the requested key."""
    source = pd.DataFrame({"row_id": [1, 2], "value": [10, 20]})

    class _Strategy:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

        def build(self, data: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"row_id": data["row_id"], "target": [0, 1]})

    monkeypatch.setitem(get_target_module.TARGET_STRATEGIES, ("synthetic_target", "v9"), _Strategy)

    result = get_target_module.get_target_with_row_id(source, ("synthetic_target", "v9"))

    assert _Strategy.instances == 1
    assert list(result.columns) == ["row_id", "target"]
    assert result["target"].tolist() == [0, 1]


def test_get_target_with_row_id_raises_for_missing_registry_key() -> None:
    """Raise ``ConfigError`` when the requested target key is not registered."""
    source = pd.DataFrame({"row_id": [1], "value": [10]})

    with pytest.raises(ConfigError, match="Target strategy for key"):
        get_target_module.get_target_with_row_id(source, ("unknown_target", "v404"))


def test_get_target_with_row_id_propagates_strategy_build_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate strategy build exceptions so callers can handle downstream failures."""
    source = pd.DataFrame({"row_id": [1], "value": [10]})

    class _FailingStrategy:
        def build(self, data: pd.DataFrame) -> pd.DataFrame:
            _ = data
            raise RuntimeError("build failed")

    monkeypatch.setitem(get_target_module.TARGET_STRATEGIES, ("failing_target", "v1"), _FailingStrategy)

    with pytest.raises(RuntimeError, match="build failed"):
        get_target_module.get_target_with_row_id(source, ("failing_target", "v1"))
