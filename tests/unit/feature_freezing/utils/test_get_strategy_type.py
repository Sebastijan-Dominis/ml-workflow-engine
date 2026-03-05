"""Unit tests for extracting freeze strategy type from raw config."""

import pytest
from ml.exceptions import ConfigError
from ml.feature_freezing.utils.get_strategy_type import get_strategy_type

pytestmark = pytest.mark.unit


def test_get_strategy_type_returns_string_value_when_present() -> None:
    """Return configured strategy type when field exists and is a string."""
    assert get_strategy_type({"type": "tabular"}) == "tabular"


def test_get_strategy_type_raises_when_type_field_missing() -> None:
    """Reject configs that omit required `type` strategy key."""
    with pytest.raises(ConfigError, match="Missing 'type' field"):
        get_strategy_type({})


def test_get_strategy_type_raises_when_type_field_is_not_string() -> None:
    """Reject configs where strategy type is not represented as a string."""
    with pytest.raises(ConfigError, match="Expected 'type' field to be a string"):
        get_strategy_type({"type": 123})
