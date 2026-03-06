"""Unit tests for schema validation pipeline component."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.components.schema_validation.validator import SchemaValidator
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_schema_validator_returns_input_dataframe_when_required_columns_exist() -> None:
    """Pass through the same dataframe object when schema requirements are met."""
    df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "extra": [5, 6]})
    validator = SchemaValidator(required_features=["f1", "f2"])

    result = validator.transform(df)

    assert result is df


def test_schema_validator_raises_data_error_with_missing_columns_in_order() -> None:
    """Report missing columns in required-feature order for actionable debugging."""
    df = pd.DataFrame({"f1": [1], "other": [2]})
    validator = SchemaValidator(required_features=["missing_a", "f1", "missing_b"])

    with pytest.raises(
        DataError,
        match=r"Missing columns: \['missing_a', 'missing_b'\]",
    ):
        validator.transform(df)


def test_schema_validator_logs_error_before_raising(caplog: pytest.LogCaptureFixture) -> None:
    """Emit an error log containing missing-column details before raising."""
    df = pd.DataFrame({"present": [1]})
    validator = SchemaValidator(required_features=["present", "absent"])

    with caplog.at_level("ERROR", logger="ml.components.schema_validation.validator"), pytest.raises(
        DataError
    ):
        validator.transform(df)

    assert "Model input schema violation. Missing columns: ['absent']" in caplog.text
