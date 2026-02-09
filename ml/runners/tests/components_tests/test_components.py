"""Tests for components shipped in `ml.components`.

Currently includes validation behavior checks for the
`cancellation_v1.SchemaValidator` component.
"""

import pandas as pd
import pytest

from ml.components.cancellation_v1 import SchemaValidator as SV1


def test_cancellation_schema_validator_raises_on_missing_columns() -> None:
    """SchemaValidator should raise when required columns are missing from input."""

    X = pd.DataFrame({"a": [1], "b": [2]})
    validator = SV1(required_columns=["a", "b", "c"])

    with pytest.raises(ValueError):
        validator.transform(X)