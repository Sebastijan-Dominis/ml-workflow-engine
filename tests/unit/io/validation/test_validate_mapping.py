"""Unit tests for required-field dictionary validation helper."""

import pytest
from ml.exceptions import DataError
from ml.io.validation.validate_mapping import ensure_required_fields_present_in_dict

pytestmark = pytest.mark.unit


def test_ensure_required_fields_present_in_dict_accepts_complete_mapping() -> None:
    """Pass validation when all required mapping keys are present."""
    ensure_required_fields_present_in_dict(
        input_dict={"a": 1, "b": 2},
        required_fields=["a", "b"],
    )


def test_ensure_required_fields_present_in_dict_raises_for_missing_fields() -> None:
    """Raise `DataError` listing missing required mapping keys."""
    with pytest.raises(DataError, match="missing required fields: b, c"):
        ensure_required_fields_present_in_dict(
            input_dict={"a": 1},
            required_fields=["a", "b", "c"],
        )
