"""Unit tests for required-field dictionary validation helper."""

import pytest
from ml.exceptions import DataError
from ml.io.validation.validate_mapping import ensure_required_fields_present_in_dict

pytestmark = pytest.mark.unit


def test_ensure_required_fields_present_in_dict_accepts_complete_mapping() -> None:
    """Test that ensure_required_fields_present_in_dict does not raise an error when all required fields are present in the input dictionary. The test calls ensure_required_fields_present_in_dict with a sample input dictionary that contains all the required fields, and asserts that no exception is raised, confirming that the function correctly identifies when all required fields are present and does not raise an error in this case."""
    ensure_required_fields_present_in_dict(
        input_dict={"a": 1, "b": 2},
        required_fields=["a", "b"],
    )


def test_ensure_required_fields_present_in_dict_raises_for_missing_fields() -> None:
    """Test that ensure_required_fields_present_in_dict raises a DataError when any required fields are missing from the input dictionary. The test calls ensure_required_fields_present_in_dict with a sample input dictionary that is missing some required fields, and asserts that a DataError is raised with a message indicating which fields are missing, confirming that the function correctly identifies missing required fields."""
    with pytest.raises(DataError, match="missing required fields: b, c"):
        ensure_required_fields_present_in_dict(
            input_dict={"a": 1},
            required_fields=["a", "b", "c"],
        )
