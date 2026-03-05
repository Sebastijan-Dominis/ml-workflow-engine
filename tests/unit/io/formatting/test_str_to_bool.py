"""Unit tests for the str_to_bool function in ml.io.formatting.str_to_bool. The tests verify that the function correctly parses various representations of truthy and falsy values, and that it raises a UserError when an unrecognized string is provided. The tests use pytest's parametrize feature to test multiple input values for both truthy and falsy cases, as well as a test for the error case."""
import pytest
from ml.exceptions import UserError
from ml.io.formatting.str_to_bool import str_to_bool

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["yes", "true", "t", "1", "YES", "TrUe"])
def test_str_to_bool_parses_truthy_tokens(value: str) -> None:
    """Test that str_to_bool correctly parses various representations of truthy values."""
    assert str_to_bool(value) is True


@pytest.mark.parametrize("value", ["no", "false", "f", "0", "NO", "FaLsE"])
def test_str_to_bool_parses_falsy_tokens(value: str) -> None:
    """Test that str_to_bool correctly parses various representations of falsy values."""
    assert str_to_bool(value) is False


def test_str_to_bool_raises_user_error_for_unknown_token() -> None:
    """Test that str_to_bool raises a UserError when an unrecognized string is provided."""
    with pytest.raises(UserError, match="Boolean value expected for argument"):
        str_to_bool("maybe")
