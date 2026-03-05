"""Unit tests for string-to-boolean parsing."""
import pytest
from ml.exceptions import UserError
from ml.io.formatting.str_to_bool import str_to_bool

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["yes", "true", "t", "1", "YES", "TrUe"])
def test_str_to_bool_parses_truthy_tokens(value: str) -> None:
    """Verify parsing of supported truthy tokens."""
    assert str_to_bool(value) is True


@pytest.mark.parametrize("value", ["no", "false", "f", "0", "NO", "FaLsE"])
def test_str_to_bool_parses_falsy_tokens(value: str) -> None:
    """Verify parsing of supported falsy tokens."""
    assert str_to_bool(value) is False


def test_str_to_bool_raises_user_error_for_unknown_token() -> None:
    """Verify rejection of unsupported boolean tokens."""
    with pytest.raises(UserError, match="Boolean value expected for argument"):
        str_to_bool("maybe")
