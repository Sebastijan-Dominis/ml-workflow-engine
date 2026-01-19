"""Small unit tests for evaluation utilities.

These verify simple validation helpers behave as expected when keys
are missing from dictionaries.
"""

import pytest

from ml.training.evaluation_scripts.utils import assert_keys


def test_assert_keys_raises() -> None:
    """assert_keys should raise KeyError when a required key is absent."""

    d = {"a": 1, "b": 2}
    with pytest.raises(KeyError):
        assert_keys(d, ["a", "c"])