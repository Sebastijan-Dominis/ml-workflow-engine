"""Unit tests for cross-feature-set hash consistency validation."""

from types import SimpleNamespace

import pytest
from ml.exceptions import DataError
from ml.features.validation.validate_set import validate_set

pytestmark = pytest.mark.unit


def test_validate_set_passes_when_all_hashes_match() -> None:
    """Accept feature-set bundles when all observed hashes are identical."""
    feature_sets = [
        SimpleNamespace(name="booking_context_features", version="v1"),
        SimpleNamespace(name="customer_history_features", version="v1"),
    ]

    validate_set("schema", {"abc123"}, feature_sets)


def test_validate_set_raises_with_feature_context_when_hashes_differ() -> None:
    """Raise DataError with informative feature-set context when hashes mismatch."""
    feature_sets = [
        SimpleNamespace(name="booking_context_features", version="v1"),
        SimpleNamespace(name="customer_history_features", version="v2"),
    ]

    with pytest.raises(DataError, match="hashes do not match across feature sets"):
        validate_set("schema", {"abc123", "def456"}, feature_sets)
