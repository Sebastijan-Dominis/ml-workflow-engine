"""Unit tests for explainability feature/importance length validator."""

from __future__ import annotations

import numpy as np
import pytest
from ml.exceptions import DataError
from ml.runners.explainability.explainers.tree_model.utils.validators.validate_lengths import (
    validate_lengths,
)

pytestmark = pytest.mark.unit


def test_validate_lengths_accepts_matching_lengths() -> None:
    """Allow equal-length feature and importance vectors without raising."""
    feature_names = np.array(["adr", "lead_time", "total_stay"], dtype=np.str_)
    importances = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    validate_lengths(feature_names, importances)


def test_validate_lengths_raises_data_error_on_length_mismatch() -> None:
    """Raise DataError with counts when feature and importance lengths diverge."""
    feature_names = np.array(["adr", "lead_time"], dtype=np.str_)
    importances = np.array([0.8], dtype=np.float64)

    with pytest.raises(
        DataError,
        match=r"Mismatch between feature names and importances: 2 vs 1",
    ):
        validate_lengths(feature_names, importances)


def test_validate_lengths_logs_error_before_raising(caplog: pytest.LogCaptureFixture) -> None:
    """Emit an error log message before raising for mismatched lengths."""
    feature_names = np.array(["f1"], dtype=np.str_)
    importances = np.array([0.4, 0.6], dtype=np.float64)

    with caplog.at_level(
        "ERROR",
        logger=(
            "ml.runners.explainability.explainers.tree_model.utils.validators."
            "validate_lengths"
        ),
    ), pytest.raises(DataError):
        validate_lengths(feature_names, importances)

    assert "Mismatch between feature names and importances: 1 vs 2" in caplog.text
