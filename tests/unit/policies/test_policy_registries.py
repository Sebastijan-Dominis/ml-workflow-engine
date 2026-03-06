"""Unit tests for policy registry and constraint declarations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from ml.policies.data import row_id as row_id_policy
from ml.policies.model_params.catboost_constraints import (
    CATBOOST_PARAM_CONSTRAINTS,
    ParamConstraints,
)
from ml.policies.promotion.threshold_support import TASKS_SUPPORTING_THRESHOLDS

pytestmark = pytest.mark.unit


def test_param_constraints_dataclass_is_frozen() -> None:
    """Prevent accidental runtime mutation of numeric policy constraints."""
    constraint = ParamConstraints(min_value=0, max_value=1, allow_zero=False)

    with pytest.raises(FrozenInstanceError):
        constraint.min_value = 99  # type: ignore[misc]


def test_catboost_param_constraints_cover_expected_keys_with_valid_bounds() -> None:
    """Assert registry key coverage and basic min/max consistency contracts."""
    expected_keys = {
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
        "min_data_in_leaf",
        "bagging_temperature",
        "border_count",
        "colsample_bylevel",
    }

    assert set(CATBOOST_PARAM_CONSTRAINTS.keys()) == expected_keys

    for key, constraint in CATBOOST_PARAM_CONSTRAINTS.items():
        assert constraint.min_value is not None, f"{key} missing min_value"
        assert constraint.max_value is not None, f"{key} missing max_value"
        assert constraint.min_value <= constraint.max_value, f"{key} has invalid bounds"
        assert constraint.allow_negative is False


def test_threshold_support_declares_only_binary_classification() -> None:
    """Document and lock current policy scope for threshold-based decisioning."""
    assert {("classification", "binary")} == TASKS_SUPPORTING_THRESHOLDS


def test_row_id_policy_registry_has_required_dataset_and_callable() -> None:
    """Ensure row-id policy registry exposes a callable handler for hotel bookings."""
    assert "hotel_bookings" in row_id_policy.ROW_ID_REQUIRED
    assert "hotel_bookings" in row_id_policy.ROW_ID_FUNCTIONS

    handler = row_id_policy.ROW_ID_FUNCTIONS["hotel_bookings"]

    assert callable(handler)
    assert handler.__name__ == "add_row_id"
