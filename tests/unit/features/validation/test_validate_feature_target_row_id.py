"""Unit tests for row-id alignment validation between feature and target sets."""

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.features.validation.validate_feature_target_row_id import (
    validate_feature_target_row_id,
)

pytestmark = pytest.mark.unit


def test_validate_feature_target_row_id_raises_when_feature_row_id_missing() -> None:
    """Reject feature frames that do not include required `row_id` column."""
    features = pd.DataFrame({"feature_a": [1, 2]})
    target = pd.DataFrame({"row_id": [1, 2], "target": [0, 1]})

    with pytest.raises(DataError, match="Feature set is missing 'row_id' column"):
        validate_feature_target_row_id(features, target)


def test_validate_feature_target_row_id_raises_when_target_row_id_missing() -> None:
    """Reject target frames that do not include required `row_id` column."""
    features = pd.DataFrame({"row_id": [1, 2], "feature_a": [1, 2]})
    target = pd.DataFrame({"target": [0, 1]})

    with pytest.raises(DataError, match="Target data is missing 'row_id' column"):
        validate_feature_target_row_id(features, target)


def test_validate_feature_target_row_id_raises_when_any_feature_row_id_missing_in_target() -> None:
    """Reject validation when target data lacks feature row IDs."""
    features = pd.DataFrame({"row_id": [1, 2, 3], "feature_a": [10, 20, 30]})
    target = pd.DataFrame({"row_id": [1, 2], "target": [0, 1]})

    with pytest.raises(DataError, match="Row ID mismatch"):
        validate_feature_target_row_id(features, target)


def test_validate_feature_target_row_id_passes_when_all_feature_ids_exist_in_target() -> None:
    """Pass when every feature row_id is present in the target dataset."""
    features = pd.DataFrame({"row_id": [1, 2], "feature_a": [10, 20]})
    target = pd.DataFrame({"row_id": [1, 2, 99], "target": [0, 1, 1]})

    validate_feature_target_row_id(features, target)
