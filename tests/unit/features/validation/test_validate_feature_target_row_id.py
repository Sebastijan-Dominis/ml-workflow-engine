"""Unit tests for row-id alignment validation between feature and target sets."""

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.features.validation.validate_feature_target_entity_key import (
    validate_feature_target_entity_key,
)

pytestmark = pytest.mark.unit


def test_validate_feature_target_entity_key_raises_when_feature_entity_key_missing() -> None:
    """Reject feature frames that do not include required `entity_key` column."""
    features = pd.DataFrame({"feature_a": [1, 2]})
    target = pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]})

    with pytest.raises(DataError, match="Feature set is missing entity_key column"):
        validate_feature_target_entity_key(features, target, entity_key="entity_key")


def test_validate_feature_target_entity_key_raises_when_target_entity_key_missing() -> None:
    """Reject target frames that do not include required `entity_key` column."""
    features = pd.DataFrame({"entity_key": [1, 2], "feature_a": [1, 2]})
    target = pd.DataFrame({"target": [0, 1]})

    with pytest.raises(DataError, match="Target data is missing entity_key column"):
        validate_feature_target_entity_key(features, target, entity_key="entity_key")


def test_validate_feature_target_entity_key_raises_when_any_feature_entity_key_missing_in_target() -> None:
    """Reject validation when target data lacks feature row IDs."""
    features = pd.DataFrame({"entity_key": [1, 2, 3], "feature_a": [10, 20, 30]})
    target = pd.DataFrame({"entity_key": [1, 2], "target": [0, 1]})

    with pytest.raises(DataError, match="Entity key mismatch"):
        validate_feature_target_entity_key(features, target, entity_key="entity_key")


def test_validate_feature_target_entity_key_passes_when_all_feature_ids_exist_in_target() -> None:
    """Pass when every feature entity_key is present in the target dataset."""
    features = pd.DataFrame({"entity_key": [1, 2], "feature_a": [10, 20]})
    target = pd.DataFrame({"entity_key": [1, 2, 99], "target": [0, 1, 1]})

    validate_feature_target_entity_key(features, target, entity_key="entity_key")
