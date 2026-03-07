"""Unit tests for tabular feature-freezing configuration models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from ml.exceptions import ConfigError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

pytestmark = pytest.mark.unit


def _valid_payload() -> dict[str, Any]:
    """Return a minimal valid payload for ``TabularFeaturesConfig``."""
    return {
        "type": "tabular",
        "description": "Tabular freeze config for tests",
        "data": [
            {
                "name": "hotel_bookings",
                "version": "v1",
                "format": "parquet",
            }
        ],
        "feature_store_path": str(Path("feature_store")),
        "columns": ["row_id", "feature_a", "feature_b"],
        "feature_roles": {
            "categorical": ["feature_a"],
            "numerical": ["feature_b"],
            "datetime": ["row_id"],
        },
        "operators": {
            "mode": "logical",
            "names": ["operator_a"],
            "hash": "abc123",
            "required_features": {"operator_a": ["feature_a"]},
        },
        "constraints": {
            "forbid_nulls": ["row_id"],
            "max_cardinality": {"feature_a": 100},
        },
        "storage": {"format": "parquet", "compression": "snappy"},
        "lineage": {
            "created_by": "tests",
            "created_at": datetime(2026, 3, 7, 12, 0, 0).isoformat(),
        },
    }


def test_tabular_features_config_accepts_valid_payload() -> None:
    """Validate that a complete, internally consistent payload is accepted."""
    cfg = TabularFeaturesConfig.model_validate(_valid_payload())

    assert cfg.type == "tabular"
    assert cfg.storage.format == "parquet"


def test_tabular_features_config_accepts_none_operators() -> None:
    """Allow configurations that omit optional operator definitions."""
    payload = _valid_payload()
    payload["operators"] = None

    cfg = TabularFeaturesConfig.model_validate(payload)

    assert cfg.operators is None


def test_tabular_features_config_rejects_operator_name_key_mismatch() -> None:
    """Reject operators payload when names and required-feature keys diverge."""
    payload = _valid_payload()
    payload["operators"]["required_features"] = {"different_operator": ["feature_a"]}

    with pytest.raises(ConfigError, match="must match required features"):
        TabularFeaturesConfig.model_validate(payload)


def test_tabular_features_config_rejects_feature_roles_mismatch() -> None:
    """Reject payloads where union of feature roles does not equal columns."""
    payload = _valid_payload()
    payload["feature_roles"]["numerical"] = []

    with pytest.raises(ConfigError, match="Feature roles do not match included columns"):
        TabularFeaturesConfig.model_validate(payload)


def test_tabular_features_config_rejects_forbid_nulls_outside_columns() -> None:
    """Reject constraint columns listed in forbid-nulls when absent from columns."""
    payload = _valid_payload()
    payload["constraints"]["forbid_nulls"] = ["unknown_column"]

    with pytest.raises(ConfigError, match="Forbidden nulls"):
        TabularFeaturesConfig.model_validate(payload)


def test_tabular_features_config_rejects_max_cardinality_outside_columns() -> None:
    """Reject max-cardinality constraints for columns that are not included."""
    payload = _valid_payload()
    payload["constraints"]["max_cardinality"] = {"unknown_column": 5}

    with pytest.raises(ConfigError, match="Max cardinality columns"):
        TabularFeaturesConfig.model_validate(payload)


def test_tabular_features_config_rejects_operator_required_features_outside_columns() -> None:
    """Reject operator required features when they reference non-included columns."""
    payload = _valid_payload()
    payload["operators"]["required_features"] = {"operator_a": ["missing_feature"]}

    with pytest.raises(ConfigError, match="Required features"):
        TabularFeaturesConfig.model_validate(payload)
