"""Unit tests for shared base model/ensemble parameter schemas."""

import pytest
from ml.config.schemas.base_model_params import BaseEnsembleParams, BaseModelParams

pytestmark = pytest.mark.unit


def test_base_model_params_defaults_to_none_for_all_fields() -> None:
    """Test that the BaseModelParams schema defaults all fields to None when not provided."""
    cfg = BaseModelParams.model_validate({})

    assert cfg.depth is None
    assert cfg.learning_rate is None
    assert cfg.l2_leaf_reg is None
    assert cfg.random_strength is None
    assert cfg.min_data_in_leaf is None
    assert cfg.border_count is None


def test_base_model_params_accepts_partial_parameter_payload() -> None:
    """Test that the BaseModelParams schema accepts a partial payload with only some fields specified."""
    cfg = BaseModelParams.model_validate(
        {
            "depth": 6,
            "learning_rate": 0.05,
            "min_data_in_leaf": 16,
        }
    )

    assert cfg.depth == 6
    assert cfg.learning_rate == pytest.approx(0.05)
    assert cfg.min_data_in_leaf == 16
    assert cfg.border_count is None


def test_base_ensemble_params_defaults_to_none_for_all_fields() -> None:
    """Test that the BaseEnsembleParams schema defaults all fields to None when not provided."""
    cfg = BaseEnsembleParams.model_validate({})

    assert cfg.bagging_temperature is None
    assert cfg.colsample_bylevel is None


def test_base_ensemble_params_accepts_explicit_values() -> None:
    """Test that the BaseEnsembleParams schema correctly accepts and validates explicit values for its fields."""
    cfg = BaseEnsembleParams.model_validate(
        {
            "bagging_temperature": 0.9,
            "colsample_bylevel": 0.7,
        }
    )

    assert cfg.bagging_temperature == pytest.approx(0.9)
    assert cfg.colsample_bylevel == pytest.approx(0.7)
