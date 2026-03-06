"""Unit tests for search configuration schemas."""

import pytest
from ml.config.schemas.search_cfg import BroadParamDistributions, SearchConfig

pytestmark = pytest.mark.unit


def test_broad_param_distributions_to_flat_dict_uses_model_prefix() -> None:
    """Flatten broad parameter distributions using default model prefixes."""
    params = BroadParamDistributions.model_validate(
        {
            "model": {
                "depth": [4, 6],
                "learning_rate": [0.03, 0.1],
            },
            "ensemble": {
                "bagging_temperature": [0.0, 1.0],
            },
        }
    )

    flat = params.to_flat_dict()

    assert flat == {
        "Model__depth": [4, 6],
        "Model__learning_rate": [0.03, 0.1],
        "Model__bagging_temperature": [0.0, 1.0],
    }


def test_broad_param_distributions_to_flat_dict_respects_custom_prefixes() -> None:
    """Flatten parameter distributions with caller-provided prefixes."""
    params = BroadParamDistributions.model_validate(
        {
            "model": {"depth": [8]},
            "ensemble": {"bagging_temperature": [0.5]},
        }
    )

    flat = params.to_flat_dict(prefix_map={"model": "Estimator", "ensemble": "Ensemble"})

    assert flat == {
        "Estimator__depth": [8],
        "Ensemble__bagging_temperature": [0.5],
    }


def test_search_config_defaults_to_disabled_narrow_and_gpu_hardware() -> None:
    """Default narrow-search settings and hardware values as expected."""
    cfg = SearchConfig.model_validate(
        {
            "random_state": 42,
            "broad": {
                "iterations": 2,
                "n_iter": 5,
                "param_distributions": {},
            },
        }
    )

    assert cfg.narrow.enabled is False
    assert cfg.narrow.iterations == 0
    assert cfg.narrow.n_iter == 0
    assert cfg.hardware.task_type == "GPU"


def test_search_config_normalizes_hardware_task_type_case() -> None:
    """Normalize search hardware `task_type` casing."""
    cfg = SearchConfig.model_validate(
        {
            "random_state": 42,
            "broad": {
                "iterations": 1,
                "n_iter": 1,
                "param_distributions": {},
            },
            "hardware": {
                "task_type": "cpu",
                "devices": [],
            },
        }
    )

    assert cfg.hardware.task_type == "CPU"
