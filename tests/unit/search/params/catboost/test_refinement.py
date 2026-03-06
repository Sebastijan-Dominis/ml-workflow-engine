"""Unit tests for CatBoost narrow-search parameter refinement."""

from types import SimpleNamespace

import pytest
from ml.search.params.catboost.refinement import prepare_narrow_params

pytestmark = pytest.mark.unit


def test_prepare_narrow_params_refines_included_fields_for_cpu() -> None:
    """Refine included parameters using config-provided ranges for CPU searches."""
    best_params = {
        "Model__depth": 8,
        "Model__learning_rate": 0.1,
        "Model__min_data_in_leaf": 20,
        "Model__bagging_temperature": 1.0,
        "Model__border_count": 128,
        "Model__colsample_bylevel": 0.9,
    }
    narrow_cfg = SimpleNamespace(
        model=SimpleNamespace(
            depth=SimpleNamespace(include=True, offsets=[1, 2], low=2, high=12),
            learning_rate=SimpleNamespace(
                include=True,
                factors=[0.8, 1.0, 1.2],
                low=0.01,
                high=0.2,
                decimals=3,
            ),
            l2_leaf_reg=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            min_data_in_leaf=SimpleNamespace(include=True, offsets=[2, 5], low=1, high=30),
            random_strength=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            border_count=SimpleNamespace(include=True),
            colsample_bylevel=SimpleNamespace(
                include=True,
                factors=[0.9, 1.0, 1.1],
                low=0.5,
                high=1.0,
                decimals=2,
            ),
        ),
        ensemble=SimpleNamespace(
            bagging_temperature=SimpleNamespace(
                include=True,
                factors=[0.5, 1.0, 1.5],
                low=0.0,
                high=2.0,
                decimals=2,
            )
        ),
    )

    refined = prepare_narrow_params(best_params, narrow_cfg, task_type="CPU")

    assert refined == {
        "Model__depth": [6, 7, 8, 9, 10],
        "Model__learning_rate": [0.08, 0.1, 0.12],
        "Model__min_data_in_leaf": [15, 18, 20, 22, 25],
        "Model__bagging_temperature": [0.5, 1.0, 1.5],
        "Model__border_count": [64, 128, 254],
        "Model__colsample_bylevel": [0.81, 0.9, 0.99],
    }


def test_prepare_narrow_params_forces_gpu_colsample_bylevel_to_one() -> None:
    """Pin colsample_bylevel to 1.0 on GPU regardless of best-value neighborhood."""
    best_params = {
        "Model__colsample_bylevel": 0.77,
    }
    narrow_cfg = SimpleNamespace(
        model=SimpleNamespace(
            depth=SimpleNamespace(include=False, offsets=None, low=None, high=None),
            learning_rate=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            l2_leaf_reg=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            min_data_in_leaf=SimpleNamespace(include=False, offsets=None, low=None, high=None),
            random_strength=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            border_count=SimpleNamespace(include=False),
            colsample_bylevel=SimpleNamespace(
                include=True,
                factors=[0.9, 1.0, 1.1],
                low=0.5,
                high=1.0,
                decimals=2,
            ),
        ),
        ensemble=SimpleNamespace(
            bagging_temperature=SimpleNamespace(
                include=False,
                factors=None,
                low=None,
                high=None,
                decimals=None,
            )
        ),
    )

    refined = prepare_narrow_params(best_params, narrow_cfg, task_type="GPU")

    assert refined == {"Model__colsample_bylevel": [1.0]}


def test_prepare_narrow_params_ignores_missing_or_excluded_inputs() -> None:
    """Skip keys not present in best params or marked with include=False."""
    best_params = {
        "Model__learning_rate": 0.1,
    }
    narrow_cfg = SimpleNamespace(
        model=SimpleNamespace(
            depth=SimpleNamespace(include=True, offsets=[1], low=2, high=12),
            learning_rate=SimpleNamespace(
                include=False,
                factors=[0.9, 1.1],
                low=0.01,
                high=0.2,
                decimals=3,
            ),
            l2_leaf_reg=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            min_data_in_leaf=SimpleNamespace(include=False, offsets=None, low=None, high=None),
            random_strength=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
            border_count=SimpleNamespace(include=False),
            colsample_bylevel=SimpleNamespace(include=False, factors=None, low=None, high=None, decimals=None),
        ),
        ensemble=SimpleNamespace(
            bagging_temperature=SimpleNamespace(
                include=False,
                factors=None,
                low=None,
                high=None,
                decimals=None,
            )
        ),
    )

    refined = prepare_narrow_params(best_params, narrow_cfg, task_type="CPU")

    assert refined == {}
