"""Unit tests for extracting model-scoped search parameters."""

import pytest
from ml.search.utils.model_params_extraction import extract_model_params

pytestmark = pytest.mark.unit


def test_extract_model_params_keeps_only_model_prefixed_keys() -> None:
    """Return only `Model__` entries and strip exactly one prefix segment."""
    best_params = {
        "Model__depth": 8,
        "Model__l2_leaf_reg": 3.5,
        "preprocessor__num__imputer": "median",
        "random_state": 42,
    }

    result = extract_model_params(best_params)

    assert result == {"depth": 8, "l2_leaf_reg": 3.5}


def test_extract_model_params_preserves_suffix_after_first_separator() -> None:
    """Keep nested suffix fragments after removing only the first `Model__` prefix."""
    best_params = {
        "Model__tree__depth": 6,
        "Model__regularization__alpha": 0.1,
    }

    result = extract_model_params(best_params)

    assert result == {"tree__depth": 6, "regularization__alpha": 0.1}


def test_extract_model_params_returns_empty_mapping_without_model_keys() -> None:
    """Return an empty dictionary when no model-prefixed parameter is present."""
    best_params = {
        "cv": 5,
        "scoring": "roc_auc",
    }

    assert extract_model_params(best_params) == {}
