"""Unit tests for CatBoost search context guard properties."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import RuntimeMLError
from ml.search.searchers.catboost.pipeline.context import SearchContext

pytestmark = pytest.mark.unit


def _make_ctx(tmp_path: Path) -> SearchContext:
    """Create a minimal context instance with dataclass-required fields only."""
    model_cfg = cast(SearchModelConfig, SimpleNamespace())
    return SearchContext(
        model_cfg=model_cfg,
        strict=True,
        failure_management_dir=tmp_path,
    )


@pytest.mark.parametrize(
    ("property_name", "error_fragment"),
    [
        ("require_x_train", "X_train not prepared yet"),
        ("require_y_train", "y_train not prepared yet"),
        ("require_splits_info", "Splits info not prepared yet"),
        ("require_feature_lineage", "Feature lineage not prepared yet"),
        ("require_input_schema", "Input schema not prepared yet"),
        ("require_derived_schema", "Derived schema not prepared yet"),
        ("require_pipeline_cfg", "Pipeline config not loaded yet"),
        ("require_pipeline_hash", "Pipeline hash not computed yet"),
        ("require_cat_features", "Categorical features not prepared yet"),
        ("require_scoring", "Scoring function not prepared yet"),
        ("require_best_params_1", "Best parameters from broad search not available yet"),
        ("require_broad_result", "Broad search result not available yet"),
        ("require_narrow_disabled", "Narrow search enabled/disabled flag not set yet"),
        ("require_best_params", "Best parameters from narrow search not available yet"),
        ("require_narrow_result", "Narrow search result not available yet"),
    ],
)
def test_search_context_require_properties_raise_when_unset(
    property_name: str,
    error_fragment: str,
    tmp_path: Path,
) -> None:
    """Raise `RuntimeMLError` with contextual message when required fields are unset."""
    ctx = _make_ctx(tmp_path)

    with pytest.raises(RuntimeMLError, match=error_fragment):
        _ = getattr(ctx, property_name)


@pytest.mark.parametrize(
    ("field_name", "property_name", "value"),
    [
        ("X_train", "require_x_train", pd.DataFrame({"x": [1]})),
        ("y_train", "require_y_train", pd.Series([1])),
        ("splits_info", "require_splits_info", {"train_rows": 1}),
        ("feature_lineage", "require_feature_lineage", [SimpleNamespace(name="f")]),
        ("input_schema", "require_input_schema", pd.DataFrame({"feature": ["x"], "dtype": ["int64"]})),
        ("derived_schema", "require_derived_schema", pd.DataFrame({"feature": ["x2"], "source_operator": ["op"]})),
        ("pipeline_cfg", "require_pipeline_cfg", {"steps": ["SchemaValidator"]}),
        ("pipeline_hash", "require_pipeline_hash", "hash-123"),
        ("cat_features", "require_cat_features", ["country"]),
        ("scoring", "require_scoring", "roc_auc"),
        ("best_params_1", "require_best_params_1", {"Model__depth": 6}),
        ("broad_result", "require_broad_result", {"best_score": 0.7}),
        ("narrow_disabled", "require_narrow_disabled", True),
        ("best_params", "require_best_params", {"Model__depth": 7}),
        ("narrow_result", "require_narrow_result", {"best_score": 0.8}),
    ],
)
def test_search_context_require_properties_return_values_when_set(
    field_name: str,
    property_name: str,
    value: Any,
    tmp_path: Path,
) -> None:
    """Return the stored field value once required context state is populated."""
    ctx = _make_ctx(tmp_path)
    setattr(ctx, field_name, value)

    result = getattr(ctx, property_name)

    assert result is value
