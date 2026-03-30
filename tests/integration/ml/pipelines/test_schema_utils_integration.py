from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import ml.pipelines.schema_utils as mod
import pandas as pd
import pytest
from ml.pipelines.constants.pipeline_features import PipelineFeatures

pytestmark = pytest.mark.integration


def test_get_categorical_and_pipeline_features_with_and_without_segmentation() -> None:
    input_schema = pd.DataFrame({"feature": ["a", "b", "c"], "dtype": ["object", "int64", "category"]})
    derived_schema = pd.DataFrame({"feature": ["d"], "source_operator": ["op"]})

    # segmentation disabled -> include all input features
    seg_cfg = SimpleNamespace(enabled=False, include_in_model=False, filters=[])
    model_cfg = SimpleNamespace(segmentation=seg_cfg)

    features: PipelineFeatures = mod.get_pipeline_features(
        model_cfg=cast(mod.SearchModelConfig, model_cfg), input_schema=input_schema, derived_schema=derived_schema
    )

    assert features.input_features == ["a", "b", "c"]
    assert features.derived_features == ["d"]
    assert "a" in features.categorical_features

    # segmentation enabled but exclude segmentation columns from model inputs
    seg_filters = [SimpleNamespace(column="b")]
    seg_cfg2 = SimpleNamespace(enabled=True, include_in_model=False, filters=seg_filters)
    model_cfg2 = SimpleNamespace(segmentation=seg_cfg2)

    features2: PipelineFeatures = mod.get_pipeline_features(
        model_cfg=cast(mod.SearchModelConfig, model_cfg2), input_schema=input_schema, derived_schema=derived_schema
    )

    assert "b" not in features2.input_features
