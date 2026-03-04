"""Schema-derived feature list utilities for pipeline assembly."""

import pandas as pd
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.pipelines.constants.pipeline_features import PipelineFeatures


def get_categorical_features(schema: pd.DataFrame) -> list[str]:
    """Return categorical feature names inferred from schema dtypes.

    Args:
        schema: Schema dataframe containing at least ``feature`` and ``dtype``.

    Returns:
        List of feature names with categorical-compatible dtypes.
    """
    return schema.loc[schema["dtype"].isin(["object", "string", "category"]), "feature"].tolist()

def get_pipeline_features(
    model_cfg: SearchModelConfig | TrainModelConfig,
    *,
    input_schema: pd.DataFrame,
    derived_schema: pd.DataFrame
) -> PipelineFeatures:
    """Compute feature groups used by the training/inference pipeline.

    Args:
        model_cfg: Validated search or training model config.
        input_schema: Schema dataframe for raw input features.
        derived_schema: Schema dataframe for engineered features.

    Returns:
        PipelineFeatures: Structured feature groups for pipeline construction.
    """
    input_features = input_schema["feature"].tolist()
    derived_features = derived_schema["feature"].tolist()

    seg_cfg = model_cfg.segmentation
    seg_enabled = seg_cfg.enabled
    include_seg = seg_cfg.include_in_model

    # If segmentation enabled and not included, drop filter columns from features
    if seg_enabled and not include_seg:
        seg_columns = [f.column for f in seg_cfg.filters]
        model_input_features = [f for f in input_features if f not in seg_columns]
    else:
        model_input_features = input_features

    selected_features = model_input_features + derived_features

    categorical_features = get_categorical_features(input_schema[input_schema["feature"].isin(model_input_features)])

    output = PipelineFeatures(
        input_features=model_input_features,
        derived_features=derived_features,
        categorical_features=categorical_features,
        selected_features=selected_features
    )

    return output
