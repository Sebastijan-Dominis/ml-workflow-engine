"""Utilities for resolving categorical feature columns for model training."""

import pandas as pd
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig


def get_cat_features(
    model_cfg: SearchModelConfig | TrainModelConfig,
    input_schema: pd.DataFrame,
    derived_schema: pd.DataFrame
) -> list:
    """Return categorical feature names from input/derived schemas with segmentation rules.

    Args:
        model_cfg: Validated training or search model configuration.
        input_schema: Schema dataframe for raw input features.
        derived_schema: Schema dataframe for derived features.

    Returns:
        Combined list of categorical input and derived feature names.
    """

    input_categoricals = input_schema.loc[
        input_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    seg_enabled = model_cfg.segmentation.enabled
    include_seg = model_cfg.segmentation.include_in_model
    if seg_enabled and not include_seg:
        seg_columns = [f.column for f in model_cfg.segmentation.filters]
        input_categoricals = [f for f in input_categoricals if f not in seg_columns]

    return input_categoricals + derived_categoricals
