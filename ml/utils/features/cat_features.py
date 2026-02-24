import pandas as pd

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig


def get_cat_features(
    model_cfg: SearchModelConfig | TrainModelConfig,
    input_schema: pd.DataFrame, 
    derived_schema: pd.DataFrame
) -> list:
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