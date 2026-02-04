from pathlib import Path
import pandas as pd

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.io import load_and_hash_data
from ml.feature_freezing.freeze_strategies.tabular.segmentation import apply_segmentation
from ml.feature_freezing.freeze_strategies.tabular.validation import validate_min_rows, validate_min_class_count, validate_include_exclude_columns
from ml.feature_freezing.utils.operators import validate_operators

def ingest_data(config: TabularFeaturesConfig) -> tuple[pd.DataFrame, str]:
    data, data_hash = load_and_hash_data(
        Path(config.data.path),
        config.data.format
    )

    data = apply_segmentation(data, config)

    validate_min_rows(data, config.min_rows)

    if config.target.problem_type == "classification":
        validate_min_class_count(
            data[config.target.name],
            config.min_class_count
        )

    if config.operators:
        validate_operators(
            config.operators.list,
            config.operators.hash
        )

    validate_include_exclude_columns(config)

    return data, data_hash
