"""Schema loading and aggregation utilities for configured feature sets."""

import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def load_feature_set_schemas(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load input and derived schema CSVs for a single feature-set version path.

    Args:
        features_path: Feature-set version directory containing schema files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Input schema and derived schema dataframes.
    """

    input_schema_path = features_path / "input_schema.csv"
    derived_schema_path = features_path / "derived_schema.csv"
    if not input_schema_path.exists():
        msg = f"Input schema file not found at {input_schema_path}"
        logger.error(msg)
        raise DataError(msg)
    try:
        input_schema = pd.read_csv(input_schema_path)
        if not derived_schema_path.exists():
            msg = f"Derived schema file not found at {derived_schema_path}. Proceeding with empty derived schema."
            logger.warning(msg)
            derived_schema = pd.DataFrame()
        else:
            derived_schema = pd.read_csv(derived_schema_path)
        return input_schema, derived_schema
    except Exception:
        logger.exception(f"Failed to load schemas from {features_path}.")
        raise

def aggregate_schema_dfs(schemas: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate schema rows while preserving first occurrence per feature name.

    Args:
        schemas: List of schema dataframes to combine.

    Returns:
        pd.DataFrame: Aggregated schema dataframe with deduplicated features.
    """

    if not schemas:
        return pd.DataFrame()

    aggregated_rows = []
    seen_features = set()

    for schema_df in schemas:
        if "feature" not in schema_df.columns:
            msg = "Schema must contain a 'feature' column"
            logger.error(msg)
            raise DataError(msg)

        for _, row in schema_df.iterrows():
            feature = row["feature"]
            if feature not in seen_features:
                seen_features.add(feature)
                aggregated_rows.append(row)

    return pd.DataFrame(aggregated_rows).reset_index(drop=True)

def load_schemas(model_cfg: SearchModelConfig | TrainModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate input/derived schemas for all configured feature sets.

    Args:
        model_cfg: Validated model configuration containing feature-store settings.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Aggregated input and derived schemas.
    """

    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not feature_sets:
        msg = "No feature sets defined in model specifications."
        logger.error(msg)
        raise DataError(msg)

    input_schemas = []
    derived_schemas = []

    for fs in feature_sets:
        version_path = feature_store_path / fs.name / fs.version

        curr_input_schema, curr_derived_schema = load_feature_set_schemas(version_path)
        input_schemas.append(curr_input_schema)
        if not curr_derived_schema.empty:
            derived_schemas.append(curr_derived_schema)

    input_schema = aggregate_schema_dfs(input_schemas)
    derived_schema = aggregate_schema_dfs(derived_schemas)

    logger.debug(f"Successfully loaded and aggregated schemas for feature sets: {[fs.name for fs in feature_sets]}. Final schema shapes - Input: {input_schema.shape}, Derived: {derived_schema.shape}")

    return input_schema, derived_schema
