"""Schema loading and aggregation utilities for configured feature sets."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import DataError
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.utils.loaders import load_json

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

def load_feature_set_schemas(features_path: Path, file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load input and derived schema CSVs for a single feature-set version path.

    Args:
        features_path: Feature-set version directory containing schema files.
        file_path: Path to the feature set snapshot for which to load schemas.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Input schema and derived schema dataframes.
    """

    # Deliberate lazy import to avoid circular dependency with operator validation logic
    from ml.features.validation.validate_operators import validate_operators

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
            operators = []
        else:
            derived_schema = pd.read_csv(derived_schema_path)
            operators = derived_schema["source_operator"].tolist()

        snapshot_metadata = load_json(file_path / "metadata.json")
        expected_operator_hash = snapshot_metadata["operator_hash"]
        validate_operators(operators, expected_operator_hash, str(file_path))

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

def load_schemas(model_cfg: SearchModelConfig | TrainModelConfig, feature_lineage: list[FeatureLineage]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate input/derived schemas for all configured feature sets.

    Args:
        model_cfg: Validated model configuration containing feature-store settings.
        feature_lineage: Feature lineage information for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Aggregated input and derived schemas.
    """

    feature_store_path = Path(model_cfg.feature_store.path)

    if not feature_lineage:
        msg = "No feature sets defined in model configuration; cannot load schemas."
        logger.error(msg)
        raise DataError(msg)

    input_schemas = []
    derived_schemas = []

    for fs in feature_lineage:
        version_path = feature_store_path / fs.name / fs.version
        snapshot_path = version_path / fs.snapshot_id

        curr_input_schema, curr_derived_schema = load_feature_set_schemas(version_path, snapshot_path)
        input_schemas.append(curr_input_schema)
        if not curr_derived_schema.empty:
            derived_schemas.append(curr_derived_schema)

    input_schema = aggregate_schema_dfs(input_schemas)
    derived_schema = aggregate_schema_dfs(derived_schemas)

    logger.debug(f"Successfully loaded and aggregated schemas for feature sets: {[fs.name for fs in feature_lineage]}. Final schema shapes - Input: {input_schema.shape}, Derived: {derived_schema.shape}")

    return input_schema, derived_schema
