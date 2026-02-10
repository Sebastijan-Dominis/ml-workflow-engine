import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import DataError
from ml.utils.features.loading.latest_snapshot import get_latest_snapshot
from ml.utils.features.validation import validate_feature_set, validate_set
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def load_feature_set_schemas(features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_schema_path = features_path / "input_schema.csv"
    derived_schema_path = features_path / "derived_schema.csv"
    try:
        input_schema = pd.read_csv(input_schema_path)
        derived_schema = pd.read_csv(derived_schema_path)
        return input_schema, derived_schema
    except Exception:
        logger.exception(f"Failed to load schemas from {features_path}.")
        raise

def aggregate_schema_dfs(schemas: list[pd.DataFrame]) -> pd.DataFrame:
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
    feature_store_path = Path(model_cfg.feature_store.path)
    feature_sets = model_cfg.feature_store.feature_sets

    if not feature_sets:
        msg = "No feature sets defined in model specifications."
        logger.error(msg)
        raise DataError(msg)

    input_schemas = []
    derived_schemas = []

    loader_validation_hashes = set()

    for fs in feature_sets:
        version_path = feature_store_path / fs.ref.replace(".", "/") / fs.name / fs.version
        latest_snapshot = get_latest_snapshot(version_path)
        metadata = load_json(latest_snapshot / "metadata.json")

        validate_feature_set(latest_snapshot, metadata)

        curr_input_schema, curr_derived_schema = load_feature_set_schemas(version_path)
        input_schemas.append(curr_input_schema)
        derived_schemas.append(curr_derived_schema)

        loader_validation_hashes.add(metadata["loader_validation_hash"])

    validate_set("Data", loader_validation_hashes, feature_sets)
    input_schema = aggregate_schema_dfs(input_schemas)
    derived_schema = aggregate_schema_dfs(derived_schemas)

    return input_schema, derived_schema
