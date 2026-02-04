import logging
logger = logging.getLogger(__name__)
import pandas as pd
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.exceptions import PersistenceError
from ml.feature_freezing.freeze_strategies.tabular.persistence import persist_feature_snapshot
from ml.feature_freezing.freeze_strategies.tabular.persistence import save_input_schema, save_derived_schema

def persist_all(config: TabularFeaturesConfig, splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], now) -> tuple[Path, Path]:

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = splits

    snapshot_path = persist_feature_snapshot(
        config,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        now,
    )

    schema_path = config.feature_store_path

    try:
        save_input_schema(schema_path, X_train)
    except Exception as e:
        logger.exception("Failed to save input schema")
        raise PersistenceError(
            f"Could not save input schema at {schema_path}"
        ) from e

    try:
        if config.operators:
            save_derived_schema(
                schema_path,
                X_train,
                config.operators.list,
                config.operators.mode,
            )
    except Exception as e:
        logger.exception("Failed to save derived schema")
        raise PersistenceError(
            f"Could not save derived schema at {schema_path}"
        ) from e

    return snapshot_path, schema_path
