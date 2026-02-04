import pandas as pd
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.feature_freezing.freeze_strategies.tabular.persistence import create_metadata
from ml.feature_freezing.freeze_strategies.tabular.io import hash_feature_set, validate_feature_set_hashes_match
from ml.utils.git import get_git_commit
from ml.feature_freezing.utils.schema import hash_data_schema

def build_metadata(config: TabularFeaturesConfig, snapshot_path: Path, schema_path: Path,
                   splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], data_hash: str, config_hash: str) -> dict:

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = splits

    train_schema_hash = hash_data_schema(X_train)
    val_schema_hash = hash_data_schema(X_val)
    test_schema_hash = hash_data_schema(X_test)

    feature_set_hash = hash_feature_set(X_train)

    validate_feature_set_hashes_match(X_val, feature_set_hash)
    validate_feature_set_hashes_match(X_test, feature_set_hash)

    git_commit = get_git_commit(Path("."))

    operators_hash = (
        config.operators.hash
        if config.operators else "none"
    )

    return create_metadata(
        snapshot_path,
        schema_path,
        data_hash,
        train_schema_hash,
        val_schema_hash,
        test_schema_hash,
        operators_hash,
        config_hash,
        feature_set_hash,
        git_commit,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        config.target.problem_type,
    )
