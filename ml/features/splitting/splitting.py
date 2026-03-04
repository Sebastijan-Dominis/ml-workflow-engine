"""Utilities for train/validation/test splitting of feature datasets."""

import logging

import pandas as pd
from ml.config.schemas.model_specs import DATA_TYPE, SplitConfig, TaskConfig
from ml.types import AllSplitsInfo, SplitInfo, TabularSplits
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

SPLIT = tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]

def random_split(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int, stratify: pd.Series | None) -> SPLIT:
    """Perform a random train-test split with optional stratification.

    Args:
        X: Feature dataframe.
        y: Target series.
        test_size: Fraction reserved for the holdout partition.
        random_state: Random seed used by sklearn split.
        stratify: Optional stratification series.

    Returns:
        SPLIT: Split tuple ``(X_train, X_test, y_train, y_test)``.
    """

    return tuple(train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify))

def split_data(X: pd.DataFrame, y: pd.Series, split_cfg: SplitConfig, test_size: float) -> SPLIT:
    """Dispatch to the configured split strategy and return split partitions.

    Args:
        X: Feature dataframe.
        y: Target series.
        split_cfg: Split strategy configuration.
        test_size: Holdout fraction for this split call.

    Returns:
        SPLIT: Split tuple ``(X_1, X_2, y_1, y_2)`` for the configured strategy.
    """

    # Expandable for future split strategies
    SPLIT_REGISTRY = {
        "random": random_split,
    }

    split_func = SPLIT_REGISTRY[split_cfg.strategy]

    X1, X2, y1, y2 = split_func(
        X, y,
        test_size=test_size,
        random_state=split_cfg.random_state,
        stratify=y if split_cfg.stratify_by and isinstance(y, pd.Series) else None
    )

    return X1, X2, y1, y2

def get_splits_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_cfg: SplitConfig,
    task_cfg: TaskConfig
) -> tuple[TabularSplits, AllSplitsInfo]:
    """Produce tabular train/val/test splits and corresponding summary statistics.

    Args:
        X: Feature dataframe.
        y: Target series.
        split_cfg: Split configuration with test/validation fractions.
        task_cfg: Task configuration used for classification-specific metrics.

    Returns:
        tuple[TabularSplits, AllSplitsInfo]: Materialized split data and split metadata.
    """

    X_train_val, X_test, y_train_val, y_test = split_data(
        X,
        y,
        split_cfg,
        test_size=split_cfg.test_size,
    )

    relative_val_size = (
        split_cfg.val_size /
        (1.0 - split_cfg.test_size)
    )

    X_train, X_val, y_train, y_val = split_data(
        X_train_val,
        y_train_val,
        split_cfg,
        test_size=relative_val_size,
    )

    splits = TabularSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test
    )

    train_rows = len(X_train)
    val_rows = len(X_val)
    test_rows = len(X_test)

    train_info = SplitInfo(n_rows=train_rows)
    val_info = SplitInfo(n_rows=val_rows)
    test_info = SplitInfo(n_rows=test_rows)

    if task_cfg.type == "classification":
        train_info.class_distribution = y_train.value_counts(normalize=True).to_dict()
        val_info.class_distribution = y_val.value_counts(normalize=True).to_dict()
        test_info.class_distribution = y_test.value_counts(normalize=True).to_dict()

        train_info.positive_rate = y_train.mean()
        val_info.positive_rate = y_val.mean()
        test_info.positive_rate = y_test.mean()

    splits_info = AllSplitsInfo(
        train=train_info,
        val=val_info,
        test=test_info
    )

    return splits, splits_info

def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    split_cfg: SplitConfig,
    data_type: DATA_TYPE,
    task_cfg: TaskConfig
) -> tuple[TabularSplits, AllSplitsInfo]:
    """Route split logic by data type and return splits with metadata.

    Args:
        X: Feature dataframe.
        y: Target series.
        split_cfg: Split configuration.
        data_type: Dataset modality selector.
        task_cfg: Task configuration for split metadata generation.

    Returns:
        tuple[TabularSplits, AllSplitsInfo]: Computed splits and associated split info.
    """

    if data_type == "tabular":
        splits, splits_info = get_splits_tabular(X, y, split_cfg=split_cfg, task_cfg=task_cfg)
        logger.info(f"Data split into train/val/test. Splits info:\n{splits_info}")
        return splits, splits_info
    elif data_type == "time-series":
        raise NotImplementedError("Time-series split not implemented yet.")
