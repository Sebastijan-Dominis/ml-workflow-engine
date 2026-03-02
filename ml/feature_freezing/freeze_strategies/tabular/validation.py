"""Validation helpers for tabular feature constraints and data types."""

import logging

import pandas as pd

from ml.exceptions import DataError
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig
from ml.utils.features.validation.normalize_dtype import normalize_dtype

logger = logging.getLogger(__name__)
        
def validate_input_no_nulls(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    """Validate configured no-null constraints on selected feature columns.

    Args:
        X: Feature dataframe or series.
        config: Tabular feature-freezing configuration.

    Returns:
        None: Raises on validation failure.
    """

    forbidden_nulls = config.constraints.forbid_nulls
    if forbidden_nulls:
        for col in forbidden_nulls:
            if col in X.columns and X[col].isnull().any():
                msg = f"Feature {col} contains null values, which is forbidden by constraints."
                logger.error(msg)
                raise DataError(msg)
    logger.debug("Null value validation passed for all features.")
        
def validate_max_cardinality(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    """Validate categorical cardinality does not exceed configured limits.

    Args:
        X: Feature dataframe or series.
        config: Tabular feature-freezing configuration.

    Returns:
        None: Raises on validation failure.
    """

    categorical_features = config.feature_roles.categorical
    max_cardinality = config.constraints.max_cardinality

    if max_cardinality:
        for col in categorical_features:
            if col in X.columns:
                cardinality = X[col].nunique()
                if cardinality > max_cardinality.get(col, float('inf')):
                    msg = f"Categorical feature {col} exceeds max cardinality of {max_cardinality.get(col, float('inf'))} with {cardinality} unique values."
                    logger.error(msg)
                    raise DataError(msg)
    logger.debug("Max cardinality validation passed for all categorical features.")

def validate_constraints(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    """Run all configured feature constraint validations.

    Args:
        X: Feature dataframe or series.
        config: Tabular feature-freezing configuration.

    Returns:
        None: Raises on validation failure.
    """

    validate_input_no_nulls(X, config)
    validate_max_cardinality(X, config)

def validate_data_types(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    """Validate feature dtypes against configured role-specific allowlists.

    Args:
        X: Feature dataframe or series.
        config: Tabular feature-freezing configuration.

    Returns:
        None: Raises on validation failure.
    """

    categorical_features = config.feature_roles.categorical
    numerical_features = config.feature_roles.numerical
    datetime_features = config.feature_roles.datetime
    allowed_categorical_types = ["object", "category", "bool", "string"]
    allowed_numerical_types = ["int64", "float64", "int32", "float32", "int16", "int8"]
    allowed_datetime_types = ["datetime64[ns]", "datetime64[ns, UTC]"]

    for col in categorical_features:
        if col in X.columns:
            actual_dtype = normalize_dtype(X[col].dtype)
            if actual_dtype not in allowed_categorical_types:
                msg = f"Categorical feature {col} has invalid dtype {X[col].dtype}"
                logger.error(msg)
                raise DataError(msg)
        
    for col in numerical_features:
        if col in X.columns:
            actual_dtype = normalize_dtype(X[col].dtype)
            if actual_dtype not in allowed_numerical_types:
                msg = f"Numerical feature {col} has invalid dtype {X[col].dtype}"
                logger.error(msg)
                raise DataError(msg)
        
    for col in datetime_features:
        if col in X.columns:
            actual_dtype = normalize_dtype(X[col].dtype)
            if actual_dtype not in allowed_datetime_types:
                msg = f"Datetime feature {col} has invalid dtype {X[col].dtype}"
                logger.error(msg)
                raise DataError(msg)
            
    logger.debug("Data type validation passed for all features.")