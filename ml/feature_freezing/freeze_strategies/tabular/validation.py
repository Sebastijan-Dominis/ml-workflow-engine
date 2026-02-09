import logging

import numpy as np
import pandas as pd

from ml.exceptions import DataError, UserError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

logger = logging.getLogger(__name__)

def validate_min_rows(data: pd.DataFrame, min_rows: int):
    if not min_rows:
        logger.warning("Minimum rows constraint not set.")
    
    logger.debug(f"Validating minimum rows: data has {len(data)} rows, minimum required is {min_rows}.")
    
    if len(data) < min_rows:
        msg = f"Data has {len(data)} rows, which is less than the minimum required {min_rows} rows."
        logger.error(msg)
        raise DataError(msg)

def validate_min_class_count(y: pd.Series, min_class_count: int):
    if y.nunique() < 2:
        msg = "Target variable must have at least two classes for classification."
        logger.error(msg)
        raise DataError(msg)
    
    if not min_class_count:
        logger.warning("Minimum class count constraint not set.")
    
    logger.debug(f"Validating minimum class count: minimum required is {min_class_count}.")
    
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < min_class_count:
            msg = f"Class {cls} has {count} instances, which is less than the minimum required {min_class_count}."
            logger.error(msg)
            raise DataError(msg)
        else:
            logger.debug(f"Class {cls} has {count} instances, which meets the minimum required {min_class_count}.")

def validate_include_exclude_columns(config: TabularFeaturesConfig):
    include = set(config.columns.include)
    exclude = set(
        (config.columns.exclude.leaky or []) +
        (config.columns.exclude.useless or []) +
        [config.target.name]
    )
    intersection = include.intersection(exclude)
    if intersection:
        msg = f"Columns {intersection} are present in both include and exclude lists."
        logger.error(msg)
        raise UserError(msg)
    
def normalize_dtype(dtype) -> str:
    """
    Normalize any pandas dtype (including extension dtypes) to a string.
    """
    # Handle categorical
    if hasattr(dtype, "categories") and hasattr(dtype, "ordered"):
        return "category"

    # Handle nullable string dtype
    if str(dtype) == "string[python]" or str(dtype) == "string":
        return "object"

    # Handle nullable integers (Int64, Int32, Int16, Int8)
    if str(dtype).startswith("Int") or str(dtype).startswith("UInt"):
        return "int64"

    if np.issubdtype(dtype, np.integer):
        return "int64"
    if np.issubdtype(dtype, np.floating):
        return "float64"
    if np.issubdtype(dtype, np.bool_):
        return "bool"
    if np.issubdtype(dtype, np.object_):
        return "object"
    if np.issubdtype(dtype, np.datetime64):
        return "datetime64[ns]"
    return str(dtype)

def validate_target(y: pd.Series, config: TabularFeaturesConfig):
    if y.isnull().any():
        msg = "Target variable contains null values."
        logger.error(msg)
        raise DataError(msg)
    
    actual_dtype = normalize_dtype(y.dtype)
    allowed = config.target.allowed_dtypes
    if actual_dtype not in allowed:
        msg = f"Target variable has dtype {y.dtype}, expected one of {allowed}."
        logger.error(msg)
        raise DataError(msg)
    
    positive_class = config.target.classes.positive_class if config.target.classes else None
    if positive_class is not None and positive_class not in y.unique():
        msg = f"Positive class {positive_class} not found in target variable."
        logger.error(msg)
        raise DataError(msg)
    
    if config.target.problem_type == "classification":
        return  # No further checks for classification

    target_constraints = config.target.constraints
    min_val = target_constraints.min_value
    max_val = target_constraints.max_value
    if min_val is not None and y.min() < min_val:
        msg = f"Target min {y.min()} < allowed min {min_val}"
        logger.error(msg)
        raise DataError(msg)
    if max_val is not None and y.max() > max_val:
        msg = f"Target max {y.max()} > allowed max {max_val}"
        logger.error(msg)
        raise DataError(msg)
        
def validate_input_no_nulls(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    forbidden_nulls = config.constraints.forbid_nulls
    if forbidden_nulls:
        for col in forbidden_nulls:
            if col in X.columns and X[col].isnull().any():
                msg = f"Feature {col} contains null values, which is forbidden by constraints."
                logger.error(msg)
                raise DataError(msg)
        
def validate_max_cardinality(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
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

def validate_constraints(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
    validate_input_no_nulls(X, config)
    validate_max_cardinality(X, config)

def validate_data_types(X: pd.DataFrame | pd.Series, config: TabularFeaturesConfig):
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