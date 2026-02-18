import logging

import pandas as pd

from ml.data.utils.config.schemas.interim import (Cleaning, DataSchema,
                                                  Invariants)
from ml.exceptions import DataError

logger = logging.getLogger(__name__)


def normalize_columns(df: pd.DataFrame, cleaning: Cleaning):
    if cleaning.lowercase_columns:
        df.columns = df.columns.str.lower()
    if cleaning.strip_strings:
        df.columns = df.columns.str.strip()
    if cleaning.replace_spaces_in_columns:
        df.columns = df.columns.str.replace(" ", "_")
    if cleaning.replace_dashes_in_columns:
        df.columns = df.columns.str.replace("-", "_")
    return df

def enforce_schema(df: pd.DataFrame, *, schema: DataSchema, drop_missing_ints: bool) -> pd.DataFrame:
    try:
        schema_cols = schema.model_dump().keys()
        missing_cols = set(schema_cols) - set(df.columns)
        if missing_cols:
            msg = f"Dataframe is missing columns required by the schema: {missing_cols}"
            logger.error(msg)
            raise DataError(msg)
        
        extra_cols = set(df.columns) - set(schema_cols)
        if extra_cols:
            logger.warning(f"Dataframe has extra columns not defined in the schema: {extra_cols}. These columns will be dropped.")
            df = df.drop(columns=extra_cols)

        for col, dtype in schema.model_dump().items():
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col])
            else:
                if dtype.startswith("int") and df[col].isnull().any():
                    if drop_missing_ints:
                        missing_count = df[col].isnull().sum()
                        df.dropna(subset=[col], inplace=True)
                        logger.warning(f"Column '{col}' had {missing_count} missing values and drop_missing_ints is True, so rows with missing values in this column have been dropped.")
                    else:
                        df[col] = df[col].astype("float64")
                        logger.warning(f"Column '{col}' has missing values and cannot be converted to integer due to drop_missing_ints=False. Converted to float instead.")
                else:
                    df[col] = df[col].astype(dtype)
        return df
    except Exception as e:
        msg = f"Error enforcing data schema on dataframe. "
        logger.error(msg + f"Details: {str(e)}")
        raise DataError(msg) from e

def clean_data(df: pd.DataFrame, invariants: Invariants):
    try:
        invariants_dict = invariants.model_dump()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
            if col in invariants_dict and invariants_dict[col] is not None:
                rules = invariants_dict[col]
                if hasattr(rules, 'min') and rules.min is not None:
                    invalid_rows = df[df[col] < rules.min]
                    logger.warning("Dropping %d rows where %s < %s", len(invalid_rows), col, rules.min)
                    df.drop(invalid_rows.index, inplace=True)
                if hasattr(rules, 'max') and rules.max is not None:
                    invalid_rows = df[df[col] > rules.max]
                    logger.warning("Dropping %d rows where %s > %s", len(invalid_rows), col, rules.max)
                    df.drop(invalid_rows.index, inplace=True)
                if hasattr(rules, 'allowed_values') and rules.allowed_values is not None:
                    invalid_rows = df[~df[col].isin(rules.allowed_values)]
                    logger.warning("Dropping %d rows where %s not in allowed values", len(invalid_rows), col)
                    df.drop(invalid_rows.index, inplace=True)
        return df
    except Exception as e:
        msg = f"Error cleaning data according to invariants. "
        logger.error(msg + f"Details: {str(e)}")
        raise DataError(msg) from e