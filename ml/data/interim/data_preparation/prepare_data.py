import logging

import pandas as pd

from ml.data.config.schemas.interim import (Cleaning, DataSchema,
                                                  Invariants)
from ml.exceptions import DataError
from ml.registry.op_map import OP_MAP

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

# Keeps missing values - this step doesn't concern with them
def clean_data(df: pd.DataFrame, invariants: Invariants):
    try:
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()
            rules = getattr(invariants, col, None)
            if rules is None:
                logger.debug(f"No invariants specified for column '{col}'. Skipping cleaning for this column.")
                continue
            logger.debug(f"Applying invariants for column '{col}': {rules}")
            if rules.min is not None:
                valid_rows = OP_MAP[rules.min.op](df[col], rules.min.value)
                valid_rows = valid_rows | df[col].isna() | (df[col] == "")
                invalid_rows = df[~valid_rows]
                if not invalid_rows.empty:
                    logger.warning("Dropping %d rows where %s < %s. Row indices: %s", len(invalid_rows), col, rules.min.value, invalid_rows.index.tolist())
                df = df[valid_rows]
            if rules.max is not None:
                valid_rows = OP_MAP[rules.max.op](df[col], rules.max.value)
                valid_rows = valid_rows | df[col].isna() | (df[col] == "")
                invalid_rows = df[~valid_rows]
                if not invalid_rows.empty:
                    logger.warning("Dropping %d rows where %s > %s. Row indices: %s", len(invalid_rows), col, rules.max.value, invalid_rows.index.tolist())
                df = df[valid_rows]
            if rules.allowed_values is not None:
                allowed_values = [x for x in rules.allowed_values if x is not None]
                mask = df[col].isin(allowed_values) | df[col].isna() | (df[col] == "")
                invalid_rows = df[~mask]
                if not invalid_rows.empty:
                    logger.warning("Dropping %d rows where %s not in allowed values. Row indices: %s",
                            len(invalid_rows), col, invalid_rows.index.tolist())
                df = df[mask]
        return df
    except Exception as e:
        msg = f"Error cleaning data according to invariants. "
        logger.error(msg + f"Details: {str(e)}")
        raise DataError(msg) from e
