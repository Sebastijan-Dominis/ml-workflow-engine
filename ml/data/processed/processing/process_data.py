import logging

import pandas as pd

from ml.data.utils.config.schemas.processed import ProcessedConfig
from ml.exceptions import DataError, UserError
from ml.registry.row_id_registry import ROW_ID_FUNCTIONS

logger = logging.getLogger(__name__)

def remove_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    missing_cols = [col for col in columns_to_remove if col not in df.columns]
    if missing_cols:
        msg = f"Cannot remove columns {missing_cols} because they are not present in the DataFrame."
        logger.error(msg)
        raise DataError(msg)
    
    return df.drop(columns=columns_to_remove)

def add_row_id(df: pd.DataFrame, cfg: ProcessedConfig) -> tuple[pd.DataFrame, dict]:
    row_id_fn = ROW_ID_FUNCTIONS.get(cfg.data.name)
    if row_id_fn:
        return row_id_fn(df)
    else:
        msg = f"Row ID generation is not implemented for data '{cfg.data.name}'."
        logger.error(msg)
        raise UserError(msg)