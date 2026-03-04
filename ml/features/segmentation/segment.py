"""Segmentation utilities for filtering datasets by configured criteria."""

import logging

import pandas as pd
from ml.config.schemas.model_specs import SegmentationConfig
from ml.exceptions import DataError, UserError
from ml.registries.catalogs import OP_MAP

logger = logging.getLogger(__name__)

def apply_segmentation(data: pd.DataFrame, seg_cfg: SegmentationConfig) -> pd.DataFrame:
    """Apply configured segmentation filters and remove segmentation columns.

    Args:
        data: Input dataframe before segmentation filters.
        seg_cfg: Segmentation configuration containing enabled flag and filters.

    Returns:
        pd.DataFrame: Filtered dataframe with segmentation columns removed.
    """

    if not seg_cfg.enabled:
        logger.debug("Segmentation step is disabled. Returning original data.")
        return data

    df = data.copy()

    for f in seg_cfg.filters:
        col = f.column
        op = f.op
        val = f.value

        if op not in OP_MAP:
            msg = f"Unsupported segmentation op: {op}"
            logger.error(msg)
            raise UserError(msg)

        if col not in df.columns:
            msg = f"Segmentation column {col} not found in data."
            logger.error(msg)
            raise DataError(msg)

        values = val if isinstance(val, (list, tuple, set)) else [val]
        for v in values:
            if v not in df[col].unique():
                msg = f"Segmentation value {v} for column {col} not found in data."
                logger.error(msg)
                raise DataError(msg)

        df = df[OP_MAP[op](df[col], val)]
        logger.debug(f"Applied segmentation filter: {col} {op} {val}. Remaining rows: {len(df)}")

    # drop segmentation columns from output, as they are not expected to be used as features, and may cause data leakage if used in training
    try:
        seg_cols = {f.column for f in seg_cfg.filters}
        df = df.drop(columns=list(seg_cols), errors='raise')
    except KeyError as e:
        msg = "Segmentation column not found in data."
        logger.exception(msg)
        raise DataError(msg) from e

    logger.info(f"Applied the segmentation step. Resulting shape: {df.shape}")

    return df
