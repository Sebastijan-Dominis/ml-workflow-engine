import logging

import pandas as pd

from ml.exceptions import DataError, UserError
from ml.config.validation_schemas.model_specs import SegmentationConfig

logger = logging.getLogger(__name__)

def apply_segmentation(data: pd.DataFrame, seg_cfg: SegmentationConfig) -> pd.DataFrame:
    if not seg_cfg.enabled:
        return data

    df = data.copy()

    OP_MAP = {
        "eq": lambda s, v: s == v,
        "neq": lambda s, v: s != v,
        "in": lambda s, v: s.isin(v),
        "not_in": lambda s, v: ~s.isin(v),
        "gt": lambda s, v: s > v,
        "gte": lambda s, v: s >= v,
        "lt": lambda s, v: s < v,
        "lte": lambda s, v: s <= v,
    }

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

    # drop segmentation columns from output, as they are not expected to be used as features, and may cause data leakage if used in training
    try:
        seg_cols = {f.column for f in seg_cfg.filters}
        df = df.drop(columns=list(seg_cols), errors='raise')
    except KeyError as e:
        msg = f"Segmentation column not found in data: {e}"
        logger.error(msg)
        raise DataError(msg)

    return df