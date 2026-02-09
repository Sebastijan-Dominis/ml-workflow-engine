import logging

import pandas as pd

from ml.exceptions import DataError, UserError
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

logger = logging.getLogger(__name__)

def apply_segmentation(data: pd.DataFrame, config: TabularFeaturesConfig) -> pd.DataFrame:
    seg_cfg = config.segmentation
    
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

    return df