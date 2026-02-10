import hashlib
import logging

import pandas as pd

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def hash_y(y) -> str:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            msg = f"hash_y only supports Series or single-column DataFrame, got {y.shape[1]} columns"
            logger.error(msg)
            raise DataError(msg)
        y = y.iloc[:, 0]

    if not isinstance(y, pd.Series):
        msg = f"hash_y expects a pandas Series or single-column DataFrame, got {type(y)}"
        logger.error(msg)
        raise DataError(msg)

    h = hashlib.sha256()

    # Numeric types
    if pd.api.types.is_numeric_dtype(y):
        # Use numpy array of floats to handle nullable dtypes safely
        arr = y.to_numpy(dtype="float64", copy=False)
        h.update(arr.tobytes())

    # Categorical types
    elif isinstance(y.dtype, pd.CategoricalDtype):
        h.update(y.cat.codes.to_numpy(dtype="int32", copy=False).tobytes())
        cat_bytes = b''.join(c.encode('utf-8') for c in y.cat.categories.astype(str))
        h.update(cat_bytes)

    # String/Object types
    elif pd.api.types.is_string_dtype(y) or pd.api.types.is_object_dtype(y):
        for val in y:
            h.update(str(val).encode('utf-8'))

    else:
        msg = f"Unsupported dtype for hashing: {y.dtype}"
        logger.error(msg)
        raise DataError(msg)

    return h.hexdigest()