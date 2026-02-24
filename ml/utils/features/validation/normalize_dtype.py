import numpy as np

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