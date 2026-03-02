"""Small helpers for safe string conversion during metadata hashing."""

def safe(val) -> str:
    """Convert optional values to a deterministic string representation.

    Args:
        val: Value to normalize for hashing serialization.

    Returns:
        Deterministic string representation.
    """

    return "None" if val is None else str(val)