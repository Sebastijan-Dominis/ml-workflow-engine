"""Datetime formatting helpers for filesystem-safe timestamp strings."""

def iso_no_colon(dt):
    """Return ISO timestamp without colon characters in the time component.

    Args:
        dt: Datetime-like object supporting ``isoformat``.

    Returns:
        Filesystem-safe ISO timestamp string.
    """

    return dt.isoformat(timespec="seconds").replace(":", "-")
