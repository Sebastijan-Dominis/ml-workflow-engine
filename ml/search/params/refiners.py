"""Primitive refinement functions for integer/float search neighborhoods."""

import logging

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def refine_int(center, offsets, low, high):
    """Generate bounded integer neighborhood around center using offsets.

    Args:
        center: Center integer value.
        offsets: Integer offsets around center.
        low: Minimum allowed value.
        high: Maximum allowed value.

    Returns:
        list[int]: Sorted bounded integer candidates.
    """

    if not isinstance(center, int):
        msg = f"Expected integer center value, got {center} of type {type(center)}"
        logger.error(msg)
        raise ConfigError(msg)

    values = {center}
    for o in offsets:
        values.add(center + o)
        values.add(center - o)
    return sorted(v for v in values if low <= v <= high)

def refine_float_mult(center, factors, low, high, decimals=5):
    """Generate bounded multiplicative float neighborhood around center.

    Args:
        center: Center numeric value.
        factors: Multipliers to apply around center.
        low: Minimum allowed value.
        high: Maximum allowed value.
        decimals: Rounding precision.

    Returns:
        list[float]: Sorted bounded float candidates.
    """

    if not isinstance(center, (float, int)):
        msg = f"Expected numeric center value, got {center} of type {type(center)}"
        logger.error(msg)
        raise ConfigError(msg)

    values = set()
    for f in factors:
        v = round(center * f, decimals)
        if low <= v <= high:
            values.add(round(v, decimals))
    values.add(round(center, decimals))
    return sorted(values)

def refine_border_count(center):
    """Refine GPU-safe CatBoost border_count using adjacent allowed values.

    Args:
        center: Current border_count value.

    Returns:
        list[int]: Neighboring allowed border_count values including center.
    """

    options = [32, 64, 128, 254]
    if center in options:
        idx = options.index(center)
        refined = set()
        if idx > 0:
            refined.add(options[idx - 1])
        refined.add(center)
        if idx < len(options) - 1:
            refined.add(options[idx + 1])
        return sorted(refined)
    else:
        msg = f"border_count value {center} is not in allowed options {options}"
        logger.error(msg)
        raise ConfigError(msg)
