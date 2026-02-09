import logging

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def refine_int(center, offsets, low, high):
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