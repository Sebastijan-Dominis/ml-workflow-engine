import logging

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

def get_default_int_params(param_cfg, default_offsets, default_low, default_high):
    offsets_cfg = param_cfg.offsets if param_cfg.offsets is not None else default_offsets
    if not offsets_cfg:
        offsets_cfg = default_offsets
    low_cfg = param_cfg.low if param_cfg.low is not None else default_low
    if not low_cfg:
        low_cfg = default_low
    high_cfg = param_cfg.high if param_cfg.high is not None else default_high
    if not high_cfg:
        high_cfg = default_high
    return offsets_cfg, low_cfg, high_cfg

def get_default_float_params(param_cfg, default_factors, default_low, default_high, default_decimals):
    factors_cfg = param_cfg.factors if param_cfg.factors is not None else default_factors
    if not factors_cfg:
        factors_cfg = default_factors
    low_cfg = param_cfg.low if param_cfg.low is not None else default_low
    if not low_cfg:
        low_cfg = default_low
    high_cfg = param_cfg.high if param_cfg.high is not None else default_high
    if not high_cfg:
        high_cfg = default_high
    decimals_cfg = param_cfg.decimals
    if decimals_cfg is None:
        decimals_cfg = default_decimals
    elif decimals_cfg <= 0:
        msg = "Decimal places for float refinement must be a positive integer."
        logger.error(msg)
        raise ConfigError(msg)
    return factors_cfg, low_cfg, high_cfg, decimals_cfg