import logging

from  ml.exceptions import UserError

logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    msg = f"Boolean value expected for argument, got '{v}'. Please use one of: yes, true, t, 1, no, false, f, 0 (case-insensitive)."
    logger.error(msg)
    raise UserError(msg)