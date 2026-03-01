import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def ensure_required_fields_present_in_dict(
    *,
    input_dict: dict,
    required_fields: list[str]
) -> None:
    missing_fields = [field for field in required_fields if field not in input_dict]

    if missing_fields:
        msg = f"dictionary for is missing required fields: {', '.join(missing_fields)}"
        logger.error(msg)
        raise DataError(msg)