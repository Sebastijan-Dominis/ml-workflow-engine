import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.search.search import SearchRecord

logger = logging.getLogger(__name__)

def validate_search_record(record_raw: dict) -> SearchRecord:
    try:
        record = SearchRecord(**record_raw)
        logger.debug("Validated search record.")
        return record
    except Exception as e:
        msg = "Error validating search record."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
