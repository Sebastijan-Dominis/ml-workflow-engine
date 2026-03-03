"""Factory helper for resolving registered searcher implementations."""

import logging

from ml.exceptions import PipelineContractError
from ml.registries import SEARCHERS
from ml.search.searchers.base import Searcher

logger = logging.getLogger(__name__)

def get_searcher(key: str) -> Searcher:
    """Instantiate the searcher registered for the provided algorithm key.

    Args:
        key: Searcher registry key identifying the algorithm implementation.

    Returns:
        Instantiated searcher implementation.
    """

    searcher_cls = SEARCHERS.get(key)

    if not searcher_cls:
        msg = f"No searcher registered for algorithm {key}."
        logger.error(msg)
        raise PipelineContractError(msg)

    searcher = searcher_cls()

    logger.debug(
        "Using searcher %s for algorithm=%s",
        searcher.__class__.__name__,
        key,
    )

    return searcher