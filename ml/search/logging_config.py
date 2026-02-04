"""Simple logging setup used by scripts in the notebooks folder.

Provides a lightweight helper to configure the root logger with a
standard timestamped format and INFO log level. Call ``setup_logging()``
at the start of CLI entrypoints to ensure consistent logs across search runs and utilities.
"""

import logging
from pathlib import Path
# TODO: consider dynamically assigning the log file name and level
def setup_logging():
    """Configure the Python logging subsystem with a standard format.

    The function applies a basic configuration setting the level to
    ``INFO`` and a compact formatter including timestamp, logger name,
    and log level. It is intentionally minimal to avoid interfering with
    downstream applications that may reconfigure logging.
    """

    Path("logs/ml/search").mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename="logs/ml/search/search.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )