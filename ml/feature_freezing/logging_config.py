"""Simple logging setup used by feature freezing scripts.

Provides a lightweight helper to configure the root logger with a
standard timestamped format and INFO log level. Call ``setup_logging()``
at the start of CLI entrypoints to ensure consistent logs across
feature freezing runs and utilities.
"""

import logging


def setup_logging():
    """Configure the Python logging subsystem with a standard format.

    The function applies a basic configuration setting the level to
    ``INFO`` and a compact formatter including timestamp, logger name,
    and log level. It is intentionally minimal to avoid interfering with
    downstream applications that may reconfigure logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        filename="logs/ml/feature_freezing/feature_freezing.log",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )