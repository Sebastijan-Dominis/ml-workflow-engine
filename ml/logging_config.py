"""Shared logging configuration helpers for scripts and pipeline entrypoints."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(path: Path, level: int = logging.INFO) -> None:
    """Configure root logger to write to *path* at the given *level*.

    Args:
        path: Destination log file path.
        level: Logging level.

    Returns:
        None: Configures global logging side effects.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        filename=str(path),
        format=LOG_FORMAT,
    )


def add_file_handler(
    path: Path,
    level: int = logging.INFO,
) -> logging.FileHandler:
    """Attach an additional file handler to the root logger.

    Useful when the final log destination is only known after the
    initial ``setup_logging`` call (e.g. after an experiment-id or
    snapshot-id has been generated).

    Args:
        path: Destination log file.
        level: Optional handler log level.

    Returns:
        logging.FileHandler: The handler that was added.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(path))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(level)

    logging.getLogger().addHandler(handler)
    return handler

def bootstrap_logging(level=logging.INFO):
    """Initialize console/root logging without a file destination.

    This is useful for early-stage CLI startup before run-specific output
    directories are known.

    Args:
        level: Logging level for the root logger.

    Returns:
        None: Configures global logging side effects.
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
    )