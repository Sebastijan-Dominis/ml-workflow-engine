import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(path: Path, level: int = logging.INFO) -> None:
    """Configure root logger to write to *path* at the given *level*."""
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

    Parameters
    ----------
    path:
        Destination log file.  Parent directories are created
        automatically.
    level:
        Optional override – defaults to the root logger's current level.

    Returns
    -------
    logging.FileHandler
        The handler that was added (handy for later removal).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(str(path))
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler.setLevel(level)

    logging.getLogger().addHandler(handler)
    return handler

def bootstrap_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
    )