"""Dataset persistence helpers for interim and processed pipeline stages."""

import logging
from pathlib import Path

import pandas as pd

from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig
from ml.exceptions import ConfigError, PersistenceError

logger = logging.getLogger(__name__)

# Currently we only support saving in parquet format, but we can easily add more formats in the future if needed
SAVE_FORMAT = {
    "parquet": pd.DataFrame.to_parquet,
}

def save_data(df: pd.DataFrame, *, config: InterimConfig | ProcessedConfig, data_dir: Path) -> Path:
    """Persist dataframe using output settings from validated stage config.

    Args:
        df: Dataframe to persist.
        config: Validated interim or processed configuration.
        data_dir: Target run directory where output file is written.

    Returns:
        Path: Absolute path to the written data file.
    """

    if not config.data.output.format in SAVE_FORMAT:
        msg = f"Unsupported output format: {config.data.output.format}"
        logger.error(msg)
        raise ConfigError(msg)
    data_path = data_dir / config.data.output.path_suffix
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    save_func = SAVE_FORMAT[config.data.output.format]
    compression = config.data.output.compression
    try:
        if save_func == pd.DataFrame.to_parquet:
            save_func(df, data_path, compression=compression)
        return data_path
    except Exception as e:
        msg = f"Error saving data to {data_path} with format {config.data.output.format} and compression {compression}. "
        logger.error(msg + f"Details: {str(e)}")        
        raise PersistenceError(msg) from e