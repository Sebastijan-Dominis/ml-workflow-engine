import hashlib
import logging
import pickle
from pathlib import Path

from catboost import CatBoostClassifier, CatBoostRegressor
from prophet import Prophet
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def _hash_file(path: Path, method="sha256") -> str:
    h = hashlib.new(method)
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def _hash_catboost(model: CatBoostClassifier | CatBoostRegressor, method="sha256", temp_dir: Path | None = None) -> str:
    temp_path = temp_dir / "temp_model.cbm" if temp_dir else Path("temp_model.cbm")
    model.save_model(temp_path)
    hash_value = _hash_file(temp_path, method)
    if not temp_dir:
        temp_path.unlink()
    return hash_value

def _hash_prophet(model: Prophet, method="sha256", temp_dir: Path | None = None) -> str:
    """Hash a Prophet model by pickling it."""
    h = hashlib.new(method)
    h.update(pickle.dumps(model))
    return h.hexdigest()

def _hash_sklearn_pipeline(pipeline: Pipeline, method="sha256", temp_dir: Path | None = None) -> str:
    """
    Hash an sklearn Pipeline by serializing it in-memory using pickle.
    """
    h = hashlib.new(method)
    serialized = pickle.dumps(pipeline)  # <-- in-memory bytes
    h.update(serialized)
    return h.hexdigest()