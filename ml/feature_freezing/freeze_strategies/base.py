import hashlib
from abc import ABC, abstractmethod
from typing import Tuple

import yaml

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml.config.compute_config_hash import compute_config_hash

class FreezeStrategy(ABC):
    @abstractmethod
    def freeze(self, config: TabularFeaturesConfig, *, timestamp: str, snapshot_id: str, start_time: float) -> Tuple:
        pass

    @staticmethod
    def hash_config(config: TabularFeaturesConfig) -> str:
        return compute_config_hash(config)