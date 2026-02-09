import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import yaml

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig


class FreezeStrategy(ABC):
    @abstractmethod
    def freeze(self, config: TabularFeaturesConfig, *, timestamp: str, snapshot_id: str, start_time: float) -> Tuple:
        pass

    @staticmethod
    def hash_config(config: TabularFeaturesConfig) -> str:
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()