from abc import ABC, abstractmethod

from ml.config.compute_config_hash import compute_config_hash
from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.tabular.config.models import \
    TabularFeaturesConfig


class FreezeStrategy(ABC):
    @abstractmethod
    def freeze(self, config: TabularFeaturesConfig, *, timestamp: str, snapshot_id: str, start_time: float) -> FreezeOutput:
        pass

    @staticmethod
    def hash_config(config: TabularFeaturesConfig) -> str:
        return compute_config_hash(config)