"""A module for defining function return types in post-promotion pipelines."""
from dataclasses import dataclass

import pandas as pd

from ml.modeling.models.feature_lineage import FeatureLineage
from ml.promotion.config.registry_entry import RegistryEntry


@dataclass
class ModelRegistryInfo:
    """A dataclass to hold model registry information."""
    prod_meta: RegistryEntry | None
    stage_meta: RegistryEntry | None

@dataclass
class PrepareFeaturesReturn:
    """A dataclass to hold the return values of the prepare_features function."""
    features: pd.DataFrame
    entity_key: str
    feature_lineage: list[FeatureLineage]
    target: pd.Series
