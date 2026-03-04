import logging

from ml.exceptions import DataError
from ml.modeling.models.feature_lineage import FeatureLineage

logger = logging.getLogger(__name__)

def validate_and_construct_feature_lineage(feature_lineage_raw: list[dict]) -> list[FeatureLineage]:
    try:
        feature_lineage = [FeatureLineage(**entry) for entry in feature_lineage_raw]
        logger.debug(f"Constructed FeatureLineage objects for {len(feature_lineage)} feature sets.")
        return feature_lineage
    except TypeError as e:
        msg = f"Error constructing FeatureLineage objects from raw metadata. Raw entries: {feature_lineage_raw}."
        logger.exception(msg)
        raise DataError(msg) from e
