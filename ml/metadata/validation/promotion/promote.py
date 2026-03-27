"""A module for validating promotion metadata against Pydantic models for staging and production."""
import logging
from typing import Literal

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.promotion.promote import (
    ProductionPromotionMetadata,
    StagingPromotionMetadata,
)
from ml.promotion.constants.promotion_metadata_dict import PromotionMetadataDict

logger = logging.getLogger(__name__)

def validate_promotion_metadata(promotion_metadata_raw: PromotionMetadataDict, stage: Literal["production", "staging"]) -> ProductionPromotionMetadata | StagingPromotionMetadata:
    """Validate raw promotion metadata against Pydantic models for staging and production.

    Args:
        promotion_metadata_raw: Raw promotion metadata dictionary to validate.
        stage: The stage for which to validate the metadata.

    Returns:
        Validated promotion metadata as either ProductionPromotionMetadata or StagingPromotionMetadata.

    Raises:
        RuntimeMLError: If validation fails against both staging and production models.
    """
    validated_metadata: ProductionPromotionMetadata | StagingPromotionMetadata
    if stage == "production":
        try:
            validated_metadata = ProductionPromotionMetadata.model_validate(promotion_metadata_raw)
            logger.debug("Promotion metadata successfully validated against ProductionPromotionMetadata model.")
            return validated_metadata
        except Exception as e:
            msg = "Failed to validate promotion metadata against ProductionPromotionMetadata model."
            logger.exception(msg)
            raise RuntimeMLError(msg) from e
    elif stage == "staging":
        try:
            validated_metadata = StagingPromotionMetadata.model_validate(promotion_metadata_raw)
            logger.debug("Promotion metadata successfully validated against StagingPromotionMetadata model.")
            return validated_metadata
        except Exception as e:
            msg = "Failed to validate promotion metadata against StagingPromotionMetadata model."
            logger.exception(msg)
            raise RuntimeMLError(msg) from e
