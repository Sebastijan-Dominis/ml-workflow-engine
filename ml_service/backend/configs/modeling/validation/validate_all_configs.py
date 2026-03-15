from ml.config.schemas.model_specs import ModelSpecs
from ml_service.backend.configs.modeling.models.configs import (
    RawConfigsWithLineage,
    SearchConfigForValidation,
    TrainConfigForValidation,
    ValidatedConfigs,
)


def validate_all_configs(data_with_lineage: RawConfigsWithLineage) -> ValidatedConfigs:
    try:
        model_specs = ModelSpecs(**data_with_lineage.model_specs)
        search = SearchConfigForValidation(**data_with_lineage.search)
        training = TrainConfigForValidation(**data_with_lineage.training)
        return ValidatedConfigs(model_specs=model_specs, search=search, training=training)
    except Exception as e:
        raise ValueError(f"Config validation error: {str(e)}") from e
