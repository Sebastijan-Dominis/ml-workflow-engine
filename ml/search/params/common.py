from ml.validation_schemas.search_cfg import BroadParamDistributions


def flatten_search_params(search_params: dict) -> dict[str, list]:
    obj = BroadParamDistributions(
        model=search_params.get("model", {}),
        ensemble=search_params.get("ensemble", {})
    )
    return obj.to_flat_dict()