def extract_model_params(best_params):
    return {
        k.split("__", 1)[1]: v
        for k, v in best_params.items()
        if k.startswith("Model__")
    }