# General imports
import logging
logger = logging.getLogger(__name__)
import joblib
from pathlib import Path

def save_model(model, cfg: dict):
    model_name = f"{cfg['problem']}_{cfg['segment']['name']}_{cfg['version']}"

    path = Path(f"ml/artifacts/{cfg['problem']}/{cfg['segment']['name']}/{cfg['version']}")

    # Step 1 - Ensure the path for saving the model exist
    path.mkdir(parents=True, exist_ok=True)

    # Step 2 - Save the trained model
    model_file = path/f"model.joblib"

    # Step 2.1 - Warn if target model file already exists
    if model_file.exists():
        logger.warning(
            f"Model file for {model_name} already exists "
            "and will be overwritten."
        )
    
    # Step 2.2 - Write the model to a joblib file
    try:
        joblib.dump(model, model_file)
        logger.info(f"Model for {model_name} successfully saved to {model_file}.")
    except Exception:
        logger.exception(f"Error saving model to {model_file}")
        raise