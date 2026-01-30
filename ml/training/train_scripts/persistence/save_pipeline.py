# General imports
import logging
logger = logging.getLogger(__name__)
import joblib
from pathlib import Path

def save_pipeline(pipeline, cfg: dict):
    model_name = f"{cfg['problem']}_{cfg['segment']['name']}_{cfg['version']}"

    path = Path(f"ml/artifacts/{cfg['problem']}/{cfg['segment']['name']}/{cfg['version']}")

    # Step 1 - Ensure the path for saving the pipeline exist
    path.mkdir(parents=True, exist_ok=True)

    # Step 2 - Save the trained pipeline
    pipeline_file = path/f"pipeline.joblib"

    # Step 2.1 - Warn if target pipeline file already exists
    if pipeline_file.exists():
        logger.warning(
            f"Pipeline file for model {model_name} already exists "
            "and will be overwritten."
        )
    
    # Step 2.2 - Write the pipeline to a joblib file
    try:
        joblib.dump(pipeline, pipeline_file)
        logger.info(f"Pipeline for model {model_name} successfully saved to {pipeline_file}.")
    except Exception:
        logger.exception(f"Error saving pipeline to {pipeline_file}")
        raise