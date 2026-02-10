import logging

from ml.utils.runtime.runtime_snapshot import get_conda_env_export, hash_environment

logger = logging.getLogger(__name__)

def validate_conda_envs_match(runtime_info: dict) -> None:
    env_export = get_conda_env_export()
    conda_env_hash = hash_environment(env_export)

    expected_env_hash = runtime_info.get("environment", {}).get("conda_env_hash", "<unknown>")

    if expected_env_hash != conda_env_hash:
        logger.warning(
            f"Current conda environment hash {conda_env_hash} does not match expected {expected_env_hash}. "
            "Reproducibility may be affected."
        )
    else:
        logger.debug("Conda environment hash matches expected.")