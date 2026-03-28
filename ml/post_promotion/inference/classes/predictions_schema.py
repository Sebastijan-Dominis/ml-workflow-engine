"""A module for defining the schema of predictions in the inference pipeline."""

# Hardcoded, since these are not expected to change very often.
# If changes become common, consider a refactor to make these configurable.
# For now, simply update this file when changes are needed, and ensure that
# all components of the pipeline are updated accordingly.
SCHEMA_VERSION = "V1"

BASE_EXPECTED_COLUMNS = [
    "run_id",
    "prediction_id",
    "timestamp",
    "model_stage",
    "model_version",
    "entity_id",
    "input_hash",
    "prediction",
    "schema_version"
]

PROBA_PREFIX = "proba_"
