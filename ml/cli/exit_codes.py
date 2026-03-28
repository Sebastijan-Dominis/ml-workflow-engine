"""Canonical process exit codes for project CLIs.

The constants in this module define stable numeric exit codes for broad
failure categories so shell scripts and orchestrators can react reliably.
"""

EXIT_SUCCESS = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_DATA_ERROR = 3
EXIT_PIPELINE_ERROR = 4
EXIT_SEARCH_ERROR = 5
EXIT_TRAINING_ERROR = 6
EXIT_EVALUATION_ERROR = 7
EXIT_EXPLAINABILITY_ERROR = 8
EXIT_PERSISTENCE_ERROR = 9
EXIT_INFERENCE_ERROR = 10
EXIT_MONITORING_ERROR = 11
