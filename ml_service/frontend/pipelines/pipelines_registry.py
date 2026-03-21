"""A module that defines the registry of pipelines available in the frontend, including their names, endpoints, and argument schemas."""
from ml_service.backend.pipelines.models.pipelines_cli_args import (
    BuildInterimDatasetInput,
    BuildProcessedDatasetInput,
    EvaluateInput,
    ExecuteAllDataPreprocessingInput,
    ExecuteAllExperimentsWithLatestInput,
    ExecuteExperimentWithLatestInput,
    ExplainInput,
    FreezeAllFeatureSetsInput,
    FreezeFeaturesInput,
    PromoteInput,
    RegisterRawSnapshotInput,
    RunAllWorkflowsInput,
    SearchInput,
    TrainInput,
)

FRONTEND_PIPELINES_REGISTRY = [
    {
        "name": "Register Raw Snapshot",
        "endpoint": "pipelines/register_raw_snapshot",
        "args_schema": RegisterRawSnapshotInput,
        "field_metadata": {
            "data": {
                "placeholder": "Data name, e.g., hotel_bookings",
                "optional": False
            },
            "version": {
                "placeholder": "Data version, e.g., v1",
                "optional": False
            },
            "snapshot_id": {
                "placeholder": "Snapshot ID for tracking the data version (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the run (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            }
        }
    },
    {
        "name": "Build Interim Dataset",
        "endpoint": "pipelines/build_interim_dataset",
        "args_schema": BuildInterimDatasetInput,
        "field_metadata": {
            "data": {
                "placeholder": "Data name, e.g., hotel_bookings",
                "optional": False
            },
            "version": {
                "placeholder": "Data version, e.g., v1",
                "optional": False
            },
            "raw_snapshot_id": {
                "placeholder": "Snapshot ID for the raw data version to use (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the data (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            }
        }
    },
    {
        "name": "Build Processed Dataset",
        "endpoint": "pipelines/build_processed_dataset",
        "args_schema": BuildProcessedDatasetInput,
        "field_metadata": {
            "data": {
                "placeholder": "Data name, e.g., hotel_bookings",
                "optional": False
            },
            "version": {
                "placeholder": "Data version, e.g., v1",
                "optional": False
            },
            "interim_snapshot_id": {
                "placeholder": "Snapshot ID for the interim data version to use (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the data (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            }
        }
    },
    {
        "name": "Freeze Feature Set",
        "endpoint": "pipelines/freeze_feature_set",
        "args_schema": FreezeFeaturesInput,
        "field_metadata": {
            "feature_set": {
                "placeholder": "Feature set name, e.g., base_features",
                "optional": False
            },
            "version": {
                "placeholder": "Feature set version, e.g., v1",
                "optional": False
            },
            "snapshot_binding_key": {
                "placeholder": "Optional key for a snapshot binding to define which snapshot to load for each dataset",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the feature set (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            }
        }
    },
    {
        "name": "Search",
        "endpoint": "pipelines/search",
        "args_schema": SearchInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "experiment_id": {
                "placeholder": "Experiment ID to use for this run (default -> new unique experiment ID). Format if provided: '20240101T120000_abcdef12'",
                "optional": True
            },
            "snapshot_binding_key": {
                "placeholder": "Optional key for a snapshot binding to define which snapshot to load for each dataset",
                "optional": True
            },
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = default ~ none)",
                "optional": True,
                "value": "default"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the experiment (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            },
            "clean_up_failure_management": {
                "label": "Whether to clean up failure management folder after successful run (default = True)",
                "optional": True,
                "value": True
            },
            "overwrite_existing": {
                "label": "Whether to overwrite existing experiment data if the experiment ID already exists (default = False)",
                "optional": True,
                "value": False
            }
        }
    },
    {
        "name": "Train",
        "endpoint": "pipelines/train",
        "args_schema": TrainInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "snapshot_binding_key": {
                "placeholder": "Optional key for a snapshot binding to define which snapshot to load for each dataset",
                "optional": True
            },
            "train_run_id": {
                "placeholder": "Train run id (dir name under experiments/{problem}/{segment}/{version}) (default = latest)",
                "optional": True
            },
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = default ~ none)",
                "optional": True,
                "value": "default"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "experiment_id": {
                "placeholder": "Experiment id (dir name under experiments/{problem}/{segment}/{version}) (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "clean_up_failure_management": {
                "label": "Whether to clean up failure management folder after successful run (default = True)",
                "optional": True,
                "value": True
            },
            "overwrite_existing": {
                "label": "Whether to overwrite existing train run data if the train run ID already exists (default = False)",
                "optional": True,
                "value": False
            }
        }
    },
    {
        "name": "Evaluate",
        "endpoint": "pipelines/evaluate",
        "args_schema": EvaluateInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = default ~ none)",
                "optional": True,
                "value": "default"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "experiment_id": {
                "placeholder": "Experiment id (dir name under experiments/{problem}/{segment}/{version}) (default = latest)",
                "optional": True
            },
            "train_id": {
                "placeholder": "Train id (dir name under experiments/{problem}/{segment}/{version}/{snapshot_id}/training) (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            }
        }
    },
    {
        "name": "Explain",
        "endpoint": "pipelines/explain",
        "args_schema": ExplainInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = default ~ none)",
                "optional": True,
                "value": "default"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "experiment_id": {
                "placeholder": "Experiment id (dir name under experiments/{problem}/{segment}/{version}) (default = latest)",
                "optional": True
            },
            "train_id": {
                "placeholder": "Train id (dir name under experiments/{problem}/{segment}/{version}/{snapshot_id}/training) (default = latest)",
                "optional": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "top_k": {
                "placeholder": "Number of top features to include in the explainability output (default = settings-specified value)",
                "optional": True
            }
        }
    },
    {
        "name": "Promote",
        "endpoint": "pipelines/promote",
        "args_schema": PromoteInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "experiment_id": {
                "placeholder": "Experiment id (dir name under experiments/{problem}/{segment}/{version})",
                "optional": False
            },
            "train_run_id": {
                "placeholder": "Train run id (dir name under experiments/{problem}/{segment}/{version}/{experiment_id}/training)",
                "optional": False
            },
            "eval_run_id": {
                "placeholder": "Eval run id (dir name under experiments/{problem}/{segment}/{version}/{experiment_id}/evaluation)",
                "optional": False
            },
            "explain_run_id": {
                "placeholder": "Explain run id (dir name under experiments/{problem}/{segment}/{version}/{experiment_id}/explainability)",
                "optional": False
            },
            "stage": {
                "placeholder": "Stage of the promotion (staging or production)",
                "optional": False
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            }
        }
    },
    {
        "name": "Execute All Data Preprocessing",
        "endpoint": "pipelines/execute_all_data_preprocessing",
        "args_schema": ExecuteAllDataPreprocessingInput,
        "field_metadata": {
            "skip_if_existing": {
                "label": "Whether to skip running a data preprocessing pipeline if at least one run already exists for the dataset (default = True)",
                "optional": True,
                "value": True
            }
        }
    },
    {
        "name": "Freeze All Feature Sets",
        "endpoint": "pipelines/freeze_all_feature_sets",
        "args_schema": FreezeAllFeatureSetsInput,
        "field_metadata": {
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the feature sets (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            },
            "skip_if_existing": {
                "label": "Whether to skip freezing if at least one freeze folder already exists for the feature set (default = True)",
                "optional": True,
                "value": True
            }
        }
    },
    {
        "name": "Execute Experiment With Latest",
        "endpoint": "pipelines/execute_experiment_with_latest",
        "args_schema": ExecuteExperimentWithLatestInput,
        "field_metadata": {
            "problem": {
                "placeholder": "Model problem, e.g., 'no_show'",
                "optional": False
            },
            "segment": {
                "placeholder": "Model segment name, e.g., 'city_hotel_online_ta'",
                "optional": False
            },
            "version": {
                "placeholder": "Model version, e.g., 'v1'",
                "optional": False
            },
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default dev)",
                "optional": True,
                "value": "dev"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the experiment (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            },
            "clean_up_failure_management": {
                "label": "Whether to clean up failure management folder after successful run (default = True)",
                "optional": True,
                "value": True
            },
            "experiment_id": {
                "placeholder": "Experiment ID to use for this run (default -> new unique experiment ID). Format if provided: '20240101T120000_abcdef12'",
                "optional": True
            },
            "overwrite_existing": {
                "label": "Whether to overwrite existing metadata and runtime snapshot files if they already exist in the target directory (default = False)",
                "optional": True,
                "value": False
            },
            "top_k": {
                "placeholder": "Number of top features to include in the explainability output (default = settings-specified value)",
                "optional": True
            }
        }
    },
    {
        "name": "Execute All Experiments With Latest",
        "endpoint": "pipelines/execute_all_experiments_with_latest",
        "args_schema": ExecuteAllExperimentsWithLatestInput,
        "field_metadata": {
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = dev)",
                "optional": True,
                "value": "dev"
            },
            "strict": {
                "label": "Whether to run in strict mode, which includes strict validation that may be computationally expensive (default = True)",
                "optional": True,
                "value": True
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the experiments (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            },
            "clean_up_failure_management": {
                "label": "Whether to clean up failure management folder after successful run (default = True)",
                "optional": True,
                "value": True
            },
            "overwrite_existing": {
                "label": "Whether to overwrite existing metadata and runtime snapshot files if they already exist in the target directory (default = False)",
                "optional": True,
                "value": False
            },
            "top_k": {
                "placeholder": "Number of top features to include in the explainability output (default = settings-specified value)",
                "optional": True
            },
            "skip_if_existing": {
                "label": "Whether to skip running an experiment if at least one experiment folder exists for the model (default = True)",
                "optional": True,
                "value": True
            }
        }
    },
    {
        "name": "Run All Workflows",
        "endpoint": "pipelines/run_all_workflows",
        "args_schema": RunAllWorkflowsInput,
        "field_metadata": {
            "env": {
                "placeholder": "Environment to run the script in (dev/test/prod) (default = dev)",
                "optional": True,
                "value": "dev"
            },
            "logging_level": {
                "optional": True,
                "value": "INFO"
            },
            "owner": {
                "placeholder": "Owner of the run (default = 'Sebastijan')",
                "optional": True,
                "value": "Sebastijan"
            },
            "skip_if_existing": {
                "label": "Whether to skip running a workflow if at least one run already exists for any of the steps in the workflow (default = True)",
                "optional": True,
                "value": True
            }
        }
    }
]
