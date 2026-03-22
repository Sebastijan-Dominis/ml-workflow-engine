"""A module that defines the registry of scripts available in the frontend, including their names, endpoints, and argument schemas."""
from ml_service.backend.scripts.models.script_cli_args import (
    CheckImportLayersInput,
    CheckNamingConventionsInput,
    GenerateColsForRowIdFingerprintInput,
    GenerateFakeDataInput,
    GenerateOperatorHashInput,
    GenerateSnapshotBindingInput,
)

FRONTEND_SCRIPTS_REGISTRY = [
    {
        "name": "Check Import Layers",
        "endpoint": "/scripts/check_import_layers",
        "args_schema": CheckImportLayersInput,
    },
    {
        "name": "Check Naming Conventions",
        "endpoint": "/scripts/check_naming_conventions",
        "args_schema": CheckNamingConventionsInput,
    },
    {
        "name": "Generate Columns for Row ID Fingerprint",
        "endpoint": "/scripts/generate_cols_for_row_id_fingerprint",
        "args_schema": GenerateColsForRowIdFingerprintInput,
    },
    {
        "name": "Generate Fake Data",
        "endpoint": "/scripts/generate_fake_data",
        "args_schema": GenerateFakeDataInput,
        "field_metadata": {
            "data": {
                "placeholder": "Name of the dataset to generate (e.g. 'hotel_bookings')",
                "optional": False
            },
            "version": {
                "placeholder": "Version of the dataset to generate (e.g. 'v1')",
                "optional": False
            },
            "snapshot_id": {
                "placeholder": "Snapshot ID to use for generation (default: 'latest')",
                "optional": True
            },
            "num_rows": {
                "placeholder": "Number of rows to generate (default: 50000)",
                "optional": True
            },
            "include_old": {
                "label": "Include old data in generation",
                "optional": True,
                "value": False
            },
            "model_path": {
                "placeholder": "Path to the CTGAN model to use for generation (default: 'synthetizers/2026-03-21T02-18-36_b682275d/ctgan_model.pkl')",
                "optional": True
            },
            "strict_missing": {
                "label": "Strict missing values (only generate missing values if they were present in the original data)",
                "optional": True,
                "value": False
            },
            "strict_quality": {
                "label": "Strict quality threshold (only generate data that meets the quality threshold)",
                "optional": True,
                "value": True
            },
            "seed": {
                "placeholder": "Random seed for generation (default: 42)",
                "optional": True
            },
            "quality_threshold": {
                "placeholder": "Quality threshold for generated data (0-1, default: 0.7)",
                "optional": True
            },
            "epochs": {
                "placeholder": "Number of epochs to train the CTGAN model (default: 400)",
                "optional": True
            },
            "batch_target_size": {
                "placeholder": "Target batch size for training the CTGAN model (default: 4000)",
                "optional": True
            },
            "save_model": {
                "label": "Save the trained CTGAN model",
                "optional": True,
                "value": True
            }
        }
    },
    {
        "name": "Generate Operator Hash",
        "endpoint": "/scripts/generate_operator_hash",
        "args_schema": GenerateOperatorHashInput,
        "field_metadata": {
            "operators": {
                "placeholder": "List of comma-separated operators to hash (e.g. 'TotalStay, ArrivalDate')",
                "optional": False
            }
        }
    },
    {
        "name": "Generate Snapshot Binding",
        "endpoint": "/scripts/generate_snapshot_binding",
        "args_schema": GenerateSnapshotBindingInput,
    }
]
