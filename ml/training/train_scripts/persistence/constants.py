"""File-system constants used by persistence helpers.

Centralised string constants for artifact directories make it easier to
change storage locations and keeps other modules free of hard-coded
paths. Values are relative to the repository root.
"""

model_dir = f"ml/models/trained"
metadata_dir = f"ml/models/metadata"
explainability_dir = f"ml/models/explainability"