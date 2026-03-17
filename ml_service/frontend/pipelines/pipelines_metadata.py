"""A module to define the metadata for the frontend pipelines, which is then used to dynamically generate the UI components for each pipeline."""
from types import UnionType
from typing import Union, get_args, get_origin

from ml_service.frontend.pipelines.pipelines_registry import FRONTEND_PIPELINES_REGISTRY

FRONTEND_PIPELINES = []

def is_boolean_field(model_field) -> bool:
    ann = model_field.annotation

    # plain bool
    if ann is bool:
        return True

    # union / optional bool
    origin = get_origin(ann)
    if origin in (Union, UnionType):
        args = get_args(ann)
        if bool in args:
            return True

    return False

for p in FRONTEND_PIPELINES_REGISTRY:
    fields = []
    for field_name, model_field in p["args_schema"].model_fields.items():
        type_hint = str(model_field.annotation).replace("typing.", "")
        if field_name == "logging_level":
            fields.append({
                "name": field_name,
                "type": "dropdown",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "value": model_field.default if model_field.default is not None else "INFO"
            })
        elif is_boolean_field(model_field):
            fields.append({"name": field_name, "type": "boolean", "value": model_field.default})
        else:
            fields.append({"name": field_name, "type": "text", "placeholder": field_name})
    FRONTEND_PIPELINES.append({"name": p["name"], "endpoint": p["endpoint"], "fields": fields})
