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

def is_number_field(model_field):
    ann = model_field.annotation
    origin = get_origin(ann)

    if ann in (int, float):
        return True

    if origin in (Union, UnionType):
        args = get_args(ann)
        return any(a in (int, float) for a in args)

    return False

for p in FRONTEND_PIPELINES_REGISTRY:
    fields = []
    field_meta = p.get("field_metadata", {})

    for field_name, model_field in p["args_schema"].model_fields.items():
        meta = field_meta.get(field_name, {})
        is_optional = meta.get("optional")

        type_hint = str(model_field.annotation).replace("typing.", "")

        if field_name == "logging_level":
            fields.append({
                "name": field_name,
                "type": "dropdown",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "value": model_field.default if model_field.default is not None else "INFO",
                "bold": True,
                "optional": is_optional
            })
        elif field_name == "env":
            fields.append({
                "name": field_name,
                "type": "dropdown",
                "options": ["dev", "test", "prod", "default"],
                "value": model_field.default if model_field.default is not None else "default",
                "bold": True,
                "optional": is_optional,
                "label": "env (default ~ none)"
            })
        elif field_name == "stage":
            fields.append({
                "name": field_name,
                "type": "dropdown",
                "options": ["staging", "production"],
                "value": model_field.default if model_field.default is not None else "production",
                "bold": True,
                "optional": is_optional,
                "label": "stage"
            })
        elif is_boolean_field(model_field):
            fields.append({
                "name": field_name,
                "type": "boolean",
                "value": meta.get("value", model_field.default),
                "label": meta.get("label", field_name),
                "optional": is_optional
            })
        elif is_number_field(model_field):
            fields.append({
                "name": field_name,
                "type": "number",
                "placeholder": meta.get("placeholder", field_name),
                "optional": is_optional
            })
        else:
            fields.append({
                "name": field_name,
                "type": "text",
                "placeholder": meta.get("placeholder", field_name),
                "optional": is_optional
            })
    FRONTEND_PIPELINES.append({
        "name": p["name"],
        "endpoint": p["endpoint"],
        "fields": fields
    })
