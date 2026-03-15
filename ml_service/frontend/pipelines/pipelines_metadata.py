from ml_service.frontend.pipelines.pipelines_registry import FRONTEND_PIPELINES_REGISTRY

# Transform Pydantic model into simple metadata for frontend
FRONTEND_PIPELINES = []

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
        elif type_hint == "bool":
            fields.append({"name": field_name, "type": "boolean", "value": model_field.default})
        else:
            fields.append({"name": field_name, "type": "text", "placeholder": field_name})
    FRONTEND_PIPELINES.append({"name": p["name"], "endpoint": p["endpoint"], "fields": fields})
