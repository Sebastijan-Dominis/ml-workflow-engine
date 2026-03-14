import json

from registries.pipelines_for_endpoint_registration import PIPELINES_FOR_ENDPOINT_REGISTRATION

frontend_ready = []

for p in PIPELINES_FOR_ENDPOINT_REGISTRATION:
    fields = []
    for field_name, model_field in p["args_schema"].model_fields.items():
        default = getattr(model_field, "default", None)

        if type(default).__name__ == "PydanticUndefinedType":
            default = None

        field_type = str(model_field.annotation).replace("typing.", "")
        fields.append({
            "name": field_name,
            "type": field_type,
            "default": default,
        })
    frontend_ready.append({"name": p["name"], "fields": fields})

with open("frontend_pipeline_schema.json", "w") as f:
    json.dump(frontend_ready, f, indent=2)

print("Pipeline schema exported successfully to frontend_pipeline_schema.json")
