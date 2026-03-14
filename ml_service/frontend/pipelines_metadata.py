import json
from pathlib import Path

with open(Path(__file__).parent / "frontend_pipeline_schema.json") as f:
    raw_pipelines = json.load(f)

FRONTEND_PIPELINES = []

for p in raw_pipelines:
    fields = []
    for field in p["fields"]:
        if field["name"] == "logging_level":
            fields.append({
                "name": field["name"],
                "type": "dropdown",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "value": field["default"] or "INFO"
            })
        elif field["type"].lower() == "bool":
            fields.append({
                "name": field["name"],
                "type": "boolean",
                "value": bool(field["default"]) if field["default"] is not None else False
            })
        else:
            fields.append({
                "name": field["name"],
                "type": "text",
                "placeholder": field["name"],
                "value": field.get("default")
            })
    FRONTEND_PIPELINES.append({
        "name": p["name"],
        "fields": fields
    })
