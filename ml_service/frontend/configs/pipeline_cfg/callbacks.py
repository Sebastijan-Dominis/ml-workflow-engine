"""Callbacks for Pipeline Config Editor page."""

import os

import dash_bootstrap_components as dbc
import requests
import yaml
from dash import Input, Output, State
from ml_service.frontend.configs.pipeline_cfg.layout import PAGE_PREFIX

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    """Register all callbacks for the pipeline config page."""

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children"),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open"),
        Output(f"{PAGE_PREFIX}-config-editor", "value", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-validate-btn", "n_clicks"),
        State(f"{PAGE_PREFIX}-data-type-input", "value"),
        State(f"{PAGE_PREFIX}-algorithm-input", "value"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def validate_config(_, data_type, algorithm, yaml_text):
        if not data_type or not algorithm:
            return dbc.Alert("Data type and algorithm are required.", color="danger"), False, yaml_text

        try:
            parsed = yaml.safe_load(yaml_text)
            pipeline_version = parsed.get("version")
            if not pipeline_version:
                raise ValueError("Missing 'version' in config YAML.")
        except Exception as e:
            return dbc.Alert(f"YAML parsing error: {str(e)}", color="danger"), False, yaml_text

        r = requests.post(
            f"{API_URL}/pipeline_cfg/validate",
            json={
                "data_type": data_type,
                "algorithm": algorithm,
                "config": yaml_text,
            },
            timeout=10,
        )

        if not r.ok:
            return dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"), False, yaml_text

        result = r.json()
        if not result.get("valid", False):
            return dbc.Alert(result.get("error", "Validation failed"), color="danger"), False, yaml_text
        if result.get("exists", False):
            return dbc.Alert(f"{data_type}/{algorithm}/{pipeline_version} already exists.", color="warning"), False, yaml_text

        normalized = yaml.safe_dump(result["normalized"], sort_keys=False)
        return dbc.Alert("Config valid.", color="success"), True, normalized

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children", allow_duplicate=True),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-confirm-write", "n_clicks"),
        State(f"{PAGE_PREFIX}-data-type-input", "value"),
        State(f"{PAGE_PREFIX}-algorithm-input", "value"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def write_config(_, data_type, algorithm, yaml_text):
        if not data_type or not algorithm:
            return dbc.Alert("Data type and algorithm are required.", color="danger"), False

        try:
            parsed = yaml.safe_load(yaml_text)
            pipeline_version = parsed.get("version")
            if not pipeline_version:
                raise ValueError("Missing 'version' in config YAML.")
        except Exception as e:
            return dbc.Alert(f"YAML parsing error: {str(e)}", color="danger"), False

        r = requests.post(
            f"{API_URL}/pipeline_cfg/write",
            json={
                "data_type": data_type,
                "algorithm": algorithm,
                "config": yaml_text,
            },
            timeout=10,
        )

        if not r.ok:
            return dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"), False

        result = r.json()
        if result.get("status") == "exists":
            return dbc.Alert(result.get("message"), color="warning"), False
        return dbc.Alert(f"Config written successfully to {result.get('path')}.", color="success"), False
