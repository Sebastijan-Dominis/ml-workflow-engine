"""Callbacks for Feature Registry Editor page."""

import os

import dash_bootstrap_components as dbc
import dotenv
import requests
import yaml
from dash import Input, Output, State
from ml_service.frontend.configs.features.layout import PAGE_PREFIX

dotenv.load_dotenv()

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    """Register all callbacks for the feature registry page."""

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children"),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open"),
        Output(f"{PAGE_PREFIX}-feature-editor", "value"),
        Input(f"{PAGE_PREFIX}-validate-btn", "n_clicks"),
        State(f"{PAGE_PREFIX}-feature-name", "value"),
        State(f"{PAGE_PREFIX}-feature-version", "value"),
        State(f"{PAGE_PREFIX}-feature-editor", "value"),
        prevent_initial_call=True,
    )
    def validate_yaml(_, name, version, yaml_text):

        r = requests.post(
            f"{API_URL}/features/validate",
            json={
                "name": name,
                "version": version,
                "config": yaml_text,
            },
            timeout=10,
        )

        if not r.ok:
            return (
                dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"),
                False,
                yaml_text,
            )

        result = r.json()

        if not result.get("valid", False):
            return dbc.Alert(result.get("error", "Validation failed with unknown error."), color="danger"), False, yaml_text

        if result.get("exists", False):
            return (
                dbc.Alert(f"{name}/{version} already exists in registry.", color="warning"),
                False,
                yaml_text,
            )

        normalized = yaml.safe_dump(result["normalized"], sort_keys=False)

        return dbc.Alert("Config valid.", color="success"), True, normalized

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children", allow_duplicate=True),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-confirm-write", "n_clicks"),
        State(f"{PAGE_PREFIX}-feature-name", "value"),
        State(f"{PAGE_PREFIX}-feature-version", "value"),
        State(f"{PAGE_PREFIX}-feature-editor", "value"),
        prevent_initial_call=True,
    )
    def write_yaml(_, name, version, yaml_text):

        r = requests.post(
            f"{API_URL}/features/write",
            json={
                "name": name,
                "version": version,
                "config": yaml_text,
            },
            timeout=10,
        )

        if not r.ok:
            return dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"), False

        result = r.json()

        if result.get("status") == "exists":
            return dbc.Alert(result.get("message"), color="warning"), False

        return dbc.Alert(f"Feature set config written successfully to {result.get('path')}.", color="success"), False
