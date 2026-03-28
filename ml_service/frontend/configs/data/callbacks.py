"""Callbacks for Data Config Editor page."""

import os

import dash_bootstrap_components as dbc
import requests
import yaml
from dash import Input, Output, State

from ml_service.frontend.configs.data.examples.interim import INTERIM_EXAMPLE
from ml_service.frontend.configs.data.examples.processed import PROCESSED_EXAMPLE
from ml_service.frontend.configs.data.layout import PAGE_PREFIX

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    """Register all callbacks for the data config page."""

    @app.callback(
        Output(f"{PAGE_PREFIX}-config-editor", "value"),
        Input(f"{PAGE_PREFIX}-config-tabs", "active_tab")
    )
    def update_editor_on_tab_change(active_tab):
        # Make sure we match full prefixed tab IDs
        if active_tab == f"{PAGE_PREFIX}-interim-tab":
            return INTERIM_EXAMPLE
        elif active_tab == f"{PAGE_PREFIX}-processed-tab":
            return PROCESSED_EXAMPLE
        return ""

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children"),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open"),
        Output(f"{PAGE_PREFIX}-config-editor", "value", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-validate-btn", "n_clicks"),
        State(f"{PAGE_PREFIX}-config-tabs", "active_tab"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def validate_config(_, active_tab, yaml_text):
        config_type = "interim" if active_tab == f"{PAGE_PREFIX}-interim-tab" else "processed"

        try:
            parsed = yaml.safe_load(yaml_text)
            dataset_name = parsed["data"]["name"]
            dataset_version = parsed["data"]["version"]
        except Exception as e:
            return dbc.Alert(f"YAML parsing error: {str(e)}", color="danger"), False, yaml_text

        r = requests.post(
            f"{API_URL}/data/validate",
            json={
                "type": config_type,
                "name": dataset_name,
                "version": dataset_version,
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
            return dbc.Alert(f"{dataset_name}/{dataset_version} already exists.", color="warning"), False, yaml_text

        normalized = yaml.safe_dump(result["normalized"], sort_keys=False)
        return dbc.Alert("Config valid.", color="success"), True, normalized

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children", allow_duplicate=True),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-confirm-write", "n_clicks"),
        State(f"{PAGE_PREFIX}-config-tabs", "active_tab"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def write_config(_, active_tab, yaml_text):
        config_type = "interim" if active_tab == f"{PAGE_PREFIX}-interim-tab" else "processed"

        try:
            parsed = yaml.safe_load(yaml_text)
            dataset_name = parsed["data"]["name"]
            dataset_version = parsed["data"]["version"]
        except Exception as e:
            return dbc.Alert(f"YAML parsing error: {str(e)}", color="danger"), False

        r = requests.post(
            f"{API_URL}/data/write",
            json={
                "type": config_type,
                "name": dataset_name,
                "version": dataset_version,
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