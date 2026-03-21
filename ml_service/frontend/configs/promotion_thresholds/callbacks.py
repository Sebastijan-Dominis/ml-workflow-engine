"""Callbacks for Promotion Thresholds Editor page."""

import os

import dash_bootstrap_components as dbc
import requests
import yaml
from dash import Input, Output, State
from ml_service.frontend.configs.promotion_thresholds.layout import PAGE_PREFIX

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    """Register all callbacks for the promotion thresholds page."""

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children"),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open"),
        Output(f"{PAGE_PREFIX}-config-editor", "value", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-validate-btn", "n_clicks"),
        State(f"{PAGE_PREFIX}-problem-type-input", "value"),
        State(f"{PAGE_PREFIX}-segment-input", "value"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def validate_config(_, problem_type, segment, yaml_text):
        if not problem_type or not segment:
            return dbc.Alert("Problem type and segment are required.", color="danger"), False, yaml_text

        r = requests.post(
            f"{API_URL}/promotion_thresholds/validate",
            json={
                "problem_type": problem_type,
                "segment": segment,
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
            return dbc.Alert(f"{problem_type}/{segment}/ already exists.", color="warning"), False, yaml_text

        normalized = yaml.safe_dump(result["normalized"], sort_keys=False)
        return dbc.Alert("Config valid.", color="success"), True, normalized

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children", allow_duplicate=True),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-confirm-write", "n_clicks"),
        State(f"{PAGE_PREFIX}-problem-type-input", "value"),
        State(f"{PAGE_PREFIX}-segment-input", "value"),
        State(f"{PAGE_PREFIX}-config-editor", "value"),
        prevent_initial_call=True
    )
    def write_config(_, problem_type, segment, yaml_text):
        if not problem_type or not segment:
            return dbc.Alert("Problem type and segment are required.", color="danger"), False

        r = requests.post(
            f"{API_URL}/promotion_thresholds/write",
            json={
                "problem_type": problem_type,
                "segment": segment,
                "config": yaml_text,
            },
            timeout=10,
        )

        if not r.ok:
            return dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"), False

        result = r.json()
        if result.get("status") == "exists":
            return dbc.Alert(result.get("message"), color="warning"), False
        return dbc.Alert("Config written successfully.", color="success"), False
