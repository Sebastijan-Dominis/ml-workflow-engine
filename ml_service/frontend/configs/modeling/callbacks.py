"""Callbacks for modeling config editor."""

import os

import dash_bootstrap_components as dbc
import dotenv
import requests
import yaml
from dash import Input, Output, State

from ml_service.frontend.configs.modeling.config_examples import CONFIG_EXAMPLES_REGISTRY
from ml_service.frontend.configs.modeling.layout import PAGE_PREFIX

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    """Register all callbacks for the modeling config page."""

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children"),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open"),
        *[Output(f"{PAGE_PREFIX}-{name}", "value") for name in CONFIG_EXAMPLES_REGISTRY],
        Input(f"{PAGE_PREFIX}-validate-btn", "n_clicks"),
        *[State(f"{PAGE_PREFIX}-{name}", "value") for name in CONFIG_EXAMPLES_REGISTRY],
        prevent_initial_call=True,
    )
    def validate_yaml(_, *yaml_values):
        """Validate the YAML configs by sending them to the backend for validation."""
        keys = list(CONFIG_EXAMPLES_REGISTRY.keys())
        yaml_texts = dict(zip(keys, yaml_values, strict=True))

        r = requests.post(
            f"{API_URL}/modeling/validate",
            json={
                "model_specs": yaml_texts.get("Model Specs", ""),
                "search": yaml_texts.get("Search", ""),
                "training": yaml_texts.get("Training", ""),
            },
            timeout=10,
        )

        if not r.ok:
            return (
                dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"),
                False,
                *yaml_values
            )

        result = r.json()
        if not result.get("valid", False):
            return (
                dbc.Alert(result.get("error", "Unknown error"), color="danger"),
                False,
                *yaml_values
            )

        normalized = result.get("normalized", {})

        normalized_yaml = [
            yaml.safe_dump(normalized.get("model_specs", {}), sort_keys=False),
            yaml.safe_dump(normalized.get("search", {}), sort_keys=False),
            yaml.safe_dump(normalized.get("training", {}), sort_keys=False),
        ]

        return (
            dbc.Alert("Config is valid.", color="success"),
            True,
            *normalized_yaml
        )

    @app.callback(
        Output(f"{PAGE_PREFIX}-validation-result", "children", allow_duplicate=True),
        Output(f"{PAGE_PREFIX}-confirm-modal", "is_open", allow_duplicate=True),
        Input(f"{PAGE_PREFIX}-confirm-write", "n_clicks"),
        *[State(f"{PAGE_PREFIX}-{name}", "value") for name in CONFIG_EXAMPLES_REGISTRY],
        prevent_initial_call=True,
    )
    def write_yaml(_, *yaml_values):
        """Write the YAML configs to disk by sending them to the backend for writing."""
        yaml_texts = dict(zip(CONFIG_EXAMPLES_REGISTRY.keys(), yaml_values, strict=True))

        model_specs = yaml_texts.get("Model Specs", "")
        search = yaml_texts.get("Search", "")
        training = yaml_texts.get("Training", "")

        r = requests.post(
            f"{API_URL}/modeling/write",
            json={
                "model_specs": model_specs,
                "search": search,
                "training": training
            },
            timeout=10,
        )

        if not r.ok:
            return (
                dbc.Alert(f"Backend error {r.status_code}: {r.text}", color="danger"),
                False
            )

        result = r.json()
        paths = result.get("paths", {})

        return (
            dbc.Alert(
                f"Written:\n{paths.get('model_specs')}\n{paths.get('search')}\n{paths.get('training')}",
                color="success"
            ),
            False
        )
