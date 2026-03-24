"""Callbacks for the file viewer page."""
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import requests
from dash import Input, Output, State

PAGE_PREFIX = "/file_viewer"
API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

def register_callbacks(app):
    @app.callback(
        Output(f"{PAGE_PREFIX}-viewer", "value"),
        Output(f"{PAGE_PREFIX}-viewer", "mode"),
        Output(f"{PAGE_PREFIX}-status", "children"),
        Input(f"{PAGE_PREFIX}-load-btn", "n_clicks"),
        State(f"{PAGE_PREFIX}-path-input", "value"),
        prevent_initial_call=True,
    )
    def load_file(_, path_str):
        if not path_str:
            return "", "yaml", dbc.Alert("Path required", color="danger")

        path = Path(path_str).as_posix()

        try:
            r = requests.post(f"{API_URL}/file_viewer/load", json={"path": path}, timeout=10)
        except Exception as e:
            return "", "yaml", dbc.Alert(f"Backend unreachable: {e}", color="danger")

        if not r.ok:
            return "", "yaml", dbc.Alert(f"{r.status_code}: {r.text}", color="danger")

        result = r.json()
        return result["content"], result["mode"], dbc.Alert(f"Loaded {result['path']}", color="success")
