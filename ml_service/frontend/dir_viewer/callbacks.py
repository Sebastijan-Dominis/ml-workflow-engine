import os

import dash_bootstrap_components as dbc
import requests
from dash import Input, Output, State

PAGE_PREFIX = "/dir_viewer"
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
    def load_dir(_, path_str):
        if not path_str:
            return "", "yaml", dbc.Alert("Path required", color="danger")

        try:
            r = requests.post(f"{API_URL}/dir_viewer/load", json={"path": path_str}, timeout=10)
        except Exception as e:
            return "", "yaml", dbc.Alert(f"Backend unreachable: {e}", color="danger")

        if not r.ok:
            return "", "yaml", dbc.Alert(f"{r.status_code}: {r.text}", color="danger")

        result = r.json()
        return result["tree_yaml"], "yaml", dbc.Alert(f"Loaded directory tree for {result['path']}", color="success")
