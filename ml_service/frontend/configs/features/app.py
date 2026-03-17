"""Dash app for editing feature registry."""

import os

import dash
import dash_ace
import dash_bootstrap_components as dbc
import dotenv
import requests
import yaml
from dash import Input, Output, State, html
from ml_service.frontend.configs.features.config_example import EXAMPLE_CONFIG

dotenv.load_dotenv()

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [

        html.H2(
            "Feature Registry Editor",
            style={
                "textAlign": "center",
                "color": "#050525",
                "fontWeight": "bold",
                "fontSize": "2.5rem",
                "marginBottom": "40px",
            },
        ),

        dbc.Col(
            [
                dbc.Input(
                    id="feature-name",
                    placeholder="Feature set name",
                    style={
                        "fontSize": "1.25rem",
                        "padding": "10px",
                        "marginRight": "10px",
                    }
                ),
                dbc.Input(
                    id="feature-version",
                    placeholder="Version (e.g. v1)",
                    style= {
                        "fontSize": "1.25rem",
                        "padding": "10px",
                        "marginLeft": "10px",
                    }
                )
            ],
            width=4,
            style={
                "margin": "0 auto",
                "marginBottom": "20px",
                "display": "flex",
                "gap": "10px",
            }
        ),

        dash_ace.DashAceEditor(
            id="feature-editor",
            mode="yaml",
            theme="github",
            tabSize=2,
            fontSize=20,
            height="1000px",
            value=EXAMPLE_CONFIG,
            setOptions={
                "showLineNumbers": True,
                "highlightActiveLine": True,
            },
            style={
                "margin": "30px auto",
                "backgroundColor": "#f8f9fa",
            },
        ),

        dbc.Row(
            dbc.Button(
                "Validate",
                id="validate-btn",
                color="primary",
                style={"width": "150px", "fontSize": "20px"},
            ),
            style={
                "margin": "0 auto",
                "width": "50%",
                "marginTop": "20px",
                "justifyContent": "center",
            },
        ),

        html.Div(id="validation-result"),

        dbc.Modal(
            [
                dbc.ModalHeader("Confirm Write"),
                dbc.ModalBody("Config validated. Write to registry?"),
                dbc.ModalFooter(
                    dbc.Button("Confirm", id="confirm-write", color="danger")
                ),
            ],
            id="confirm-modal",
            is_open=False,
        ),
    ],
    fluid=True,
    style={
        "backgroundColor": "#8fa0d8",
        "minHeight": "100vh",
        "paddingTop": "45px",
        "paddingBottom": "50px",
    },
)

@app.callback(
    Output("validation-result", "children"),
    Output("confirm-modal", "is_open"),
    Output("feature-editor", "value"),
    Input("validate-btn", "n_clicks"),
    State("feature-name", "value"),
    State("feature-version", "value"),
    State("feature-editor", "value"),
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

    # Handle HTTP errors first
    if not r.ok:
        return (
            dbc.Alert(
                f"Backend error {r.status_code}: {r.text}",
                color="danger"
            ),
            False,
            yaml_text,
        )

    result = r.json()

    # Handle validation failure
    if not result.get("valid", False):
        error_msg = result.get("error", "Validation failed with unknown error.")
        return dbc.Alert(error_msg, color="danger"), False, yaml_text

    if result["exists"]:
        return (
            dbc.Alert(
                f"{name}/{version} already exists in registry.",
                color="warning",
            ),
            False,
            yaml_text,
        )

    normalized = yaml.safe_dump(result["normalized"], sort_keys=False)

    return dbc.Alert("Config valid.", color="success"), True, normalized

@app.callback(
    Output("validation-result", "children", allow_duplicate=True),
    Output("confirm-modal", "is_open", allow_duplicate=True),
    Input("confirm-write", "n_clicks"),
    State("feature-name", "value"),
    State("feature-version", "value"),
    State("feature-editor", "value"),
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
        return (
            dbc.Alert(
                f"Backend error {r.status_code}: {r.text}",
                color="danger"
            ),
            False
        )

    result = r.json()

    if result["status"] == "exists":
        return dbc.Alert(result["message"], color="warning"), False

    return dbc.Alert("Feature config written.", color="success"), False

if __name__ == "__main__":
    app.run(debug=True, port=8052, host="0.0.0.0")
