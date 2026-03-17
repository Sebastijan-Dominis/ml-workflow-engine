import os

import dash
import dash_ace
import dash_bootstrap_components as dbc
import requests
import yaml
from dash import Input, Output, State, html
from ml_service.frontend.configs.promotion_thresholds.config_example import EXAMPLE_CONFIG

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([

    html.H2(
        "Promotion Thresholds Editor",
        style={
            "textAlign": "center",
            "color": "#050525",
            "fontWeight": "bold",
            "fontSize": "2.5rem",
            "marginBottom": "40px",
        },
    ),

    dbc.Row([
                dbc.Col([
                    dbc.Label("Problem Type", style={"fontWeight": "bold", "fontSize": "1.1rem"}, html_for="problem-type-input"),
                    dbc.Input(id="problem-type-input", placeholder="Enter problem type...", type="text"),
                ]),
                dbc.Col(
                    [
                        dbc.Label("Segment", style={"fontWeight": "bold", "fontSize": "1.1rem"}, html_for="segment-input"),
                        dbc.Input(id="segment-input", placeholder="Enter segment...", type="text"),
                    ],
                    style={
                        "marginBottom": "20px",
                    }
                ),
            ],
            style={
                "width": "40%",
                "margin": "0 auto",
                "marginBottom": "20px",
                "display": "flex",
                "gap": "2rem",
            }
        ),

    dash_ace.DashAceEditor(
        id="config-editor",
        mode="yaml",
        value=EXAMPLE_CONFIG,
        theme="github",
        tabSize=2,
        fontSize=20,
        height="700px",
        setOptions={"showLineNumbers": True, "highlightActiveLine": True},
        style={"margin": "30px auto", "backgroundColor": "#f8f9fa"},
    ),

    dbc.Row(
        dbc.Button("Validate", id="validate-btn", color="primary", style={"width": "150px", "fontSize": "20px"}),
        style={"margin": "0 auto", "width": "50%", "marginTop": "20px", "justifyContent": "center"},
    ),

    html.Div(id="validation-result"),

    dbc.Modal([
        dbc.ModalHeader("Confirm Write"),
        dbc.ModalBody("Config validated. Write to disk?"),
        dbc.ModalFooter(dbc.Button("Confirm", id="confirm-write", color="danger")),
    ], id="confirm-modal", is_open=False),

], fluid=True, style={"background-color": "#8fa0d8", "min-height": "100vh", "paddingTop": "45px", "paddingBottom": "50px"})


# Validate pipeline config
@app.callback(
    Output("validation-result", "children"),
    Output("confirm-modal", "is_open"),
    Output("config-editor", "value", allow_duplicate=True),
    Input("validate-btn", "n_clicks"),
    State("problem-type-input", "value"),
    State("segment-input", "value"),
    State("config-editor", "value"),
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


# Write pipeline config
@app.callback(
    Output("validation-result", "children", allow_duplicate=True),
    Output("confirm-modal", "is_open", allow_duplicate=True),
    Input("confirm-write", "n_clicks"),
    State("problem-type-input", "value"),
    State("segment-input", "value"),
    State("config-editor", "value"),
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


if __name__ == "__main__":
    app.run(debug=True, port=8055, host="0.0.0.0")
