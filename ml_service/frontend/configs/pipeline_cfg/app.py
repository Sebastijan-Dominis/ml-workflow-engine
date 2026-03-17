import os

import dash
import dash_ace
import dash_bootstrap_components as dbc
import requests
import yaml
from dash import Input, Output, State, html
from ml_service.frontend.configs.pipeline_cfg.config_example import EXAMPLE_CONFIG

API_URL = os.getenv("ML_SERVICE_BACKEND_URL", "http://localhost:8000")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([

    html.H2(
        "Pipeline Config Editor",
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
                    dbc.Label("Data Type", style={"fontWeight": "bold", "fontSize": "1.1rem"}, html_for="data-type-input"),
                    dbc.Input(id="data-type-input", placeholder="Enter data type...", type="text"),
                ]),
                dbc.Col(
                    [
                        dbc.Label("Algorithm", style={"fontWeight": "bold", "fontSize": "1.1rem"}, html_for="algorithm-input"),
                        dbc.Input(id="algorithm-input", placeholder="Enter algorithm...", type="text"),
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

], fluid=True, style={"backgroundColor": "#8fa0d8", "minHeight": "100vh", "paddingTop": "45px", "paddingBottom": "50px"})


# Validate pipeline config
@app.callback(
    Output("validation-result", "children"),
    Output("confirm-modal", "is_open"),
    Output("config-editor", "value", allow_duplicate=True),
    Input("validate-btn", "n_clicks"),
    State("data-type-input", "value"),
    State("algorithm-input", "value"),
    State("config-editor", "value"),
    prevent_initial_call=True
)
def validate_config(_, data_type, algorithm, yaml_text):
    if not data_type or not algorithm:
        return dbc.Alert("Data type and algorithm are required.", color="danger"), False, yaml_text

    # Parse YAML to extract version
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
    if result["exists"]:
        return dbc.Alert(f"{data_type}/{algorithm}/{pipeline_version} already exists.", color="warning"), False, yaml_text

    normalized = yaml.safe_dump(result["normalized"], sort_keys=False)
    return dbc.Alert("Config valid.", color="success"), True, normalized


# Write pipeline config
@app.callback(
    Output("validation-result", "children", allow_duplicate=True),
    Output("confirm-modal", "is_open", allow_duplicate=True),
    Input("confirm-write", "n_clicks"),
    State("data-type-input", "value"),
    State("algorithm-input", "value"),
    State("config-editor", "value"),
    prevent_initial_call=True
)
def write_config(_, data_type, algorithm, yaml_text):
    if not data_type or not algorithm:
        return dbc.Alert("Data type and algorithm are required.", color="danger"), False

    # Parse YAML to extract version
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
    return dbc.Alert("Config written successfully.", color="success"), False


if __name__ == "__main__":
    app.run(debug=True, port=8054, host="0.0.0.0")
