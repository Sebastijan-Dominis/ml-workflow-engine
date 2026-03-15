import os

import dash
import dash_ace
import dash_bootstrap_components as dbc
import dotenv
import requests
import yaml
from dash import Input, Output, State, html
from ml_service.frontend.configs.modeling.config_examples import CONFIG_EXAMPLES_REGISTRY

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container(
    [
        html.H2(
            "Modeling Config Editor",
            style={
                "text-align": "center",
                "color": "#050525",
                "font-weight": "bold",
                "font-size": "2.5rem",
                "margin-bottom": "40px",
            }
        ),
        dbc.Row(
            [
                dbc.Col(
                  [
                    html.H2(
                        name,
                        style={
                            "text-align": "center",
                            "color": "#050525",
                            "font-weight": "bold",
                            "font-size": "2rem",
                        }
                    ),
                    dash_ace.DashAceEditor
                    (
                      id=name,
                      mode="yaml",
                      theme="github",
                      tabSize=2,
                      setOptions={
                          "showLineNumbers": True,
                          "highlightActiveLine": True,
                      },
                      fontSize=20,
                      height="1000px",
                      style={
                          "margin": "30px auto 30px auto",
                          "display": "block",
                          "background-color": "#f8f9fa",
                      },
                      value=content
                    )
                  ]
                )
                for name, content in CONFIG_EXAMPLES_REGISTRY.items()
            ]
        ),

        dbc.Row(
            dbc.Button(
                "Validate",
                id="validate-btn",
                color="primary",
                style={"width": "150px", "font-size": "20px"}
            ),
            style={
                "margin": "0 auto",
                "width": "50%",
                "margin-top": "20px",
                "justify-content": "center",
            }
        ),

        html.Div(id="validation-result", className="mt-3"),

        dbc.Modal(
            [
                dbc.ModalHeader("Confirm Write"),
                dbc.ModalBody("Config validated. Write YAML to disk?"),
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
        "background-color": "#8fa0d8",
        "min-height": "100vh",
        "padding-top": "45px",
        "padding-bottom": "50px",
    }
)


@app.callback(
    Output("validation-result", "children"),
    Output("confirm-modal", "is_open"),
    *[Output(name, "value") for name in CONFIG_EXAMPLES_REGISTRY],
    Input("validate-btn", "n_clicks"),
    *[State(name, "value") for name in CONFIG_EXAMPLES_REGISTRY],
    prevent_initial_call=True,
)
def validate_yaml(_, *yaml_values):

    keys = list(CONFIG_EXAMPLES_REGISTRY.keys())
    yaml_texts = dict(zip(keys, yaml_values, strict=True))

    r = requests.post(
        f"{API_URL}/modeling/validate",
        json={
            "model_specs": yaml_texts["Model Specs"],
            "search": yaml_texts["Search"],
            "training": yaml_texts["Training"],
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
    Output("validation-result", "children", allow_duplicate=True),
    Output("confirm-modal", "is_open", allow_duplicate=True),
    Input("confirm-write", "n_clicks"),
    *[State(name, "value") for name in CONFIG_EXAMPLES_REGISTRY],
    prevent_initial_call=True,
)
def write_yaml(_, *yaml_values):
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
          dbc.Alert(
            f"Backend error {r.status_code}: {r.text}",
            color="danger"
          ),
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



if __name__ == "__main__":
    app.run(debug=True, port=8051, host="0.0.0.0")
