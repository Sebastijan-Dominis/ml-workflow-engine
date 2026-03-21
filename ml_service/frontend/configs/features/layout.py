"""Layout for Feature Registry Editor page."""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html
from ml_service.frontend.configs.features.config_example import EXAMPLE_CONFIG

PAGE_PREFIX = "feature_registry"

def build_layout():
    """Builds the layout for the feature registry editor page."""

    return dbc.Container(
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
                        id=f"{PAGE_PREFIX}-feature-name",
                        placeholder="Feature set name",
                        style={
                            "fontSize": "1.25rem",
                            "padding": "10px",
                            "marginRight": "10px",
                        }
                    ),
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-feature-version",
                        placeholder="Version (e.g. v1)",
                        style={
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
                id=f"{PAGE_PREFIX}-feature-editor",
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
                    id=f"{PAGE_PREFIX}-validate-btn",
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
            html.Div(id=f"{PAGE_PREFIX}-validation-result"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Confirm Write"),
                    dbc.ModalBody("Config validated. Write to registry?"),
                    dbc.ModalFooter(
                        dbc.Button("Confirm", id=f"{PAGE_PREFIX}-confirm-write", color="danger")
                    ),
                ],
                id=f"{PAGE_PREFIX}-confirm-modal",
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
