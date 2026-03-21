"""Layout for Modeling Config Editor page."""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html

from ml_service.frontend.configs.modeling.config_examples import CONFIG_EXAMPLES_REGISTRY

PAGE_PREFIX = "modeling_configs"

def build_layout():
    """Builds the layout for the modeling config editor page."""
    rows = [
        dbc.Col(
            [
                html.H2(
                    name,
                    style={
                        "textAlign": "center",
                        "color": "#050525",
                        "fontWeight": "bold",
                        "fontSize": "2rem",
                    }
                ),
                dash_ace.DashAceEditor(
                    id=f"{PAGE_PREFIX}-{name}",
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
                        "backgroundColor": "#f8f9fa",
                    },
                    value=content
                )
            ]
        )
        for name, content in CONFIG_EXAMPLES_REGISTRY.items()
    ]

    layout = dbc.Container(
        [
            html.H2(
                "Modeling Config Editor",
                style={
                    "textAlign": "center",
                    "color": "#050525",
                    "fontWeight": "bold",
                    "fontSize": "2.5rem",
                    "marginBottom": "40px",
                }
            ),
            dbc.Row(rows),
            dbc.Row(
                dbc.Button(
                    "Validate",
                    id=f"{PAGE_PREFIX}-validate-btn",
                    color="primary",
                    style={"width": "150px", "fontSize": "20px"}
                ),
                style={
                    "margin": "0 auto",
                    "width": "50%",
                    "marginTop": "20px",
                    "justifyContent": "center",
                }
            ),
            html.Div(id=f"{PAGE_PREFIX}-validation-result", className="mt-3"),
            dbc.Modal(
                [
                    dbc.ModalHeader("Confirm Write"),
                    dbc.ModalBody("Config validated. Write YAML to disk?"),
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
        }
    )
    return layout
