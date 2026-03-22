"""Layout for Pipeline Config Editor page."""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html
from ml_service.frontend.configs.pipeline_cfg.config_example import EXAMPLE_CONFIG

PAGE_PREFIX = "/pipeline_cfg"

def build_layout():
    """Builds the layout for the pipeline config editor page."""
    return dbc.Container(
        [
            html.H1(
                "Pipeline Config Editor",
                style={
                    "textAlign": "center",
                    "marginBottom": "40px",
                },
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label(
                        "Data Type", style={"fontWeight": "bold", "fontSize": "1.25rem"}, html_for=f"{PAGE_PREFIX}-data-type-input"
                    ),
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-data-type-input",
                        placeholder="Enter data type...",
                        type="text",
                        style={
                                "fontSize": "1.25rem",
                                "padding": "10px",
                            }
                    ),
                ]),
                dbc.Col([
                    dbc.Label(
                        "Algorithm", style={"fontWeight": "bold", "fontSize": "1.25rem"}, html_for=f"{PAGE_PREFIX}-algorithm-input"
                    ),
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-algorithm-input",
                        placeholder="Enter algorithm...",
                        type="text",
                        style={
                            "fontSize": "1.25rem",
                            "padding": "10px",
                        }
                    ),
                ], style={"marginBottom": "20px"})
            ],
            style={
                "width": "40%",
                "margin": "0 auto",
                "marginBottom": "20px",
                "display": "flex",
                "gap": "2rem",
            }),
            dash_ace.DashAceEditor(
                id=f"{PAGE_PREFIX}-config-editor",
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
                dbc.Button("Validate", id=f"{PAGE_PREFIX}-validate-btn", color="primary", style={"width": "150px", "fontSize": "20px"}),
                style={"margin": "0 auto", "width": "50%", "marginTop": "20px", "justifyContent": "center"},
            ),
            html.Div(id=f"{PAGE_PREFIX}-validation-result"),
            dbc.Modal([
                dbc.ModalHeader("Confirm Write"),
                dbc.ModalBody("Config validated. Write to disk?"),
                dbc.ModalFooter(dbc.Button("Confirm", id=f"{PAGE_PREFIX}-confirm-write", color="danger")),
            ], id=f"{PAGE_PREFIX}-confirm-modal", is_open=False),
        ],
        fluid=True,
        style={
            "backgroundColor": "#8fa0d8", ""
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px"
        }
    )
