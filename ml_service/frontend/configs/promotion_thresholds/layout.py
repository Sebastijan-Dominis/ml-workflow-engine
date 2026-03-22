"""Layout for Promotion Thresholds Editor page."""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html
from ml_service.frontend.configs.promotion_thresholds.config_example import EXAMPLE_CONFIG

PAGE_PREFIX = "/promotion_thresholds"

def build_layout():
    """Builds the layout for the promotion thresholds editor page."""
    return dbc.Container(
        [
            html.H1(
                "Promotion Thresholds Editor",
                style={
                    "textAlign": "center",
                    "marginBottom": "40px",
                },
            ),
            html.P(
                "Edit the config to define new promotion thresholds for a model. " \
                "Model is defined by problem type and segment. " \
                "All versions and snapshots of the model will be evaluated with the same promotion thresholds. " \
                "The content is pre-filled with an example config. Click validate to check if the config is valid, then confirm to save the config to disk. Read the documentation (docs/configs/promotion.md) for more details on each config field.",
                style={
                    "maxWidth": "50%",
                    "margin": "0 auto",
                    "fontSize": "1.25rem",
                    "marginBottom": "40px",
                }
            ),
            dbc.Row([
                dbc.Col([
                    dbc.Label(
                        "Problem Type",
                        style={"fontWeight": "bold", "fontSize": "1.25rem"},
                        html_for=f"{PAGE_PREFIX}-problem-type-input"
                    ),
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-problem-type-input",
                        placeholder="Enter problem type...",
                        type="text",
                        style={
                            "fontSize": "1.25rem",
                            "padding": "10px",
                        }
                    ),
                ]),
                dbc.Col([
                    dbc.Label(
                        "Segment",
                        style={"fontWeight": "bold", "fontSize": "1.25rem"},
                        html_for=f"{PAGE_PREFIX}-segment-input"
                    ),
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-segment-input",
                        placeholder="Enter segment...",
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
            "backgroundColor": "#8fa0d8",
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px"
        }
    )
