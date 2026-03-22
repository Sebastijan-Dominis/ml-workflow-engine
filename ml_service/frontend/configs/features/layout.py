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
            html.H1(
                "Feature Registry Editor",
                style={
                    "textAlign": "center",
                    "marginBottom": "40px",
                },
            ),
            html.P(
                [
                    "Generate the operator hash with generate_operator_hash.py (found here in ",
                    html.A('Scripts', href='/Scripts', className='text-decoration-underline'),
                    " as well as the scripts/ directory in the repository) and paste it in the editor below, along with the feature definitions. The content is pre-filled with an example config. Click validate to check if the config is valid, then confirm to write to the registry. Read the documentation (docs/configs/feature_registry.md) for more details on each config field.",
                ],
                style={
                    "maxWidth": "50%",
                    "margin": "0 auto",
                    "fontSize": "1.25rem",
                    "marginBottom": "40px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label(
                            "Feature Set Name",
                            style={"fontWeight": "bold", "fontSize": "1.25rem"},
                            html_for=f"{PAGE_PREFIX}-feature-name",
                        ),
                        dbc.Input(
                            id=f"{PAGE_PREFIX}-feature-name",
                            placeholder="e.g. booking_context_features",
                            type="text",
                            style={
                                "fontSize": "1.25rem",
                                "padding": "10px",
                            }
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label(
                            "Feature Set Version",
                            style={"fontWeight": "bold", "fontSize": "1.25rem"},
                            html_for=f"{PAGE_PREFIX}-feature-version",
                        ),
                        dbc.Input(
                            id=f"{PAGE_PREFIX}-feature-version",
                            placeholder="e.g. v1",
                            type="text",
                            style={
                                "fontSize": "1.25rem",
                                "padding": "10px",
                            }
                        )
                    ], style={"marginBottom": "20px"}),
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
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px",
        },
    )
