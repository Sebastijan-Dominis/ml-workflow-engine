"""Layout for Data Config Editor page."""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html

PAGE_PREFIX = "/data_configs"

def build_layout():
    """Builds the layout for the data config editor page."""
    return dbc.Container(
        [
            html.H1(
                "Data Config Editor",
                style={
                    "textAlign": "center",
                    "marginBottom": "40px",
                },
            ),
            html.P([
                "Choose which config you want to edit. The content is pre-filled with an example config. Click validate to check if the config is valid, then confirm to save the config to disk. Read the ",
                html.A('interim data config documentation', href='/Docs?doc=configs/interim_data.md', className='text-decoration-underline'),
                " (docs/configs/interim_data.md) and ",
                html.A('processed data config documentation', href='/Docs?doc=configs/processed_data.md', className='text-decoration-underline'),
                " (docs/configs/processed_data.md) for more details on each config field.",
            ],
                style={
                    "maxWidth": "50%",
                    "margin": "0 auto",
                    "fontSize": "1.25rem",
                    "marginBottom": "40px",
                }
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="Interim",
                        tab_id=f"{PAGE_PREFIX}-interim-tab",
                        tab_style={"width": "20%", "textAlign": "center"}
                    ),
                    dbc.Tab(
                        label="Processed",
                        tab_id=f"{PAGE_PREFIX}-processed-tab",
                        tab_style={"width": "20%", "textAlign": "center"}
                    ),
                ],
                id=f"{PAGE_PREFIX}-config-tabs",
                active_tab=f"{PAGE_PREFIX}-interim-tab",
                style={
                    "width": "50%",
                    "margin": "0 auto",
                    "marginBottom": "30px",
                    "marginTop": "20px",
                    "display": "flex",
                    "justifyContent": "center",
                    "gap": "30%",
                    "fontSize": "1.25rem",
                    "fontWeight": "bold",
                }
            ),
            dash_ace.DashAceEditor(
                id=f"{PAGE_PREFIX}-config-editor",
                mode="yaml",
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