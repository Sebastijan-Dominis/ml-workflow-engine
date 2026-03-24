import dash_ace
import dash_bootstrap_components as dbc
from dash import html

PAGE_PREFIX = "/file_viewer"

def build_layout():
    return dbc.Container(
        [
            html.H1("File Viewer", style={"textAlign": "center"}),

            dbc.Row(
                [
                    dbc.Input(
                        id=f"{PAGE_PREFIX}-path-input",
                        placeholder="Enter YAML/JSON path...",
                        type="text",
                        style={"fontSize": "1.2rem"},
                    ),
                    dbc.Button(
                        "Load",
                        id=f"{PAGE_PREFIX}-load-btn",
                        color="primary",
                        style={"width": "120px"},
                    ),
                ],
                style={"width": "60%", "margin": "20px auto", "gap": "1rem"},
            ),

            html.Div(id=f"{PAGE_PREFIX}-status"),

            dash_ace.DashAceEditor(
                id=f"{PAGE_PREFIX}-viewer",
                mode="yaml",
                theme="github",
                value="",
                readOnly=True,
                fontSize=16,
                height="700px",
                setOptions={"showLineNumbers": True},
                style={"marginTop": "20px"},
            ),
        ],
        fluid=True,
    )
