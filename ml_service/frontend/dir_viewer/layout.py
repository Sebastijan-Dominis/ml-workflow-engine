import dash_ace
import dash_bootstrap_components as dbc
from dash import html

PAGE_PREFIX = "/dir_viewer"

def build_layout():
    return dbc.Container(
        [
            dbc.Col(
                [
                    html.H1(
                        "Directory Viewer",
                        style={
                            "textAlign": "center"
                        }
                    ),

                    dbc.Label(
                        "Enter directory path relative to repo root:",
                        html_for=f"{PAGE_PREFIX}-path-input",
                        style={
                            "fontSize": "1.2rem"
                        }
                    ),

                    dbc.Input(
                        id=f"{PAGE_PREFIX}-path-input",
                        placeholder="e.g. experiments",
                        type="text",
                        style={
                            "fontSize": "1.2rem"
                        },
                    ),

                    dbc.Button(
                        "Load",
                        id=f"{PAGE_PREFIX}-load-btn",
                        color="primary",
                        style={
                            "width": "120px"
                        },
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
                        setOptions={
                            "showLineNumbers":
                            True
                        },
                        placeholder="Directory tree will be displayed here in YAML format after loading.",
                    ),

                    # Manual path input + prompt
                    dbc.Row(
                        [
                            dbc.Input(
                                id=f"{PAGE_PREFIX}-manual-path",
                                placeholder="Enter or paste directory path here...",
                                type="text",
                                style={
                                    "fontSize": "1.1rem",
                                    "width": "800px"
                                },
                            ),
                        ],
                    ),

                    html.P(
                        [
                            "The input field above is for convenience - it allows you to quickly write the path into it, without leaving the page, which you can then copy-paste into the ",
                            html.A(
                                "file viewer",
                                href="/File_Viewer",
                                style={
                                    "fontWeight": "bold",
                                    "color": "#007bff",
                                    "textDecoration": "underline"
                                }
                            ),
                            " to view the contents of a yaml or json file. Ensure that the path is relative to the repo root, and that the file exists.",
                        ],
                        style={
                            "width": "800px"
                        }
                    ),
                ],
                style={
                    "alignItems": "center",
                    "display": "flex",
                    "flexDirection": "column",
                    "width": "50%",
                    "margin": "0 auto",
                    "gap": "1rem"
                }
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

