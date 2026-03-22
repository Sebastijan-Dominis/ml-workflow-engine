"""Layout construction for ML pipelines dashboard."""

import dash_bootstrap_components as dbc
from dash import dcc, html

from ml_service.frontend.pipelines.pipelines_metadata import FRONTEND_PIPELINES

PAGE_PREFIX = "/pipelines"

def build_layout():
    """Builds the full layout including tabs and modals."""
    tabs = []
    modals = []

    for pipeline in FRONTEND_PIPELINES:
        form_inputs = []

        for field in pipeline["fields"]:
            input_id = f"{PAGE_PREFIX}-{pipeline['name']}-{field['name']}"
            label = dbc.Label(f"{field['name']}", html_for=input_id, className="fw-bold")

            if field["type"] == "text":
                input_component = dbc.Input(
                    id=input_id,
                    placeholder=field.get("placeholder", ""),
                    type="text",
                    name=input_id,
                    style={"backgroundColor": "#f0f0f08b"} if field.get("optional", False) else {"backgroundColor": "#ffffff"}
                )

            elif field["type"] == "boolean":
                input_component = dbc.Checkbox(
                    id=input_id,
                    name=input_id,
                    label=field.get("label", field["name"]),
                    value=field.get("value", False)
                )

            elif field["type"] == "dropdown":
                input_component = dcc.Dropdown(
                    id=input_id,
                    options=[{"label": opt, "value": opt} for opt in field["options"]],
                    value=field.get("value", None),
                    clearable=False,
                    className="mb-3",
                )
                label = dbc.Label(field.get("label", field["name"]), html_for=input_id, className="fw-bold")
                input_component = html.Div([label, input_component])

            else:
                input_component = dbc.Input(
                    id=input_id,
                    placeholder=field.get("placeholder", ""),
                    type="text",
                    name=input_id
                )

            if field["type"] not in ["dropdown"]:
                form_inputs.append(dbc.Form([label, input_component], className="mb-3"))
            else:
                form_inputs.append(input_component)

        run_btn_id = f"{PAGE_PREFIX}-{pipeline['name']}-submit"

        form_inputs.append(
            dbc.Button(
                "Run Pipeline",
                id=run_btn_id,
                color="primary",
                className="mt-2",
                style={
                    "width": "25%",
                    "height": "56px",
                    "margin": "0 auto",
                    "display": "block"
                }
            )
        )

        output_id = f"{PAGE_PREFIX}-{pipeline['name']}-output"
        form_inputs.append(html.Div(id=output_id, className="mt-3"))

        tabs.append(
            dbc.Tab(
                dbc.Card(
                    dbc.CardBody(form_inputs),
                    className="shadow-sm p-3 mb-3",
                    style={
                        "backgroundColor": "#91abd3",
                        "marginTop": "3rem"
                    }
                ),
                label=pipeline["name"],
                style={
                    "width": "50%",
                    "margin": "0 auto"
                }
            )
        )

        modal_id = f"{PAGE_PREFIX}-{pipeline['name']}-modal"
        confirm_id = f"{PAGE_PREFIX}-{pipeline['name']}-confirm"
        cancel_id = f"{PAGE_PREFIX}-{pipeline['name']}-cancel"

        modals.append(
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Confirm Run")),
                    dbc.ModalBody(f"Are you sure you want to run the '{pipeline['name']}' pipeline?"),
                    dbc.ModalFooter([
                        dbc.Button("Cancel", id=cancel_id, className="ms-auto", color="secondary"),
                        dbc.Button("Confirm", id=confirm_id, color="danger")
                    ])
                ],
                id=modal_id,
                is_open=False
            )
        )

    return dbc.Container(
        [
            html.H1("ML Pipelines Dashboard", className="text-center"),
            dbc.Tabs(
                tabs,
                style={
                    "fontSize": "1.25rem",
                    "borderRadius": "0.5rem",
                    "border": "1px solid #5674d6",
                    "backgroundColor": "#91abd3",
                    "fontWeight": "bold",
                    "width": "50%",
                    "margin": "0 auto",
                    "marginTop": "2rem"
                }
            ),
            *modals
        ],
        fluid=True,
        style={
            "backgroundColor": "#8fa0d8",
            "minHeight": "100%",
            "paddingTop": "45px",
            "paddingBottom": "50px",
        }
    )
