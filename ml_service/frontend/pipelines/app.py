import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from ml_service.frontend.pipelines.pipelines_metadata import FRONTEND_PIPELINES
from ml_service.frontend.pipelines.utils import call_pipeline

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

tabs = []
modals = []

for pipeline in FRONTEND_PIPELINES:
    form_inputs = []
    for field in pipeline["fields"]:
        input_id = f"{pipeline['name']}-{field['name']}"
        label = dbc.Label(field["name"], html_for=input_id, className="fw-bold")
        if field["type"] == "text":
            input_component = dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="text")
        elif field["type"] == "number":
            input_component = dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="number")
        elif field["type"] == "boolean":
            input_component = dbc.Checklist(
                options=[{"label": "Enable", "value": True}],
                value=[field["value"]] if field.get("value") else [],
                id=input_id,
                inline=True
            )
        elif field["type"] == "dropdown":
            input_component = dcc.Dropdown(
                id=input_id,
                options=[{"label": opt, "value": opt} for opt in field["options"]],
                value=field.get("value", None),
                clearable=False
            )
        else:
            input_component = dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="text")
        form_inputs.append(dbc.Form([label, input_component], className="mb-3"))

    run_btn_id = f"{pipeline['name']}-submit"
    form_inputs.append(dbc.Button(
        "Run Pipeline",
        id=run_btn_id,
        color="primary",
        className="mt-2",
        style={"width": "25%", "height": "56px", "margin": "0 auto", "display": "block"}
    ))
    output_id = f"{pipeline['name']}-output"
    form_inputs.append(html.Div(id=output_id, className="mt-3"))

    card_body = dbc.CardBody(form_inputs)
    tabs.append(
        dbc.Tab(
            dbc.Card(
                card_body,
                className="shadow-sm p-3 mb-3",
                style = {
                    "backgroundColor": "#91abd3",
                    "marginTop": "3rem"
                }
            ),
            label=pipeline["name"],
            style = {
                "width": "50%",
                "margin": "0 auto   "
            }
        )
    )

    modal_id = f"{pipeline['name']}-modal"
    confirm_id = f"{pipeline['name']}-confirm"
    cancel_id = f"{pipeline['name']}-cancel"
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

app.layout = dbc.Container([
        html.H1(
            "ML Pipelines Dashboard",
            className="text-center",

        ),
        dbc.Tabs(
            tabs,
            style = {
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
        "minHeight": "100vh",
    }
)


def create_pipeline_callbacks(pipeline):
    input_ids = [f"{pipeline['name']}-{f['name']}" for f in pipeline["fields"]]
    submit_id = f"{pipeline['name']}-submit"
    output_id = f"{pipeline['name']}-output"
    modal_id = f"{pipeline['name']}-modal"
    confirm_id = f"{pipeline['name']}-confirm"
    cancel_id = f"{pipeline['name']}-cancel"

    @app.callback(
        Output(modal_id, "is_open"),
        Input(submit_id, "n_clicks"),
        Input(confirm_id, "n_clicks"),
        Input(cancel_id, "n_clicks"),
        State(modal_id, "is_open"),
        prevent_initial_call=True
    )
    def toggle_modal(submit_click, confirm_click, cancel_click, is_open):
        ctx = dash.callback_context
        if not ctx.triggered:
            return is_open
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == submit_id:
            return True
        elif button_id in [cancel_id, confirm_id]:
            return False
        return is_open

    @app.callback(
        Output(output_id, "children"),
        Input(confirm_id, "n_clicks"),
        [State(iid, "value") for iid in input_ids],
        prevent_initial_call=True
    )
    def run_pipeline(n_clicks, *values):
        if n_clicks is None:
            return dash.no_update
        payload = {}
        field_names = [f["name"] for f in pipeline["fields"]]
        for k, v, f in zip(field_names, values, pipeline["fields"], strict=True):
            payload[k] = bool(v) if f["type"] == "boolean" else v
        result = call_pipeline(pipeline["endpoint"], payload)
        return dbc.Textarea(value=str(result), style={"width": "100%", "height": "200px"}, className="mt-2")


for p in FRONTEND_PIPELINES:
    create_pipeline_callbacks(p)

if __name__ == "__main__":
    app.run(debug=True)
