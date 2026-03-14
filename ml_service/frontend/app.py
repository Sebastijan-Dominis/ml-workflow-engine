import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
from ml_service.frontend.pipelines_metadata import FRONTEND_PIPELINES
from ml_service.frontend.utils import call_pipeline

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

tabs = []
for pipeline in FRONTEND_PIPELINES:
    inputs = []
    for field in pipeline["fields"]:
        input_id = f"{pipeline['name']}-{field['name']}"
        if field["type"] == "text":
            inputs.append(dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="text"))
        elif field["type"] == "number":
            inputs.append(dbc.Input(id=input_id, placeholder=field.get("placeholder", ""), type="number"))
        elif field["type"] == "boolean":
            inputs.append(dbc.Checklist(options=[{"label": field["name"], "value": True}], value=[field["value"]] if field["value"] else [], id=input_id, inline=True))
        elif field["type"] == "dropdown":
            inputs.append(dcc.Dropdown(
                id=input_id,
                options=[{"label": opt, "value": opt} for opt in field["options"]],
                value=field.get("value", None)
            ))
    inputs.append(dbc.Button("Run Pipeline", id=f"{pipeline['name']}-submit", color="primary", className="mt-2"))
    inputs.append(html.Div(id=f"{pipeline['name']}-output", className="mt-2"))
    tabs.append(dbc.Tab(dbc.Card(dbc.CardBody(inputs)), label=pipeline["name"]))

app.layout = dbc.Container([
    html.H1("ML Pipelines Dashboard"),
    dbc.Tabs(tabs)
], fluid=True)

for pipeline in FRONTEND_PIPELINES:
    input_ids = [f"{pipeline['name']}-{field['name']}" for field in pipeline["fields"]]
    submit_id = f"{pipeline['name']}-submit"
    output_id = f"{pipeline['name']}-output"

    @app.callback(
        Output(output_id, "children"),
        Input(submit_id, "n_clicks"),
        [State(input_id, "value") for input_id in input_ids],
        prevent_initial_call=True
    )
    def run_pipeline(
        n_clicks,
        *values,
        pipeline=pipeline,
        pipeline_name=pipeline["name"],
        field_names=None
    ):
        if field_names is None:
            field_names = [f["name"] for f in pipeline["fields"]]
        payload = {}
        for k, v, f in zip(field_names, values, pipeline["fields"], strict=True):
            if f["type"] == "boolean":
                payload[k] = bool(v)
            else:
                payload[k] = v
        result = call_pipeline(pipeline_name, payload)
        return dbc.Textarea(value=str(result), style={"width": "100%", "height": "200px"})

if __name__ == "__main__":
    app.run(debug=True)
