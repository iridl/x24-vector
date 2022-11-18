import flask
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound
from flex_fcst import maproom as flex_fcst
from onset import maproom as onset
from monthly import maproom as monthly
import pingrid
import os

CONFIG = pingrid.load_config(os.environ["CONFIG"])

def maproom_card(title, desc, link):
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            html.Div([
		html.A(desc, href=link),
		],
                className="card-text",
            ),
        ]),
    ])


portal = dash.Dash(
    "portal",
    external_stylesheets=[
         dbc.themes.BOOTSTRAP,
    ],
    #url_base_pathname=f"/",
    requests_pathname_prefix="/python_maproom/"
)

portal.layout = dbc.Container([
    dbc.Row(html.H1("Python Maproom Suite")),
    dbc.Row(
        dbc.Col(maproom_card(
            "Onset",
            "Onset and Cessation Date Calculation",
            "onset/"
        )),
        className="mb-1",
    ),
    dbc.Row(
        dbc.Col(maproom_card(
            "Flexible Forecast", 
            "Explore a CPT probabilistic forecast", 
            "flex-fcst/"
        ))
    ),
    dbc.Row(
        dbc.Col(maproom_card(
            "Monthly Climatology",
            "See monthly climatologies and anomalies",
            "monthly-climatology/"
        ))
    ),
])


@portal.server.route(f"/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


server = flask.Flask(__name__)


server.wsgi_app = DispatcherMiddleware(NotFound(), {
    "/python_maproom": portal.server,
    "/python_maproom/onset": onset.SERVER,
    "/python_maproom/flex-fcst": flex_fcst.SERVER,
    "/python_maproom/monthly-climatology": monthly.SERVER
})

if __name__ == "__main__":
    server.run(
       host=CONFIG["server"],
        port=CONFIG["port"],
    )
