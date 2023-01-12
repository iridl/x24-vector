import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import flask
import os

from flask_app import FLASK
from flex_fcst import maproom as flex_fcst
from monthly import maproom as monthly
from onset import maproom as onset
import pingrid

CONFIG = pingrid.load_config(os.environ["CONFIG"])

HOMEPAGE = dash.Dash(
    name='homepage',
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    server=FLASK,
    url_base_pathname='/python_maproom/',
)


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


HOMEPAGE.layout = dbc.Container([
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


@FLASK.route(f"/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


if __name__ == "__main__":
    FLASK.run(
        host=CONFIG["server"],
        port=CONFIG["port"],
        debug=False,
    )
