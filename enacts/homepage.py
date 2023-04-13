import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import flask
import os

from globals_ import FLASK, GLOBAL_CONFIG
import pingrid


APP = dash.Dash(
    name='homepage',
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    server=FLASK,
    url_base_pathname=f'{GLOBAL_CONFIG["url_path_prefix"]}/',
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


APP.layout = dbc.Container([
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
