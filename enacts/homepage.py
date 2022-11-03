from flask import Flask
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.exceptions import NotFound
from flex_fcst import maproom as flex_fcst
from onset import maproom as onset

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
    requests_pathname_prefix="/python-maproom/"
)

portal.layout = dbc.Container([
    dbc.Row(html.H1("ZMD Python Maproom Suite")),
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
])

server = Flask(__name__)

print(onset.SERVER)

server.wsgi_app = DispatcherMiddleware(NotFound(), {
   "/python-maproom": portal.server,
   "/python-maproom/onset": onset.SERVER,
   "/python-maproom/flex-fcst": flex_fcst.SERVER,
})

if __name__ == "__main__":
    server.run()
