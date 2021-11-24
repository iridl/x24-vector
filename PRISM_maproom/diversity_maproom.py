import geopandas as gpd
from geopandas import GeoDataFrame
import os
import flask
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pyaconf
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import json

DATA_path = "/data/drewr/PRISM/eBird/derived/detectionProbability/Mass_towns/"

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    #url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "PRISM Maproom"

APP.layout = layout.app_layout()

df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
candidates = df["city"].unique()
with open(f"{DATA_path}ma_towns.json") as geofile:
    towns = json.load(geofile)

@APP.callback(
    Output("choropleth", "figure"),
    [Input("candidate", "value")])
def display_choropleth(candidate):
    fig = px.choropleth(df, geojson=geofile, color=candidate, locations = "city")
    return fig


if __name__ == "__main__":
    APP.run_server(debug=True)
