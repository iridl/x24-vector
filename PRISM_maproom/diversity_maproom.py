import geopandas as gpd
from geopandas import GeoDataFrame
import os
import flask
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
from pathlib import Path
import pyaconf
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import json
import dash_leaflet as dl

DATA_path = "/data/drewr/PRISM/eBird/derived/detectionProbability/Mass_towns/"
df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
df = df[['city', 'date','eBird.DP.RF', 'eBird.DP.RF.SE']]
with open(f"{DATA_path}ma_towns.json") as geofile:
    towns = json.load(geofile)
#joined dataframes
geoDf = gpd.read_file(f"{DATA_path}ma_towns.json")
geoDf = geoDf.drop_duplicates()
geoDf = geoDf.set_index("city")
dfSel = df[df['date']== "2005-01-03"].set_index("city")
dfJoined = geoDf.join(dfSel)

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

@APP.callback(
    Output("timeSeriesPlot", "figure"),
    Output("diValue", "children"),
    Output("city_dropdown", "value"),
    Output("candidate", "value"),
    [Input("city_dropdown", "value"), Input("towns", "click_feature"), Input("candidate", "value")])
def update_timeSeries(city, feature, candidate):
    if feature is not None:
        featureString = feature['id']
        dfCity = df.loc[df['city'] == featureString]
        timeSeries = px.line(
            data_frame = dfCity,
            x = "date",
            y = candidate
        )
        timeSeries.update_traces(
            mode="markers+lines",
            hovertemplate='%{y} %{x}',
            connectgaps=False       
        )
        if candidate == "eBird.DP.RF":
            timeSeries.update_layout(
                yaxis_title="Detection robability",
                xaxis_title="dates",
                title=f"Time series plot for {city}"
            )
        if candidate == "eBird.DP.RF.SE":
            timeSeries.update_layout(
                yaxis_title="Inverse of standard error for detection probability",
                xaxis_title="dates",
                title=f"Time series plot for {city}"
            )
        if candidate is not None:
            return timeSeries, f"You clicked on {feature['id']}", featureString, candidate
        if candidate is None:
            return timeSeries, f"You clicked on {feature['id']}", featureString, candidate
    if feature is None:
        dfCity = df.loc[df['city'] == city]
        timeSeries = px.line(
            data_frame = dfCity,
            x = "date",
            y = candidate
        )
        timeSeries.update_traces(
            mode="markers+lines",
            hovertemplate='%{y} %{x}',
            connectgaps=False       
        )
        if candidate == "eBird.DP.RF":
            timeSeries.update_layout(
                yaxis_title="Detection robability",
                xaxis_title="dates",
                title=f"Time series plot for {city}"
            )
        if candidate == "eBird.DP.RF.SE":
            timeSeries.update_layout(
                yaxis_title="Inverse of standard error for detection probability",
                xaxis_title="dates",
                title=f"Time series plot for {city}"
            )
        if candidate is not None:
            return timeSeries, None, None, candidate
        if candidate is None:
            return timeSeries, None, None, candidate

def get_info(feature=None):
    header = [html.H4("Hover to see city name")]
    if not feature:
        return header
    return [html.B(feature["id"])]

@APP.callback(
    Output("info", "children"),
    [Input("towns", "hover_feature")])
def info_hover(feature):
    return get_info(feature)

#@APP.callback(
#    Output("towns", "hideout"),
#    [Input("date_dropdown", "value"), Input("towns","feature")]
#def colorMap(feature, date):
#    dfLoc = df[df["date"]==date]
#    value = feature.properties["

if __name__ == "__main__":
    APP.run_server(debug=True)
