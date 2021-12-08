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
dfSel= df.set_index("city")
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
    Output("city_dropdown", "value"),
    [Input("towns", "click_feature")])
def updateCityDD(feature):
    featureString = feature['id']
    return featureString 

@APP.callback(
    Output("timeSeriesPlot", "figure"),
    #Output("diValue", "children"),
    #Output("city_dropdown", "value"),
    #Output("candidate", "value"),
    [Input("city_dropdown", "value"), Input("candidate", "value")])
def update_timeSeries(city, candidate):
    cityString = city
    dfCity = dfJoined.loc[dfJoined.index == cityString]
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
    return timeSeries

def get_info(feature=None):
    header = [html.H4("Hover to see city name")]
    if not feature:
        return header
    return [html.H4(feature["id"])]

@APP.callback(
    Output("info", "children"),
    [Input("towns", "hover_feature")])
def info_hover(feature):
    return get_info(feature)

@APP.callback(
    Output("towns", "data"),
    [Input("date_dropdown", "value"), Input("candidate" ,"value")])
def colorMap(date, candidate):
    dfLoc = dfJoined.loc[dfJoined["date"]==date]
    dfLoc = dfLoc[['geometry', candidate]]
    dfLoc = dfLoc.rename(columns ={candidate: "diversity"})
    quantiles = [0, .1, .2, .5, .6, .8, .9]
    classes = []
    for q in quantiles:
        value = dfLoc["diversity"].quantile(q)
        valueRound = value.round(3)
        classes.append(valueRound)
    #dfLoc["color"] = "yellow"
    colorConditions = [
    (dfLoc['diversity'] < classes[0]),
    (dfLoc['diversity'] >= classes[0]) & (dfLoc['diversity'] < classes[1]),
    (dfLoc['diversity'] >= classes[1]) & (dfLoc['diversity'] < classes[2]),
    (dfLoc['diversity'] >= classes[2]) & (dfLoc['diversity'] < classes[3]),
    (dfLoc['diversity'] >= classes[3]) & (dfLoc['diversity'] < classes[4]),
    (dfLoc['diversity'] >= classes[4]) & (dfLoc['diversity'] < classes[5]),
    (dfLoc['diversity'] >= classes[5]) & (dfLoc['diversity'] < classes[6]),
    (dfLoc['diversity'] >= classes[6]),    
    ]
    colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
    dfLoc["color"] = np.select(colorConditions, colorscale)
    print(dfLoc)
    toJSON = json.loads(dfLoc.to_json())
    return toJSON

if __name__ == "__main__":
    APP.run_server(debug=True)
