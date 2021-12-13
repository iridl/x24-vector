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
import dash_leaflet.express as dlx
from  psycopg2 import connect
from datetime import datetime

CONFIG = pyaconf.load(os.environ["CONFIG"])
PFX = CONFIG["core_path"]
MATOWNS_PFX = CONFIG["maTowns_path"]
EBIRD_PFX = CONFIG["eBird_path"]
SQL_UN = CONFIG["user"]
SQL_PW = CONFIG["password"]
SQL_HN = CONFIG["hostname"]
SQL_DB = CONFIG["database"]

SQL_CONN = connect(host=SQL_HN, database=SQL_DB, user=SQL_UN, password=SQL_PW)

#this is the single eBird diversity data currently being read
df = pd.read_csv(f"{EBIRD_PFX}bhco_weekly_DP_MAtowns_05_18.csv")
df = df[['city', 'date','eBird.DP.RF', 'eBird.DP.RF.SE']]

#this is the MA towns
geoDf = gpd.read_file(f"{MATOWNS_PFX}ma_towns.json")
geoDf = geoDf.drop_duplicates()
geoDf = geoDf.set_index("city")
dfSel= df.set_index("city")
dfJoined = geoDf.join(dfSel) #here we have joined together the eBird and MA towns data to one

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

#this updates the choropleth to display by city
@APP.callback(
    Output("city_dropdown", "value"),
    [Input("towns", "click_feature")])
def updateCityDD(feature):
    if feature is not None:
        featureString = feature['id']
        return featureString
    if feature is None:
        return None
        
#updating the time series plot through the city dropdown and the diversity index dropdown
@APP.callback(
    Output("timeSeriesPlot", "figure"),
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
            yaxis_title="Detection Probability",
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

#callback to update info on hover of map
@APP.callback(
    Output("info", "children"),
    [Input("towns", "hover_feature")])
def info_hover(feature=None):
    header = [html.H4("Hover to see city name")]
    if not feature:
        return header
    return [html.H4(feature["id"])] 

#callback that updates the choropleth map from the date and diversity index dropdowns
#currently has SQL that will eventually return point data for outages although currently only print
    #the table in terminal
@APP.callback(
    Output("towns", "data"), Output("colorBar", "children"),
    [Input("date_dropdown", "value"), Input("candidate" ,"value")])
def colorMap(date, candidate):
    dfLoc = dfJoined.loc[dfJoined["date"]==date]
    dfLoc = dfLoc[['geometry', candidate]]
    dfLoc = dfLoc.rename(columns ={candidate: "diversity"})
    #outage data 
    dateSince = datetime.strptime("2010-01-01", "%Y-%m-%d")
    dateSelect = datetime.strptime(date, "%Y-%m-%d")
    dateDiff = (dateSince - dateSelect).days #beginning of the day we are looking at
    dateDiff2 = dateDiff + 1 #end of the day we are looking at
    outageDf = pd.read_sql('''
           drop table testTable, selectTable, uniquetable, countTable, joinedTable;
           create table testTable as select days_since_out, days_since_in, city_town, geo_id, reason_for_outage from ma_outage;
           Create table selectTable as select * from testTable where days_since_out between %(dateDiff)s and %(dateDiff2)s;
           create table uniquetable as select distinct city_town, geo_id from selectTable;
           create table countTable as select geo_id, count(geo_id) from selectTable group by geo_id;
           create table joinedTable as Select countTable.*, uniqueTable.city_town FROM countTable INNER JOIN uniqueTable ON countTable.geo_id = uniqueTable.geo_id;
           select * from joinedTable;''',SQL_CONN,params={"dateDiff":dateDiff, "dateDiff2":dateDiff2})
    print(outageDf)
    quantiles = [0, .1, .2, .5, .6, .8, .9]
    classes = []
    for q in quantiles:
        value = dfLoc["diversity"].quantile(q)
        valueRound = value.round(3)
        classes.append(valueRound)
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
    ctg = ["{}+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
    colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=350, height=30, position="bottomleft")
    toJSON = json.loads(dfLoc.to_json())
    return toJSON, colorbar

if __name__ == "__main__":
    APP.run_server(debug=CONFIG["mode"])
