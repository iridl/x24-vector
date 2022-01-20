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
from psycopg2 import connect
from datetime import datetime
import dash_table

CONFIG = pyaconf.load(os.environ["CONFIG"])
PFX = CONFIG["core_path"]
MATOWNS_PFX = CONFIG["maTowns_path"]
EBIRD_PFX = CONFIG["eBird_path"]
SQL_UN = CONFIG["user"]
SQL_PW = CONFIG["password"]
SQL_HN = CONFIG["hostname"]
SQL_DB = CONFIG["database"]

SQL_CONN = connect(host=SQL_HN, database=SQL_DB, user=SQL_UN, password=SQL_PW)

# Species
SPECIES_FILES = list(Path(EBIRD_PFX).glob("*.csv"))
df = pd.concat(
    (
        pd.read_csv(
            path,
            usecols=["city", "date", "eBird.DP.RF"],
            index_col=0,
        ).assign(species=path.stem.partition("_")[0])
        for path in SPECIES_FILES
    ),
)

# this is the MA towns
geoDf = gpd.read_file(f"{MATOWNS_PFX}ma_towns.json")
geoDf = geoDf.drop_duplicates()
geoDf = geoDf.set_index("city")
dfJoined = geoDf.join(
    df
)  # here we have joined together the eBird and MA towns data to one

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    # url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "PRISM Maproom"

APP.layout = layout.app_layout()

# this updates the choropleth to display by city
@APP.callback(Output("city_dropdown", "value"), [Input("towns", "click_feature")])
def updateCityDD(feature):
    if feature is not None:
        featureString = feature["id"]
        return featureString
    if feature is None:
        return None


# updating the time series plot through the city dropdown
# and the species dropdown
@APP.callback(
    Output("timeSeriesPlot", "figure"),
    [Input("city_dropdown", "value"), Input("species_dropdown", "value")],
)
def update_timeSeries(city, myspecies):
    cityString = city
    if cityString is None:
        errorFig = pgo.Figure().add_annotation(x=2, y=2,text="No Data to Display. Select a city.",font=dict(family="sans serif",size=30,color="crimson"),showarrow=False, yshift=10, xshift=60)
        return errorFig
    dfCity = dfJoined.where(dfJoined["species"] == myspecies).loc[
        dfJoined.index == cityString
    ]
    timeSeries = px.line(data_frame=dfCity, x="date", y="eBird.DP.RF")
    timeSeries.update_traces(
        mode="markers+lines", hovertemplate="%{y} %{x}", connectgaps=False
    )
    timeSeries.update_layout(
        yaxis_title="Detection Probability",
        xaxis_title="dates",
        title=f"{myspecies} in {city}",
    )
    return timeSeries


# callback to update info on hover of map
@APP.callback(Output("info", "children"), [Input("towns", "hover_feature")])
def info_hover(feature=None):
    header = [html.H4("Hover to see city name")]
    if not feature:
        return header
    return [html.H4(feature["id"])]


# callback that updates the choropleth map and outage table
@APP.callback(
    Output("towns", "data"),
    Output("colorBar", "children"),
    [Input("date_dropdown", "value"), Input("species_dropdown", "value")],
)
def colorMap(date, myspecies):
    dfLoc = dfJoined.where(dfJoined["species"] == myspecies).loc[
        dfJoined["date"] == date
    ]
    dfLoc = dfLoc[["geometry", "eBird.DP.RF"]]
    dfLoc = dfLoc.rename(columns={"eBird.DP.RF": "diversity"})
    classes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    colorConditions = [
        (dfLoc["diversity"] >= classes[0]) & (dfLoc["diversity"] < classes[1]),
        (dfLoc["diversity"] >= classes[1]) & (dfLoc["diversity"] < classes[2]),
        (dfLoc["diversity"] >= classes[2]) & (dfLoc["diversity"] < classes[3]),
        (dfLoc["diversity"] >= classes[3]) & (dfLoc["diversity"] < classes[4]),
        (dfLoc["diversity"] >= classes[4]) & (dfLoc["diversity"] < classes[5]),
        (dfLoc["diversity"] >= classes[5]) & (dfLoc["diversity"] < classes[6]),
        (dfLoc["diversity"] >= classes[6]) & (dfLoc["diversity"] < classes[7]),
        (dfLoc["diversity"] >= classes[7]) & (dfLoc["diversity"] < classes[8]),
        (dfLoc["diversity"] >= classes[8]) & (dfLoc["diversity"] < classes[9]),
        (dfLoc["diversity"] >= classes[9]),
    ]
    colorscale = [
        "#FFFDA0",
        "#FFEDA0",
        "#FED976",
        "#FEB24C",
        "#FD8D3C",
        "#FC4E2A",
        "#E31A1C",
        "#BD0026",
        "#800026",
        "#4a148c",
    ]
    dfLoc["color"] = np.select(colorConditions, colorscale)
    ctg = [str(i) for i in classes]
    colorbar = dlx.categorical_colorbar(
        categories=ctg,
        colorscale=colorscale,
        width=350,
        height=30,
        position="bottomleft",
    )
    toJSON = json.loads(dfLoc.to_json())
    return toJSON, colorbar


@APP.callback(
    Output("outagePoints", "data"),
    Output("outageTable", "children"),
    Output("dateWarning", "children"),
    [Input("date_dropdown", "value")],
)
def outagePoints(date):
    # outage data
    dateSince = datetime.strptime("2010-01-01", "%Y-%m-%d")
    dateSelect = datetime.strptime(date, "%Y-%m-%d")
    dateDiff = (dateSelect - dateSince).days  # beginning of the day we are looking at
    dateDiff2 = dateDiff + 6.99  # end of the week we are looking at
    # query and edit outage df
    outageDF = pd.read_sql(
        """WITH testTable AS (select days_since_out, days_since_in, city_town, geo_id, reason_for_outage from ma_outage) SELECT * FROM testTable WHERE days_since_out <= %(dateDiff2)s AND days_since_in >= %(dateDiff)s;""",
        SQL_CONN,
        params={"dateDiff": dateDiff, "dateDiff2": dateDiff2},
    )
    if outageDF.empty:
        warning = dbc.Alert("No available outage data for selected date.", color="warning")
        return None, None, warning
    outageDF["outageCount"] = outageDF["geo_id"].map(outageDF["geo_id"].value_counts())
    outageDF = (
        outageDF.rename(columns={"city_town": "city"}).set_index("city").sort_index()
    )
    outageDF = outageDF.drop(columns=["days_since_out", "days_since_in", "geo_id"])
    outageDFunique = outageDF[~outageDF.index.duplicated(keep="first")]
    # getting the lat / lng from the geography column in the geodataframe to use for points
    geoDf["centroid"] = geoDf.centroid
    geoDf["lon"] = geoDf["centroid"].x
    geoDf["lat"] = geoDf["centroid"].y
    geoDf.drop("centroid", axis=1, inplace=True)
    mergedDF = pd.merge(
        outageDFunique, geoDf, right_index=True, left_index=True
    )  # merged the outage and MA towns dataframes
    mergedDF = gpd.GeoDataFrame(
        mergedDF, geometry=gpd.points_from_xy(mergedDF.lon, mergedDF.lat)
    )
    outageDF = outageDF.reset_index()
    outageTable = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in outageDF.columns],
        data=outageDF.to_dict("records"),
        style_header=dict(backgroundColor="grey"),
        style_cell=dict(textAlign="left", height="20px", width="20px"),
        filter_action="native",
        sort_action="native",
        fixed_rows=dict(headers=True),
    )
    toJSON2 = json.loads(mergedDF.to_json())
    return toJSON2, outageTable, None


if __name__ == "__main__":
    APP.run_server(debug=CONFIG["mode"])
