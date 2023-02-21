import os
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_leaflet as dlf
import plotly.express as px
from controls import Block, Sentence, DateNoYear, Number

import numpy as np
from pathlib import Path
import calc
import pingrid
import pandas as pd


GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["crop_suit"]

DR_PATH = GLOBAL_CONFIG["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    # Initialization
    rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)
    center_of_the_map = [((rr_mrg["Y"][int(rr_mrg["Y"].size/2)].values)), ((rr_mrg["X"][int(rr_mrg["X"].size/2)].values))]
    lat_res = np.around((rr_mrg["Y"][1]-rr_mrg["Y"][0]).values, decimals=10)
    lat_min = np.around((rr_mrg["Y"][0]-lat_res/2).values, decimals=10)
    lat_max = np.around((rr_mrg["Y"][-1]+lat_res/2).values, decimals=10)
    lon_res = np.around((rr_mrg["X"][1]-rr_mrg["X"][0]).values, decimals=10)
    lon_min = np.around((rr_mrg["X"][0]-lon_res/2).values, decimals=10)
    lon_max = np.around((rr_mrg["X"][-1]+lon_res/2).values, decimals=10)
    lat_label = str(lat_min)+" to "+str(lat_max)+" by "+str(lat_res)+"˚"
    lon_label = str(lon_min)+" to "+str(lon_max)+" by "+str(lon_res)+"˚"

    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(lat_min, lat_max, lon_min, lon_max, lat_label, lon_label),
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                            "overflow":"scroll","height":"95vh",#column that holds text and controls
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(center_of_the_map, lon_min, lat_min, lon_max, lat_max),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
                                style={"overflow":"scroll","height": "75%"}, #box the map is in
                                className="g-0",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        results_layout(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                            "min-height": "100px",
                                            "border-style": "solid",
                                            "border-color": LIGHT_GRAY,
                                            "border-width": "1px 0px 0px 0px",
                                        },
                                    ),
                                ],
                                style={"overflow":"scroll","height":"25%"}, #box the plots are in
                                className="g-0",
                            ),
                        ],style={"overflow":"scroll","height":"95vh"},#main column for map and results
                        sm=12,
                        md=8,
                    ),
                ],
                className="g-0",
            ),
        ],
        fluid=True,
        style={"padding-left": "1px", "padding-right": "1px","height":"100%"},
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                src="assets/" + GLOBAL_CONFIG["logo"],
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / " + CONFIG["crop_suit_title"],
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        sticky="top",
        color=IRI_GRAY,
        dark=True,
    )


def controls_layout(lat_min, lat_max, lon_min, lon_max, lat_label, lon_label):
    return dbc.Container(
        [
            html.Div(
                [
                    html.H5(
                        [
                            CONFIG["crop_suit_title"],
                        ]
                    ),
                    html.P(
                        f"""
                        """
                    ),
                    dcc.Loading(html.P(id="map_description"), type="dot"),
                    html.P(
                        f"""
                        """
                    ),
                ]+[
                    html.P([html.H6(val["menu_label"]), html.P(val["description"])])
                    for key, val in CONFIG["map_text"].items()
                ]+[
                    html.P(
                        """
                        """
                    ),
                    html.H5("Dataset Documentation"),
                    html.P(
                        f"""
                        """
                    ),
                ],
                style={"position":"relative","height":"30%", "overflow":"scroll"},#box holding text
            ),
            html.H3("Controls Panel",style={"padding":".5rem"}),
            html.Div(
                [
                    Block("Pick a point",
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id = "lat_input",
                                        min=lat_min,
                                        max=lat_max,
                                        type="number",
                                    ),
                                    dbc.Label("Latitude", style={"font-size": "80%"}),
                                    dbc.Tooltip(
                                        f"{lat_label}",
                                        target="lat_input",
                                        className="tooltiptext",
                                    )]),
                                ),
                                dbc.Col(
                                    dbc.FormFloating([dbc.Input(
                                        id = "lng_input",
                                        min=lon_min,
                                        max=lon_max,
                                        type="number",
                                    ),
                                    dbc.Label("Longitude", style={"font-size": "80%"}),
                                    dbc.Tooltip(
                                        f"{lon_label}",
                                        target="lng_input",
                                        className="tooltiptext",
                                    )]),
                                ),
                                dbc.Button(id="submit_lat_lng", children='Submit'),
                            ],
                        ),
                        "Choose a map layer to view:",
                        dbc.Select(
                            id="map_choice",
                            value=list(CONFIG["map_text"].keys())[0],
                            options=[
                                {"label": val["menu_label"], "value": key}
                                for key, val in CONFIG["map_text"].items()
                            ],
                        ),
                        "Choose a target season",
                        dbc.Select(
                            id="season_choice",
                            value= "DJF",
                            options=[
                                {"label":"Dec-Feb", "value":"DJF"},
                                {"label":"Mar-May", "value":"MAM"},
                                {"label":"Jun-Aug", "value":"JJA"},
                                {"label":"Sep-Nov", "value":"SON"},
                            ],
                        ),
                    ),
                    Block(
                        "Rainfall tolerance",
                        Sentence(
                            "Total rainfall amount between",
                            Number("lowerWetThreshold", 500, min=0, max=99999),
                            "mm and",
                            Number("upperWetThreshold", 700, min=0, max=99999),
                            "mm",
                        ),
                    ),
                    Block(
                        "Temperature tolerance",
                        Sentence(
                            "Temperature range between",
                            Number("minimumTemp", 10, min=-99, max=999),
                            "C and",
                            Number("maximumTemp", 25, min=-99, max=99999),
                            "C",
                        ),
                    ),
                    Block(
                        "Optimal daily temperature",
                        Sentence(
                            "Average daily temperature of",
                            Number("avgDailyTemp", 15, min=0, max=99999),
                            "C",
                        ),
                    ),
                    Block(
                        "Season length",
                        Sentence(
                            "Maximum length of season",
                            Number("seasonLength", 75, min=0, max=99999),
                            "days",
                        ),
                    ),
                    Block(
                        "Wet days",
                        Sentence(
                            "Minimum number of wet days within a season:",
                            Number("minWetDays", 60, min=0, max=99999),
                            "days",
                        ),
                        Sentence(
                            "where a wet day is a day with rainfall more than",
                            Number("wetDayDef", 10, min=0, max=9999),
                            "mm",
                        ),
                    ),
                    Block(
                        "Dry spells",
                        Sentence(
                            "Maximum number of dry spells within the first",
                            Number("numberMonths", 3, min=0, max=12),
                            "months of the season:",
                            Number("numberDrySpells", 4, min=0, max=99),
                        ),
                        Sentence(
                            "where a dry spell is defined as a period of",
                            Number("daysInRow", 3, min=0, max=999),
                            "days with less than",
                            Number("drySpellRain", 5, min=0, max=999),
                            "mm of rainfall in each day",
                        ),
                    ),                    
                ],
                style={"position":"relative","height":"60%", "overflow":"scroll"},#box holding controls
            ),
        ], #end container
        fluid=True,
        className="scrollable-panel p-3",
        style={"overflow":"scroll","height":"100%","padding-bottom": "1rem", "padding-top": "1rem"},
    )    #style for container that is returned #95vh

def map_layout(center_of_the_map, lon_min, lat_min, lon_max, lat_max):
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(id="layers_control", position="topleft"),
                    dlf.LayerGroup(
                        [dlf.Marker(id="loc_marker", position=center_of_the_map)],
                        id="layers_group"
                    ),
                    dlf.ScaleControl(imperial=False, position="topright"),
                    dlf.Colorbar(
                        id="colorbar",
                        min=0,
                        position="bottomleft",
                        width=300,
                        height=10,
                        opacity=.8,
                    )
                ],
                id="map",
                center=center_of_the_map,
                zoom=GLOBAL_CONFIG["zoom"],
                maxBounds = [[lat_min, lon_min],[lat_max, lon_max]],
                minZoom = GLOBAL_CONFIG["zoom"] - 1,
                maxZoom = GLOBAL_CONFIG["zoom"] + 10, #this was completely arbitrary
                style={
                    "width": "100%",
                    "height": "87%",#height of the map 
                },
            ),
            html.H6(
                id="map_title"
            ),
            html.H6(
                id="hover_feature_label"
            )
        ],
        fluid=True,
        style={"padding": "0rem", "height":"100%"},#box that holds map and title
    )


def results_layout():
    return html.Div( 
        ["Print some data here" ],
        id="results_div",
    )
