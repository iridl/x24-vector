import os
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
from controls import Block, Sentence, Date, Units, Number, Text
import calc
import numpy as np
from pathlib import Path
import pingrid

CONFIG_GLOBAL = pingrid.load_config(os.environ["CONFIG"])
CONFIG = CONFIG_GLOBAL["wat_bal_monit"]
DR_PATH = CONFIG_GLOBAL["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def help_layout(buttonname, id_name, message):
    return html.Div([
        html.Label(
            f"{buttonname}:",
            id=id_name,
            style={"cursor": "pointer","font-size": "100%","padding-left":"3px"}
        ),
        dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
    ])


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
                            "overflow":"scroll","height":"95vh",
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
                                style={"overflow":"scroll","height": "45%"},
                                className="g-0",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        local_layout(),
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
                                style={"overflow":"scroll","height":"55%"},
                                className="g-0",
                            ),
                        ],style={"overflow":"scroll","height":"95vh"},
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
                                src="assets/" + CONFIG_GLOBAL["logo"],
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / " + CONFIG["app_title"],
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
            ),
            html.Div(
                [help_layout("Date", "date", "Data to map")],
                style={
                    "color": "white",
                    "position": "relative",
                    "width": "75px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="time_selection",
                        clearable=False,
                    ),
                ],style={"width":"9%","font-size":".9vw"},
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
                            CONFIG["app_title"],
                        ]
                    ),
                    html.P(
                        f"""
			The Maproom monitors recent daily water balance.
                        """
                    ),
                    dcc.Loading(html.P(id="map_description"), type="dot"),
                    html.P(
                        f"""
                        The soil-plant-water balance algorithm estimates soil moisture
                        and other characteristics of the soil and plants since planting date
                        of the current season and up to now.
                        It is driven by rainfall and the crop cultivars Kc
                        that can be changed in the Control Panel below.
                        """
                    ),
                    html.P(
                        f"""
                        Map another day of the simulation using the Date control on the top bar,
                        or by clicking a day of interest on the time series graph..
                        You can pick a day between planting and today (or last day of available data).
                        """
                    ),
                    html.P(
                        f"""
                        Pick another point to monitor evolution since planting
                        with the controls below or by clicking on the map.
                        """
                    ),
                    html.H5("Water Balance Outputs"),
                ]+[
                    html.P([html.H6(val["menu_label"]), html.P(val["description"])])
                    for key, val in CONFIG["map_text"].items()
                ]+[
                    html.H5("Dataset Documentation"),
                    html.P(
                        f"""
                        Reconstructed gridded rainfall from {CONFIG_GLOBAL["institution"]}.
                        The time series were created by combining
                        quality-controlled station observations in 
                        {CONFIG_GLOBAL["institution"]}’s archive with satellite rainfall estimates.
                        """
                    ),
                ],
                style={"position":"relative","height":"30%", "overflow":"scroll"},
            ),
            html.H3("Controls Panel",style={"padding":".5rem"}),
            html.Div(
                [
                    Block("Pick a point",
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.FormFloating([
                                        dbc.Input(
                                            id="lat_input",
                                            min=lat_min,
                                            max=lat_max,
                                            type="number",
                                        ),
                                        dbc.Label("Latitude", style={"font-size": "80%"}),
                                        dbc.Tooltip(
                                            f"{lat_label}",
                                            target="lat_input",
                                            className="tooltiptext",
                                        )
                                    ]),
                                ),
                                dbc.Col(
                                    dbc.FormFloating([
                                        dbc.Input(
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
                                        )
                                    ]),
                                ),
                                dbc.Button(id="submit_lat_lng", children='Submit'),
                            ],
                        ),
                    ),
                    Block("Water Balance Outputs to display",
                        dbc.Select(
                            id="map_choice",
                            value=list(CONFIG["map_text"].keys())[0],
                            options=[
                                {"label": val["menu_label"], "value": key}
                                for key, val in CONFIG["map_text"].items()
                            ],
                        ),
                    ),
                    Block(
                        "Current Season",
                        Sentence(
                            "Planting Date",
                            Date("planting_", 1, CONFIG["planting_month"]),
                            "for",
                            Text("crop_name", CONFIG["crop_name"]),
                            "crop cultivars: initiated at",
                        ),
                        Sentence(
                            Number("kc_init", CONFIG["kc_v"][0], min=0, max=2, html_size=4),
                            "through",
                            Number("kc_init_length", CONFIG["kc_l"][0], min=0, max=99, html_size=2),
                            "days of initialization to",
                        ),
                        Sentence(
                            Number("kc_veg", CONFIG["kc_v"][1], min=0, max=2, html_size=4),
                            "through",
                            Number("kc_veg_length", CONFIG["kc_l"][1], min=0, max=99, html_size=2),
                            "days of growth to",
                        ),
                        Sentence(
                            Number("kc_mid", CONFIG["kc_v"][2], min=0, max=2, html_size=4),
                            "through",
                            Number("kc_mid_length", CONFIG["kc_l"][2], min=0, max=99, html_size=2),
                            "days of mid-season to",
                        ),
                        Sentence(
                            Number("kc_late", CONFIG["kc_v"][3], min=0, max=2, html_size=4),
                            "through",
                            Number("kc_late_length", CONFIG["kc_l"][3], min=0, max=99, html_size=2),
                            "days of late-season to",
                        ),
                        Sentence(
                            Number("kc_end", CONFIG["kc_v"][4], min=0, max=2, html_size=4),
                        ),
                        dbc.Button(id="submit_kc", children='Submit'),
                    ),
                ],
                style={"position":"relative","height":"60%", "overflow":"scroll"},
            ),
        ],
        fluid=True,
        className="scrollable-panel p-3",
        style={"overflow":"scroll","height":"100%","padding-bottom": "1rem", "padding-top": "1rem"},
    )


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
                zoom=CONFIG_GLOBAL["zoom"],
                maxBounds = [[lat_min, lon_min],[lat_max, lon_max]],
                minZoom = CONFIG_GLOBAL["zoom"] - 1,
                maxZoom = CONFIG_GLOBAL["zoom"] + 10, #this was completely arbitrary
                style={
                    "width": "100%",
                    "height": "77%",
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
        style={"padding": "0rem", "height":"100%"},
    )


def local_layout():
    return html.Div( 
        [   
            dbc.Tabs(
                [
                    dbc.Tab(
                        [
                            dbc.Spinner(dcc.Graph(id="wat_bal_plot")),
                        ],
                        label="Evolution since planting",
                    ),
                ],
                className="mt-4",
            )
        ],
    )
