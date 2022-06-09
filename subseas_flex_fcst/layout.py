from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number

import pyaconf
import os

CONFIG = pyaconf.load(os.environ["CONFIG"])

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"
# this should be inferred from data
# but since date is read in maproom.py
# and this is called by maproom.py
# I am not clear how this should be handled
INIT_LAT = CONFIG["init_lat"]
INIT_LNG = CONFIG["init_lng"]

# Reading units

phys_units = "mm"

def app_layout():
    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(),
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(),
                                        width=12,
                                        style={
                                            "background-color": "white",
                                        },
                                    ),
                                ],
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
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
            ),
        ],
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
    )


def help_layout(buttonname, id_name, message):
    return html.Div(
        [
            html.Label(f"{buttonname}:", id=id_name, style={"cursor": "pointer"}),
            dbc.Tooltip(
                f"{message}",
                target=id_name,
                className="tooltiptext",
            ),
        ]
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Sub-Seasonal Forecast",
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                ),
                href="https://iridl.ldeo.columbia.edu",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            html.Div(
                [
                    help_layout(
                        "Probability",
                        "probability_label",
                        "Custom forecast probability choices",
                    ),
                ],
                style={
                    "color": "white",
                    "position": "relative",
                    "width": "95px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="proba",
                        clearable=False,
                        options=[
                            dict(label="exceeding", value="exceeding"),
                            dict(label="non-exceeding", value="non-exceeding"),
                        ],
                        value="exceeding",
                    )
                ],
                style={
                    "position": "relative",
                    "width": "150px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        className="var",
                        id="variable",
                        clearable=False,
                        options=[
                            dict(label="Percentile", value="Percentile"),
                            dict(label=CONFIG["variable"], value=CONFIG["variable"]),
                        ],
                        value="Percentile",
                    )
                ],
                style={
                    "position": "relative",
                    "width": "200px",
                    "display": "inline-block",
                    "padding": "10px",
                    "vertical-align": "top",
                }
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="percentile",
                        clearable=False,
                        options=[
                            dict(label="10", value=0.1),
                            dict(label="15", value=0.15),
                            dict(label="20", value=0.2),
                            dict(label="25", value=0.25),
                            dict(label="30", value=0.3),
                            dict(label="35", value=0.35),
                            dict(label="40", value=0.4),
                            dict(label="45", value=0.45),
                            dict(label="50", value=0.5),
                            dict(label="55", value=0.55),
                            dict(label="60", value=0.60),
                            dict(label="65", value=0.65),
                            dict(label="70", value=0.70),
                            dict(label="75", value=0.75),
                            dict(label="80", value=0.8),
                            dict(label="85", value=0.85),
                            dict(label="90", value=0.9),
                        ],
                        value=0.5,
                    ),
                    html.Div([" %-ile"], style={
                        "color": "white",
                    })
                ],
                id="percentile_style",
            ),
            html.Div(
                [
                    dbc.Input(
                        id="threshold",
                        type="number",
                        className="my-1",
                        debounce=True,
                        value=0,
                    ),
                    html.Div([" "+phys_units], style={
                        "color": "white",
                    })
                ],
                id="threshold_style"
            ),
            dbc.Alert(
                "Please type-in a threshold for probability of non-/exceeding",
                color="danger",
                dismissable=True,
                is_open=False,
                id="forecast_warning",
                style={
                    "margin-bottom": "8px",
                },
            ),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Select(
                                id="select",
                                value="en",
                                options=[
                                    {"label": "English", "value": "en"},
                                    {
                                        "label": "Español",
                                        "value": "es",
                                        "disabled": True,
                                    },
                                    {"label": "Française", "value": "fr"},
                                    {"label": "Русский", "value": "ru"},
                                    {"label": "العربية", "value": "ar"},
                                    {"label": "हिन्दी", "value": "hi"},
                                    {"label": "中文", "value": "zh"},
                                ],
                            ),
                        ),
                    ],
                    className="ml-auto flex-nowrap mt-3 mt-md-0",
                    align="center",
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        sticky="top",
        color=IRI_GRAY,
        dark=True,
    )


def controls_layout():
    return dbc.Container(
        [
            html.H5(
                [
                    "Sub-Seasonal Forecast",
                ]
            ),
            html.P(
                """
                The Maproom displays full distribution sub-seasonal
                forecast in different flavors
                """
            ),
            html.P(
                """
                The map shows the probability of exceeding the 50th observed percentile.
                Use the controls in the top banner to show probability of non-exceeding,
                of other observed percentiles, or of a physical threshold.
                """
            ),
            html.P(
                """
                Click the map to show forecast and observed
                probability of exceeding and distribution
                at the clicked location
                """
            ),
        ],
        fluid=True,
        className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


def map_layout():
    return dbc.Container(
        [
            dlf.Map(
                [
                    dlf.LayersControl(
                        [
                            dlf.BaseLayer(
                                dlf.TileLayer(
                                    url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
                                ),
                                name="Street",
                                checked=False,
                            ),
                            dlf.BaseLayer(
                                dlf.TileLayer(
                                    url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                                ),
                                name="Topo",
                                checked=True,
                            ),
                            dlf.Overlay(
                                dlf.TileLayer(opacity=0.8, id="fcst_layer"),
                                name="Forecast",
                                checked=True,
                            ),
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.LayerGroup(id="layers_group"),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                    dlf.Colorbar(
                        id="fcst_colorbar",
                        position="bottomleft",
                        width=300,
                        height=10,
                        min=0,
                        max=1,
                        nTicks=11,
                        opacity=1,
                        tooltip=True,
                    ),
                ],
                id="map",
                center=[INIT_LAT, INIT_LNG],
                zoom=6,
                style={
                    "width": "100%",
                    "height": "50vh",
                },
            ),
        ],
        fluid=True,
        style={"padding": "0rem"},
    )


def results_layout():
    return dbc.Tabs(
        [
            dbc.Tab(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Spinner(
                                    dcc.Graph(
                                        id="cdf_graph",
                                    ),
                                )
                            ),
                            dbc.Col(
                                dbc.Spinner(
                                    dcc.Graph(
                                        id="pdf_graph",
                                    ),
                                )
                            ),
                        ]
                    )
                ],
                label="Local Forecast and Observations Distributions",
            )
        ],
        className="mt-4",
    )
