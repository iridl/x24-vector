import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dlf
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number
import pandas as pd

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"
INIT_LAT = 42.4072
INIT_LNG = -71.3824

df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
candidates = ["eBird.DP.RF", "eBird.DP.RF.SE"]

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
                                no_gutters=True,
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
                                no_gutters=True,
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
                no_gutters=True,
            ),
            html.Div(id="coord_alert",style={'position':'fixed','bottom':'0', 'width':'60%','right':'20px'}, children=[]),
        ],
        fluid=True,
        style={"padding-left": "0px", "padding-right": "0px"},
    )


def navbar_layout():
    return dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(
                                #src="", #put logo for PRISM here?
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "PRISM / Bird Diversity and Outage data",
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="https://iridl.ldeo.columbia.edu",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Select(
                                id="select",
                                value="en",
                                bs_size="sm",
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
                    no_gutters=True,
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
                    "Some Info",
                ]
            ),
            html.P(
                """
                Explain some things about the maproom.
                """
            ),
            html.P(
                """
                Explain some more things about the maproom.
                """
            ),

            Block("Dropdown",
                  dbc.Select(
                      id="date_input",
                      value="some options",
                      bs_size="sm",
                      options=[
                          {"label": "Onset", "value": "onset"},
                          {"label": "Cessation", "value": "cessation"},
                      ],
                  ),
            ),
            html.P("Candidate:"),
            dcc.RadioItems(
                id='candidate', 
                options=[{'value': x, 'label': x} 
                    for x in candidates],
                value=candidates[0],
                labelStyle={'display': 'inline-block'}
            ),
            html.P(
                """
                Here is room for more text    
                """
            ),
            html.H5("Dataset Documentation"),
            html.P(
                """
                Some info about the data
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
                                checked=True,
                            ),
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.LayerGroup(id="layers_group"),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                ],
                id="map",
                center=[INIT_LAT, INIT_LNG],
                zoom=8,
                style={
                    "width": "100%",
                    "height": "60vh",
                },
            ),
            dcc.Graph(id="choropleth")
        ],
        fluid=True,
        style={"padding": "0rem"},
    )




def results_layout():
    return dbc.Tabs(
        [
            dbc.Tab(
                [
                    dbc.Spinner(dcc.Graph(
                        id="graph1",
                    )),
                    dbc.Spinner(dcc.Graph(
                        id="graph2",
                    ))	
                ],
                label="Graphs"
            ),

            dbc.Tab(
                dbc.Spinner(
                    dbc.Table([], id="table1", bordered=True, className="m-2")),
                label="Table"
            ),
        ],
        className="mt-4",
    )
