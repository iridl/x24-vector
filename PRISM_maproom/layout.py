import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dl
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number
import pandas as pd
import json
from dash_extensions.javascript import arrow_function, assign
import dash_leaflet.express as dlx
import geopandas as gpd

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"
INIT_LAT = 42.4072
INIT_LNG = -71.3824

df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
df = df.drop_duplicates()
df = df[['city', 'date','eBird.DP.RF', 'eBird.DP.RF.SE']]

candidates = ["eBird.DP.RF", "eBird.DP.RF.SE"]

style_handle = assign("""function(feature, context){
    style = {
        weight: 2,
        opacity: 1,
        color: 'white',
        dashArray:'3',
        fillOpacity: 0.7,
        fillColor:feature.properties["color"]};
    return style;
}""")

def app_layout():
    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(),
                        #width=17,
                        sm=12,
                        md=4,
                        style={
                            "background-color": "white",
                            "border-style": "solid",
                            "border-color": LIGHT_GRAY,
                            "border-width": "0px 1px 0px 0px",
                            "width":"10%",
                        },
                    ),
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        map_layout(),
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
                        style={"background-color": "white", "width":"70%"},
                    ),
                ],
                no_gutters=True,
            ),
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
                                src="prism_logo3.png",
                                alt="PRISM",
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
                    "Maproom Documentation",
                ]
            ),
            html.P(
                [
                    "This maproom allows the user to explore data from two different domains associated with PRISM:",
                    html.Br(),
                   """eBird derived data, from the ecology domain, and MA outage data.
                     See dataset documentation below for more on these datasets."""
                ]
            ),
            html.P(
                [
                    """
                    To view the time series for a bird species, 
                    either click on the town within the map or select from the city dropdown. 
                    You may hover over towns to see town names before selecting one to visualise.  
                    """,
                    html.Br(),
                    """
                    The choropleth map by default displays the date and diversity data selected in the controls panel. 
                    To view other diversity data, you may update the choropleth by choosing a different date or diversity index.
                    """
                ]
            ),

            Block("Select Date",
                dcc.Dropdown(id="date_dropdown",
                    options=[
                        {"label": i, "value": i} for i in df.date.unique()
                    ],
                    value= "2005-01-03"    
                ),
            ),

            Block("Select City",
                dcc.Dropdown(id="city_dropdown",
                    options=[
                        {"label": i, "value": i} for i in df.city.unique()
                    ],
                ),
            ),
            Block("Select Diversity Index",
                dcc.Dropdown(id="candidate",
                    options=[
                        {'value': x, 'label': x} for x in candidates
                    ],
                    value=candidates[0]
                ),
            ),  
            html.P( #room for more text
                """
                """
            ),
            html.H5("Dataset Documentation"),
            html.P(
                dcc.Markdown('''
                    These data describe electrical power outages and relative bird abundance 
                    for 14 outage-prone species in the state of Massachusetts. 
                    Relative bird abundance can be used as a measurement of animal activity which 
                    is an important predictor of animal-related power outages. 
 
                    Outage records with causes of "animal", "animal-other", "birds", 
                    and "squirrels" in the "Reason for Outage" outage data variable can be identified 
                    as animal-related outages. Please refer [here](https://github.com/mefeng7/Bird_Outages_MA) 
                    for analysis relating these data.
                ''')
            ),
            html.H5("Citations"),
            html.P(
                dcc.Markdown('''
                    We recommend the following citation for use of these datasets: 

                    Feng, M.-L.E., Owolabi, O.O., Schafer, L.J., Sengupta, S., Wang, L., 
                    Matteson, D.S., Che-Castaldo, J.P., and Sunter, D.A. 2021. 
                    Informing data-driven analyses of animal-related electric outages using 
                    species distribution models and community science data [Manuscript Submitted for Publication]."
                ''')
            ),
        ],
        fluid=True,
        className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )

def map_layout():
    return dbc.Container(
        [
            dl.Map(
                [
                    dl.LayersControl(
                        [
                            dl.BaseLayer(dl.TileLayer(), name="Streets", checked=True),
                            dl.BaseLayer(
                                dl.TileLayer(url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"), 
                                name="Topo", checked=False), 
                            dl.Overlay(
                                dl.LayerGroup(
                                    dl.GeoJSON(
                                        # TODO we're going from a
                                        # geopandas dataframe to json
                                        # to a python dictionary to
                                        # json. There's got to be a
                                        # more direct path.
                                        data={},#json.loads(dfJoined.to_json()), 
                                        id="towns",
                                        options=dict(style=style_handle),
                                        zoomToBounds=True,
                                        zoomToBoundsOnClick=True, #how to style click?
                                        hoverStyle=arrow_function(dict(weight=6, color='#666', dashArray='')),
                                    ),id="geoJSON"
                                ), name="GeoJSON", checked=True,
                            ),
                        ]
                    ), #layersControl
                    html.Div(id="info", className="info", 
                        style={"position": "absolute", "top": "10px", "left": "50px", "z-index": "1000"} 
                    ),html.Div(id="colorBar"),
                ],
                style={"width": "100%", "height": "50vh", "display": "block", "margin": "auto"},
                id="layersMap"
            ),
        ],
        fluid=True,
        style={"padding": "0rem", "height":"50vh"},
    )


def results_layout():
    return dbc.Tabs(
        [
            dbc.Tab(
                [
                    #html.Div(id="colorBar"),
                    dbc.Spinner(dcc.Graph(
                        id="timeSeriesPlot"
                    ))
                ],
                label="Graphs",
            ),

            dbc.Tab(
                dbc.Spinner(
                    dl.Map()
                ),label="extra tab",
            ),
        ],
        style={"width":"100%", "height": "40%", "margin":"auto"}
    )
