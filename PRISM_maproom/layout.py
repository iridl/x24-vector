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

DATA_path = "/data/drewr/PRISM/eBird/derived/detectionProbability/Mass_towns/"
df = pd.read_csv("/data/drewr/PRISM/eBird/derived/detectionProbability/originalCSV/bhco_weekly_DP_MAtowns_05_18.csv")
df = df.drop_duplicates()
df = df[['city', 'date','eBird.DP.RF', 'eBird.DP.RF.SE']]
with open(f"{DATA_path}ma_towns.json") as geofile:
    towns = json.load(geofile)
#joined dataframes
geoDf = gpd.read_file(f"{DATA_path}ma_towns.json")
geoDf = geoDf.drop_duplicates()
geoDf = geoDf.set_index("city")
dfSel = df[df['date']== "2005-01-03"].set_index("city")
dfJoined = geoDf.join(dfSel)


#getting classes for the colorscale, will have to make as a callback eventually because only doing for one data
quantiles = [0, .1, .2, .5, .6, .8, .9, 1]
classes= []
for q in quantiles:
    value = df["eBird.DP.RF"].quantile(q)
    valueRound = value.round(3)
    classes.append(valueRound) 

#setting up coloring of colorbar and polygons for choropleth map
candidates = ["eBird.DP.RF", "eBird.DP.RF.SE"]
ctg = ["{}+".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
style = dict(weight=2, opacity=1, color='white', dashArray='3', fillOpacity=0.7)
colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=350, height=30, position="bottomleft")
style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
    const value = feature.properties[colorProp];  // get value the determines the color
    for (let i = 0; i < classes.length; ++i) {
        if (value > classes[i]) {
            style.fillColor = colorscale[i];  // set the fill color according to the class
        }
    }
    return style;
}""")

#function to return infor on hover
def get_info(feature=None):
    if not feature:
        return [html.H3("Hover over city to see name")]
    return [html.H3(feature["properties"]["city"])]

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
                Click a city on the map, or select from the dropdown to get time series plot. 
                """
            ),

            Block("Select Date (currently only updates alt map)",
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
                    #value="Abington"
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
            dl.Map(
                [
                    dl.LayersControl(
                        [
                            dl.BaseLayer(dl.TileLayer(), name="Base Layer", checked=True),
                            dl.Overlay(
                                dl.LayerGroup(
                                    dl.GeoJSON(
                                        # TODO we're going from a
                                        # geopandas dataframe to json
                                        # to a python dictionary to
                                        # json. There's got to be a
                                        # more direct path.
                                        data=json.loads(dfJoined.to_json()), id="towns",
                                        options=dict(style=style_handle),
                                        zoomToBounds=True,
                                        zoomToBoundsOnClick=True, #how to style click?
                                        hoverStyle=arrow_function(dict(weight=6, color='#666', dashArray='')),
                                        hideout=dict(
                                            colorscale=colorscale,
                                            classes=classes,
                                            style=style,
                                            colorProp='eBird.DP.RF',
                                        )
                                    ),id="geoJSON"
                                ), name="GeoJSON", checked=True,
                            ),
                           # dl.Overlay(#this renders the alternate plotly express choropleth
                           #     dl.LayerGroup(
                           #         dcc.Graph(id="choropleth", figure={}), id="choroLayer"
                           #     ), name="Choropleth", checked=False,
                           # )
                        ]
                    ), #layersControl
                    html.Div(children=get_info(), id="info", className="info", 
                        style={"position": "absolute", "top": "10px", "left": "50px", "z-index": "1000"} 
                    ),colorbar,
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
                    html.Div(id="diValue"),
                    dbc.Spinner(dcc.Graph(
                        id="timeSeriesPlot"
                    ))#,
                    #dbc.Spinner(dcc.Graph(
                    #    id="choropleth", figure={}
                    #))	
                ],
                label="Graphs",
                #style={"width": "100%", "height": "50vh", "display": "block", "margin": "auto"}
            ),

            dbc.Tab(
                dbc.Spinner(
                    dl.Map()
                ),label="extra tab",
            ),
        ],
        #class_name="mt-4",
        style={"width":"100%", "height": "40%", "margin":"auto"}
    )
