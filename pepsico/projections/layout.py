from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
from fieldsets import Block, Select, PickPoint, Month, Number

from globals_ import GLOBAL_CONFIG

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"


def app_layout():

    return dbc.Container(
        [
            dcc.Location(id="location", refresh=True),
            navbar_layout(),
            dbc.Row(
                [
                    dbc.Col(
                        controls_layout(),
                        sm=12, md=4,
                        style={
                            "background-color": "white", "border-style": "solid",
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
                                        style={"background-color": "white"},
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
            html.Label(
                f"{buttonname}:", id=id_name,
                style={"cursor": "pointer","font-size": "100%","padding-left":"3px"},
            ),
            dbc.Tooltip(f"{message}", target=id_name, className="tooltiptext"),
        ]
    )


def navbar_layout():
    return dbc.Nav(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.NavbarBrand("CCA", className="ml-2")
                        ),
                    ],
                    align="center", style={"padding-left": "5px", "color": "white"},
                ),
            ),
            Block("Region",
                Select(
                    id="region",
                    options=["SAMER", "SASIA", "Thailand", "US-CA"],
                    labels=[
                        "South America",
                        "South Asia",
                        "Thailand",
                        "United States and Canada",
                    ],
                    init=3,
                ),
            ),
            PickPoint(width="8em"),
            Block("Submit",
                Block("Scenario", Select(
                    id="scenario",
                    options=["picontrol", "ssp126", "ssp370", "ssp585"],
                    init=1,
                )),
                Block("Model", Select(id="model", options=[
                    "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR", "MRI-ESM2-0",
                    "UKESM1-0-LL",
                ])),
                Block("Variable", Select(
                    id="variable",
                    options=[
                        "hurs",
                        "huss",
                        "pr",
                        "prsn",
                        "ps",
                        "rlds",
                        "sfcwind",
                        "tas",
                        "tasmax",
                        "tasmin",
                    ],
                    labels=[
                        "Near-Surface Relative Humidity",
                        "Near-Surface Specific Humidity",
                        "Precipitation",
                        "Snowfall Flux",
                        "Surface Air Pressure",
                        "Surface Downwelling Longwave Radiation",
                        "Near-Surface Wind Speed",
                        "Near-Surface Air Temperature",
                        "Daily Maximum Near-Surface Air Temperature",
                        "Daily Minimum Near-Surface Air Temperature",
                    ],
                    init=2,
                )),
                Block("Season",
                    Month(id="start_month", default="Jan"),
                    "-",
                    Month(id="end_month", default="Mar"),
                ),
                Block("Projected Years",
                    Number(
                        id="start_year",
                        default=2015,
                        min=2015,
                        max=2095,
                        width="5em",
                    ),
                    "-",
                    Number(
                        id="end_year",
                        default=2019,
                        min=2019,
                        max=2099,
                        width="5em",
                    ),
                ),
                Block("Reference Years",
                    Number(
                        id="start_year_ref",
                        default=1981,
                        min=1951,
                        max=1985,
                        width="5em",
                    ),
                    "-",
                    Number(
                        id="end_year_ref",
                        default=2010,
                        min=1980,
                        max=2014,
                        width="5em",
                    ),
                ),
                button_id="submit_controls",
            ),
            dbc.Alert(
                "Something went wrong",
                color="danger",
                dismissable=True,
                is_open=False,
                id="map_warning",
                style={"margin-bottom": "8px"},
            ),
        ],
        style={"background-color": IRI_GRAY},
    )


def controls_layout():
    return dbc.Container(
        [
            html.H5(["Climate Change Analysis"]),
            html.P(
                """
                This Maproom displays seasonal projected change of key climate
                variables with respect to historical records.
                """
            ),
            dcc.Loading(html.P(id="map_description"), type="dot"),
            html.P(
                """
                Use the controls in the top banner to choose other variables, models,
                scenarios, seasons, projected years and reference to compare with.
                """
            ),
            html.P(
                """
                Click the map (or enter coordinates) to show historical seasonal time
                series for this variable of this model, followed by a plume of
                possible projected scenarios.
                """
            ),
            html.P(
                """
                Change is expressed as the difference between average over projected
                years and average over reference historical years (in the variables
                units), except for precipitation and both humidity variables for
                which it is the relative difference (in %).
                """
            ),
        ],
        fluid=True, className="scrollable-panel",
        style={"padding-bottom": "1rem", "padding-top": "1rem"},
    )


def map_layout():
    return dbc.Container(
        [
            dcc.Loading(html.H5(
                id="map_title",
                style={
                    "text-align":"center", "border-width":"1px",
                    "border-style":"solid", "border-color":"grey",
                    "margin-top":"3px", "margin-bottom":"3px",
                },
            ),  type="dot"),
            dcc.Loading(dlf.Map(
                [
                    dlf.LayersControl(id="layers_control", position="topleft"),
                    dlf.LayerGroup(
                        [dlf.Marker(id="loc_marker", position=(0, 0))],
                        id="layers_group"
                    ),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                    dlf.Colorbar(
                        id="colorbar",
                        position="bottomleft",
                        width=300,
                        height=10,
                        min=0,
                        max=1,
                        nTicks=11,
                        opacity=1,
                        tooltip=True,
                        className="p-1",
                        style={
                            "background": "white", "border-style": "inset",
                            "-moz-border-radius": "4px", "border-radius": "4px",
                            "border-color": "LightGrey",
                        },
                    ),
                ],
                id="map",
                center=None,
                zoom=GLOBAL_CONFIG["zoom"],
                style={"width": "100%", "height": "50vh"},
            ), type="dot"),
        ],
        fluid=True,
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
                                    dcc.Graph(id="local_graph"),
                                )
                            ),
                        ]
                    )
                ],
                label="Local History and Projections",
            )
        ],
        className="mt-4",
    )
