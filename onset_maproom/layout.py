import os
import pyaconf
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
import dash_leaflet as dlf
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number

CONFIG = pyaconf.load(os.environ["CONFIG"])

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
                                className="g-0",
                            ),
                        ],
                        sm=12,
                        md=8,
                        style={"background-color": "white"},
                    ),
                ],
                className="g-0",
            ),
            html.Div(
                id="coord_alert_onset",
                style={
                    "position": "fixed",
                    "bottom": "0",
                    "width": "60%",
                    "right": "20px",
                },
                children=[],
            ),
            html.Div(
                id="coord_alert_cess",
                style={
                    "position": "fixed",
                    "bottom": "0",
                    "width": "60%",
                    "right": "20px",
                },
                children=[],
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
                                src="assets/" + CONFIG["logo"],
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / " + CONFIG["onset_and_cessation_title"],
                                className="ml-2",
                            )
                        ),
                    ],
                    align="center",
                    className="g-0",
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
                                size="sm",
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
                    className="ml-auto flex-nowrap mt-3 mt-md-0 g-0",
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
                    CONFIG["onset_and_cessation_title"],
                ]
            ),
            html.P(
                """
                The Maproom explores historical rainy season onset and
                cessation dates based on user-defined definitions. The
                date when the rainy season starts is critical to
                agriculture planification, in particular for planting.
                """
            ),
            html.P(
                """
                By enabling the exploration of the history of onset
                dates, the Maproom allows to understand the spatial
                and temporal variability of this phenomenon and
                therefore characterize the risk for a successful
                agricultural campaign associated with it.
                """
            ),
            Block("Pick a point",
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.FormFloating([dbc.Input(
                                id = "latInput",
                                type="number",
                            ),
                            dbc.Label(id="latLab", style={"font-size": "80%"})]),
                        ),
                        dbc.Col(
                            dbc.FormFloating([dbc.Input(
                                id = "lngInput",
                                type="number",
                            ),
                            dbc.Label(id="lonLab", style={"font-size": "80%"})]),
                        ),
                        dbc.Button(id="submitLatLng", n_clicks=0, children='Submit'),
                    ],
                ),
            ),
            #Block(
            #    "Date",
            #    dbc.Select(
            #        id="date_input",
            #        value="onset",
            #        bs_size="sm",
            #        options=[
            #            {"label": "Onset", "value": "onset"},
            #            {"label": "Cessation", "value": "cessation"},
            #        ],
            #    ),
            #),
            Block(
                "Map shows",
                dbc.Select(
                    id="yearly_stats_input",
                    value="monit",
                    # bs_size="sm",
                    options=[
                        {"label": "Has germinating rain occured?", "value": "monit"},
                        {"label": "Climatological Mean", "value": "mean"},
                        #as of now, xr.std doesn't know how to deal with NaT
                        #{"label": "Climatological Standard deviation", "value": "stddev"},
                        {"label": "Climatological Probability of exceedance", "value": "pe"},
                    ],
                ),
                html.P(
                    Sentence(
                        Number("probExcThresh1", 30, min=0),
                        "days since Early Start"
                    ),
                    id="pet_style"
                )
            ),
            Block(
                "Onset Date Search Period",
                Sentence(
                    "From Early Start date of",
                    Date("search_start_", 1, CONFIG["default_search_month"]),
                    "and within the next",
                    Number("searchDays", 90, min=0, max=9999), "days",
                ),
            ),
            Block(
                "Wet Day Definition",
                Sentence(
                    "Rainfall amount greater than",
                    Number("wetThreshold", 1, min=0, max=99999),
                    "mm",
                ),
            ),
            Block(
                "Onset Date Definition",
                Sentence(
                    "First window of",
                    Number("runningDays", CONFIG["default_running_days"], min=0, max=999),
                    "days that totals",
                    Number("runningTotal", 20, min=0, max=99999),
                    "mm or more and with at least",
                    Number("minRainyDays", CONFIG["default_min_rainy_days"], min=0, max=999),
                    "wet days and that is not followed by a",
                    Number("dryDays", 7, min=0, max=999),
                    "day dry spell within the next",
                    Number("drySpell", 21, min=0, max=9999),
                    "days",
                ),
            ),
            Block(
                "Cessation Date Definition",
                Sentence(
                    "First date after",
                    Date("start_cess_", 1, "Sep"),
                    "in",
                    Number("searchDaysCess", 90, min=0, max=99999),
                    "days when the soil water balance falls below",
                    Number("waterBalanceCess", 5, min=0, max=999),
                    "mm for a period of",
                    Number("drySpellCess", 3, min=0, max=999),
                    "days",
                ),
                ison=CONFIG["ison_cess_date_hist"]
            ),
            html.P(
                """
                The definition of the onset can be set up in the
                Control Bar at the top and is looking at a
                significantly wet event (e.g. 20mm in 3 days) that is
                not followed by a dry spell (e.g. 7-day dry spell in
                the following 21 days). The actual date is the first
                wet day of the wet event. The onset date is computed
                on-the-fly for each year according to the definition,
                and is expressed in days since an early start date
                (e.g. Jun. 1st). The search for the onset date is made
                from that early start date and for a certain number of
                following days (e.g. 60 days). The early start date
                serves as a reference and should be picked so that it
                is ahead of the expected onset date.
                """
            ),
            html.P(
                """
                Then the map shows yearly statistics of the onset
                date: the mean (by default), standard deviation or
                probability of exceeding a chosen number of
                days. Clicking on the map will then produce a local
                yearly time series of onset dates, as well as a table
                with the actual date (as opposed to days since early
                start); and a probability of exceeding graph.
                """
            ),
            html.P(
                """
                Note that if the criteria to define the onset date are
                not met within the search period, the analysis will
                return a missing value. And if the analysis returns 0
                (days since the early start), it is likely that the
                early start date picked is within the rainy season.
                """
            ),
            html.H5("Dataset Documentation"),
            html.P(
                """
                Reconstructed gridded rainfall from
                """+CONFIG["institution"]+""".
                The time series were created by combining
                quality-controlled station observations in 
                """+CONFIG["institution"]+"""’s
                archive with satellite rainfall estimates.
                """
            ),
        ],
        fluid=True,
        className="scrollable-panel p-3",
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
                                dlf.TileLayer(
                                    opacity=.8,
                                    id="onset_layer",
                                ),
                                name="Onset",
                                checked=True,
                            ),
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.LayerGroup(id="layers_group"),
                    dlf.ScaleControl(imperial=False, position="topright"),
                    dlf.Colorbar(
                        id="colorbar",
                        min=0,
                        position="bottomleft",
                        width=300,
                        height=10,
                        opacity=.8,
                    ),
                ],
                id="map",
                zoom=CONFIG["zoom"],
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
                    dbc.Spinner(dcc.Graph(id="onsetDate_plot")),
                    dbc.Spinner(dcc.Graph(id="probExceed_onset")),
                ],
                label="Onset Date",
            ),
            dbc.Tab(
                [
                    dbc.Spinner(dcc.Graph(id="cessDate_plot")),
                    dbc.Spinner(dcc.Graph(id="probExceed_cess")),
                ],
                id="cess_dbct",
                label="Cessation Date",
            ),
        ],
        className="mt-4",
    )

    # return html.Img(
    #     style={"width": "600px"},
    #     src=(
    #         "https://iridl.ldeo.columbia.edu/dlcharts/render/"
    #         "905cdac6e87a58586967e115a18e615d01530ddd?_wd=1200px&_ht=600px"
    #         "&_langs=en&_mimetype=image%2Fpng"
    #         "&region=bb%3A39.375%3A7.125%3A39.625%3A7.375%3Abb"
    #         "&waterBalanceCess=3&drySpellCess=10&plotrange2=15"
    #     ),
    # )
