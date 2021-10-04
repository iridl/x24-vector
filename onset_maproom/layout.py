import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table
import dash_leaflet as dlf
import plotly.express as px
from widgets import Block, Sentence, Date, Units, Number

IRI_BLUE = "rgb(25,57,138)"
IRI_GRAY = "rgb(113,112,116)"
LIGHT_GRAY = "#eeeeee"
INIT_LAT = 9.03
INIT_LNG = 38.74


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
                                    html.Div(id="coord_alert", children=[]),
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
                                src="assets/Ethiopia_IRI_98x48.png",
                                height="30px",
                            )
                        ),
                        dbc.Col(
                            dbc.NavbarBrand(
                                "Climate and Agriculture / Onset Maproom",
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
                    "Historical Onset and Cessation Date ",
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

            Block("Date",
                  dbc.Select(
                      id="date_input",
                      value="onset",
                      bs_size="sm",
                      options=[
                          {"label": "Onset", "value": "onset"},
                          {"label": "Cessation", "value": "cessation"},
                      ],
                  ),
            ),

            Block("Yearly Statistics",
                  dbc.Select(
                      id="yearly_stats_input",
                      value="mean",
                      bs_size="sm",
                      options=[
                          {"label": "Mean", "value": "mean"},
                          {"label": "Standard deviation", "value": "stddev"},
                          {"label": "Probability of exceedance", "value": "pe"},
                      ],
                  ),
                  dbc.Collapse(
                      Sentence(
                          Number("probExcThresh1", 30, min=0, max=999),
                          "days since Early Start, as",
                          Units("poeunits"),
                      ),
                      id="probability-collapse",
                  )
            ),

            Block("Search Period",
                  Sentence(
                      "From Early Start date of",
                      Date("search_start_", 1, "Jun"),
                      "and within the next",
                      Number("searchDays", 90, min=0, max=9999),
                      "days"
                  )
            ),

            Block("Wet Day Definition",
                  Sentence(
                      "Rainfall amount greater than",
                      Number("wetThreshold", 1, min=0, max=99999),
                      "mm",
                  )
            ),

            Block("Onset Date Definition",
                  Sentence(
                      "First window of",
                      Number("runningDays", 5, min=0, max=999),
                      "days that totals",
                      Number("runningTotal", 20, min=0, max=99999),
                      "mm or more and with at least",
                      Number("minRainyDays", 3, min=0, max=999),
                      "wet days and that is not followed by a",
                      Number("dryDays", 7, min=0, max=999),
                      "day dry spell within the next",
                      Number("drySpell", 21, min=0, max=9999),
                      "days",
                  )
            ),

            Block("Cessation Date Definition",
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
                  )
            ),

            Block("Local Plots Range",
                  Sentence(
                      Number("plotrange1", 0, min=0, max=9999),
                      "to",
                      Number("plotrange2", 60, min=0, max=9999),
                      "days"
                  )
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
                is ahead of the expected onset date. Generally, two
                rainy seasons are expected in the region: Belg
                (Feb-May) and Kiremt (Jun-Sep).
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
                Reconstructed rainfall on a 0.0375˚ x 0.0375˚ lat/lon
                grid (about 4km) from National Meteorology Agency. The
                time series (1981 to 2017) were created by combining
                quality-controlled station observations in NMA’s
                archive with satellite rainfall estimates.
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
                        ],
                        position="topleft",
                        id="layers_control",
                    ),
                    dlf.LayerGroup(id="layers_group"),
                    dlf.ScaleControl(imperial=False, position="bottomleft"),
                ],
                id="map",
                center=[INIT_LAT, INIT_LNG],
                zoom=7,
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
                    dbc.Spinner(html.Img(
                        style={"width": "600px"},
                        id="onset_date_graph",
                        src="",
                    )),
                    dbc.Spinner(html.Img(
                        style={"width": "600px"},
                        id="onset_date_exceeding",
                        src=""
                    )),
                ],
                label="Onset Dates"
            ),

            dbc.Tab(
                [
                    dbc.Spinner(dcc.Graph(
                        id="plotly_onset_test",
                    )),
                    dbc.Spinner(dcc.Graph(
                        id="probExceed_graph",
                    ))	
                ],
                label="New onset alg"
            ),

            dbc.Tab(
                [
                    dbc.Spinner(html.Img(
                        style={"width": "600px"},
                        id="cess_date_graph",
                        src="",
                    )),
                    dbc.Spinner(html.Img(
                        style={"width": "600px"},
                        id="cess_date_exceeding",
                        src=""
                    ))
                ],
                label="Cessation Dates"
            ),

            dbc.Tab(
                dbc.Spinner(
                    dbc.Table([], id="onset_cess_table", bordered=True, className="m-2")),
                label="Table"
            ),

            dbc.Tab(
                html.A("PDF Batch Report", id="pdf_link", href="#", target='_blank'),
                label="PDF"
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
