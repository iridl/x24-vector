import os
import flask
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pyaconf
import pingrid
import layout
import charts
import calc
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd

CONFIG = pyaconf.load(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

#Reads daily data

CONFIG = pyaconf.load(os.environ["CONFIG"])
DR_PATH = CONFIG["daily_rainfall_path"]
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "Onset Maproom"

APP.layout = layout.app_layout()

@APP.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@APP.callback(
    Output("probability-collapse", "is_open"),
    Output("probExcThresh1", "value"),
    Output("poeunits", "value"),
    Input("yearly_stats_input", "value"),
)
def change_form_layout(value):
    return value == "pe", 30, "/percent"

def get_coords(click_lat_lng):
    if click_lat_lng is not None:
        return click_lat_lng
    else:
        return [layout.INIT_LAT, layout.INIT_LNG]

@APP.callback(
    Output("layers_group", "children"),
    Input("map", "click_lat_lng")
)
def map_click(click_lat_lng):
    lat_lng = get_coords(click_lat_lng)

    return dlf.Marker(
        position=lat_lng,
        children=dlf.Tooltip("({:.3f}, {:.3f})".format(*lat_lng))
    )

@APP.callback(
    Output("plotly_onset_test", "figure"),
    Input("map", "click_lat_lng"),
    Input("earlyStartDay", "value"),
    Input("earlyStartMonth", "value"),
    Input("searchDays", "value"),
    Input("wetThreshold", "value"),
    Input("runningDays", "value"),
    Input("runningTotal", "value"),
    Input("minRainyDays", "value"),
    Input("dryDays", "value"),
    Input("drySpell", "value"),
)
def onset_plot(click_lat_lng, earlyStartDay, earlyStartMonth, searchDays, wetThreshold, runningDays, runningTotal, minRainyDays, dryDays,drySpell):
    lat, lng = get_coords(click_lat_lng)
    ds = rr_mrg.sel(X=lng, Y=lat, method="nearest")
    onset = calc.onset_date(ds.precip, int(earlyStartDay), \
        calc.strftimeb2int(earlyStartMonth), searchDays, \
        wetThreshold, runningDays, runningTotal, minRainyDays, dryDays, drySpell)
    onsetDate = onset["T"] + onset
    year = pd.DatetimeIndex(onsetDate['T']).year
    onsetMD = onsetDate.dt.strftime("1999-%m-%d").to_dataframe(name="Onset Date")
    graph = px.line(data_frame=onsetMD,x=onsetMD.index, y="Onset Date")
    graph.update_traces(mode="markers+lines") # , hovertemplate=None)
    graph.update_layout(
        yaxis=dict(tickformat="%b %d"), 
        xaxis_title="Year", 
        yaxis_title="Onset Date",
        title= f"Starting dates of {int(earlyStartDay)} {earlyStartMonth} season {year.min()}-{year.max()} ([{lat}E - {lng}N])"
    )
    return graph

@APP.callback(
    Output("onset_date_graph", "src"),
    Output("onset_date_exceeding", "src"),
    Output("cess_date_graph", "src"),
    Output("cess_date_exceeding", "src"),
    Output("pdf_link", "href"),
    Output("onset_cess_table", "children"),
    Input("map", "click_lat_lng"),
    Input("earlyStartDay", "value"),
    Input("earlyStartMonth", "value"),
    Input("searchDays", "value"),
    Input("wetThreshold", "value"),
    Input("runningDays", "value"),
    Input("runningTotal", "value"),
    Input("minRainyDays", "value"),
    Input("dryDays", "value"),
    Input("drySpell", "value"),
    Input("earlyCessDay", "value"),
    Input("earlyCessMonth", "value"),
    Input("searchDaysCess", "value"),
    Input("waterBalanceCess", "value"),
    Input("drySpellCess", "value"),
    Input("plotrange1", "value"),
    Input("plotrange2", "value"),
)
def update_charts(click_lat_lng, earlyStartDay, earlyStartMonth, searchDays, wetThreshold, \
                  runningDays, runningTotal, minRainyDays, dryDays, \
                  drySpell, earlyCessDay, earlyCessMonth, searchDaysCess, waterBalanceCess, drySpellCess \
    , plotrange1, plotrange2):
    lat, lng = get_coords(click_lat_lng)
    params = {
        "earlyStart": str(earlyStartDay) + " " + earlyStartMonth,
        "searchDays": searchDays,
        "wetThreshold": wetThreshold,
        "runningDays": runningDays,
        "runningTotal": runningTotal,
        "minRainyDays": minRainyDays,
        "dryDays": dryDays,
        "drySpell": drySpell,
        "earlyCess": earlyCessDay + " " + earlyCessMonth,
        "searchDaysCess": searchDaysCess,
        "waterBalanceCess": waterBalanceCess,
        "drySpellCess": drySpellCess,
        "plotrange1": plotrange1,
        "plotrange2": plotrange2
    }
    
#    od_test = calc.onset_date(rr_mrg.precip, int(earlyStartDay), calc.strftimeb2int(earlyStartMonth), params["searchDays"], params["wetThreshold"], params["runningDays"], params["runningTotal"], params["minRainyDays"], params["dryDays"], params["drySpell"])
#    print(od_test)

    try:
        tab_data = charts.table(lat, lng, params)
        table_header = [
            html.Thead(html.Tr([html.Th("Year"),
                                html.Th("Onset Date"),
                                html.Th("Cessation Date")]))
        ]
        table_body = html.Tbody(
            [ html.Tr([html.Td(r[0]), html.Td(r[1]), html.Td(r[2])]) for r in tab_data ]
        )
        table_elem = table_header + [ table_body ]
    except:
        table_elem = []

    return [
        charts.onset_date(lat, lng, params),
        charts.prob_exceed(lat, lng, params),
        charts.cess_date(lat, lng, params),
        charts.cess_exceed(lat, lng, params),
        charts.pdf(lat, lng, params),
        table_elem
    ]




if __name__ == "__main__":
    APP.run_server(debug=CONFIG["mode"] != "prod")
