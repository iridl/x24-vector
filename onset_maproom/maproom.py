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
#import datetime as dt

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

def round_latLng(coord):
    value = float(coord)
    value = round(value,4)
    return value

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
    Output("probExceed_graph", "figure"),
    Output("coord_alert", "children"),
    Input("map", "click_lat_lng"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("wetThreshold", "value"),
    Input("runningDays", "value"),
    Input("runningTotal", "value"),
    Input("minRainyDays", "value"),
    Input("dryDays", "value"),
    Input("drySpell", "value"),
)
def onset_plots(click_lat_lng, search_start_day, search_start_month, searchDays, wetThreshold,
                runningDays, runningTotal, minRainyDays, dryDays,drySpell):
    lat, lng = get_coords(click_lat_lng)
    try:
        precip = rr_mrg.precip.sel(X=lng, Y=lat, method="nearest", tolerance=0.04)
    except KeyError:
        errorFig = pgo.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=30,color="crimson"),showarrow=False, yshift=10, xshift=60)
        alert1 = dbc.Alert("The point you have chosen is not within the bounding box of this dataset. Please choose a different point.", color="danger", dismissable=True)
        return errorFig, errorFig, alert1
    precip.load()
    try:
        onset_delta = calc.seasonal_onset_date(precip, int(search_start_day),
            calc.strftimeb2int(search_start_month), int(searchDays),
            int(wetThreshold), int(runningDays), int(runningTotal),
            int(minRainyDays), int(dryDays), int(drySpell), time_coord="T")
    except TypeError:
        errorFig = pgo.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=30,color="crimson"),showarrow=False, yshift=10, xshift=60)
        alert1 = dbc.Alert("Please ensure all input boxes are filled for the calculation to run.", color="danger", dismissable=True)
        return errorFig, errorFig, alert1 #dash.no_update to leave the plat as-is and not show no data display    
    onsetDate = (onset_delta["T"] + onset_delta["onset_delta"])
    onsetDate = pd.DataFrame(onsetDate.values, columns = ['onset'])
    year = pd.DatetimeIndex(onsetDate["onset"]).year
    onsetMD = onsetDate["onset"].dt.strftime("2000-%m-%d").astype('datetime64[ns]').to_frame(name="onset")
    onsetMD['Year'] = year
    earlyStart = pd.to_datetime(f'2000-{search_start_month}-{search_start_day}', yearfirst=True)
    try:
        cumsum = calc.probExceed(onsetMD, earlyStart)
    except IndexError:
        errorFig = pgo.Figure().add_annotation(x=2, y=2,text="No Data to Display",font=dict(family="sans serif",size=30,color="crimson"),showarrow=False,yshift=10, xshift=60)
        alert1 = dbc.Alert("The dataset at the chosen coordinates is empty (NaN). Please choose a different point.", color="danger", dismissable=True)
        return errorFig, errorFig, alert1
    onsetDate_graph = px.line(
        data_frame=onsetMD,
        x="Year", 
        y="onset",
    )
    onsetDate_graph.update_traces(
        mode="markers+lines",
        hovertemplate='%{y} %{x}',
        connectgaps=False
    )
    onsetDate_graph.update_layout(
        yaxis=dict(tickformat="%b %d"), 
        xaxis_title="Year", 
        yaxis_title="Onset Date",
        title= f"Starting dates of {int(search_start_day)} {search_start_month} season {year.min()}-{year.max()} ({round_latLng(lat)}N,{round_latLng(lng)}E)"
    )
    probExceed_graph = px.line(
        data_frame=cumsum,
        x="Days",
        y="probExceed",
    )
    probExceed_graph.update_traces(
        mode="markers+lines",
        hovertemplate= 'Days since Early Start Date: %{x}'+'<br>Probability: %{y:.0%}'
    )
    probExceed_graph.update_layout(
        yaxis=dict(tickformat=".0%"),
        yaxis_title="Probability of Exceeding",
        xaxis_title=f"Onset Date [days since {search_start_day} {search_start_month}]"
    )
    return onsetDate_graph, probExceed_graph, None
    
@APP.callback(
    Output("onset_date_graph", "src"),
    Output("onset_date_exceeding", "src"),
    Output("cess_date_graph", "src"),
    Output("cess_date_exceeding", "src"),
    Output("pdf_link", "href"),
    Output("onset_cess_table", "children"),
    Input("map", "click_lat_lng"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("wetThreshold", "value"),
    Input("runningDays", "value"),
    Input("runningTotal", "value"),
    Input("minRainyDays", "value"),
    Input("dryDays", "value"),
    Input("drySpell", "value"),
    Input("start_cess_day", "value"),
    Input("start_cess_month", "value"),
    Input("searchDaysCess", "value"),
    Input("waterBalanceCess", "value"),
    Input("drySpellCess", "value"),
    Input("plotrange1", "value"),
    Input("plotrange2", "value"),
)
def update_charts(click_lat_lng, search_start_day, search_start_month, searchDays, wetThreshold,
                  runningDays, runningTotal, minRainyDays, dryDays,
                  drySpell, start_cess_day, start_cess_month, searchDaysCess, waterBalanceCess, drySpellCess,
                  plotrange1, plotrange2):
    lat, lng = get_coords(click_lat_lng)
    params = {
        "earlyStart": str(search_start_day) + " " + search_start_month,
        "searchDays": searchDays,
        "wetThreshold": wetThreshold,
        "runningDays": runningDays,
        "runningTotal": runningTotal,
        "minRainyDays": minRainyDays,
        "dryDays": dryDays,
        "drySpell": drySpell,
        "earlyCess": start_cess_day + " " + start_cess_month,
        "searchDaysCess": searchDaysCess,
        "waterBalanceCess": waterBalanceCess,
        "drySpellCess": drySpellCess,
        "plotrange1": plotrange1,
        "plotrange2": plotrange2
    }
    
#    od_test = calc.onset_date(rr_mrg.precip, int(search_start_day), calc.strftimeb2int(search_start_month), params["searchDays"], params["wetThreshold"], params["runningDays"], params["runningTotal"], params["minRainyDays"], params["dryDays"], params["drySpell"])
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
