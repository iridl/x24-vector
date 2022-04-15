import os
import flask
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pyaconf
import pingrid 
import layout
import calc
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import urllib
import math
from widgets import Sentence, Number

CONFIG = pyaconf.load(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

# Reads daily data

CONFIG = pyaconf.load(os.environ["CONFIG"])
DR_PATH = CONFIG["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.
TOLERANCE = math.sqrt(2 * (RESOLUTION / 2) ** 2)

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
APP.title = CONFIG["app_title"]

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
    Output("map", "center"),
    Output("latInput", "min"),
    Output("latInput", "max"),
    Output("lngInput", "min"),
    Output("lngInput", "max"),
    Output("latLab", "children"),
    Output("lonLab", "children"),
    Input("submitLatLng","n_clicks"),
)
def initialize(toto):
    center_of_the_map = [((rr_mrg.Y[0]+rr_mrg.Y[-1])/2).values, ((rr_mrg.X[0]+rr_mrg.X[-1])/2).values]
    lat_res = np.around((rr_mrg.Y[1]-rr_mrg.Y[0]).values, decimals=10)
    lat_min = str(np.around((rr_mrg.Y[0]-lat_res/2).values, decimals=10))
    lat_max = str(np.around((rr_mrg.Y[-1]+lat_res/2).values, decimals=10))
    lon_res = np.around((rr_mrg.X[1]-rr_mrg.X[0]).values, decimals=10)
    lon_min = str(np.around((rr_mrg.X[0]-lon_res/2).values, decimals=10))
    lon_max = str(np.around((rr_mrg.X[-1]+lon_res/2).values, decimals=10))
    lat_label = lat_min+" to "+lat_max+" by "+str(lat_res)+"˚"
    lon_label = lon_min+" to "+lon_max+" by "+str(lon_res)+"˚"
    return center_of_the_map, lat_min, lat_max, lon_min, lon_max, lat_label, lon_label


@APP.callback(
    Output("pet_style", "style"),
    Input("yearly_stats_input", "value"),
)

def display_pet_control(yearly_stats_input):

    if yearly_stats_input == "pe":
        pet_style={"display": "flex"}
    else:
        pet_style={"display": "none"}
    return pet_style


@APP.callback(
    Output("probExcThresh1", "max"),
    Input("searchDays", "value"),
)

def pet_control_max(searchDays):

    return searchDays


def get_coords(click_lat_lng):
    if click_lat_lng is not None:
        return click_lat_lng
    else:
        return [(rr_mrg.Y[0].values+rr_mrg.Y[-1].values)/2, (rr_mrg.X[0].values+rr_mrg.X[-1].values)/2]


def round_latLng(coord):
    value = float(coord)
    value = round(value, 4)
    return value

@APP.callback(Output("map", "click_lat_lng"), Input("submitLatLng","n_clicks"), State("latInput", "value"), State("lngInput", "value"))
def inputCoords(n_clicks,latitude,longitude):
    if latitude is None:
        return None
    else:
        lat_lng = [latitude, longitude]
        return lat_lng

@APP.callback(Output("layers_group", "children"), Output("latInput","value"), Output("lngInput","value"),Input("map", "click_lat_lng"))
def map_click(click_lat_lng):
    lat_lng = get_coords(click_lat_lng)
    return dlf.Marker(
        position=lat_lng, children=dlf.Tooltip("({:.3f}, {:.3f})".format(*lat_lng))
    ), round(lat_lng[0],4), round(lat_lng[1],4)


@APP.callback(
    Output("onsetDate_plot", "figure"),
    Output("probExceed_onset", "figure"),
    Output("coord_alert_onset", "children"),
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
def onset_plots(
    click_lat_lng,
    search_start_day,
    search_start_month,
    searchDays,
    wetThreshold,
    runningDays,
    runningTotal,
    minRainyDays,
    dryDays,
    drySpell,
):
    lat1, lng1 = get_coords(click_lat_lng)
    try:
        precip = rr_mrg.precip.sel(X=lng1, Y=lat1, method="nearest", tolerance=TOLERANCE)
        isnan = np.isnan(precip).sum().sum()
        if isnan > 0:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="No Data to Display",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            alert1 = dbc.Alert(
                "Onset alert: The dataset at the chosen coordinates is empty (NaN). Please choose a different point.",
                color="danger",
                dismissable=True,
            )
            return errorFig, errorFig, alert1
    except KeyError:
        errorFig = pgo.Figure().add_annotation(
            x=2,
            y=2,
            text="No Data to Display",
            font=dict(family="sans serif", size=30, color="crimson"),
            showarrow=False,
            yshift=10,
            xshift=60,
        )
        alert1 = dbc.Alert(
            "Onset alert: The point you have chosen is not within the bounding box of this dataset. Please choose a different point.",
            color="danger",
            dismissable=True,
        )
        return errorFig, errorFig, alert1
    precip.load()
    try:
        onset_delta = calc.seasonal_onset_date(
            precip,
            int(search_start_day),
            calc.strftimeb2int(search_start_month),
            int(searchDays),
            int(wetThreshold),
            int(runningDays),
            int(runningTotal),
            int(minRainyDays),
            int(dryDays),
            int(drySpell),
            time_coord="T",
        )
    except TypeError:
        errorFig = pgo.Figure().add_annotation(
            x=2,
            y=2,
            text="No Data to Display",
            font=dict(family="sans serif", size=30, color="crimson"),
            showarrow=False,
            yshift=10,
            xshift=60,
        )
        alert1 = dbc.Alert(
            "Onset alert: Please ensure all input boxes are filled for the calculation to run.",
            color="danger",
            dismissable=True,
        )
        return (
            errorFig,
            errorFig,
            alert1,
        )  # dash.no_update to leave the plat as-is and not show no data display
    onsetDate = onset_delta["T"] + onset_delta["onset_delta"]
    onsetDate = pd.DataFrame(onsetDate.values, columns=["onset"])
    year = pd.DatetimeIndex(onsetDate["onset"]).year
    onsetMD = (
        onsetDate["onset"]
        .dt.strftime("2000-%m-%d")
        .astype("datetime64[ns]")
        .to_frame(name="onset")
    )
    onsetMD["Year"] = year
    earlyStart = pd.to_datetime(
        f"2000-{search_start_month}-{search_start_day}", yearfirst=True
    )
    try:
        cumsum = calc.probExceed(onsetMD, earlyStart)
    except IndexError:
        errorFig = pgo.Figure().add_annotation(
            x=2,
            y=2,
            text="No Data to Display",
            font=dict(family="sans serif", size=30, color="crimson"),
            showarrow=False,
            yshift=10,
            xshift=60,
        )
        alert1 = dbc.Alert(
            "Onset alert: The calculations for these coordinates have returned an empty dataset. Please choose a different point.",
            color="danger",
            dismissable=True,
        )
        return errorFig, errorFig, alert1
    onsetDate_graph = px.line(
        data_frame=onsetMD,
        x="Year",
        y="onset",
    )
    onsetDate_graph.update_traces(
        mode="markers+lines", hovertemplate="%{y} %{x}", connectgaps=False
    )
    onsetDate_graph.update_layout(
        yaxis=dict(tickformat="%b %d"),
        xaxis_title="Year",
        yaxis_title="Onset Date",
        title=f"Onset dates found after {search_start_month} {int(search_start_day)}, at ({round_latLng(lat1)}N,{round_latLng(lng1)}E)",
    )
    probExceed_onset = px.line(
        data_frame=cumsum,
        x="Days",
        y="probExceed",
    )
    probExceed_onset.update_traces(
        mode="markers+lines",
        hovertemplate="Days since Early Start Date: %{x}" + "<br>Probability: %{y:.0%}",
    )
    probExceed_onset.update_layout(
        yaxis=dict(tickformat=".0%"),
        yaxis_title="Probability of Exceeding",
        xaxis_title=f"Onset Date [days since {search_start_day} {search_start_month}]",
    )
    return onsetDate_graph, probExceed_onset, None

@APP.callback(
    Output("cessDate_plot", "figure"),
    Output("probExceed_cess", "figure"),
    Output("cess_dbct","tab_style"),
    Output("coord_alert_cess", "children"),
    Input("map", "click_lat_lng"),
    Input("start_cess_day", "value"),
    Input("start_cess_month", "value"),
    Input("searchDaysCess", "value"),
    Input("waterBalanceCess", "value"),
    Input("drySpellCess", "value"),
)
def cess_plots(
    click_lat_lng,
    start_cess_day,
    start_cess_month,
    searchDaysCess,
    waterBalanceCess,
    drySpellCess,
):
    if not CONFIG["ison_cess_date_hist"]:
        tab_style = {"display": "none"}
        return {}, {}, tab_style, None
    else:
        tab_style = {}
        lat, lng = get_coords(click_lat_lng)
        try:
            precip = rr_mrg.precip.sel(X=lng, Y=lat, method="nearest", tolerance=TOLERANCE)
            isnan = np.isnan(precip).sum().sum()
            if isnan > 0:
                errorFig = pgo.Figure().add_annotation(
                    x=2,
                    y=2,
                    text="No Data to Display",
                    font=dict(family="sans serif", size=30, color="crimson"),
                    showarrow=False,
                    yshift=10,
                    xshift=60,
                )
                alert1 = dbc.Alert(
                    "Cess alert: The dataset at the chosen coordinates is empty (NaN). Please choose a different point.",
                    color="danger",
                    dismissable=True,
                )
                return errorFig, errorFig, tab_style, alert1
        except KeyError:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="No Data to Display",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            alert1 = dbc.Alert(
                "Cess alert: The point you have chosen is not within the bounding box of this dataset. Please choose a different point.",
                color="danger",
                dismissable=True,
            )
            return errorFig, errorFig, tab_style, alert1
        precip.load()
        try:
            soil_moisture = calc.water_balance(precip, 5,60,0,time_coord="T").to_array(name="soil moisture") #convert to array to use xr.array functionality in calc.py
            cess_delta = calc.seasonal_cess_date(
                soil_moisture,
                int(start_cess_day),
                calc.strftimeb2int(start_cess_month),
                int(searchDaysCess),
                int(waterBalanceCess),
                int(drySpellCess),
                time_coord="T",
            )
        except TypeError:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="No Data to Display",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            alert1 = dbc.Alert(
                "Cess alert: Please ensure all input boxes are filled for the calculation to run.",
                color="danger",
                dismissable=True,
            )
            return (
                errorFig, errorFig, tab_style,
                alert1,
            )  # dash.no_update to leave the plat as-is and not show no data display
        cessDate = cess_delta["T"] + cess_delta["cess_delta"]
        cessDate = pd.DataFrame(cessDate.values, columns=["precip"])  #'cess' ?
        year = pd.DatetimeIndex(cessDate["precip"]).year
        cessMD = (
            cessDate["precip"]
            .dt.strftime("2000-%m-%d")
            .astype("datetime64[ns]")
            .to_frame(name="cessation")
        )
        cessMD["Year"] = year
        earlyStart = pd.to_datetime(
            f"2000-{start_cess_month}-{start_cess_day}", yearfirst=True
        )
        try:
            cumsum = calc.probExceed(cessMD, earlyStart)
        except IndexError:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="No Data to Display",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            alert1 = dbc.Alert(
                "Cess alert: The calculations for these coordinates have returned an empty dataset. Please choose a different point.",
                color="danger",
                dismissable=True,
            )
            return errorFig, errorFig, tab_style, alert1
        cessDate_graph = px.line(
            data_frame=cessMD,
            x="Year",
            y="cessation",
        )
        cessDate_graph.update_traces(
            mode="markers+lines", hovertemplate="%{y} %{x}", connectgaps=False
        )
        cessDate_graph.update_layout(
            yaxis=dict(tickformat="%b %d"),
            xaxis_title="Year",
            yaxis_title="Cessation Date",
            title=f"Starting dates of {int(start_cess_day)} {start_cess_month} season {year.min()}-{year.max()} ({round_latLng(lat)}N,{round_latLng(lng)}E)",
        )
        probExceed_cess = px.line(
            data_frame=cumsum,
            x="Days",
            y="probExceed",
        )
        probExceed_cess.update_traces(
            mode="markers+lines",
            hovertemplate="Days since Early Start Date: %{x}" + "<br>Probability: %{y:.0%}",
        )
        probExceed_cess.update_layout(
            yaxis=dict(tickformat=".0%"),
            yaxis_title="Probability of Exceeding",
            xaxis_title=f"Cessation Date [days since {start_cess_day} {start_cess_month}]",
        )
        return cessDate_graph, probExceed_cess, tab_style, None


# TODO can we make this any more redundant?
@APP.callback(
    Output("onset_layer", "url"),
    Input("yearly_stats_input", "value"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("wetThreshold", "value"),
    Input("runningDays", "value"),
    Input("runningTotal", "value"),
    Input("minRainyDays", "value"),
    Input("dryDays", "value"),
    Input("drySpell", "value"),
    Input("probExcThresh1", "value")
)
def onset_tile_url(
        yearly_stats_input,
        search_start_day,
        search_start_month,
        search_days,
        wet_thresh,
        wet_spell_length,
        wet_spell_thresh,
        min_wet_days,
        dry_spell_length,
        dry_spell_search,
        probExcThresh1
):
    qstr = urllib.parse.urlencode({
        "yearly_stats_input": yearly_stats_input,
        "search_start_day": search_start_day,
        "search_start_month": search_start_month,
        "search_days": search_days,
        "wet_thresh": wet_thresh,
        "wet_spell_length": wet_spell_length,
        "wet_spell_thresh": wet_spell_thresh,
        "min_wet_days": min_wet_days,
        "dry_spell_length": dry_spell_length,
        "dry_spell_search": dry_spell_search,
        "probExcThresh1": probExcThresh1
    })
    return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}"


@SERVER.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def onset_tile(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    yearly_stats_input = parse_arg("yearly_stats_input")
    search_start_day = parse_arg("search_start_day", int)
    search_start_month1 = parse_arg("search_start_month", calc.strftimeb2int)
    search_days = parse_arg("search_days", int)
    wet_thresh = parse_arg("wet_thresh", float)
    wet_spell_length = parse_arg("wet_spell_length", int)
    wet_spell_thresh = parse_arg("wet_spell_thresh", float)
    min_wet_days = parse_arg("min_wet_days", int)
    dry_spell_length = parse_arg("dry_spell_length", int)
    dry_spell_search = parse_arg("dry_spell_search", int)
    probExcThresh1 = parse_arg("probExcThresh1", int)

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)
    # row numbers increase as latitude decreases
    y_max = pingrid.tile_bottom_mercator(ty, tz)
    y_min = pingrid.tile_bottom_mercator(ty + 1, tz)

    precip = rr_mrg.precip

    if (
            # When we generalize this to other datasets, remember to
            # account for the possibility that longitudes wrap around,
            # so a < b doesn't always mean that a is west of b.
            x_min > precip['X'].max() or
            x_max < precip['X'].min() or
            y_min > precip['Y'].max() or
            y_max < precip['Y'].min()
    ):
        return pingrid.empty_tile()

    if yearly_stats_input == "monit":
        precip_tile = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = precip_tile["T"].where(
            lambda x: (x.dt.day == search_start_day) & (x.dt.month == search_start_month1),
            drop=True,
        )
        precip_tile = precip_tile.sel({"T": slice(search_start_dm.values[0], None)})
    else:
        precip_tile = rr_mrg.precip

    precip_tile = precip_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()

    mymap_min = np.timedelta64(0)
    mymap_max = np.timedelta64(search_days, 'D')
    mycolormap = pingrid.RAINBOW_COLORMAP

    if yearly_stats_input == "monit":
        mymap = calc.onset_date(
            precip_tile,
            wet_thresh,
            wet_spell_length,
            wet_spell_thresh,
            min_wet_days,
            dry_spell_search+1,
            dry_spell_search
        )
    else:
        onset_dates = calc.seasonal_onset_date(
            precip_tile,
            search_start_day,
            search_start_month1,
            search_days,
            wet_thresh,
            wet_spell_length,
            wet_spell_thresh,
            min_wet_days,
            dry_spell_length,
            dry_spell_search,
        )
        if yearly_stats_input == "mean":
            mymap = onset_dates.onset_delta.mean("T")
        if yearly_stats_input == "stddev":
            mymap = onset_dates.onset_delta.std("T", skipna=True)
        if yearly_stats_input == "pe":
            mymap = (
                onset_dates.onset_delta.fillna(
                    np.timedelta64(search_days+1, 'D')
                ) > np.timedelta64(probExcThresh1, 'D')
            ).mean("T") * 100
            mymap_min = 0
            mymap_max = 100
            mycolormap = pingrid.CORRELATION_COLORMAP
            mymap.attrs["colormapkey"] = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
    mymap.attrs["colormap"] = mycolormap
    mymap = mymap.rename(X="lon", Y="lat")
    mymap.attrs["scale_min"] = mymap_min
    mymap.attrs["scale_max"] = mymap_max
    result = pingrid.tile(mymap, tx, ty, tz)

    return result


@APP.callback(
    Output("colorbar", "children"),
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("yearly_stats_input", "value")
)
def set_colorbar(search_start_day, search_start_month, search_days, yearly_stats_input):
    if yearly_stats_input == "pe":
        thresholds = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
        return (
            f"Probabily of onset date to be {search_days} past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.CORRELATION_COLORMAP, thresholds=thresholds),
            int(100),
            [i for i in range(0, int(100) + 1) if i % 10 == 0],
        )
    else:
        return (
            f"Onset date in days past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP),
            int(search_days),
            [i for i in range(0, int(search_days) + 1) if i % 10 == 0],
        )


if __name__ == "__main__":
    APP.run_server(
        host=CONFIG["listen_address"],
        port=CONFIG["listen_port"],
        debug=CONFIG["mode"] != "prod",
        processes=CONFIG["dev_processes"],
        threaded=False,
    )
