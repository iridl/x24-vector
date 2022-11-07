import os
import flask
import dash
from dash import dcc
from dash import html
from dash import ALL
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pingrid 
from . import layout
from . import calc
import plotly.graph_objects as pgo
import plotly.express as px
import pandas as pd
import numpy as np
import urllib
import math
from .widgets import Sentence, Number

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon

CONFIG = pingrid.load_config(os.environ["ONSET_CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = "/tile"

with psycopg2.connect(**CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(CONFIG['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data

DR_PATH = CONFIG["rr_mrg_zarr_path"]
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.12.1/css/all.css",
    ],
    requests_pathname_prefix=f"/python_maproom{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = CONFIG["app_title"]

APP.layout = layout.app_layout()


def adm_borders(shapes):
    with psycopg2.connect(**CONFIG["db"]) as conn:
        s = sql.Composed(
            [
                sql.SQL("with g as ("),
                sql.SQL(shapes),
                sql.SQL(
                    """
                    )
                    select
                        g.label, g.key, g.the_geom
                    from g
                    """
                ),
            ]
        )
        df = pd.read_sql(s, conn)

    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    df["the_geom"] = df["the_geom"].apply(
        lambda x: x if isinstance(x, MultiPolygon) else MultiPolygon([x])
    )
    shapes = df["the_geom"].apply(shapely.geometry.mapping)
    for i in df.index: #this adds the district layer as a label in the dict
        shapes[i]['label'] = df['label'][i]
    return {"features": shapes}


def make_adm_overlay(adm_name, adm_sql, adm_color, adm_lev, adm_weight, is_checked=False):
    border_id = {"type": "borders_adm", "index": adm_lev}
    return dlf.Overlay(
        dlf.GeoJSON(
            id=border_id,
            data=adm_borders(adm_sql),
            options={
                "fill": True,
                "color": adm_color,
                "weight": adm_weight,
                "fillOpacity": 0
            },
        ),
        name=adm_name,
        checked=is_checked,
    )


@APP.callback(
    Output("layers_control", "children"),
    Input("map_choice", "value"),
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
def make_map(
        map_choice,
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
        "map_choice": map_choice,
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
    return [
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
    ] + [
        make_adm_overlay(
            adm["name"],
            adm["sql"],
            adm["color"],
            i+1,
            len(CONFIG["shapes_adm"])-i,
            is_checked=adm["is_checked"]
        )
        for i, adm in enumerate(CONFIG["shapes_adm"])
    ] + [
        dlf.Overlay(
            dlf.TileLayer(
                url=f"/python_maproom/onset{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
                opacity=1,
            ),
            name="Onset",
            checked=True,
        ),
    ]


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
    Output("pet_input_wrapper", "style"),
    Input("map_choice", "value"),
)
def display_pet_control(map_choice):

    if map_choice == "pe":
        pet_input_wrapper={"display": "flex"}
    else:
        pet_input_wrapper={"display": "none"}
    return pet_input_wrapper


@APP.callback(
    Output("probExcThresh1", "max"),
    Input("searchDays", "value"),
)
def pet_control_max(searchDays):

    return searchDays


@APP.callback(
    Output("pet_units", "children"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
)
def write_pet_units(search_start_day, search_start_month):

    return "days after " + search_start_month + " " + search_start_day


def round_latLng(coord):
    value = float(coord)
    value = round(value, 4)
    return value


@APP.callback(
    Output("hover_feature_label", "children"),
    Input({"type": "borders_adm", "index": ALL}, "hover_feature")
)
def write_hover_adm_label(adm_loc):
    location_description = "the map will return location name"
    for i, adm in enumerate(adm_loc):
        if adm is not None:
            location_description = adm['geometry']['label']
    return f'Mousing over {location_description}'


@APP.callback(
    Output("map_title", "children"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("map_choice", "value"),
    Input("probExcThresh1", "value")
)
def write_map_title(search_start_day, search_start_month, map_choice, probExcThresh1):

    if map_choice == "monit":
        search_start_month1 = calc.strftimeb2int(search_start_month)
        first_day = calc.sel_day_and_month(
            rr_mrg.precip["T"][-366:-1], int(search_start_day), search_start_month1
        )[0].dt.strftime("%Y-%m-%d").values
        last_day = rr_mrg.precip["T"][-1].dt.strftime("%Y-%m-%d").values
        mytitle = (
            "Onset date found between " + first_day + " and " + last_day
            + " in days since " + first_day
        )
    if map_choice == "mean":
        mytitle = (
            "Climatological Onset date in days since " 
            + search_start_month + " " + search_start_day
        )
    if map_choice == "stddev":
        mytitle = (
            "Climatological Onset date standard deviation in days since "
            + search_start_month + " " + search_start_day
        )
    if map_choice == "pe":
        mytitle = (
            "Climatological probability that Onset date is " + probExcThresh1  + " days past "
            + search_start_month + " " + search_start_day
        )
    return mytitle


@APP.callback(
    Output("map_description", "children"),
    Input("map_choice", "value"),
)
def write_map_description(map_choice):
    return CONFIG["map_text"][map_choice]["description"]    


@APP.callback(
    Output("loc_marker", "position"),
    Output("lat_input", "value"),
    Output("lng_input", "value"),
    Input("submit_lat_lng","n_clicks"),
    Input("map", "click_lat_lng"),
    State("lat_input", "value"),
    State("lng_input", "value")
)
def pick_location(n_clicks, click_lat_lng, latitude, longitude):
    if dash.ctx.triggered_id == None:
        lat = rr_mrg.precip["Y"][int(rr_mrg.precip["Y"].size/2)].values
        lng = rr_mrg.precip["X"][int(rr_mrg.precip["X"].size/2)].values
    else:
        if dash.ctx.triggered_id == "map":
            lat = click_lat_lng[0]
            lng = click_lat_lng[1]
        else:
            lat = latitude
            lng = longitude
        try:
            nearest_grid = pingrid.sel_snap(rr_mrg.precip, lat, lng)
            lat = nearest_grid["Y"].values
            lng = nearest_grid["X"].values
        except KeyError:
            lat = lat
            lng = lng
    return [lat, lng], lat, lng


@APP.callback(
    Output("onsetDate_plot", "figure"),
    Output("probExceed_onset", "figure"),
    Output("germination_sentence", "children"),
    Input("loc_marker", "position"),
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
    marker_pos,
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
    lat1 = marker_pos[0]
    lng1 = marker_pos[1]
    try:
        precip = pingrid.sel_snap(rr_mrg.precip, lat1, lng1)
        isnan = np.isnan(precip).sum().sum()
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            germ_sentence = ""
            return error_fig, error_fig, germ_sentence
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        germ_sentence = ""
        return error_fig, error_fig, germ_sentence
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
        error_fig = pingrid.error_fig(error_msg="Please ensure all input boxes are filled for the calculation to run.")
        germ_sentence = ""
        return (
            error_fig,
            error_fig,
            germ_sentence
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
        error_fig = pingrid.error_fig(error_msg="No dates found for this location")
        germ_sentence = ""
        return error_fig, error_fig, germ_sentence
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
    precip = precip.isel({"T": slice(-366, None)})
    search_start_dm = calc.sel_day_and_month(precip["T"], int(search_start_day), calc.strftimeb2int(search_start_month))
    precip = precip.sel({"T": slice(search_start_dm.values[0], None)})
    germ_rains_date = calc.onset_date(
        precip,
        int(wetThreshold),
        int(runningDays),
        int(runningTotal),
        int(minRainyDays),
        int(dryDays),
        0
    )
    germ_rains_date = germ_rains_date + germ_rains_date["T"]
    if pd.isnull(germ_rains_date):
        germ_sentence = (
            "Germinating rains have not yet occured as of "
            + precip["T"][-1].dt.strftime("%B %d, %Y").values
        )
    else:
        germ_sentence = (
            "Germinating rains have occured on "
            + germ_rains_date.dt.strftime("%B %d, %Y").values
        )
    return onsetDate_graph, probExceed_onset, germ_sentence

@APP.callback(
    Output("cessDate_plot", "figure"),
    Output("probExceed_cess", "figure"),
    Output("cess_dbct","tab_style"),
    Input("loc_marker", "position"),
    Input("start_cess_day", "value"),
    Input("start_cess_month", "value"),
    Input("searchDaysCess", "value"),
    Input("waterBalanceCess", "value"),
    Input("drySpellCess", "value"),
)
def cess_plots(
    marker_pos,
    start_cess_day,
    start_cess_month,
    searchDaysCess,
    waterBalanceCess,
    drySpellCess,
):
    if not CONFIG["ison_cess_date_hist"]:
        tab_style = {"display": "none"}
        return {}, {}, tab_style
    else:
        tab_style = {}
        lat = marker_pos[0]
        lng = marker_pos[1]
        try:
            precip = pingrid.sel_snap(rr_mrg.precip, lat, lng)
            isnan = np.isnan(precip).sum().sum()
            if isnan > 0:
                error_fig = pingrid.error_fig(error_msg="Data missing at this location")
                return error_fig, error_fig, tab_style
        except KeyError:
            error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
            return error_fig, error_fig, tab_style
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
            error_fig = pingrid.error_fig(error_msg="Please ensure all input boxes are filled for the calculation to run.")
            return (
                error_fig, error_fig, tab_style,
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
            error_fig = pingrid.error_fig(error_msg="No dates found for this location")
            return error_fig, error_fig, tab_style
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
        return cessDate_graph, probExceed_cess, tab_style


@SERVER.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def onset_tile(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    map_choice = parse_arg("map_choice")
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
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)

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
        return pingrid.image_resp(pingrid.empty_tile())

    if map_choice == "monit":
        precip_tile = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = calc.sel_day_and_month(precip_tile["T"], search_start_day, search_start_month1)
        precip_tile = precip_tile.sel({"T": slice(search_start_dm.values[0], None)})
    else:
        precip_tile = rr_mrg.precip

    precip_tile = precip_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()

    mymap_min = np.timedelta64(0)
    mycolormap = pingrid.RAINBOW_COLORMAP

    if map_choice == "monit":
        mymap = calc.onset_date(
            precip_tile,
            wet_thresh,
            wet_spell_length,
            wet_spell_thresh,
            min_wet_days,
            dry_spell_length,
            0
        )
        mymap_max = np.timedelta64((precip_tile["T"][-1] - precip_tile["T"][0]).values, 'D')
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
        if map_choice == "mean":
            mymap = onset_dates.onset_delta.mean("T")
            mymap_max = np.timedelta64(search_days, 'D')
        if map_choice == "stddev":
            mymap = onset_dates.onset_delta.dt.days.std(dim="T", skipna=True)
            mymap_min = 0
            mymap_max = int(search_days/3)
        if map_choice == "pe":
            mymap = (
                onset_dates.onset_delta.fillna(
                    np.timedelta64(search_days+1, 'D')
                ) > np.timedelta64(probExcThresh1, 'D')
            ).mean("T") * 100
            mymap_min = 0
            mymap_max = 100
            mycolormap = pingrid.CORRELATION_COLORMAP
            mymap.attrs["colormapkey"] = pingrid.CORRELATION_COLORMAP_KEY
    mymap.attrs["colormap"] = mycolormap
    mymap = mymap.rename(X="lon", Y="lat")
    mymap.attrs["scale_min"] = mymap_min
    mymap.attrs["scale_max"] = mymap_max
    result = pingrid.tile(mymap, tx, ty, tz, clip_shape)

    return result


@APP.callback(
    Output("colorbar", "children"),
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("map_choice", "value")
)
def set_colorbar(search_start_day, search_start_month, search_days, map_choice):
    if map_choice == "pe":
        thresholds = pingrid.CORRELATION_COLORMAP_KEY
        return (
            f"Probabily of onset date to be {search_days} past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.CORRELATION_COLORMAP, thresholds=thresholds),
            int(100),
            [i for i in range(0, int(100) + 1) if i % 10 == 0],
        )
    if map_choice == "mean":
        return (
            f"Onset date in days past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP),
            int(search_days),
            [i for i in range(0, int(search_days) + 1) if i % 10 == 0],
        )
    if map_choice == "stddev":
        return (
            f"Onset date standard deviation in days past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP),
            int(int(search_days)/3),
            [i for i in range(0, int(int(search_days)/3) + 1) if i % 10 == 0],
        )
    if map_choice == "monit":
        precip = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = calc.sel_day_and_month(precip["T"], int(search_start_day), calc.strftimeb2int(search_start_month))
        mymap_max = np.timedelta64((precip["T"][-1] - search_start_dm).values[0], 'D').astype(int)
        return (
            f"Germinating rains date in days past {search_start_day} {search_start_month}",
            pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP),
            mymap_max,
            [i for i in range(0, mymap_max + 1) if i % 25 == 0],
        )


if __name__ == "__main__":
    APP.run_server(
        host=CONFIG["listen_address"],
        port=CONFIG["listen_port"],
        debug=CONFIG["mode"] != "prod",
        processes=CONFIG["dev_processes"],
        threaded=False,
    )
