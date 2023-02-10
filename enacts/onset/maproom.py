import os
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
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
import datetime

from globals_ import FLASK

GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["onset"]

PFX = f'{GLOBAL_CONFIG["url_path_prefix"]}{CONFIG["core_path"]}'
TILE_PFX = f"{PFX}/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data

DR_PATH = f"{GLOBAL_CONFIG['zarr_path']}{GLOBAL_CONFIG['vars']['precip'][0]}"
RR_MRG_ZARR = Path(DR_PATH)
rr_mrg = calc.read_zarr_data(RR_MRG_ZARR)

# Assumes that grid spacing is regular and cells are square. When we
# generalize this, don't make those assumptions.
RESOLUTION = rr_mrg['X'][1].item() - rr_mrg['X'][0].item()
# The longest possible distance between a point and the center of the
# grid cell containing that point.

APP = dash.Dash(
    __name__,
    server=FLASK,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Onset Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = CONFIG["app_title"]

APP.layout = layout.app_layout()


def adm_borders(shapes):
    with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
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
    Input("start_cess_day", "value"),
    Input("start_cess_month", "value"),
    Input("search_days_cess", "value"),
    Input("water_balance_cess", "value"),
    Input("dry_spell_cess", "value"),
    Input("prob_exc_thresh1", "value"),
    Input("prob_exc_thresh2", "value"),
    Input("prob_exc_thresh3", "value"),
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
        start_cess_day,
        start_cess_month,
        search_days_cess, 
        water_balance_cess,
        dry_spell_cess,
        prob_exc_thresh1,
        prob_exc_thresh2,
        prob_exc_thresh3,
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
        "start_cess_day": start_cess_day,
        "start_cess_month": start_cess_month,
        "search_days_cess": search_days_cess,
        "water_balance_cess": water_balance_cess,
        "dry_spell_cess": dry_spell_cess,
        "prob_exc_thresh1": prob_exc_thresh1,
        "prob_exc_thresh2": prob_exc_thresh2,
        "prob_exc_thresh3": prob_exc_thresh3,
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
            len(GLOBAL_CONFIG["shapes_adm"])-i,
            is_checked=adm["is_checked"]
        )
        for i, adm in enumerate(GLOBAL_CONFIG["shapes_adm"])
    ] + [
        dlf.Overlay(
            dlf.TileLayer(
                url=f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
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
    Output("pet_input_wrapper1", "style"),
    Output("pet_input_wrapper2", "style"),
    Output("pet_input_wrapper3", "style"),
    Input("map_choice", "value"),
)
def display_pet_control(map_choice):

    pet_input_wrapper1={"display": "none"}
    pet_input_wrapper2={"display": "none"}
    pet_input_wrapper3={"display": "none"}
    if map_choice == "pe":
        pet_input_wrapper1={"display": "flex"}
    elif map_choice == "length_pe":
        pet_input_wrapper2={"display": "flex"}
    elif map_choice == "total_pe":
        pet_input_wrapper3={"display": "flex"}
    return pet_input_wrapper1, pet_input_wrapper2, pet_input_wrapper3


@APP.callback(
    Output("pet_units1", "children"),
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
    Input("prob_exc_thresh1", "value"),
    Input("prob_exc_thresh2", "value"),
    Input("prob_exc_thresh3", "value"),
)
def write_map_title(search_start_day, search_start_month, map_choice, pet1, pet2, pet3):

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
            "Climatological probability that Onset date is " + pet1  + " days past "
            + search_start_month + " " + search_start_day
        )
    if map_choice == "length_mean":
        mytitle = (
            "Climatological Length of season in days"
        )
    if map_choice == "length_stddev":
        mytitle = (
            "Climatological Length of season standard deviation in days"
        )
    if map_choice == "length_pe":
        mytitle = (
            "Climatological probability that season is shorter than "
            + pet2  + " days"
        )
    if map_choice == "total_mean":
        mytitle = (
            "Climatological Total seasonal precipitation in mm"
        )
    if map_choice == "total_stddev":
        mytitle = (
            "Climatological Total seasoanl precipitation standard deviation in mm"
        )
    if map_choice == "total_pe":
        mytitle = (
            "Climatological probability that it rains less than "
            + pet3  + " mm in season"
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
        error_fig = pingrid.error_fig(
            error_msg="Please ensure all input boxes are filled for the calculation to run."
        )
        germ_sentence = ""
        return (
            error_fig,
            error_fig,
            germ_sentence
        )  # dash.no_update to leave the plat as-is and not show no data display
    onset_date_graph = pgo.Figure()
    onset_date_graph.add_trace(
        pgo.Scatter(
            x=onset_delta["T"].dt.year.values,
            y=onset_delta["onset_delta"].dt.days.values,
            customdata=(onset_delta["T"] + onset_delta["onset_delta"]).dt.strftime("%-d %B %Y"),
            hovertemplate="%{customdata}",
            name="",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    onset_date_graph.update_traces(mode="lines", connectgaps=False)
    onset_date_graph.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Onset Date in days since {search_start_month} {int(search_start_day)}",
        title=f"Onset dates found after {search_start_month} {int(search_start_day)}, at ({round_latLng(lat1)}N,{round_latLng(lng1)}E)",
    )
    quantiles = np.arange(1, onset_delta["T"].size + 1) / (onset_delta["T"].size + 1)
    onset_quant = onset_delta["onset_delta"].dt.days.quantile(quantiles, dim="T")
    cdf_graph = pgo.Figure()
    cdf_graph.add_trace(
        pgo.Scatter(
            x=onset_quant.values,
            y=(1 - quantiles),
            customdata=(datetime.datetime(
                2000,
                calc.strftimeb2int(search_start_month),
                int(search_start_day)
            ) + pd.to_timedelta(onset_quant, "D")).strftime("%-d %B"),
            hovertemplate="%{y:.0%} chance of exceeding %{customdata}",
            name="",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    cdf_graph.update_layout(
        xaxis_title=f"Onset Date in days since {search_start_month} {int(search_start_day)}",
        yaxis_title="Probability of exceeding",
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
    #return onsetDate_graph, probExceed_onset, germ_sentence
    return onset_date_graph, cdf_graph, germ_sentence


@APP.callback(
    Output("cessDate_plot", "figure"),
    Output("probExceed_cess", "figure"),
    Output("cess_dbct","tab_style"),
    Input("loc_marker", "position"),
    Input("start_cess_day", "value"),
    Input("start_cess_month", "value"),
    Input("search_days_cess", "value"),
    Input("water_balance_cess", "value"),
    Input("dry_spell_cess", "value"),
)
def cess_plots(
    marker_pos,
    start_cess_day,
    start_cess_month,
    search_days_cess,
    water_balance_cess,
    dry_spell_cess,
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
                int(search_days_cess),
                int(water_balance_cess),
                int(dry_spell_cess),
                time_coord="T",
            )
        except TypeError:
            error_fig = pingrid.error_fig(error_msg="Please ensure all input boxes are filled for the calculation to run.")
            return (
                error_fig, error_fig, tab_style,
            )  # dash.no_update to leave the plat as-is and not show no data display
        cess_date_graph = pgo.Figure()
        cess_date_graph.add_trace(
            pgo.Scatter(
                x=cess_delta["T"].dt.year.values,
                y=cess_delta["cess_delta"].squeeze().dt.days.values,
                customdata=(cess_delta["T"] + cess_delta["cess_delta"]).dt.strftime("%-d %B %Y"),
                hovertemplate="%{customdata}",
                name="",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cess_date_graph.update_traces(mode="lines", connectgaps=False)
        cess_date_graph.update_layout(
            xaxis_title="Year",
            yaxis_title=f"Cessation Date in days since {start_cess_month} {int(start_cess_day)}",
            title=f"Cessation dates found after {start_cess_month} {int(start_cess_day)}, at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
        )
        quantiles = np.arange(1, cess_delta["T"].size + 1) / (cess_delta["T"].size + 1)
        cess_quant = cess_delta["cess_delta"].dt.days.quantile(quantiles, dim="T").squeeze()
        cdf_graph = pgo.Figure()
        cdf_graph.add_trace(
            pgo.Scatter(
                x=cess_quant.values,
                y=(1 - quantiles),
                customdata=(datetime.datetime(
                    2000,
                    calc.strftimeb2int(start_cess_month),
                    int(start_cess_day)
                ) + pd.to_timedelta(cess_quant, "D")).strftime("%-d %B"),
                hovertemplate="%{y:.0%} chance of exceeding %{customdata}",
                name="",
                line=pgo.scatter.Line(color="blue"),
            )
        )
        cdf_graph.update_traces(mode="lines", connectgaps=False)
        cdf_graph.update_layout(
            xaxis_title=f"Cessation Date in days since {start_cess_month} {int(start_cess_day)}",
            yaxis_title="Probability of exceeding",
        )
        return cess_date_graph, cdf_graph, tab_style


@FLASK.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
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
    start_cess_day = parse_arg("start_cess_day", int)
    start_cess_month1 = parse_arg("start_cess_month", calc.strftimeb2int)
    search_days_cess = parse_arg("search_days_cess", int)
    water_balance_cess = parse_arg("water_balance_cess", float)
    dry_spell_cess = parse_arg("dry_spell_cess", int)
    prob_exc_thresh1 = parse_arg("prob_exc_thresh1", int)
    prob_exc_thresh2 = parse_arg("prob_exc_thresh2", int)
    prob_exc_thresh3 = parse_arg("prob_exc_thresh3", int)

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

    map_min = np.timedelta64(0) if map_choice in ["monit", "mean"] else 0
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
        map_max = np.timedelta64((precip_tile["T"][-1] - precip_tile["T"][0]).values, 'D')
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
        if ("length" in map_choice) | ("total" in map_choice):
            soil_moisture = calc.water_balance(
                precip_tile, 5, 60, 0
            )["soil_moisture"]
            cess_dates = calc.seasonal_cess_date(
                soil_moisture,
                start_cess_day,
                start_cess_month1,
                search_days_cess,
                water_balance_cess,
                dry_spell_cess,
            )
            if "length" in map_choice:
                if cess_dates["T"][0] < onset_dates["T"][0]:
                    cess_dates = cess_dates.isel({"T": slice(1, None)})
                    if cess_dates["T"].size != onset_dates["T"].size:
                        onset_dates = onset_dates.isel({"T": slice(None, -2)})
                seasonal_length = (
                    (cess_dates["T"] + cess_dates["cess_delta"]).drop_indexes("T")
                    - (onset_dates["T"] + onset_dates["onset_delta"]).drop_indexes("T")
                ) #.astype("timedelta64[D]")
        if map_choice == "mean":
            mymap = onset_dates.onset_delta.mean("T")
            map_max = np.timedelta64(search_days, 'D')
        if map_choice == "stddev":
            mymap = onset_dates.onset_delta.dt.days.std(dim="T", skipna=True)
            map_max = int(search_days/3)
        if map_choice == "pe":
            mymap = (
                onset_dates.onset_delta.fillna(
                    np.timedelta64(search_days+1, 'D')
                ) > np.timedelta64(prob_exc_thresh1, 'D')
            ).mean("T") * 100
            map_max = 100
            mycolormap = pingrid.CORRELATION_COLORMAP
        if map_choice == "length_mean":
            mymap = seasonal_length.mean("T")
            map_max = np.timedelta64(int(CONFIG["map_text"][map_choice]["map_max"]), 'D')
        if map_choice == "length_stddev":
            mymap = seasonal_length.dt.days.std(dim="T", skipna=True)
            map_max = CONFIG["map_text"][map_choice]["map_max"]
        if map_choice == "length_pe":
            mymap = (seasonal_length < np.timedelta64(prob_exc_thresh2, 'D')).mean("T") * 100
            map_max = 100
            mycolormap = pingrid.CORRELATION_COLORMAP
    mymap.attrs["colormap"] = mycolormap
    mymap = mymap.rename(X="lon", Y="lat")
    mymap.attrs["scale_min"] = map_min
    mymap.attrs["scale_max"] = map_max
    result = pingrid.tile(mymap, tx, ty, tz, clip_shape)

    return result


@APP.callback(
    Output("colorbar", "colorscale"),
    Output("colorbar", "max"),
    Output("colorbar", "tickValues"),
    Output("colorbar", "unit"),
    Input("search_start_day", "value"),
    Input("search_start_month", "value"),
    Input("searchDays", "value"),
    Input("map_choice", "value")
)
def set_colorbar(search_start_day, search_start_month, search_days, map_choice):
    colorbar = pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP)
    if "pe" in map_choice:
        colorbar = pingrid.to_dash_colorscale(pingrid.CORRELATION_COLORMAP)
        tick_freq = 10
        map_max = 100
        unit = "%"
    if map_choice == "mean":
        tick_freq = 10
        map_max = int(search_days)
        unit = "days" 
    if map_choice == "stddev":
        tick_freq = 5
        map_max = int(int(search_days)/3)
        unit = "days"
    if map_choice == "length_mean":
        tick_freq = 20
        map_max = CONFIG["map_text"][map_choice]["map_max"]
        unit = "days"
    if map_choice == "length_stddev":
        tick_freq = 5
        map_max = CONFIG["map_text"][map_choice]["map_max"]
        unit = "days"
    if map_choice == "monit":
        precip = rr_mrg.precip.isel({"T": slice(-366, None)})
        search_start_dm = calc.sel_day_and_month(
            precip["T"], int(search_start_day), calc.strftimeb2int(search_start_month)
        )
        map_max = np.timedelta64(
            (precip["T"][-1] - search_start_dm).values[0], 'D'
        ).astype(int)
        tick_freq = 10
        unit = "days"
    return colorbar, map_max, [i for i in range(0, map_max + 1) if i % tick_freq == 0], unit


if __name__ == "__main__":
    APP.run_server(
        debug=GLOBAL_CONFIG["mode"] != "prod",
        processes=GLOBAL_CONFIG["dev_processes"],
        threaded=False,
    )
