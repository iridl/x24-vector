import os
import flask
import dash
from dash import ALL
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pingrid 
import layout_monit
import calc
import plotly.graph_objects as pgo
import pandas as pd
import numpy as np
import urllib

import xarray as xr
import agronomy as ag

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon

CONFIG_GLOBAL = pingrid.load_config(os.environ["CONFIG"])
CONFIG = CONFIG_GLOBAL["wat_bal_monit"]

PFX = CONFIG["core_path"]
TILE_PFX = "/tile"

with psycopg2.connect(**CONFIG_GLOBAL["db"]) as conn:
    s = sql.Composed([sql.SQL(CONFIG_GLOBAL['shapes_adm'][0]['sql'])])
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
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Water Balance Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = CONFIG["app_title"]

APP.layout = layout_monit.app_layout()


def adm_borders(shapes):
    with psycopg2.connect(**CONFIG_GLOBAL["db"]) as conn:
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
    for i in df.index:
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
    Output("time_selection", "options"),
    Output("time_selection", "value"),
    Input("planting_day","value"),
    Input("planting_month", "value"),
    Input("wat_bal_plot", "clickData"),
    State("time_selection", "options"),
)
def update_time_sel(planting_day, planting_month, graph_click, current_options):
    if dash.ctx.triggered_id == "wat_bal_plot":
        time_range = current_options
        the_value = graph_click["points"][0]["x"]
    else:
        time_range = rr_mrg.precip["T"].isel({"T": slice(-366, None)})
        p_d = time_range.where(
            lambda x: (x.dt.day == int(planting_day))
            & (x.dt.month == calc.strftimeb2int(planting_month)),
            drop=True
        ).squeeze()
        time_range = time_range.where(
            time_range >= p_d, drop=True
        ).dt.strftime("%-d %b %y").values
        the_value = time_range[-1]
    return time_range, the_value


@APP.callback(
    Output("layers_control", "children"),
    Input("map_choice", "value"),
    Input("time_selection", "value"),
    Input("submit_kc", "n_clicks"),
    State("planting_day", "value"),
    State("planting_month", "value"),
    State("kc_init", "value"),
    State("kc_init_length", "value"),
    State("kc_veg", "value"),
    State("kc_veg_length", "value"),
    State("kc_mid", "value"),
    State("kc_mid_length", "value"),
    State("kc_late", "value"),
    State("kc_late_length", "value"),
    State("kc_end", "value"),
)
def make_map(
        map_choice,
        the_date,
        n_clicks,
        planting_day,
        planting_month,
        kc_init,
        kc_init_length,
        kc_veg,
        kc_veg_length,
        kc_mid,
        kc_mid_length,
        kc_late,
        kc_late_length,
        kc_end,
):
    qstr = urllib.parse.urlencode({
        "map_choice": map_choice,
        "the_date": the_date,
        "n_clicks": n_clicks,
        "planting_day": planting_day,
        "planting_month": planting_month,
        "kc_init": kc_init,
        "kc_init_length": kc_init_length,
        "kc_veg": kc_veg,
        "kc_veg_length": kc_veg_length,
        "kc_mid": kc_mid,
        "kc_mid_length": kc_mid_length,
        "kc_late": kc_late,
        "kc_late_length": kc_late_length,
        "kc_end": kc_end,
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
            len(CONFIG_GLOBAL["shapes_adm"])-i,
            is_checked=adm["is_checked"]
        )
        for i, adm in enumerate(CONFIG_GLOBAL["shapes_adm"])
    ] + [
        dlf.Overlay(
            dlf.TileLayer(
                url=f"{TILE_PFX}/{{z}}/{{x}}/{{y}}?{qstr}",
                opacity=1,
            ),
            name="Water_Balance",
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
    Input("map_choice", "value"),
    Input("crop_name", "value"),
)
def write_map_title(map_choice, crop_name):
    return f"{CONFIG['map_text'][map_choice]['menu_label']} for {crop_name}"


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
    Output("wat_bal_plot", "figure"),
    Input("loc_marker", "position"),
    Input("map_choice", "value"),
    Input("submit_kc", "n_clicks"),
    State("planting_day", "value"),
    State("planting_month", "value"),
    State("crop_name", "value"),
    State("kc_init", "value"),
    State("kc_init_length", "value"),
    State("kc_veg", "value"),
    State("kc_veg_length", "value"),
    State("kc_mid", "value"),
    State("kc_mid_length", "value"),
    State("kc_late", "value"),
    State("kc_late_length", "value"),
    State("kc_end", "value"),
)
def wat_bal_plots(
    marker_pos,
    map_choice,
    n_clicks,
    planting_day,
    planting_month,
    crop_name,
    kc_init,
    kc_init_length,
    kc_veg,
    kc_veg_length,
    kc_mid,
    kc_mid_length,
    kc_late,
    kc_late_length,
    kc_end,
):
    lat = marker_pos[0]
    lng = marker_pos[1]
    kc_periods = pd.TimedeltaIndex(
        [0, int(kc_init_length), int(kc_veg_length), int(kc_mid_length), int(kc_late_length)], unit="D"
    )
    kc_params = xr.DataArray(data=[
        float(kc_init), float(kc_veg), float(kc_mid), float(kc_late), float(kc_end)
    ], dims=["kc_periods"], coords=[kc_periods])
    precip = rr_mrg.precip.isel({"T": slice(-366, None)})
    p_d = precip["T"].where(
        lambda x: (x.dt.day == int(planting_day))
        & (x.dt.month == calc.strftimeb2int(planting_month)),
        drop=True
    ).squeeze(drop=True).rename("p_d")
    precip = precip.where(precip["T"] >= p_d, drop=True)
    try:
        precip = pingrid.sel_snap(precip, lat, lng)
        isnan = np.isnan(precip).sum().sum()
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            return error_fig
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        return error_fig
    precip.load()
    try:
        sm, drainage, et_crop = ag.soil_plant_water_balance(
            precip,
            5,
            60,
            60./3.,
            kc_params=kc_params,
            planting_date=p_d,
        )
    except TypeError:
        error_fig = pingrid.error_fig(
            error_msg="Please ensure all input boxes are filled for the calculation to run."
        )
        return error_fig
    if map_choice == "sm":
        myts = sm
    elif map_choice == "drainage":
        myts = drainage
    elif map_choice == "et_crop":
        myts = et_crop
    wat_bal_graph = pgo.Figure()
    wat_bal_graph.add_trace(
        pgo.Scatter(
            x=myts["T"].dt.strftime("%-d %b %y"),
            y=myts.values,
            hovertemplate="%{y} on %{x}",
            name="",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    wat_bal_graph.update_traces(
        mode="lines",
        connectgaps=False,
    )
    wat_bal_graph.update_layout(
        xaxis_title="Time",
        yaxis_title=f"{CONFIG['map_text'][map_choice]['menu_label']} [{CONFIG['map_text'][map_choice]['units']}]",
        title=f"{CONFIG['map_text'][map_choice]['menu_label']} for {crop_name} at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
    )
    return wat_bal_graph


@SERVER.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def wat_bal_tile(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    map_choice = parse_arg("map_choice")
    the_date = parse_arg("the_date", str)
    planting_day = parse_arg("planting_day", int)
    planting_month1 = parse_arg("planting_month", calc.strftimeb2int)
    kc_init = parse_arg("kc_init", float)
    kc_init_length = parse_arg("kc_init_length", int)
    kc_veg = parse_arg("kc_init", float)
    kc_veg_length = parse_arg("kc_init_length", int)
    kc_mid = parse_arg("kc_mid", float)
    kc_mid_length = parse_arg("kc_mid_length", int)
    kc_late = parse_arg("kc_late", float)
    kc_late_length = parse_arg("kc_late_length", int)
    kc_end = parse_arg("kc_end", float)

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)
    # row numbers increase as latitude decreases
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)

    precip = rr_mrg.precip.isel({"T": slice(-366, None)})
    p_d = precip["T"].where(
        lambda x: (x.dt.day == int(planting_day))
        & (x.dt.month == planting_month1),
        drop=True
    ).squeeze(drop=True).rename("p_d")
    precip = precip.sel(T=slice(p_d.dt.strftime("%-d %b %y"), the_date))

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

    precip_tile = precip.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()

    kc_periods = pd.TimedeltaIndex(
        [0, kc_init_length, kc_veg_length, kc_mid_length, kc_late_length], unit="D"
    )
    kc_params = xr.DataArray(
        data=[kc_init, kc_veg, kc_mid, kc_late, kc_end], dims=["kc_periods"], coords=[kc_periods]
    )

    mymap_min = 0
    mymap_max = 60
    mycolormap = pingrid.RAINFALL_COLORMAP

    sm, drainage, et_crop = ag.soil_plant_water_balance(
        precip_tile,
        5,
        60,
        60./3.,
        kc_params=kc_params,
        planting_date=p_d,
    )
    if map_choice == "sm":
        mymap = sm
    elif map_choice == "drainage":
        mymap = drainage
    elif map_choice == "et_crop":
        mymap = et_crop
    else:
       raise Exception("can not enter here")
    mymap = mymap.isel(T=-1)
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
    Input("map_choice", "value"),
)
def set_colorbar(
    map_choice,
):
    mymap_max = 60
    return (
        f"{CONFIG['map_text'][map_choice]['menu_label']} [{CONFIG['map_text'][map_choice]['units']}]",
        pingrid.to_dash_colorscale(pingrid.RAINFALL_COLORMAP),
        mymap_max,
        [i for i in range(0, mymap_max + 1) if i % int(mymap_max/12) == 0],
    )


if __name__ == "__main__":
    APP.run_server(
        host=CONFIG_GLOBAL["server"],
        port=CONFIG_GLOBAL["port"],
        debug=CONFIG_GLOBAL["mode"] != "prod",
        processes=CONFIG_GLOBAL["dev_processes"],
        threaded=False,
    )
