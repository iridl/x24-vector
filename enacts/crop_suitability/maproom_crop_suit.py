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
import layout_crop_suit
import calc
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
import xarray as xr

GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["crop_suit"]

PFX = CONFIG["core_path"]
TILE_PFX = "/tile"

with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
    s = sql.Composed([sql.SQL(GLOBAL_CONFIG['shapes_adm'][0]['sql'])])
    df = pd.read_sql(s, conn)
    clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

# Reads daily data
rr_mrg = calc.read_zarr_data(Path(GLOBAL_CONFIG["rr_mrg_zarr_path"]))
tmin_mrg = calc.read_zarr_data(Path(GLOBAL_CONFIG["tmin_zarr_path"]))
tmax_mrg = calc.read_zarr_data(Path(GLOBAL_CONFIG["tmax_zarr_path"]))
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
        {"name": "description", "content": "Crop Suitability Maproom"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = CONFIG["app_title"]

APP.layout = layout_crop_suit.app_layout()


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
    Input("target_season", "value"),
)
def make_map(
        map_choice,
        target_season,
):
    qstr = urllib.parse.urlencode({
        "map_choice": map_choice,
        "target_season": target_season,
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
            name="Crop Suitability",
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

def sel_year_season(data,target_season,target_year):
    seasonal = data.sel(T=data['T.season']==target_season)
    if target_season == "DJF": #need december of previous year right now this does not work
        seasonal_year = seasonal.sel(T=(seasonal['T.year']==(target_year-1 or target_year))).load()
    else:
        seasonal_year = seasonal.sel(T=seasonal['T.year']==target_year).load()    
    return seasonal_year

def crop_suitability(
    rainfall_data,
    min_wet_days,
    wet_day_def,
    tmax_data,
    tmin_data,
    lower_wet_threshold,
    upper_wet_threshold,
    max_temp,
    min_temp,
    temp_range,
    target_season,
):  
    #seasonal_year_precip = sel_year_season(rainfall_data,target_season,target_year)
    #seasonal_year_tmax = sel_year_season(tmax_data,target_season,target_year)
    #seasonal_year_tmin = sel_year_season(tmin_data,target_season,target_year)
    seasonal_precip = rainfall_data.sel(T=rainfall_data['T.season']==target_season).load()
    seasonal_tmax = tmax_data.sel(T=tmax_data['T.season']==target_season).load()
    seasonal_tmin = tmin_data.sel(T=tmin_data['T.season']==target_season).load()

    sum_precip = seasonal_precip.groupby("T.year").sum("T")
    avg_tmax = seasonal_tmax.groupby("T.year").mean("T")
    avg_tmin = seasonal_tmin.groupby("T.year").mean("T")
    
    avg_daily_temp_range = (
        seasonal_tmax["temp"] - seasonal_tmin["temp"]
    ).groupby("T.year").mean("T")

    min_total_wet_days = xr.where(
        seasonal_precip["precip"] >= float(wet_day_def),1,0
    ).groupby("T.year").sum("T")

#    max_total_dry_spells = xr.where(
#        seasonal_precip["precip"] <= float(dry_spell_rain),1,0
#    )

    crop_suitability = avg_tmax.copy(data=None).drop_vars("temp")

    total_precip_range = xr.where(
        np.logical_and(
            sum_precip["precip"] <= float(upper_wet_threshold), 
            sum_precip["precip"] >= float(lower_wet_threshold)
        ),1, 0)
    tmax = xr.where(avg_tmax["temp"] <= float(max_temp), 1, 0)
    tmin = xr.where(avg_tmin["temp"] >= float(min_temp), 1, 0)
    avg_temp_range = xr.where(avg_daily_temp_range <= float(temp_range), 1, 0)
    wet_days = xr.where(min_total_wet_days >= float(min_wet_days), 1, 0)

    crop_suitability = crop_suitability.assign(max_temp = tmax, min_temp = tmin, temp_range = avg_temp_range,precip_range = total_precip_range, wet_days = wet_days)
    crop_suitability['crop_suit'] = (crop_suitability['max_temp'] + crop_suitability['min_temp'] + crop_suitability['temp_range'] + crop_suitability['precip_range'] + crop_suitability['wet_days']) / 5
    crop_suitability = crop_suitability.dropna(dim="year", how="any")
    print(crop_suitability) 
    return crop_suitability

@APP.callback(
    Output("timeseries_graph","figure"),
    Input("loc_marker", "position"),
    Input("map_choice","value"),
    Input("target_year","value"),
    Input("target_season","value"),
    Input("lower_wet_threshold","value"),
    Input("upper_wet_threshold","value"),
    Input("minimum_temp","value"),
    Input("maximum_temp","value"),
    Input("temp_range","value"),
    Input("season_length","value"),
    Input("min_wet_days","value"),
    Input("wet_day_def","value"),
    Input("number_months","value"),
    Input("number_dry_spells","value"),
    Input("days_in_row","value"),
    Input("dry_spell_rain","value"),
)
def timeseries_plot(
    loc_marker,
    map_choice,
    target_year,
    target_season,
    lower_wet_threshold,
    upper_wet_threshold,
    minimum_temp,
    maximum_temp,
    temp_range,
    season_length,
    min_wet_days,
    wet_day_def,
    number_months,
    number_dry_spells,
    days_in_row,
    dry_spell_rain
):
    lat1 = loc_marker[0]
    lng1 = loc_marker[1]

    if map_choice == "suitability_map":
        data = crop_suitability(rr_mrg,min_wet_days,wet_day_def,tmax_mrg,tmin_mrg,lower_wet_threshold,upper_wet_threshold,maximum_temp,minimum_temp,temp_range,target_season)
    if map_choice == "precip_map":
        data = rr_mrg
    if map_choice == "tmax_map":
        data = tmax_mrg
    if map_choice == "tmin_map":
        data = tmin_mrg
    
    try:
        if map_choice == "precip_map":
            data_var = pingrid.sel_snap(data.precip, lat1, lng1)
            isnan = np.isnan(data_var).sum().sum()
        elif map_choice == "suitability_map":
            data_var = pingrid.sel_snap(data.crop_suit, lat1, lng1)
            isnan = np.isnan(data_var).sum().sum()            
        else:
            data_var = pingrid.sel_snap(data.temp, lat1, lng1)
            isnan = np.isnan(data_var).sum().sum()
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            germ_sentence = ""
            return error_fig, error_fig, germ_sentence
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        germ_sentence = ""
        return error_fig, error_fig, germ_sentence

    if map_choice == "suitability_map":
        seasonal_suit = data_var
        timeseries_plot = pgo.Figure()
        timeseries_plot.add_trace(
            pgo.Scatter(
                x = seasonal_suit["year"].values,
                y = seasonal_suit.values,
                line=pgo.scatter.Line(color="blue"),
            )
        )
        timeseries_plot.update_traces(mode="lines", connectgaps=False)
        timeseries_plot.update_layout(
            xaxis_title = "years",
            yaxis_title = f"{CONFIG['map_text'][map_choice]['data_var']} ({CONFIG['map_text'][map_choice]['units']})",
            title = f"{CONFIG['map_text'][map_choice]['menu_label']} seasonal climatology timeseries plot"
        ) 
    else:
        data_var.load()
    
        seasonal_var = data_var.sel(T=data_var['T.season']==target_season)
        seasonal_mean = seasonal_var.groupby("T.year").mean("T")
        
        timeseries_plot = pgo.Figure()
        timeseries_plot.add_trace(
            pgo.Scatter(
                x = seasonal_mean["year"].values,
                y = seasonal_mean.values,
                line=pgo.scatter.Line(color="blue"),
            )
        )
        timeseries_plot.update_traces(mode="lines", connectgaps=False)
        timeseries_plot.update_layout(
            xaxis_title = "years",
            yaxis_title = f"{CONFIG['map_text'][map_choice]['data_var']} ({CONFIG['map_text'][map_choice]['units']})",
            title = f"{CONFIG['map_text'][map_choice]['menu_label']} seasonal climatology timeseries plot"
        )

    return timeseries_plot

@SERVER.route(f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>")
def overlay_layers(tz, tx, ty):
    parse_arg = pingrid.parse_arg
    map_choice = parse_arg("map_choice")
    target_season = parse_arg("target_season")   

    x_min = pingrid.tile_left(tx, tz)
    x_max = pingrid.tile_left(tx + 1, tz)

    # row numbers increase as latitude decreases
    y_max = pingrid.tile_top_mercator(ty, tz)
    y_min = pingrid.tile_top_mercator(ty + 1, tz)
    
#    if map_choice == "precip_map": # eventually I wanted users to
#        data_tile = rr_mrg.precip  # choose what type of data to 
#    if map_choice == "tmax_map":   # see displayed but for now there
#        data_tile = tmax_mrg.temp  # was an issue with the data
#    if map_choice == "tmin_map":
#        data_tile = tmin_mrg.temp
#    if map_choice == "suitability_map":
#        data_tile = tmin_mrg.temp

#    data_season = data_tile.groupby('T.season').mean("T")
#    data_season = data_tile.sel(season=target_season).values

    data_tile = rr_mrg.precip
    if (
            # When we generalize this to other datasets, remember to
            # account for the possibility that longitudes wrap around,
            # so a < b doesn't always mean that a is west of b.
            x_min > data_tile['X'].max() or
            x_max < data_tile['X'].min() or
            y_min > data_tile['Y'].max() or
            y_max < data_tile['Y'].min()
    ):
        return pingrid.image_resp(pingrid.empty_tile())
    
    data_tile = data_tile.sel(
        X=slice(x_min - x_min % RESOLUTION, x_max + RESOLUTION - x_max % RESOLUTION),
        Y=slice(y_min - y_min % RESOLUTION, y_max + RESOLUTION - y_max % RESOLUTION),
    ).compute()
    print(data_tile.min)
    print(data_tile.max)    
    mymap_min = float(0) #np.timedelta64(0)
    mymap_max = float(25) #np.timedelta64(100)

    mycolormap = pingrid.RAINBOW_COLORMAP

#    mymap = data_tile.groupby('T.season').mean("T")
#    mymap = data_tile.sel(season=target_season).values

    mymap = data_tile.mean("T") # as a start I am just trying to get
                                # printed a mean precip map before

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
    mymap_max = 60 #taw.max()
    return (
        f"{CONFIG['map_text'][map_choice]['menu_label']} [{CONFIG['map_text'][map_choice]['units']}]",
        pingrid.to_dash_colorscale(pingrid.RAINFALL_COLORMAP),
        mymap_max,
        [i for i in range(0, mymap_max + 1) if i % int(mymap_max/12) == 0],
    )

if __name__ == "__main__":
    APP.run_server(
        debug=GLOBAL_CONFIG["mode"] != "prod",
        processes=GLOBAL_CONFIG["dev_processes"],
        threaded=False,
    )
