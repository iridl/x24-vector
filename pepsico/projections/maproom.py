import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import pingrid
from pingrid import CMAPS
from . import layout
import plotly.graph_objects as pgo
import xarray as xr
import pandas as pd
import dash_leaflet as dlf
import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from globals_ import FLASK, GLOBAL_CONFIG

def register(FLASK, config):
    PFX = f"{GLOBAL_CONFIG['url_path_prefix']}/{config['core_path']}"
    TILE_PFX = f"{PFX}/tile"

    # App

    APP = dash.Dash(
        __name__,
        server=FLASK,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
        ],
        url_base_pathname=f"{PFX}/",
        meta_tags=[
            {"name": "description", "content": "Forecast"},
            {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        ],
    )
    APP.title = "Forecast"

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


    def make_adm_overlay(
        adm_name, adm_sql, adm_color, adm_lev, adm_weight, is_checked=False,
    ):
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
            Output("lat_input", "min"),
            Output("lat_input", "max"),
            Output("lat_input_tooltip", "children"),
            Output("lng_input", "min"),
            Output("lng_input", "max"),
            Output("lng_input_tooltip", "children"),
            Output("map", "center"),
            Input("location", "pathname"),
    )
    def initialize(path):
        scenario = "historical"
        model = "GFDL-ESM4"
        variable = "pr"
        data = xr.open_zarr(
            f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted'
            f'/global/monthly/{scenario}/{model}/zarr/{variable}'
        )[variable]
        center_of_the_map = [
            ((data["Y"][int(data["Y"].size/2)].values)),
            ((data["X"][int(data["X"].size/2)].values)),
        ]
        lat_res = (data["Y"][0 ]- data["Y"][1]).values
        lat_min = str((data["Y"][-1] - lat_res/2).values)
        lat_max = str((data["Y"][0] + lat_res/2).values)
        lon_res = (data["X"][1] - fcst_mu["X"][0]).values
        lon_min = str((data["X"][0] - lon_res/2).values)
        lon_max = str((data["X"][-1] + lon_res/2).values)
        lat_label = lat_min + " to " + lat_max + " by " + str(lat_res) + "˚"
        lon_label = lon_min + " to " + lon_max + " by " + str(lon_res) + "˚"
        
        return (
            lat_min, lat_max, lat_label, lon_min, lon_max, lon_label,
            center_of_the_map,
        )


    @APP.callback(
    Output("map_title", "children"),
    )
    def write_map_title():
        return "MAP TITLE"


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
        # Reading
        scenario = "historical"
        model = "GFDL-ESM4"
        variable = "pr"
        data = xr.open_zarr(
            f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted'
            f'/global/monthly/{scenario}/{model}/zarr/{variable}'
        )[variable]
        if dash.ctx.triggered_id == None:
            lat = data["Y"][int(data["Y"].size/2)].values
            lng = data["X"][int(data["X"].size/2)].values
        else:
            if dash.ctx.triggered_id == "map":
                lat = click_lat_lng[0]
                lng = click_lat_lng[1]
            else:
                lat = latitude
                lng = longitude
            try:
                nearest_grid = pingrid.sel_snap(data, lat, lng)
                lat = nearest_grid["Y"].values
                lng = nearest_grid["X"].values
            except KeyError:
                lat = lat
                lng = lng
        return [lat, lng], lat, lng


    @APP.callback(
        Output("cdf_graph", "figure"),
        Input("loc_marker", "position"),
    )
    def local_plots(marker_pos):
        lat = marker_pos[0]
        lng = marker_pos[1]
        return error_fig


    @APP.callback(
        Output("fcst_colorbar", "colorscale"),
    )
    def draw_colorbar():
        return (pingrid.to_dash_colorscale(pingrid.RAINBOW_COLORMAP)


    @APP.callback(
        Output("layers_control", "children"),
        Output("forecast_warning", "is_open"),
    )
    def make_map():
        try:
            send_alarm = False
            url_str = f"{TILE_PFX}/{{z}}/{{x}}/{{y}}}"
        except:
            url_str= ""
            send_alarm = True
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
                len(GLOBAL_CONFIG["datasets"]["shapes_adm"])-i,
                is_checked=adm["is_checked"]
            )
            for i, adm in enumerate(GLOBAL_CONFIG["datasets"]["shapes_adm"])
        ] + [
            dlf.Overlay(
                dlf.TileLayer(url=url_str, opacity=1),
                name="Forecast",
                checked=True,
            ),
        ], send_alarm


    # Endpoints

    @FLASK.route(
        f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>",
        endpoint=f"{config['core_path']}"
    )
    def fcst_tiles(tz, tx, ty):
        # Reading
        scenario = "historical"
        model = "GFDL-ESM4"
        variable = "pr"
        data = xr.open_zarr(
            f'/Data/data24/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted'
            f'/global/monthly/{scenario}/{model}/zarr/{variable}'
        )[variable][T=-1]
        with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
            s = sql.Composed(
                [sql.SQL(GLOBAL_CONFIG['datasets']['shapes_adm'][0]['sql'])]
            )
            df = pd.read_sql(s, conn)
            clip_shape = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))[0]

        resp = pingrid.tile(data, tx, ty, tz, clip_shape)
        return resp
