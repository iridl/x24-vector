import os
import flask
import dash
import glob
import re
from datetime import datetime, timedelta
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import dash_leaflet as dlf
from pathlib import Path
import pingrid
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np
import cptio
import xarray as xr
from scipy.stats import t, norm, rankdata
import pandas as pd
import predictions

CONFIG = pingrid.load_config(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]
DATA_PATH = CONFIG["results_path"]

# App

SERVER = flask.Flask(__name__)
APP = dash.Dash(
    __name__,
    server=SERVER,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
    ],
    url_base_pathname=f"{PFX}/",
    meta_tags=[
        {"name": "description", "content": "Seasonal Forecast"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "Sub-Seasonal Forecast"

APP.layout = layout.app_layout

#Should I move this function into the predictions.py file where I put the other funcs?
#if we do so maybe I should redo the func to be more flexible since it is hard coded to read each file separately..
def read_cptdataset(lead_time, start_date, y_transform=CONFIG["y_transform"]):
    fcst_mu = predictions.sel_cpt_file(
        DATA_PATH,
        CONFIG["forecast_mu_file_pattern"],
        lead_time,
        start_date,
        CONFIG["start_format_in"],
    )
    fcst_mu_name = list(fcst_mu.data_vars)[0]
    fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = predictions.sel_cpt_file(
        DATA_PATH,
        CONFIG["forecast_var_file_pattern"],
        lead_time,
        start_date,
        CONFIG["start_format_in"],
    )
    fcst_var_name = list(fcst_var.data_vars)[0]
    fcst_var = fcst_var[fcst_var_name]
    obs = (predictions.sel_cpt_file(
        DATA_PATH,
        CONFIG["obs_file_pattern"],
        lead_time,
        start_date,
        CONFIG["start_format_in"],
    )).squeeze()
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    if y_transform:
        hcst = (predictions.sel_cpt_file(
            DATA_PATH,
            CONFIG["hcst_file_pattern"],
            lead_time,
            start_date,
            CONFIG["start_format_in"],
        )).squeeze()
        hcst_name = list(hcst.data_vars)[0]
        hcst = hcst[hcst_name]
    else:
        hcst=None
    return fcst_mu, fcst_var, obs, hcst



@APP.callback(
    Output("percentile_style", "style"),
    Output("threshold_style", "style"),
    Input("variable", "value")
)
def display_relevant_control(variable):

    displayed_style={
        "position": "relative",
        "width": "190px",
        "display": "flex",
        "padding": "10px",
        "vertical-align": "top",
    }
    if variable == "Percentile":
        style_percentile=displayed_style
        style_threshold={"display": "none"}
    else:
        style_percentile={"display": "none"}
        style_threshold=displayed_style
    return style_percentile, style_threshold


@APP.callback(
    Output("lead_time","options"),
    Output("lead_time","value"),
    Input("start_date","value"),
)
def target_range_options(start_date):
    leads_values = list(CONFIG["leads"].values())
    leads_keys = list(CONFIG["leads"])
    start_date = pd.to_datetime(start_date)
    leads_dict = {}
    for idx, lead in enumerate(leads_keys):
        target_range = predictions.target_range_format(
            leads_values[idx],
            leads_keys[idx],
            start_date,
            CONFIG["target_period_length"],
            CONFIG["time_units"],
        )
        leads_dict.update({lead:target_range})
    return leads_dict, list(CONFIG["leads"])[0]


@APP.callback(
   Output("map_title","children"),
   Input("start_date","value"),
   Input("lead_time","value"),
   Input("lead_time","options"),
)
def write_map_title(start_date, lead_time, lead_time_options):
    target_period = lead_time_options.get(lead_time)
    return f'{target_period} {CONFIG["variable"]} Forecast issued {start_date}'


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
    start_dates = predictions.cpt_starts_list(
        DATA_PATH,
        CONFIG["forecast_mu_file_pattern"],
        CONFIG["start_regex"],
        format_in=CONFIG["start_format_in"],
        format_out=CONFIG["start_format_out"],
    )
    fcst_mu = predictions.sel_cpt_file(
        DATA_PATH,
        CONFIG["forecast_mu_file_pattern"],
        list(CONFIG["leads"])[0],
        start_dates[-1],
        CONFIG["start_format_in"],
    )
    if dash.ctx.triggered_id == None:
        lat = fcst_mu["Y"][int(fcst_mu["Y"].size/2)].values
        lng = fcst_mu["X"][int(fcst_mu["X"].size/2)].values
    else:
        if dash.ctx.triggered_id == "map":
            lat = click_lat_lng[0]
            lng = click_lat_lng[1]
        else:
            lat = latitude
            lng = longitude
        try:
            nearest_grid = pingrid.sel_snap(fcst_mu, lat, lng)
            lat = nearest_grid["Y"].values
            lng = nearest_grid["X"].values
        except KeyError:
            lat = lat
            lng = lng
    return [lat, lng], lat, lng


@APP.callback(
    Output("cdf_graph", "figure"),
    Output("pdf_graph", "figure"),
    Input("loc_marker", "position"),
    Input("start_date","value"),
    Input("lead_time","value"),
)
def local_plots(marker_pos, start_date, lead_time):
    # Reading
    lat = marker_pos[0]
    lng = marker_pos[1]
    fcst_mu, fcst_var, obs, hcst = read_cptdataset(lead_time, start_date, y_transform=CONFIG["y_transform"])
    # Errors handling
    try:
        fcst_mu = pingrid.sel_snap(fcst_mu, lat, lng)
        fcst_var = pingrid.sel_snap(fcst_var, lat, lng)
        obs = pingrid.sel_snap(obs, lat, lng)
        isnan = np.isnan(fcst_mu).sum() + np.isnan(obs).sum()
        if CONFIG["y_transform"]:
            hcst = pingrid.sel_snap(hcst, lat, lng)
            isnan_yt = np.isnan(hcst).sum()
            isnan = isnan + isnan_yt
        if isnan > 0:
            error_fig = pingrid.error_fig(error_msg="Data missing at this location")
            return error_fig, error_fig
    except KeyError:
        error_fig = pingrid.error_fig(error_msg="Grid box out of data domain")
        return error_fig, error_fig

    target_range = predictions.target_range_format(
        CONFIG["leads"][lead_time],
        lead_time,
        pd.to_datetime(start_date),
        CONFIG["target_period_length"],
        CONFIG["time_units"],
    )
    # CDF from 499 quantiles
    quantiles = np.arange(1, 500) / 500
    quantiles = xr.DataArray(
        quantiles, dims="percentile", coords={"percentile": quantiles}
    )

    # Obs CDF
    obs_q, obs_mu = xr.broadcast(quantiles, obs.mean(dim="T"))
    obs_stddev = obs.std(dim="T")
    obs_ppf = xr.apply_ufunc(
        norm.ppf,
        obs_q,
        kwargs={"loc": obs_mu, "scale": obs_stddev},
    ).rename("obs_ppf")
    # Obs quantiles
    obs_quant = obs.quantile(quantiles, dim="T")

    # Forecast CDF
    fcst_q, fcst_mu = xr.broadcast(quantiles, fcst_mu)
    try:
        fcst_dof = int(fcst_var.attrs["dof"])
    except:
        fcst_dof = obs["T"].size - 1
    if CONFIG["y_transform"]:
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
        # fcst variance is hindcast variance weighted by (1+xvp)
        # but data files don't have xvp neither can we recompute it from them
        # thus xvp=0 is an approximation, acceptable dixit Simon Mason
        # The line below is thus just a reminder of the above
        xvp = 0
        fcst_var = hcst_err_var * (1 + xvp)
    fcst_ppf = xr.apply_ufunc(
        t.ppf,
        fcst_q,
        fcst_dof,
        kwargs={
            "loc": fcst_mu,
            "scale": np.sqrt(fcst_var),
        },
    ).rename("fcst_ppf")
    poe = fcst_ppf["percentile"] * -1 + 1
    # Graph for CDF
    cdf_graph = pgo.Figure()
    cdf_graph.add_trace(
        pgo.Scatter(
            x=fcst_ppf.squeeze().values,
            y=poe,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.values,
            y=poe,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs (parametric)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_quant.values,
            y=poe,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs (empirical)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    if CONFIG["time_units"] == "days":
        start_date_pretty = (pd.to_datetime(start_date)).strftime("%-d %b %Y")
    else:
        start_date_pretty = (pd.to_datetime(start_date)).strftime("%b %Y")
    cdf_graph.update_layout(
        xaxis_title=f'{CONFIG["variable"]} ({fcst_mu.attrs["units"]})',
        yaxis_title="Probability of exceeding",
        title={
            "text": f'{target_range} forecast issued {start_date_pretty} <br> at ({fcst_mu["Y"].values}N,{fcst_mu["X"].values}E)',
            "font": dict(size=14),
        },
    )

    # PDF from 499 ppf values

    fcst_pdf = xr.apply_ufunc(
        t.pdf,
        fcst_ppf,
        fcst_dof,
        kwargs={
            "loc": fcst_mu,
            "scale": np.sqrt(fcst_var),
        },
    ).rename("fcst_pdf")

    obs_pdf = xr.apply_ufunc(
        norm.pdf,
        obs_ppf,
        kwargs={"loc": obs_mu, "scale": obs_stddev},
    ).rename("obs_pdf")
    # Graph for PDF
    pdf_graph = pgo.Figure()
    pdf_graph.add_trace(
        pgo.Scatter(
            x=fcst_ppf.squeeze().values,
            y=fcst_pdf.squeeze().values,
            customdata=poe,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    pdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.values,
            y=obs_pdf.values,
            customdata=poe,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.attrs["units"],
            name="obs",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    pdf_graph.update_traces(mode="lines", connectgaps=False)
    pdf_graph.update_layout(
        xaxis_title=f'{CONFIG["variable"]} ({fcst_mu.attrs["units"]})',
        yaxis_title="Probability density",
        title={
            "text": f'{target_range} forecast issued {start_date_pretty} <br> at ({fcst_mu["Y"].values}N,{fcst_mu["X"].values}E)',
            "font": dict(size=14),
        },
    )
    return cdf_graph, pdf_graph


@APP.callback(
    Output("fcst_colorbar", "colorscale"),
    Input("proba", "value"),
    Input("variable", "value"),
    Input("percentile", "value")
)
def draw_colorbar(proba, variable, percentile):

    fcst_cdf = xr.DataArray()
    if variable == "Percentile":
        if proba == "exceeding":
            percentile = 1 - percentile
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_POE_COLORMAP
        else:
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_PNE_COLORMAP
    else:
        fcst_cdf.attrs["colormap"] = pingrid.CORRELATION_COLORMAP
    fcst_cs = pingrid.to_dash_colorscale(fcst_cdf.attrs["colormap"])
    return fcst_cs


@APP.callback(
    Output("fcst_layer", "url"),
    Output("forecast_warning", "is_open"),
    Input("proba", "value"),
    Input("variable", "value"),
    Input("percentile", "value"),
    Input("threshold", "value"),
    Input("start_date","value"),
    Input("lead_time","value")
)
def fcst_tile_url_callback(proba, variable, percentile, threshold, start_date, lead_time):

    try:
        if variable != "Percentile":
            if threshold is None:
                return "", True
            else:
                return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/{float(threshold)}/{start_date}/{lead_time}", False
        else:
            return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/0.0/{start_date}/{lead_time}", False
    except:
        return "", True


# Endpoints

@SERVER.route(
    f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float(signed=True):threshold>/<start_date>/<lead_time>"
)
def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold, start_date,lead_time):
    # Reading
    fcst_mu, fcst_var, obs, hcst = read_cptdataset(lead_time, start_date, y_transform=CONFIG["y_transform"])

    # Obs CDF
    if variable == "Percentile":
        obs_mu = obs.mean(dim="T")
        obs_stddev = obs.std(dim="T")
        obs_ppf = xr.apply_ufunc(
            norm.ppf,
            percentile,
            kwargs={"loc": obs_mu, "scale": obs_stddev},
        )
    else:
        obs_ppf = threshold
    # Forecast CDF
    try:
        fcst_dof = int(fcst_var.attrs["dof"])
    except:
        fcst_dof = obs["T"].size - 1
    if CONFIG["y_transform"]:
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
        # fcst variance is hindcast variance weighted by (1+xvp)
        # but data files don't have xvp neither can we recompute it from them
        # thus xvp=0 is an approximation, acceptable dixit Simon Mason
        # The line below is thus just a reminder of the above
        xvp = 0
        fcst_var = hcst_err_var * (1 + xvp)

    fcst_cdf = xr.DataArray( # pingrid.tile expects a xr.DA but obs_ppf is never that
        data = xr.apply_ufunc(
            t.cdf,
            obs_ppf,
            fcst_dof,
            kwargs={
                "loc": fcst_mu,
                "scale": np.sqrt(fcst_var),
            },
        ),
        # Naming conventions for pingrid.tile
        coords = fcst_mu.rename({"X": "lon", "Y": "lat"}).coords,
        dims = fcst_mu.rename({"X": "lon", "Y": "lat"}).dims
    # pingrid.tile wants 2D data
    ).squeeze("T")
    if "S" in fcst_cdf.dims:
        fcst_cdf = fcst_cdf.squeeze("S")
    # Depending on choices:
    # probabilities symmetry around 0.5
    # choice of colorscale (dry to wet, wet to dry, or correlation)
    # translation of "near normal" to
    if variable == "Percentile":
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
            percentile = 1 - percentile
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_POE_COLORMAP
        else:
            fcst_cdf.attrs["colormap"] = pingrid.RAIN_PNE_COLORMAP
    else:
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
        fcst_cdf.attrs["colormap"] = pingrid.CORRELATION_COLORMAP
    fcst_cdf.attrs["scale_min"] = 0
    fcst_cdf.attrs["scale_max"] = 1
    clipping = None
    resp = pingrid.tile(fcst_cdf, tx, ty, tz, clipping)
    return resp


if __name__ == "__main__":
    import warnings
    warnings.simplefilter('error')
    APP.run_server(CONFIG["server"], CONFIG["port"], debug=CONFIG["mode"] != "prod")
