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
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np
from pycpt import Report
import xarray as xr
from scipy.stats import t, norm, rankdata

CONFIG = pyaconf.load(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]

# App

CONFIG = pyaconf.load(os.environ["CONFIG"])

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
        {"name": "description", "content": "Seasonal Forecast"},
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
APP.title = "Seasonal Forecast"

APP.layout = layout.app_layout()


# Reading functions

def read_cpt_outputs(outputs_path, mos_config):
    results_path = outputs_path
    result = Report(Path(results_path), cpt_config={"MOS": mos_config})
    result.open(CONFIG["forecast_mu_file"])
    fcst_mu = result.forecast_mu
    fcst_mu_name = list(fcst_mu.keys())[0]
    fcst_mu = fcst_mu[fcst_mu_name]
    result.open(CONFIG["forecast_var_file"])
    fcst_var = result.forecast_variance
    fcst_var_name = list(fcst_var.keys())[0]
    fcst_var = fcst_var[fcst_var_name]
    result.open(CONFIG["obs_file"])
    obs = result.predictand
    obs_name = list(obs.keys())[0]
    obs = obs[obs_name]
    result.open(CONFIG["xccamap_file"])
    xccamap = result.xccamap
    xccamap_name = list(xccamap.keys())[0]
    xccamap = xccamap[xccamap_name]
    return fcst_mu, fcst_mu_name, fcst_var, fcst_var_name, obs, obs_name, xccamap, xccamap_name


def read_cpt_outputs_y_transform(outputs_path, mos_config):
    results_path = outputs_path
    result = Report(Path(results_path), cpt_config={"MOS": mos_config})
    result.open(CONFIG["hcst_file"])
    hcst = result.crossvalidated_hindcast_values
    hcst_name = list(hcst.keys())[0]
    hcst = hcst[hcst_name]
    result.open(CONFIG["ccacorr_file"])
    ccacorr = result.ccacorr
    ccacorr_name = list(ccacorr.keys())[0]
    ccacorr = ccacorr[ccacorr_name]
    result.open(CONFIG["fcst_pred_file"])
    fcst_pred = result.original_fcst_predictor
    fcst_pred_name = list(fcst_pred.keys())[0]
    fcst_pred = fcst_pred[fcst_pred_name]
    return hcst, hcst_name, ccacorr, ccacorr_name, fcst_pred, fcst_pred_name


@APP.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


def get_coords(click_lat_lng):
    if click_lat_lng is not None:
        return click_lat_lng
    else:
        return [layout.INIT_LAT, layout.INIT_LNG]


def round_latLng(coord):
    value = float(coord)
    value = round(value, 1)
    return value


@APP.callback(Output("layers_group", "children"), Input("map", "click_lat_lng"))
def map_click(click_lat_lng):
    lat_lng = get_coords(click_lat_lng)

    return dlf.Marker(
        position=lat_lng, children=dlf.Tooltip("({:.3f}, {:.3f})".format(*lat_lng))
    )


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
    Output("cdf_graph", "figure"),
    Output("pdf_graph", "figure"),
    Input("map", "click_lat_lng"),
)
def local_plots(click_lat_lng):

    lat, lng = get_coords(click_lat_lng)

    # Reading
    (fcst_mu, fcst_mu_name, fcst_var, fcst_var_name, obs, obs_name, xccamap, xccamap_name) = read_cpt_outputs(
        CONFIG["results_path"], CONFIG["cpt_mos"]
    )
    if CONFIG["y_transform"]:
        hcst, hcst_name, ccacorr, ccacorr_name, fcst_pred, fcst_pred_name = read_cpt_outputs_y_transform(
            CONFIG["results_path"], CONFIG["cpt_mos"]
        )

    # Spatial Tolerance for lat/lon selection clicking on map
    half_res = (fcst_mu["X"][1] - fcst_mu["X"][0]) / 2
    tol = np.sqrt(2 * np.square(half_res)) 

    # Errors handling
    try:
        isnan = (np.isnan(fcst_mu.sel(
            X=lng, Y=lat, method="nearest", tolerance=tol.values
        )).sum()) + (np.isnan(obs.sel(
            X=lng, Y=lat, method="nearest", tolerance=tol.values
        )).sum()) + (np.isnan(xccamap.sel(
            X=lng, Y=lat, method="nearest", tolerance=tol.values
        )).sum())
        if CONFIG["y_transform"]:
            isnan_yt = (np.isnan(hcst.sel(
                X=lng, Y=lat, method="nearest", tolerance=tol.values
            )).sum()) + (np.isnan(fcst_pred.sel(
                X=lng, Y=lat, method="nearest", tolerance=tol.values
            )).sum())
            isnan = isnan + isnan_yt
        if isnan > 0:
            errorFig = pgo.Figure().add_annotation(
                x=2,
                y=2,
                text="Data missing at this location",
                font=dict(family="sans serif", size=30, color="crimson"),
                showarrow=False,
                yshift=10,
                xshift=60,
            )
            return errorFig, errorFig
    except KeyError:
        errorFig = pgo.Figure().add_annotation(
            x=2,
            y=2,
            text="Grid box out of data domain",
            font=dict(family="sans serif", size=30, color="crimson"),
            showarrow=False,
            yshift=10,
            xshift=60,
        )
        return errorFig, errorFig

    fcst_mu = fcst_mu.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    fcst_var = fcst_var.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    obs = obs.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
    xccamap = xccamap.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)

    # Get Issue month and Target season
    ts = fcst_mu.attrs["T"]
    im = (
        (
            fcst_mu["T"]
            - np.timedelta64(int(CONFIG["lead"]), "M").astype("timedelta64[ns]")
        )
        .dt.strftime("%b %Y")
        .values[0]
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
    fcst_dof = obs["T"].size - xccamap["Mode"].size - 1
    # Y transform correction
    if CONFIG["y_transform"]:
        hcst = hcst.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
        fcst_pred = fcst_pred.sel(X=lng, Y=lat, method="nearest", tolerance=tol.values)
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
        xvp = (1 / obs["T"].size) + np.square(ccacorr * xccamap * fcst_pred).sum(dim="Mode")
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

    # Graph for CDF
    cdf_graph = pgo.Figure()
    cdf_graph.add_trace(
        pgo.Scatter(
            x=fcst_ppf.where(lambda x: x >= 0).squeeze().values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.units,
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.where(lambda x: x >= 0).values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.units,
            name="obs (parametric)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.add_trace(
        pgo.Scatter(
            x=obs_quant.values,
            y=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{y:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.units,
            name="obs (empirical)",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    cdf_graph.update_traces(mode="lines", connectgaps=False)
    cdf_graph.update_layout(
        xaxis_title=fcst_mu_name + " " + "(" + fcst_mu.units + ")",
        yaxis_title="Probability of exceeding",
        title={
            "text": f"{ts} forecast issued {im} at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
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
            x=fcst_ppf.where(lambda x: x >= 0).squeeze().values,
            y=fcst_pdf.squeeze().values,
            customdata=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.units,
            name="forecast",
            line=pgo.scatter.Line(color="red"),
        )
    )
    pdf_graph.add_trace(
        pgo.Scatter(
            x=obs_ppf.where(lambda x: x >= 0).values,
            y=obs_pdf.values,
            customdata=fcst_ppf["percentile"] * -1 + 1,
            hovertemplate="%{customdata:.0%} chance of exceeding"
            + "<br>%{x:.1f} "
            + fcst_mu.units,
            name="obs",
            line=pgo.scatter.Line(color="blue"),
        )
    )
    pdf_graph.update_traces(mode="lines", connectgaps=False)
    pdf_graph.update_layout(
        xaxis_title=fcst_mu_name + " " + "(" + fcst_mu.units + ")",
        yaxis_title="Probability density",
        title={
            "text": f"{ts} forecast issued {im} at ({round_latLng(lat)}N,{round_latLng(lng)}E)",
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
            fcst_cdf.attrs["colormap"] = CONFIG["colormap"]
        else:
            fcst_cdf.attrs["colormap"] = CONFIG["colormapflipped"]
        thresholds = np.array([
            0,
            (percentile-0.05)/3,
            2*(percentile-0.05)/3,
            percentile-0.05,
            percentile-0.05+1/256.,
            percentile+0.05-1/256.,
            percentile+0.05,
            percentile+0.05+(1-(percentile+0.05))/3,
            percentile+0.05+2*(1-(percentile+0.05))/3,
            1
        ])
    else:
        fcst_cdf.attrs["colormap"] = CONFIG["colormapcorrel"]
        thresholds = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
    fcst_cs = pingrid.to_dash_colorscale(fcst_cdf.attrs["colormap"], thresholds=thresholds)
    return fcst_cs


@APP.callback(
    Output("fcst_layer", "url"),
    Output("forecast_warning", "is_open"),
    Input("proba", "value"),
    Input("variable", "value"),
    Input("percentile", "value"),
    Input("threshold", "value")
)
def fcst_tile_url_callback(proba, variable, percentile, threshold):

    try:
        if variable != "Percentile":
            if threshold is None:
                return "", True
            else:
                return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/{float(threshold)}", False
        else:
            return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/90.0", False
    except:
        return "", True


# Endpoints


@SERVER.route(
    f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float:threshold>"
)
def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold):

    # Reading
    (fcst_mu, fcst_mu_name, fcst_var, fcst_var_name, obs, obs_name, xccamap, xccamap_name) = read_cpt_outputs(
        CONFIG["results_path"], CONFIG["cpt_mos"]
    )

    # Get Issue month and Target season
    ts = fcst_mu.attrs["T"]
    im = (
        (
            fcst_mu["T"]
            - np.timedelta64(int(CONFIG["lead"]), "M").astype("timedelta64[ns]")
        )
        .dt.strftime("%b %Y")
        .values[0]
    )

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
    fcst_dof = obs["T"].size - xccamap["Mode"].size - 1
    # Y transform correction
    if CONFIG["y_transform"]:
        hcst, hcst_name, ccacorr, ccacorr_name, fcst_pred, fcst_pred_name = read_cpt_outputs_y_transform(CONFIG["results_path"], CONFIG["cpt_mos"])
        hcst_err_var = (np.square(obs - hcst).sum(dim="T")) / fcst_dof
        xvp = (1 / obs["T"].size) + np.square(ccacorr * xccamap * fcst_pred).sum(dim="Mode")
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
    # Depending on choices:
    # probabilities symmetry around 0.5
    # choice of colorscale (dry to wet, wet to dry, or correlation)
    # translation of "near normal" to
    if variable == "Percentile":
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
            percentile = 1 - percentile
            fcst_cdf.attrs["colormap"] = CONFIG["colormap"]
        else:
            fcst_cdf.attrs["colormap"] = CONFIG["colormapflipped"]
        fcst_cdf.attrs["colormapkey"] = np.array([
            0,
            (percentile-0.05)/3,
            2*(percentile-0.05)/3,
            percentile-0.05,
            percentile-0.05+1/256.,
            percentile+0.05-1/256.,
            percentile+0.05,
            percentile+0.05+(1-(percentile+0.05))/3,
            percentile+0.05+2*(1-(percentile+0.05))/3,
            1
        ])
    else:
        if proba == "exceeding":
            fcst_cdf = 1 - fcst_cdf
        fcst_cdf.attrs["colormap"] = CONFIG["colormapcorrel"]
        fcst_cdf.attrs["colormapkey"] = np.array([0, 0.1, 0.2, 0.35, 0.45, 0.45+1/256., 0.55-1/256., 0.55, 0.7, 0.85, 1])
    fcst_cdf.attrs["scale_min"] = 0
    fcst_cdf.attrs["scale_max"] = 1
    clipping = None
    resp = pingrid.tile(fcst_cdf, tx, ty, tz, clipping)
    return resp


if __name__ == "__main__":
    APP.run_server(CONFIG["server"], CONFIG["port"], debug=CONFIG["mode"] != "prod")
