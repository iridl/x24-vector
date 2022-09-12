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
import pyaconf
import pingrid
import layout
import plotly.graph_objects as pgo
import plotly.express as px
import numpy as np
import cptio
import xarray as xr
from scipy.stats import t, norm, rankdata
import pandas as pd

CONFIG = pyaconf.load(os.environ["CONFIG"])

PFX = CONFIG["core_path"]
TILE_PFX = CONFIG["tile_path"]
ADMIN_PFX = CONFIG["admin_path"]
DATA_PATH = CONFIG["results_path"]

# App

CONFIG = pyaconf.load(os.environ["CONFIG"])

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

def selFile(dataPath, filePattern, leadTime, startDate):
    filesNameList = glob.glob(f'{dataPath}/{filePattern}')
    pattern = f"{startDate}_{leadTime}"
    for idx, i in enumerate(filesNameList):
        x = re.search(f"{pattern}",filesNameList[idx])
        if x:
            #x = x.group()
            fileName = filesNameList[idx]
            fileSelected = cptio.open_cptdataset(fileName)
            break
    startDT = datetime.strptime(startDate, "%b-%d-%Y")
    fileSelected = fileSelected.expand_dims({"S":[startDT]})
    return fileSelected #string name of the full file path

def read_cptdataset(leadTime, startDate, y_transform=CONFIG["y_transform"]): #add leadTime and startDate as inputs
    fcst_mu = selFile(DATA_PATH, CONFIG["forecast_mu_filePattern"], leadTime, startDate)
    fcst_mu_name = list(fcst_mu.data_vars)[0]
    fcst_mu = fcst_mu[fcst_mu_name]
    fcst_var = selFile(DATA_PATH, CONFIG["forecast_var_filePattern"], leadTime, startDate)
    fcst_var_name = list(fcst_var.data_vars)[0]
    fcst_var = fcst_var[fcst_var_name]
    obs = (selFile(DATA_PATH, CONFIG["obs_filePattern"], leadTime, startDate)).squeeze()
    obs_name = list(obs.data_vars)[0]
    obs = obs[obs_name]
    if y_transform:
        hcst = (selFile(DATA_PATH, CONFIG["hcst_filePattern"], leadTime, startDate)).squeeze()
        hcst_name = list(hcst.data_vars)[0]
        hcst = hcst[hcst_name]
    else:
        hcst=None
    return fcst_mu, fcst_var, obs, hcst

def getTargets(issueDate, leadTime):
    # Get Issue date and Target season
    issue_date_td = pd.to_datetime(issueDate) #fcst_var["S"].values
    issue_date = issue_date_td[0].strftime("%-d %b %Y")
    #for leads they are currently set to be the difference in days fron target_start to issue date
    if leadTime == "wk1":
        lead_time = 1
    elif leadTime == "wk2":
        lead_time = 8
    elif leadTime == "wk3":
        lead_time = 15
    elif leadTime == "wk4":
        lead_time = 22
    target_start = (issue_date_td + timedelta(days=lead_time))[0].strftime("%-d %b %Y")
    target_end = (issue_date_td + timedelta(days=(lead_time+CONFIG["target_period_length"])))[0].strftime("%-d %b %Y")

    return issue_date, target_start, target_end

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
    Output("map", "click_lat_lng"),
    Output("layers_group", "children"),
    Output("latInput", "value"),
    Output("lngInput", "value"),
    Input("submitLatLng","n_clicks"),
    Input("map", "click_lat_lng"),
    Input("startDate","value"),
    Input("leadTime","value"),
    State("latInput", "value"),
    State("lngInput", "value")
)
def local_plots(n_clicks, click_lat_lng, startDate, leadTime, latitude, longitude):
    # Reading
    fcst_mu, fcst_var, obs, hcst = read_cptdataset(leadTime, startDate, y_transform=CONFIG["y_transform"])
    if click_lat_lng is None: #Map was not clicked
        if n_clicks == 0: #Button was not clicked (that's landing page)
            lat = (fcst_mu.Y[0].values+fcst_mu.Y[-1].values)/2
            lng = (fcst_mu.X[0].values+fcst_mu.X[-1].values)/2
        else: #Button was clicked
            lat = latitude
            lng = longitude
    else: #Map was clicked
        lat = click_lat_lng[0]
        lng = click_lat_lng[1]
    # Errors handling
    try:
        half_res = (fcst_mu["X"][1] - fcst_mu["X"][0]) / 2
        tol = np.sqrt(2 * np.square(half_res)).values
        nearest_grid = fcst_mu.sel(X=lng, Y=lat, method="nearest", tolerance=tol)
        lat = nearest_grid.Y.values
        lng = nearest_grid.X.values
        fcst_mu = fcst_mu.sel(X=lng, Y=lat, method="nearest", tolerance=tol)
        fcst_var = fcst_var.sel(X=lng, Y=lat, method="nearest", tolerance=tol)
        obs = obs.sel(X=lng, Y=lat, method="nearest", tolerance=tol)
        isnan = np.isnan(fcst_mu).sum() + np.isnan(obs).sum()
        if CONFIG["y_transform"]:
            hcst = hcst.sel(X=lng, Y=lat, method="nearest", tolerance=tol)
            isnan_yt = np.isnan(hcst).sum()
            isnan = isnan + isnan_yt
        if isnan > 0:
            errorFig = pingrid.error_fig(error_msg="Data missing at this location")
            return errorFig, errorFig, None, dlf.Marker(position=[lat, lng]), lat, lng
    except KeyError:
        errorFig = pingrid.error_fig(error_msg="Grid box out of data domain")
        return errorFig, errorFig, None, dlf.Marker(position=[lat, lng]), lat, lng

    issue_date, target_start, target_end = getTargets(fcst_var["S"].values, leadTime)
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
    fcst_dof = int(fcst_var.attrs["dof"]) #int(dofVar)
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
    cdf_graph.update_layout(
        xaxis_title=f'{CONFIG["variable"]} ({fcst_mu.attrs["units"]})',
        yaxis_title="Probability of exceeding",
        title={
            "text": f"{target_start} - {target_end} forecast issued {issue_date} <br> at ({fcst_mu.Y.values}N,{fcst_mu.X.values}E)",
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
            "text": f"{target_start} - {target_end} forecast issued {issue_date} <br> at ({fcst_mu.Y.values}N,{fcst_mu.X.values}E)",
            "font": dict(size=14),
        },
    )
    return cdf_graph, pdf_graph, None, dlf.Marker(position=[lat, lng]), lat, lng


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
    Input("startDate","value"),
    Input("leadTime","value")
)
def fcst_tile_url_callback(proba, variable, percentile, threshold, startDate, leadTime):

    try:
        if variable != "Percentile":
            if threshold is None:
                return "", True
            else:
                return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/{float(threshold)}/{startDate}/{leadTime}", False
        else:
            return f"{TILE_PFX}/{{z}}/{{x}}/{{y}}/{proba}/{variable}/{percentile}/0.0/{startDate}/{leadTime}", False
    except:
        return "", True


# Endpoints

@SERVER.route(
    f"{TILE_PFX}/<int:tz>/<int:tx>/<int:ty>/<proba>/<variable>/<float:percentile>/<float(signed=True):threshold>/<startDate>/<leadTime>"
)
def fcst_tiles(tz, tx, ty, proba, variable, percentile, threshold, startDate,leadTime):
    # Reading
    fcst_mu, fcst_var, obs, hcst = read_cptdataset(leadTime, startDate, y_transform=CONFIG["y_transform"])

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
    fcst_dof = int(fcst_var.attrs["dof"])
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
