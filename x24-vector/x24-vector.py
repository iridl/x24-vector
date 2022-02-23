import sys
import json
import time
from io import BytesIO
import pandas as pd
import numpy as np
from PIL import Image
import requests
import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash.dependencies import Output, Input


csv_path = sys.argv[1]


# Data


crops = ["barley", "maize", "sorghum", "teff", "wheat"]

categories = ["Complete Failure", "Poor", "Mediocre", "Average", "Good", "Very Good"]
colorscale = ["#fffd38", "#81fa30", "#29fd2f", "#3ccb3e", "#288a2a", "#1b703f"]

df = pd.read_csv(csv_path)
df["tooltip"] = df.apply(
    lambda d: "({:.3f}, {:.3f}), {}, {}, {}".format(
        d["Lon"], d["Lat"], d["Region"], d["Woreda"], d["Kebele"]
    ),
    axis=1,
)
# d["popup"] = d["Region"]
print(df.info())

regions = df["Region"].unique()
expansions = df["Expansion or Current"].unique()


def filtered_data(df, regions, expansions):
    df_filtered = df[
        df["Region"].isin(regions) & df["Expansion or Current"].isin(expansions)
    ]
    dicts = df_filtered.to_dict("rows")
    geojson = dlx.dicts_to_geojson(dicts, lon="Lon", lat="Lat")
    geobuf = dlx.geojson_to_geobuf(geojson)
    return geobuf


# Layout

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname="/x24-vector/")


app.layout = html.Div(
    [
        dl.Map(
            [
                dl.LayersControl(
                    [
                        dl.BaseLayer(
                            dl.TileLayer(
                                url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png"
                            ),
                            name="street",
                            checked=True,
                        ),
                        dl.BaseLayer(
                            dl.TileLayer(
                                url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
                            ),
                            name="topo",
                            checked=False,
                        ),
                        dl.Overlay(
                            dl.ImageOverlay(
                                url="",
                                bounds=[[3.3, 32.6], [15.2, 48.1]],
                                id="img",
                            ),
                            name="wrsi",
                            checked=True,
                        ),
                        dl.Overlay(
                            dl.GeoJSON(
                                format="geobuf",
                                cluster=True,
                                id="points",
                                zoomToBoundsOnClick=True,
                                superClusterOptions={"radius": 100},
                            ),
                            name="points",
                            checked=True,
                        ),
                    ],
                    position="topleft",
                ),
                dlx.categorical_colorbar(
                    id="colorbar",
                    categories=categories,
                    colorscale=colorscale,
                    width=300,
                    height=30,
                    position="bottomleft",
                ),
                dl.ScaleControl(imperial=False),
            ],
            center=(8, 40),
            zoom=6,
            style={"width": "100%", "height": "100%", "position": "absolute"},
        ),
        html.Div(
            [
                html.H2("—Maproom x24.0"),
                html.Label("Crops:"),
                dcc.Dropdown(
                    options=[dict(label=v.capitalize(), value=v) for v in crops],
                    value="barley",
                    id="crops",
                    clearable=True,
                ),
                html.Br(),
                html.Label("Opacity:"),
                dcc.Slider(
                    id="opacity",
                    min=0.25,
                    max=1.0,
                    step=0.05,
                    marks={0.25: "25%", 0.5: "50%", 0.75: "75%", 1: "100%"},
                    value=0.5,
                ),
                html.Br(),
                html.Label("Regions:"),
                dcc.Checklist(
                    id="regions",
                    options=[dict(label=v, value=v) for v in regions],
                    value=regions,
                    labelStyle={"display": "block"},
                ),
                html.Br(),
                html.Label("Expansion:"),
                dcc.Checklist(
                    id="expansions",
                    options=[dict(label=v, value=v) for v in expansions],
                    value=expansions,
                    labelStyle={"display": "block"},
                ),
                html.Br(),
                html.Label("Point:"),
                html.Div(id="text", style={"margin": "3px"}),
            ],
            id="info",
            className="info",
            style={
                "position": "absolute",
                "top": "10px",
                "right": "10px",
                "z-index": "1000",
                "height": "80%",
                "width": "300px",
                "pointer-events": "auto",
                "padding-left": "25px",
                "padding-right": "25px",
            },
        ),
    ]
)


# Callbacks


@app.callback(Output("text", "children"), [Input("points", "hover_feature")])
def info_hover(feature):
    rs = []
    if feature is not None and not feature["properties"]["cluster"]:
        cs = feature["geometry"]["coordinates"]
        rs.append(html.B("Lon"))
        rs.append(": ")
        rs.append(str(round(cs[0], 3)))
        rs.append(", ")
        rs.append(html.B("Lat"))
        rs.append(": ")
        rs.append(str(round(cs[1], 3)))
        rs.append(html.Br())
        for k, v in feature["properties"].items():
            if v is not None and k not in ("cluster", "tooltip"):
                rs.append(html.B(k.capitalize()))
                rs.append(": ")
                rs.append(v)
                rs.append(html.Br())
        time.sleep(1)
    else:
        rs.append(html.I("Hover over a marker."))
    return rs


@app.callback(Output("img", "url"), [Input("crops", "value")])
def update1(crop):
    return f"/datalibrary/{crop}"


@app.callback(
    [Output("img", "opacity"), Output("colorbar", "opacity")],
    [Input("opacity", "value")],
)
def update2(opacity):
    return opacity, opacity


@app.callback(
    Output("points", "data"), [Input("regions", "value"), Input("expansions", "value")]
)
def update3(regions, expansions):
    return filtered_data(df, regions, expansions)


# Endpoints


@server.route(f"/datalibrary/<crop>")
def datalibrary(crop):
    r = requests.get(
        f"http://iridl.ldeo.columbia.edu/home/.remic/.Leap/.WRSI/.Meher/.FinalIcat/Crop/({crop.capitalize()})/VALUE/X/Y/fig-/colors/-fig//XOVY/1/psdef//plotaxislength/5000/psdef//plotborder/0/psdef/.png"
    )
    f_in = BytesIO(r.content)
    img = Image.open(f_in)
    img = img.convert("RGBA")
    imgnp = np.array(img)
    white = np.sum(imgnp[:, :, :3], axis=2)
    white_mask = np.where(white == 255 * 3, 1, 0)
    alpha = np.where(white_mask, 0, imgnp[:, :, -1])
    imgnp[:, :, -1] = alpha
    img = Image.fromarray(np.uint8(imgnp))
    f_out = BytesIO()
    img.save(f_out, "PNG")
    f_out.seek(0)
    return flask.send_file(f_out, mimetype="image/png")


if __name__ == "__main__":
    app.run_server()
