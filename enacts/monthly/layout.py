"""
The `layout.py` file is used to def a layout function, which is called within the main maproom.py file where the maproom is run.

The `layout()` function includes any code which defines the layout of the maproom. It should not include any callbacks or directly reference
the data be loaded into the maproom.
"""

# Import libraries used
import os
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dlf
import os
import xarray as xr
import json
import pandas as pd

import psycopg2
from psycopg2 import sql
import shapely
from shapely import wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely import geometry

import pingrid
from . import ui_components # Import `ui_components.py` which has UI controls tailored to maprooms.

GLOBAL_CONFIG = pingrid.load_config(os.environ["CONFIG"])
CONFIG = GLOBAL_CONFIG["monthly"]

def get_shapes(query):
    with psycopg2.connect(**GLOBAL_CONFIG["db"]) as conn:
        s = sql.SQL(query)
        df = pd.read_sql(s, conn)

    df["the_geom"] = df["the_geom"].apply(lambda x: wkb.loads(x.tobytes()))
    df["the_geom"] = df["the_geom"].apply(
        # lambda x: ((x if isinstance(x, MultiPolygon) else MultiPolygon([x]))
        lambda x: (x
                   .simplify(0.01, preserve_topology=False))
    )
    shapes = df["the_geom"].apply(shapely.geometry.mapping)
    for i in df.index: #this adds the district layer as a label in the dict
        shapes[i]['label'] = df['label'][i]
    return {"features": shapes}


# Loading the geometries for the admin layers in the map
SHAP = {
    level['name']: get_shapes(level['sql'])
    for level in GLOBAL_CONFIG['shapes_adm']
}


def layout(): # Defining the function that will be called in the layout section of  `maproom.py`.
    return dbc.Container([ # The function will return the dash bootstrap container, and all of its contents.
       dbc.Row(html.H1(CONFIG["map_title"])), # First of two rows (horizontal) which is the title bar of the maproom.

       dbc.Row([ # second of two rows (horizontal), which contains the rest of the maproom (the map and controls column).

           dbc.Col( # Now we divide the second row into two columns. The first column contains the controls.
               [
         # Within the controls column, we add four `Blocks` which are the controls themselves.
               ui_components.Block( # This block has title 'Variable' and includes a dropdown with the options listed
                 "Variable", ui_components.Select("variable",["Rainfall","Maximum Temperature","Minimum Temperature"])
               ),
               ui_components.Block( # Block containing dropdown to select from months of the year.
                  "Month", ui_components.Month("mon0","jan")
               ),
           ], width=4), # End of first column; width defined here determines the width of the column.
           dbc.Col( # The second of two columns. This column contains the map.
                dbc.Container( # Container that holds the leaflet map element.
                    [ # This list holds all of the children elements within the container.
                        dlf.Map( # Dash leaflet map.
                             [
                                dlf.LayersControl(
                                    [
                                        dlf.BaseLayer(
                                            dlf.TileLayer(
                                                opacity=0.6,
                                                url="https://cartodb-basemaps-{s}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png", # Cartodb street map.
                                            ),
                                            name="Street",
                                            checked=True,
                                        ),
                                        dlf.BaseLayer(
                                            dlf.TileLayer(
                                                opacity=0.6,
                                                url="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png" # opentopomap topography map.
                                            ),
                                            name="Topo",
                                            checked=False,
                                        ),
                                        dlf.Overlay(
                                            dlf.TileLayer(
                                                id="map_raster",
                                            ),
                                            name="Raster",
                                            checked=True,
                                        ),
                                    ] + [
                                        dlf.Overlay(
                                           dlf.GeoJSON(
                                               data=v,
                                               options={
                                                  "fill": False,
                                                  "color": "black",
                                                  "weight": .5,
                                                  "fillOpacity": 0,
                                               },
                                           ),
                                           name=k,
                                           checked=True,
                                        )
                                        for k, v in SHAP.items()
                                    ],
                                    position="topleft", # Where the layers control button is placed.
                                    id="map_layers_control",
                                ),
                                 dlf.LayerGroup(dlf.Marker(id="loc_marker",position=GLOBAL_CONFIG['map_center']),id="marker_layer"),
                                dlf.ScaleControl(imperial=False, position="topright"), # Define scale bar
                                dlf.Colorbar( # Define map color bar
                                    id="map_colorbar",
                                    min=0,
                                    position="bottomleft",
                                    width=300,
                                    height=10,
                                    opacity=.7,
                                ),
                             ],
                             id="map", # Finishing defining the dlf Map element.
                             style={ # The css style applied to the map
                                 "width": "100%",
                                 "height": "50vh",
                             },
                             center=GLOBAL_CONFIG['map_center'], # Where the center of the map will be upon loading the maproom.
                             zoom=GLOBAL_CONFIG['zoom'],
                        ),
                        dbc.Spinner(dcc.Graph(id="plot"))
                    ],
                    fluid=True,
                    style={"padding": "0rem"},
                )
           ),
       ]),
    ])
