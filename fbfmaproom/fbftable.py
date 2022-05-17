import os
import time
import io
import datetime
import random
import urllib.parse
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr
import flask
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
import __about__ as about
import pyaconf
import dash_bootstrap_components as dbc
from collections import OrderedDict

def gen_table(tcs, summary, data):
    return html.Table(
        [
            gen_head(tcs, summary),
            gen_body(tcs, data)
        ], className="supertable", style={"overflow":"auto", "height": 700, "display":"block"}
    )

def gen_header(el):
    return html.Th(el['name'])
    # if el['dynamic'] is None:
    #     return html.Th(el['name'])
    # else:
    #     return html.Th(html.Select(
    #         [
    #             html.Option(v, k, selected=k == el['dynamic']['value'])
    #             for k, v in el['dynamic']['options'].items()
    #         ],
    #         id=el['id']))



def gen_head(tcs, summary, pre_tooltips=None, col_tooltips=None):
    # if pre_tooltips is not None:
    #     assert len(pre) == len(pre_tooltips), "wrong number of tooltips"

    # if col_tooltips is not None:
    #     assert len(col) == len(col_tooltips), "wrong number of tooltips"

    headers = summary[list(tcs.keys())].values[:-1]

    return html.Thead([
        html.Tr(
            [html.Th(headers[r][c]) for c in range(len(headers[r]))]
        )
        for r in range(len(headers))
    ] + [ html.Tr(
        [ gen_header(c) for c in tcs.values() ]

    )
    ], style={"position": "sticky", "top": "0"})


def gen_body(tcs, data):
    def Style(col, row):
        sty = tcs[col]['style']
        if sty is not None:
            assert callable(sty), f"column {col} style field is not a function"
            return sty(row)
        else:
            return ""

    return html.Tbody([
        html.Tr([
            html.Td(row[col], className=Style(col, row)) for col in tcs.keys()
        ])
        for row in data.to_dict(orient="records")
    ])
