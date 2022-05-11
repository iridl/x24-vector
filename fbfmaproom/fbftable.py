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

def gen_table(summary, cols, data):
    return html.Table(
        [
            gen_head(summary, cols),
            gen_body(data)
        ], className="supertable", style={"overflow":"auto", "height": 700, "display":"block"}
    )

def gen_header(el):
    if el['dynamic'] is None:
        return html.Th(el['name'])
    else:
        return html.Th(html.Select(
            [
                html.Option(i['label'], value=i['value'],
                            selected="selected" if i['value'] == el['dynamic']['value'] else "")
                for i in el['dynamic']['options']
            ],
            id=el['id']))



def gen_head(pre, cols, pre_tooltips=None, col_tooltips=None):
    if pre_tooltips is not None:
        assert len(pre) == len(pre_tooltips), "wrong number of tooltips"

    if col_tooltips is not None:
        assert len(col) == len(col_tooltips), "wrong number of tooltips"

    assert len(pre[0]) == len(cols), "pre-headers not same length as column headers"
    return html.Thead([
        html.Tr(
            [html.Th(pre[p][0])] + [html.Th(pre[p][c]) for c in range(1,len(pre[p]))]
        )
        for p in range(len(pre))
    ] + [ html.Tr(
        [ gen_header(c) for c in cols ]

    )
    ], style={"position": "sticky", "top": "0"})


def gen_body(data, style=None):
    if style is not None:
        assert all(callable(x) for x in style), "style is not an array of functions"
        # data is expected to be rectangular in row major order, so we test the
        # length of the first row to get the number of columns
        assert len(data[0]) == len(style), "style function array is wrong length"

    def sty(row, col):
        if style is not None:
            return style[col](data, row, col)
        else:
            return {}

    return html.Tbody([
        html.Tr([
            html.Td(data[row][col], style=sty(row, col))
            for col in range(len(data[row]))
        ])
        for row in range(len(data))
    ])
    # ], style={"height": "500px", "overflow-y": "scroll", "display": "block"})
