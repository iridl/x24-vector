import cftime
from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import datetime
import uuid
from collections import OrderedDict

def gen_table(tcs, dfs, data, thresholds, severity, final_season):
    return html.Table(
        [
            gen_head(tcs, dfs),
            gen_body(tcs, data, thresholds, severity, final_season)
        ], className="supertable"
    )

def head_cell(child, tool=None):
    if tool is not None:
        obj_id = "target-" + str(uuid.uuid4())
        return [ html.Div(child, id=obj_id),
                 dbc.Tooltip(tool, target=obj_id, className="tooltiptext") ]
    else:
        return child

def gen_select_header(col, options, value):
    return html.Select(
        [
            html.Option(v, k, selected=k == value)
            for k, v in options.items()
        ],
        id=col
    )

def gen_head(tcs, dfs):
    return html.Thead(
        [
            html.Tr([
                html.Th(head_cell(
                    c['name'] + (f" ({c['units']})" if c.get('units') else ''),
                    c['tooltip']
                )) for c in tcs.values()
            ])
        ] +
        [
            html.Tr([
                html.Th(row[col]) for col in tcs.keys()
            ])
            for row in dfs.to_dict(orient="records")
        ] 
    )


def gen_body(tcs, data, thresholds, severity, final_season):

    def fmt(col, row):
        f = tcs[col].get('format', lambda x: x)
        return f(row[col])

    def class_name(col_name, row):
        return cell_class(
            col_name, row, severity, thresholds.get(col_name),
            tcs[col_name].get('lower_is_worse'), final_season
        )

    return html.Tbody([
        html.Tr([
            html.Td(fmt(col, row), className=class_name(col, row)) for col in tcs.keys()
        ])
        for row in data.to_dict(orient="records")
    ])


def cell_class(col_name, row, severity, thresh, lower_is_worse, final_season):
    val = row[col_name]
    highlight = (
        (thresh is not None) and
        (
            (lower_is_worse and val <= thresh) or
            (not lower_is_worse and val >= thresh)
        )
    )
    excluded = (
        final_season is not None and
        row['time'] > final_season
    )
    if not highlight and not excluded:
        cls = ''
    elif not highlight and excluded:
        cls = 'cell-excluded'
    elif highlight and not excluded:
        cls = f'cell-severity-{severity}'
    elif highlight and excluded:
        cls = f'cell-excluded-severity-{severity}'
    else:
        assert False

    return cls
