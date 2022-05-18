from dash import Dash, dcc, html
from dash.dependencies import Output, Input, State, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import uuid
from collections import OrderedDict

def gen_table(tcs, dfs, data):
    return html.Table(
        [
            gen_head(tcs, dfs),
            gen_body(tcs, data)
        ], className="supertable", style={"overflow":"auto", "height": 700, "display":"block"}
    )

def head_cell(text, tool=None):
    if tool is not None:
        obj_id = "target-" + str(uuid.uuid4())
        return [ html.Div(text, id=obj_id),
                 dbc.Tooltip(tool, target=obj_id, className="tooltiptext") ]
    else:
        return text


def gen_header(el):
    return html.Th(head_cell(el['name'], el['tooltip']))
    # if el['dynamic'] is None:
    #     return html.Th(el['name'])
    # else:
    #     return html.Th(html.Select(
    #         [
    #             html.Option(v, k, selected=k == el['dynamic']['value'])
    #             for k, v in el['dynamic']['options'].items()
    #         ],
    #         id=el['id']))



def gen_head(tcs, dfs):
    return html.Thead([
        html.Tr(
            [ html.Th(head_cell(row[col], row['tooltip']) if i == 0 else row[col])
              for i, col in enumerate(tcs.keys()) ]
        )
        for row in dfs.to_dict(orient="records")
    ] + [ html.Tr(
        [ gen_header(c) for c in tcs.values() ]

    )
    ], style={"position": "sticky", "top": "0"})


def gen_body(tcs, data):
    def style(col, row):
        sty = tcs[col]['style']
        if sty is not None:
            assert callable(sty), f"column {col} style field is not a function"
            return sty(row)
        else:
            return ""

    return html.Tbody([
        html.Tr([
            html.Td(row[col], className=style(col, row)) for col in tcs.keys()
        ])
        for row in data.to_dict(orient="records")
    ])
