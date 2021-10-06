import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table as table

def Number(id, default, min=0, max=5):
    return [ dbc.Input(id=id, type="number", min=min, max=max,
                     bs_size="sm", className="my-1",debounce=True,  value=str(default)) ]

def Date(id, defaultDay, defaultMonth):
    return [
        dbc.Input(id=id + "day", type="number", min=1, max=31,
                  bs_size="sm", className="my-1", debounce=True, value=str(defaultDay)),
        dbc.Select(id=id + "month", value=defaultMonth, bs_size="sm", className="my-1",
                   options=[
                       {"label": "January", "value": "Jan"},
                       {"label": "February", "value": "Feb"},
                       {"label": "March", "value": "Mar"},
                       {"label": "April", "value": "Apr"},
                       {"label": "May", "value": "May"},
                       {"label": "June", "value": "Jun"},
                       {"label": "July", "value": "Jul"},
                       {"label": "August", "value": "Aug"},
                       {"label": "September", "value": "Sep"},
                       {"label": "October", "value": "Oct"},
                       {"label": "November", "value": "Nov"},
                       {"label": "December", "value": "Dec"},
                   ],
        )
    ]

def Units(id):
    return [
        dbc.Select(id=id, value="/percent", bs_size="sm", className="my-1",
                   options=[
                       {"label": "fraction", "value": "/unitless"},
                       {"label": "%", "value": "/percent"},
                   ],
        )
    ]

def Sentence(*elems):
    tail = (len(elems) % 2) == 1
    groups = []
    start = 0

    if not isinstance(elems[0], str):
        start = 1
        tail = (len(elems) % 2) == 0
        groups.append(dbc.FormGroup(elems[0], className="mr-2"))

    for i in range(start, len(elems) - (1 if tail else 0), 2):
        assert isinstance(elems[i], str)
        groups.append(
            dbc.FormGroup(
                [dbc.Label(elems[i], size="sm", className="mr-2")] + elems[i + 1],
                className="mr-2")
        )

    if tail:
        assert isinstance(elems[-1], str)
        groups.append(
            dbc.FormGroup([
                dbc.Label(elems[-1], size="sm", className="mr-2"),
            ], className="mr-2")
        )

    return dbc.Form(groups, inline=True)


def Block(title, *body):
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(body),
    ], className="mb-4")
