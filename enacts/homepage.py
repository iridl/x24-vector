import flask
import os

from globals_ import FLASK, GLOBAL_CONFIG
import pingrid

template = '''
{% for item in maprooms %}
<a href="{{item["path"]}}">{{item["title"]}}</a><br>
{% endfor %}
'''

maprooms = [
    {
        "title": config['title'],
        "path": f'{GLOBAL_CONFIG["url_path_prefix"]}{config["core_path"]}',
    }
    for config in GLOBAL_CONFIG["maprooms"].values()
    if config is not None
]

@FLASK.route(GLOBAL_CONFIG["url_path_prefix"] + "/")
def homepage():
    return flask.render_template_string(template, maprooms=maprooms)
