import flask
import os

from globals_ import FLASK, GLOBAL_CONFIG
import pingrid

template = '''
{% for item in maprooms %}
<a href="{{item["path"]}}">{{item["title"]}}</a><br>
{% endfor %}
'''

maprooms = []

for name, config in GLOBAL_CONFIG["maprooms"].items():
    if config is not None:
        try:
            one_config = {
                "title": config['title'],
                "path": f'{GLOBAL_CONFIG["url_path_prefix"]}{config["core_path"]}',
            }
        except KeyError as e:
            raise Exception(f'configuration of maproom "{name}" is incomplete') from e
        maprooms.append(one_config)


@FLASK.route(GLOBAL_CONFIG["url_path_prefix"] + "/")
def homepage():
    return flask.render_template_string(template, maprooms=maprooms)
