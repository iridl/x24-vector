import flask
import os

from flex_fcst import maproom as flex_fcst
from globals_ import FLASK
import homepage
from monthly import maproom as monthly
from onset import maproom as onset
import pingrid


CONFIG = pingrid.load_config(os.environ["CONFIG"])


@FLASK.route(f"/health")
def health_endpoint():
    return flask.jsonify({'status': 'healthy', 'name': 'python_maproom'})


if __name__ == "__main__":
    if CONFIG["mode"] != "prod":
        import warnings
        warnings.simplefilter("error")
        debug = True
    else:
        debug = False

    FLASK.run(
        CONFIG["dev_server_interface"],
        CONFIG["dev_server_port"],
        debug=debug,
        extra_files=os.environ["CONFIG"].split(":"),
        processes=CONFIG["dev_processes"],
        threaded=False,
    )
