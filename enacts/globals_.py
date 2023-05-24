import flask
import pingrid
import os

FLASK = flask.Flask("enactsmaproom")

GLOBAL_CONFIG = pingrid.load_config("config-defaults.yaml:" + os.environ["CONFIG"])
