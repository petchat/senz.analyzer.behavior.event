# coding: utf-8

from leancloud import Engine
from app import app
import json

engine = Engine(app)


@engine.define
def hello(**params):
    if "name" in params:
        return "Hello, {}!".format(params["name"])
    else:
        return "Hello, LeanCloud!"

@engine.define
def event(**params):

    return json.dumps({})
