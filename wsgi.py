# coding: utf-8

import os

import leancloud
from wsgiref import simple_server

from app import app
from cloud import engine

# from event_analyzer_lib.test import testFunction
from event_analyzer_lib.core import *
APP_ID = os.environ['LC_APP_ID']
MASTER_KEY = os.environ['LC_APP_MASTER_KEY']
PORT = int(os.environ['LC_APP_PORT'])


leancloud.init(APP_ID, master_key=MASTER_KEY)

application = engine


if __name__ == '__main__':
    # Be runnable locally.
    print rebuildEvent(event_type="shopping#mall", algo_type="GMMHMM")
    app.debug = True
    server = simple_server.make_server('localhost', PORT, application)
    server.serve_forever()
