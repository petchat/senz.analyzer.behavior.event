# coding: utf-8

import os

import leancloud
from wsgiref import simple_server

from app import app
from cloud import engine

from config import APP_ID, MASTER_KEY

from gevent import monkey
monkey.patch_all()

leancloud.init(APP_ID, master_key=MASTER_KEY)

application = engine


if __name__ == '__main__':
    # Be runnable locally.
    app.debug = True
    server = simple_server.make_server('0.0.0.0', 9010, application)
    server.serve_forever()
