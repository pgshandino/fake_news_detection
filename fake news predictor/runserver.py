# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:22:55 2020

@author: Shanding Gershinen
"""


"""
This script runs the FakeNews application using a development server.
"""

from os import environ
from FakeNews import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, debug=True)
