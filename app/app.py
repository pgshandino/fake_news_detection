# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:48:00 2020

@author: Shanding Gershinen
"""


import gzip
import dill
from flask import Flask


app = Flask(__name__)


@app.route('/')
def index():
    
    with gzip.open('data\model.dill.gz', 'rb') as f:
        model = dill.load(f)
    return 'Accuracy is {}'.format(model.predict(['I think I am going to love it here']))


if __name__ =='__main__':
    
    app.run(debug=True)