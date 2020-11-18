# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:50:02 2020

@author: Shanding Gershinen
"""


from sklearn.base import BaseEstimator, TransformerMixin

class BinaryMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        
        pattern = {'No match': 0, 'Match': 1}
        Xt.to_csv('data/EDA_Data.csv', compression='zip')
        for column in Xt.columns:
            if Xt[column].nunique() == 2:
                Xt[column].map(pattern)
        
        return Xt
    
    
    