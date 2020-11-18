# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 13:34:25 2020

@author: Shanding Gershinen
"""


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from string import punctuation
import spacy

nlp = spacy.load('en_core_web_md')
stopwords = spacy.lang.en.STOP_WORDS


class Clean(BaseEstimator, TransformerMixin):
    """
    Input:
        dict, DataFrame
    
    Output:
        DataFrame
    
        
    Clean text data.
    Performs mulltiple transformation on a dataframe.
    
    The input to this transformer should be a dictionary ('key' : 'Value') or a 
    DataFrame containing string characcters to be transformed. Lemmentization and 
    stemming and stopwords removal are among the transformations applied to the 
    DataFrame. This creates a new DataFrame with containing cleaned text.
    
    By default, this transformer performs:
        - Removes punctuation marks present in a string or document.
        - Remove stopwords from the string or document.
        - Lemmentize the string or document. 
    After successfull compleetion of the transformation, a new DataFrame is 
    returned.
    
    The dictionary fed to the transformer MUST have a 'title' and 'text' keys 
    while a DataFrame must contain 'text' and 'title' columns. 
    
    
    Examples
    --------
    Given a dataset with two features, we let the transformer clean and transform
    the data into a clean DataFrame.
    
    tt = TextTransformer()
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    tt.fit(X)

    """

    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        
        return self
    
    def transform(self, X):
        Xt = X.copy()
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
        Xt.headline = list(nlp.pipe(Xt.headline))
        
#       Remove stop words from the text data
        Xt.headline = Xt.headline.apply(lambda row: [token.text for token in row if not token.is_stop])

#       Lower text
        Xt.headline = Xt.headline.apply(lambda row: [token.lower() for token in row])

#       Remove punctuations from the title title and text columns
        Xt.headline = Xt.headline.apply(lambda row: nlp(' '.join([token for token in row if not token in punctuation])))


#       Lemmatize the text data
        Xt.headline = Xt.headline.apply(lambda row: ' '.join([token.lemma_ for token in row]))
        
        
        return Xt.headline
