# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:29:07 2020

@author: Shanding Gershinen
"""


import requests
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.matcher import PhraseMatcher
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.STOP_WORDS

def get_animal_name():
    try:
        with open('data/animals.txt', 'r') as f:
            animals = f.read()
            animals = animals.split(' \n')[:-1]
    except:
        animals = requests.get('https://gist.githubusercontent.com/atduskgreg/3cf8ef48cb0d29cf151bedad81553a54/raw/82f142562cf50b0f6fb8010f890b2f934093553e/animals.txt').text.split('\n')
        animals = set([animal.capitalize() for animal in animals])
        animals.remove('List')
        animals = list(animals)
        with open('animals.txt', 'w') as f:
            [f.write(animal+' \n ') for animal in animals]
    return animals

animals = get_animal_name()


class Animal(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.animals = get_animal_name()
        return self
    def transform(self, X):
        Xt = X.copy()
        
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
        #Xt.headline = list(nlp.pipe(Xt.headline))
        
        
        Xt['animal'] = Xt['headline'].apply(self.animal_matcher)
        return Xt

    def animal_matcher(self, text):
        matcher = PhraseMatcher(nlp.vocab)
        doc = nlp(text)
    
        # Create pattern Doc objects and add them to the matcher
        # This is the faster version of: [nlp(country) for country in COUNTRIES]
        patterns = list(nlp.pipe(self.animals))
        matcher.add("ANIMAL", None, *patterns)
    
        # Call the matcher on the test document and print the result
        matches = matcher(doc)
        for match_id, start, end in matches:
            if doc[start:end] == []:
                return 'No Match'
            else: 
                return 'Match'
#        print([doc[start:end] for match_id, start, end in matches])