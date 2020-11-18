# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:37:44 2020

@author: Shanding Gershinen
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import requests
# Import the PhraseMatcher and initialize it
from spacy.matcher import PhraseMatcher

import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.STOP_WORDS
from spacy.matcher import Matcher



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



class CheckList(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        Xt['check_list'] = Xt['headline'].apply(self.check_matcher)
        return Xt
    def check_matcher(self, doc):
        doc_ = nlp(doc)
        matcher_ = Matcher(nlp.vocab)
        pattern_1 = [{'POS':'NUM'}, {'POS':'NOUN'}, {'POS':'DET'}]
        pattern_2 = [{'POS':'NUM'}, {'POS':'NOUN'}, {'POS':'ADJ'}]
        matcher_.add('check_list', None, pattern_1, pattern_2)
        matches_ = matcher_(doc_)

        if matches_ == []:
            return 'No match'
        else:
            for matchId, start, end in matches_:
                string_id = nlp.vocab.strings[matchId]
                span = doc_[start:end]
#                 print(matchId, string_id, start, end, span.text)
                return 'Match'


class CheckYouOrI(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        Xt['check_you_i'] = Xt['headline'].apply(self.check_matcher)
        return Xt
    def check_matcher(self, doc):
        doc_ = nlp(doc)
        matcher_ = Matcher(nlp.vocab)
        pattern_1 = [{'LOWER':'you'}]
        pattern_2 = [{'LOWER':'i'}]
        pattern_3 = [{'LOWER':'why'}]
        pattern_4 = [{'LOWER':'this'}]
        pattern_5 = [{'LOWER':'my'}]
        pattern_6 = [{'LOWER':'why'}, {'LOWER':'this'}]
        matcher_.add('check_you_i', None, pattern_1, pattern_2, pattern_3, pattern_4, pattern_5, pattern_6)
        matches_ = matcher_(doc_)

        if matches_ == []:
            return 'No match'
        else:
            for matchId, start, end in matches_:
                string_id = nlp.vocab.strings[matchId]
                span = doc_[start:end]
#                 print(matchId, string_id, start, end, span.text)
                return 'Match'
            
            

class Animal(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        Xt['animal'] = Xt['headline'].apply(self.check_matcher)
        return Xt
    def check_matcher(self, doc):
        matcher = PhraseMatcher(nlp.vocab)
        doc = nlp(doc)
    
        # Create pattern Doc objects and add them to the matcher
        # This is the faster version of: [nlp(country) for country in COUNTRIES]
        patterns = list(nlp.pipe(animals))
        matcher.add("ANIMAL", None, *patterns)
    
        # Call the matcher on the test document and print the result
        matches_ = matcher(doc)
        print([doc[start:end] for match_id, start, end in matches_])
    
        if matches_ == []:
            return 'No match'
        else:
            for matchId, start, end in matches_:
                string_id = nlp.vocab.strings[matchId]
                span = doc_[start:end]
#                 print(matchId, string_id, start, end, span.text)
                return 'Match'
            
            
