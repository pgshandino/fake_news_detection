# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:11:10 2020

@author: Shanding Gershinen
"""


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.STOP_WORDS
from spacy.matcher import Matcher



class Lists(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
        Xt.headline = list(nlp.pipe(Xt.headline))
        
        Xt['List'] = Xt['headline'].apply(self.check_matcher)
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
