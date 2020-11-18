# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:56:31 2020

@author: Shanding Gershinen
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:16:20 2020

@author: Shanding Gershinen
"""


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.STOP_WORDS
from spacy.matcher import Matcher

class Shock(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        Xt = X.copy()
        
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt)
        Xt.headline = list(nlp.pipe(Xt.headline))
        
        Xt['Shock'] = Xt['headline'].apply(self.check_matcher)
        return Xt
    def check_matcher(self, doc):
        doc_ = nlp(doc)
        matcher_ = Matcher(nlp.vocab)
        pattern_1 = [{'LOWER':'why'}]
        pattern_2 = [{'LOWER':'this'}]
        pattern_3 = [{'LOWER':'why'}, {'LOWER':'this'}]
        matcher_.add('shock', None, pattern_1, pattern_2, pattern_3)

        if matches_ == []:
            return 'No match'
        else:
            for matchId, start, end in matches_:
                string_id = nlp.vocab.strings[matchId]
                span = doc_[start:end]
#                 print(matchId, string_id, start, end, span.text)
                return 'Match'