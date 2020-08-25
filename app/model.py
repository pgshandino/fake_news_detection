# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:48:26 2020

@author: Shanding Gershinen
"""


import dill
import time
from string import punctuation
import re
import gzip
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer



lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')


class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Input:
        dict, DataFrame
    
    Output:
        DataFrame
    
        
    Encode categorical integer features as a one-hot numeric array.
    Performs mulltiple transformation on a dataframe.
    
    The input to this transformer should be a dictionary ('key' : 'Value') or a 
    DataFrame containing string characcters to be transformed. Lemmentization and 
    stemming and stopwords removal are among the transformations applied to the 
    DataFrame. This creates a new DataFrame with containing cleaned text.
    
    By default, this transformer performs:
        - Removes hashtags (strings atarting with #)
        - Removes mentions (strings starting with @)
        - Removes hyperlinks (strings starting with 'http(s):// ')
        - Removes punctuation marks present in a string or document.
        - Tokenize the Dataframe.
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
        self.new_X = X.copy()
        
        if not isinstance(self.new_X, pd.DataFrame):
            self.new_X = pd.DataFrame(self.new_X)

#       Remove all hash tags from the main text data
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'#\w+','', x['text']), axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: re.sub(r'#\w+','', x['title']), axis=1)
    
#       Since mentions have already been collected in the mention colummc, mentions should be removed from the text data
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'@[a-zA-Z0-9]+','', x['text']), axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: re.sub(r'@[a-zA-Z0-9]+','', x['title']), axis=1)
    
#       remove all hyperlinks in the tweets
        self.new_X['text'] = self.new_X.apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', x['text']), axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', x['title']), axis=1)
        
#         Remove punctuations from the title title and text columns
        self.new_X['title'] = self.new_X.apply(lambda x: "".join([word.lower() for word in x['title'] if word not in punctuation]), axis=1)
        self.new_X['text'] = self.new_X.apply(lambda x: "".join([word.lower() for word in x['text'] if word not in punctuation]), axis=1)
        
#       Tokenize the text data
        self.new_X['text'] = self.new_X.apply(lambda x: nltk.word_tokenize(x['text']), axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: nltk.word_tokenize(x['title']), axis=1)
    
#       Remove stop words from the text data
        self.new_X['text'] = self.new_X.apply(lambda x: [word for word in x['text'] if word not in stopwords], axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: [word for word in x['title'] if word not in stopwords], axis=1)
    
#       Lemmatize the text data
        self.new_X['text'] = self.new_X.apply(lambda x: [lemm.lemmatize(word) for word in x['text']], axis=1)
        self.new_X['text'] = self.new_X.apply(lambda x: ' '.join(x['text']), axis=1)
        
        self.new_X['title'] = self.new_X.apply(lambda x: [lemm.lemmatize(word) for word in x['title']], axis=1)
        self.new_X['title'] = self.new_X.apply(lambda x: ' '.join(x['title']), axis=1)
        
        
        return self.new_X

def fit_model(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        model = func()
        model.fit(X_train, y_train)
        time_elapsed = time.time() - start_time
        print('Training time : {}'.format(time_elapsed))
        print('Training accuracy : {}'.format(model.score(X_train, y_train)))
        print('Testing accuracy : {}'.format(model.score(X_test, y_test)))
        
        return model
    
    return wrapper


@fit_model
def create_model():
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    clf = LogisticRegression()
    
    pipe = Pipeline(
            steps=[
                    #('transformer', transformer),
                    ('vectorizer', vectorizer),
                    ('classifier', clf)
                    ]
            )
    return pipe

def serialize_model():
    model = create_model()
    
    with gzip.open('data/model.dill.gz', 'wb') as f:
        dill.dump(model, f, recurse=True)
        
        
if __name__ == '__main__':
    df = pd.read_csv('data/news.zip', compression='zip')
    print(df.columns)
    
    transformer = TextTransformer()
    df = transformer.fit_transform(df)
    
    X = df.text
    y = df.label
    grid_params = dict(classifier__penalty = ['l1', 'l2'], 
                   classifier__C = [0.0001, 0.001, 0.01, 0.1, 10, 100, 1000],
                   classifier__solver = ['liblinear', 'saga']
                   )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=0)
    
    model = create_model()
    
    
    kfold = StratifiedKFold(n_splits=10,
                             random_state=1).split(X_train, y_train)
    
    gs = GridSearchCV(estimator=model, 
                      param_grid=grid_params,
                      scoring='accuracy', 
                      cv=10)
    
    scores = cross_val_score(estimator=gs, 
                             X=X_train,
                             y=y_train,
                             cv=10, 
                             n_jobs=-1)
    
    
    serialize_model( )
    
    