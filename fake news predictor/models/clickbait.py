# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:13:54 2020

@author: Shanding Gershinen
"""


import dill
import time
import gzip
import pandas as pd
from transformers import clean, YIWT, List, animal, shock, mapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 


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
    Tr = clean.Clean()
    print(X_train)
    
    pipe = Pipeline(
            steps=[
                    ('clean', Tr),
                    ('vectorizer', vectorizer),
                    ('classifier', clf)
                    ]
            )
    return pipe


@fit_model
def create_model_2():
    
    youI = YIWT.YouOrI()
    lists = List.Lists()
    animals = animal.Animal()
    shocks = shock.Shock()
    clf = LogisticRegression()
    mapp = mapper.BinaryMapper()
    
    pipe = Pipeline(
            steps=[
                    ('Animal', animals),
                    ('List', lisits),
                    ('Shock', shocks),
                    ('YouI', youI),
                    ('mapper', mapp),
                    ('classifier', clf)
                    ]
            )
    return pipe


def split_data(data, t_size=0.2, r_state=42):
    
    #print('Cleaning Data...')
    #transformer = Clean.TextTransformer()
    #data = transformer.fit_transform(data)
    
    X = data.headline
    y = data.clickbait
    
    print('Done cleaning data')
    print('')
    print('Splitting data into train and test sets')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=r_state)
    
    print('splitting data into train and test size completed.')
    print('********** Info *********')
    print('Data size: {}'.format(len(X)))
    print('Train size: {}% equivalent to {}'.format(100-(t_size*100), len(X_train)))
    print('Test size: {}% equivalent to {}'.format(t_size*100, len(X_test)))
    return X_train, X_test, y_train, y_test



def serialize_model():
    model = create_model_2()
    
    with gzip.open('data/clickbait2_model.dill.gz', 'wb') as f:
        dill.dump(model, f, recurse=True)
        
        
def eval_model():
    with gzip.open('data/clickbait_model.dill.gz', 'rb') as f:
        model = dill.load(f)
    
        
if __name__ == '__main__':
    print('Loading data')
    df = pd.read_csv('data/609158_1090983_compressed_clickbait_data.csv.zip', compression='zip')
    print('Done Loading Data')
    print(df.columns)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    
    
    
    model = serialize_model()
    
    '''
    #hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
    
    
    grid_params = dict(classifier__penalty = ['l1', 'l2'], 
                   classifier__C = [0.0001, 0.001, 0.01, 0.1, 10, 100, 1000],
                   classifier__solver = ['liblinear', 'saga']
                   )
    
    
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
    '''