#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import gzip
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['savefig.dpi'] = 144
sns.set()


# In[2]:


import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix


lemm = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')


# In[3]:


import re
from string import punctuation


# In[4]:


with zipfile.ZipFile('news.zip') as f:
    df = pd.read_csv(f.open('news.csv'))
#     pd.read_csv(f)



real_pct = (len(df.loc[df.label == 'REAL'])/len(df))*100
fake_pct = (len(df.loc[df.label == 'FAKE'])/len(df))*100
print('Percentage of Real news in dataset = {}\nPercentage of Fake news in dataset = {}'.format(real_pct, fake_pct))


# In[10]:


class TextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        self.new_X = X.copy()
        
        if not isinstance(self.new_X, pd.DataFrame):
            self.new_X = pd.DataFrame(self.new_X)
        
#       Create a new feature mention, representing the names mentioned in the text denotede by any word which is preeeded by the @ symbol
        self.new_X['mention'] = self.new_X.apply(lambda x: re.findall(r'@[a-zA-Z0-9]+', x['text']), axis=1)
        self.new_X['title_mention'] = self.new_X.apply(lambda x: re.findall(r'@[a-zA-Z0-9]+', x['title']), axis=1)
    
#       Create a new feature to store the hash tags
        self.new_X['hashtag'] = self.new_X.apply(lambda x: re.findall(r'#\w+', x['text']), axis=1)
        self.new_X['hashtag'] = self.new_X.apply(lambda x: ' '.join(x['hashtag']), axis=1)
        
        self.new_X['title_hashtag'] = self.new_X.apply(lambda x: re.findall(r'#\w+', x['title']), axis=1)
        self.new_X['title_hashtag'] = self.new_X.apply(lambda x: ' '.join(x['title_hashtag']), axis=1)
        
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


# In[33]:


tt = TextTransformer()
data = tt.fit_transform(df)


# In[35]:


tfidf = TfidfVectorizer(ngram_range=(1, 2))
X_text = tfidf.fit_transform(data.text)
X_title = tfidf.fit_transform(data.title)


# In[36]:


y = df.label
# X = X.toarray()


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)


# In[46]:


lr = LogisticRegression()


# In[47]:


lr.fit(X_train, y_train)


# In[48]:


y_pred = lr.predict(X_test)


# In[ ]:





# In[49]:


score = accuracy_score(y_test, y_pred)


# In[50]:


print(score)


# In[43]:


print(score)


# In[44]:


confusion_matrix(y_test, y_pred, labels=y.unique())


# In[57]:


# lr.predict(tfidf.fit_transform(['I think i an gonna like it here']))


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# In[31]:


transform = TextTransformer()
tfidf = TfidfVectorizer(ngram_range=(1, 2))
model = LogisticRegression()


# In[32]:


pipe = Pipeline(steps=[
    ('transformer', transform),
    ('vectorizer', tfidf),
    ('model', model)
])


# In[33]:


pipe.fit(X_train, y_train)


# In[34]:


pred = pipe.predict(X_test)


# In[35]:


pipe.predict(['I think I am going to love it here!'])


# In[37]:


accuracy_score(y_test, pred)


# In[64]:


abc = AdaBoostClassifier()
bag = BaggingClassifier()
gbc = GradientBoostingClassifier()
rfc = RandomForestClassifier()
lr = LogisticRegression()


# In[65]:


lr.fit(X_train, y_train)


# In[68]:


lr_pred = lr.predict(X_test)


# In[69]:


abc.fit(X_train, y_train)


# In[70]:


abc_pred = abc.predict(X_test)


# In[72]:


bag.fit(X_train, y_train)


# In[73]:


bag_pred = bag.predict(X_test)


# In[74]:


gbc.fit(X_train, y_train)


# In[75]:


gbc_pred = gbc.predict(X_test)


# In[76]:


rfc.fit(X_train, y_train)


# In[77]:


rfc_pred = rfc.predict(X_test)


# In[79]:


(accuracy_score(y_test, lr_pred),
accuracy_score(y_test, abc_pred), 
 accuracy_score(y_test, bag_pred), 
 accuracy_score(y_test, rfc_pred))


# In[82]:


print(' Logistic Regression : {} \n AdaBoostClassifier : {} \n BaggingClassifier : {} \n GradientBoostingClassifier : {} \n RandomForestClassifier : {}'.format(accuracy_score(y_test, lr_pred),accuracy_score(y_test, abc_pred), accuracy_score(y_test, bag_pred), accuracy_score(y_test, gbc_pred), accuracy_score(y_test, rfc_pred)))


# In[ ]:





