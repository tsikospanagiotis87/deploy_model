#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pickle


# ---- Train the Model on Train DataSet

# In[2]:


df_train = pd.read_csv('df_full_train')


# In[3]:


train_features = ['paymentmethod', 'contract', 'techsupport', 'onlinesecurity', 'internetservice', 'tenure', 'monthlycharges', 'totalcharges', 'churn']


# In[4]:


df_train = df_train[train_features]


# In[5]:


df_train


# In[6]:


ct = ColumnTransformer(transformers= [
    ('ohe', OneHotEncoder(sparse_output= False, handle_unknown= 'ignore', drop= 'first'), ['paymentmethod', 'contract', 'techsupport', 'onlinesecurity', 'internetservice']),
    ('scaler', StandardScaler(), ['tenure', 'monthlycharges', 'totalcharges'])],
                       verbose_feature_names_out= False
                      )


# In[7]:


X_train = ct.fit_transform(df_train)
y_train = df_train['churn'].values


# In[8]:


model = LogisticRegression(C= 1, max_iter= 1000)


# In[9]:


model.fit(X_train, y_train)


# ---- Save the Model in a binary file

# In[10]:


C = 1.0
model_file = f'churn_model_C={C}.bin'


# In[11]:


with open(model_file, 'wb') as file:
    pickle.dump((ct, model), file)

