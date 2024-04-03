#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Classification Credit Card Default Taiwan
# 
# <p>Data Source:  UC Irvine Machine Learning Repository</p>
# <p>Default of credit card clients.  24 features. 30K instances</p>
# <p>Customer default payments in Taiwan.</p>
# <br>
# <p>Reduced number of features used in model</p>
# <p></p>Prep for web application</p>

# ## Load Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Read csv file

df = pd.read_csv('creditcardml_cleaned.csv')

# ## Data Preparation

# Select features based on prior feature importance
df_slim = df[['LIMIT_BAL','PAY_0', 'BILL_AMT1','PAY_AMT1','PAY_AMT2','PAY_AMT3', 'PAY_AMT4', 'default']].copy()
df_slim

# ## Split dataframe into X (feature) and y (target).  

X = df_slim.drop('default', axis=1)
y= df_slim['default']

# ## Address imbalance in Default Values

from imblearn.over_sampling import RandomOverSampler

#Oversampling & fit
ros = RandomOverSampler()
X_res,y_res = ros.fit_resample(X,y)

# ## Split in Train and Test

from sklearn.model_selection import train_test_split

# Split the data into training and test sets
# set random_state so that train data will be constant For every run
# test_size = 0.2.  20% of data will be used for testing, 80% for training

X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size = 0.33, random_state = 42)

# ## Model Build

from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import RobustScaler
#from sklearn.pipeline import Pipeline
#from sklearn.feature_selection import VarianceThreshold # Feature selector

rfc_model = RandomForestClassifier(random_state = 42)

#rfc_model = Pipeline([('scaler', RobustScaler()),('selector', VarianceThreshold()) ,('forest', RandomForestClassifier(random_state = 42))])

# fit
rfc_model.fit(X_train, y_train)

# Predict
rfc_pred = rfc_model.predict(X_test)


# ## Save Model

import joblib
# Use the dump() function to save the model
# Compress file due to size
joblib.dump(rfc_model,'cc_rfc_model_jl.sav.bz2',compress=('bz2',2))
#joblib.dump(rfc_model,'cc_rfc_model_jl.sav')