# Credit Card Default Taiwan
# Machine Learning Classification Model
# Random Forest Classifier 

# Load Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier


# Write Title for Streamlit App
st.write("""
# Simple Prediction App 

This app predicts the default of credit card clients 

* **Data Source**: UCI Machine Learning Repository 
* **Dataset**: [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **Classification Model**:  RandomForestClassifier
""")

# Write Streamlit Sidebar Title
st.sidebar.header('User Input Parameters')

# Define User Input Feature Function
def user_input_features():
    LIMIT_BAL = st.sidebar.number_input("Credit Card Limit (NT$)",min_value = 10000, max_value=1000000,value=167484,step=1000)
   
    PAY_0 = st.sidebar.number_input('Current Payment Status',min_value=-2, max_value = 8, value = -1, step = 1)
                                    
    BILL_AMT1 = st.sidebar.number_input('Current Statement Balance (NT$)', min_value=-165580, max_value=964511,value=51223,step=1000)
                                        
    PAY_AMT1 = st.sidebar.number_input('Amount of Previous Payment (NT$)',min_value = 0, max_value=873552, value=5663, step=1000)                         
    PAY_AMT2 = st.sidebar.number_input('Amount of Payment 2 months prior (NT$)',min_value=0,max_value=1684259,value=5921,step=1000)
   
    PAY_AMT3 = st.sidebar.number_input('Amount of Payment 3 months prior (NT$)',min_value=0,max_value=896040,value=5225,step=1000)
   
    PAY_AMT4 = st.sidebar.number_input('Amount of Payment 4 months prior (NT$)',min_value=0,max_value=621000,value=4826,step=1000)

    data = {'LIMIT_BAL':LIMIT_BAL,
            'PAY_0': PAY_0,
            'BILL_AMT1': BILL_AMT1,
            'PAY_AMT1': PAY_AMT1,
            'PAY_AMT2': PAY_AMT2,
            'PAY_AMT3': PAY_AMT3,
            'PAY_AMT4': PAY_AMT4
            }

    features = pd.DataFrame(data, index=[0])
    return features 

# Capture user selected features from sidebar
df_input = user_input_features()

# Write User Input Parameters
st.subheader('User Input Parameters')
st.write(df_input)
st.write("___")  

# Load Model
# Cache resource so that is loads once
@st.cache_resource
def load_model():
    model = joblib.load('cc_rfc_model_jl.sav.bz2')
    return model

load_joblib_model = load_model()


#load_joblib_model = joblib.load('cc_rfc_model_jl.sav')

# Predict with model
prediction = load_joblib_model.predict(df_input)
predict_proba = load_joblib_model.predict_proba(df_input)

# List possible outcomes with labels
st.subheader('Outcome labels and corresponding index')

labels = np.array(['No Default','Default'])
st.write(pd.DataFrame(labels))

# List Prediction outcome
st.subheader('Prediction of Default')
st.write(labels[prediction])

# List Prediction Probability
st.subheader('Prediction Probability')
st.write(predict_proba)