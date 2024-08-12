# Default of Credit Card Clients Web Application
This is a web application that predicts the default of credit card clients.

[Default of Credit Card Clients Web Application](https://turbo-guacamole-gdvxswlcaaf48pasuqiiec.streamlit.app/)

## Project Overview
- This project uses the Credit Card Machine Learning Classification project to create a web application.
- [Credit Card Machine Learning Classification](https://github.com/Sarah269/glowing-dollop/tree/main/Credit%20Card%20Machine%20Learning)

## Data Source
- UCI Machine Learning Repository
  - Dataset:  [Default of Credit Card Clients](https://archive.ics.uci.edu/datasets?search=Default%20of%20Credit%20Card%20Clients)
 
## Tools
- Anaconda
- Python
  - Pandas
  - RandomForestclassifier
  - Joblib
  - Streamlit
- Replit
 
## Data Preparation
- This dataset has 23 features and 30000 observations.  Based on prior analysis, the features were reduced to 8 with the same accuracy score as the 23 features.
- This app loads a data model instead of regenerating the model after each parameter change.

## User Story
As an analyst,  I want to input parameters related to a credit card account into an application,
so that I can get a prediction on whether the credit card holder is likely to default on their payments.

  Acceptance Criteria:
  1.  The application allows me to select or input appropriate values for each parameter.
  2.  Upon submission of the parameters, the application processes the input and generates a prediction.
  3.  The prediction is displayed on the screen, indicating whether the credit card holder is likely to default.

As an analyst, I want to see the probability of the prediction's accuracy alongside the prediction itself,
so that I can better understand the confidence level of the model's output and make informed decisions.

  Acceptance Criteria:
  1.  After submitting the parameters, the application displays a prediction result indicating the likelihood of default.
  2.  Alongside the prediction, the application provides a probability score representing the accuracy of the prediction.

## Reference
- [Dataprofessor: Build 12 Data Science Apps with Python and Streamlit](https://www.youtube.com/watch?v=JwSS70SZdyM)
- [Dataprofessor Population Dashboard](https://github.com/dataprofessor/population-dashboard/tree/master)
- [aloofness T: Dump Large Datasets and Machine Learning Models with Joblib and Pickle](https://aloofness54.medium.com/dump-large-datasets-and-machine-learning-models-with-joblib-and-pickle-9fb73970114a)

  
  
