import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve

st.title("🤍 Predicting onset of Cardiovascular disease from data?")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title('Introduction')
st.subheader('Cardiovascular disease and research aim')
#first paraphraph
st.write('Cardiovascular disease (CVD) is the leading cause of death globally, accounting for' \
' approximately 38\% of global deaths (1, WHO (2025). The term CVD includes a range of diseases, such as stroke,' \
' heart attack or peripheral arterial disease. CVD is often caused by a combination of unhealthy' \
' lifestyle factors. Early detection of patients with high risk of developing CVD is therefore very' \
' important, as modifying lifestyle factors can greatly decrease the risk of developing a disease under the CVD umbrella.'
)
#Second paragraph - research aim/description of model
st.write('Through the use of data analysis and different AI/machine learning approaches, this project' \
'aims to quantify which baseline factors are associated with earlier developments of CVD, and whether this ' \
' risk can be predicted accurately through these factors.')

st.title('Methodology: Preperation of our data')
st.subheader('Data source')
st.write('The data used for our analysis and machine learning models are sourced from the Framingham Heart Study'
' (2, add reference to the datset). Add more stuff about the dataset, when, what is measured etc.')
# This is the second parapgraph about which variables we selected and why.
st.subheader('Data selection')
st.write('We selected the following features based on.... data leakage. ')
"""
- Age
- Sex (male, female)
- Systolic blood pressure
- Diastolic blood pressure
- Total cholesterol
- BMI
- Smoking status (smoker, non-smoker)
- Cigaretted smoked per day
- Diabetes (yes, no)
- Glucose levels
- Taking blood pressure medication (yes/no)
- Is hypertensive (yes/no)
- Heart rate
"""
st.write('As the target variable CVD was entered retroactively in the dataset (if a The target variable was created using the following logic:')

# 1. Load dataset from link
cvd = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
cols = [
    "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", "BMI",
    "CURSMOKE", "CIGPDAY", "DIABETES", "GLUCOSE",
    "BPMEDS", "PREVHYP", "HEARTRTE", "CVD", "TIMECVD"
    ] #PREVCHD, etc. not includes as would introduce data leakage since these are apart of CVD
st.title('References')
st.write(' 1. World Health Organisation (2025), "Cardiovascular diseases (CVDs)", pls check how to do manual reference https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)')


